// ==============================================
// transfer.c — Single-Byte Transfer Implementation
// ==============================================
//
// Handles the complete handshake for one byte transfer:
//   SI path: Pull READY, clock 8 bits out, wait KBACK, release READY
//   SO path: Detect KBRQ, pull READY, clock 8 bits in, release READY
//
// All ISRs and hardware interaction for transfers live here.
// The protocol layer only sees TransferStatus and receivedByte.
//
// Interrupt Strategy:
//   KBRQ (INT0): Only enabled when waiting for edges (TS_IDLE, TS_SO_FIN)
//   KBACK: Polled directly in TS_SI_BUSY — no ISR needed
//
#include "transfer.h"
#include "hardware.h"

#include <avr/interrupt.h>

// For micros() — remove when porting away from Arduino
#include <Arduino.h>

// ==============================================
// Transfer Timer Control
// ==============================================

static inline void startTransferTimer() {
    TIMER_CONTROL_B |= (1 << TIMER_PRESCALE_1);
}

static inline void stopTransferTimer() {
    TIMER_CONTROL_B &= ~(1 << TIMER_PRESCALE_1);
    TIMER_COUNTER = 0;
}

// ==============================================
// ISR Flags — Set by ISR, read by state machine
// ==============================================
//
// Only KBRQ flags remain. KBACK is polled directly.
//
// These flags are only meaningful when the KBRQ interrupt is enabled:
//   - TS_IDLE: waiting for kbrqRising
//   - TS_SO_FIN: waiting for kbrqFalling
//

static volatile bool kbrqRising = false;    // Flag: KBRQ just went HIGH
static volatile bool kbrqFalling = false;   // Flag: KBRQ just went LOW

// ==============================================
// KBRQ Flag Control
// ==============================================
//
// Clear flags before enabling interrupt to ensure we only
// see edges that occur after we start watching.
//

static inline void clearKBRQFlags() {
    kbrqRising = false;
    kbrqFalling = false;
}

// ==============================================
// Bit Transfer State — Used by Timer ISR
// ==============================================
//
// These are the low-level variables for clocking bits.
// The timer ISR reads/writes these directly.
//

static volatile bool siPending = false;         // Flag: protocol layer queued a byte
static volatile uint8_t siPendingByte = 0;      // The byte waiting to be sent
static volatile uint8_t dataOut = 0;            // Byte we're sending
static volatile uint8_t dataIn = 0;             // Byte we're receiving
static volatile uint8_t bitIndex = 0;           // Current bit position (0-7)
static volatile bool transferComplete = false;  // Flag: 8-bit transfer finished
static volatile bool clockPhase = true;         // SCK phase: true = HIGH (idle), false = LOW

// ==============================================
// Start a Bit Transfer
// ==============================================
//
// Resets bit-level state and starts the timer.
// After this, the timer ISR takes over and clocks 8 bits.
//
// Note: KBRQ interrupt is already disabled when this is called
// (disabled when leaving TS_IDLE or at start of SI path).
//
static void startBitTransfer(uint8_t byteToSend) {
    noInterrupts();
    
    dataOut = byteToSend;
    dataIn = 0;
    bitIndex = 0;
    transferComplete = false;
    clockPhase = true;
    
    startTransferTimer();
    
    interrupts();
}

// ==============================================
// Stop a Bit Transfer
// ==============================================
//
// Called from timer ISR after all 8 bits are transferred.
// At this point:
//   - SCK just went HIGH (rising edge)
//   - Clock is in correct idle state (HIGH)
//
static void stopBitTransfer() {
    transferComplete = true;
}

// ==============================================
// Timer Setup
// ==============================================
//
// Timer1 is a 16-bit counter that counts up.
// In CTC mode, we set a target (OCR1A). When counter hits target:
//   1. Counter resets to 0
//   2. Interrupt fires
//   3. ISR stops timer, toggles SCK, restarts timer
//
// This "one-shot" approach prevents extra clock pulses if ISR is delayed.
//
static void setupTransferTimer() {
    // Disable pin toggle mode (we toggle SCK manually)
    TIMER_CONTROL_A = 0;
    
    // CTC mode, stopped (no prescaler yet)
    TIMER_CONTROL_B = (1 << TIMER_CTC_BIT);
    
    // Compare value for ~6.4µs half-period
    TIMER_COMPARE = TIMER_COMPARE_VALUE;
    
    // Reset counter
    TIMER_COUNTER = 0;
    
    // Enable compare match interrupt
    TIMER_INT_MASK = (1 << TIMER_INT_ENABLE);
}

// ==============================================
// External Interrupt Setup
// ==============================================
//
// Configures INT0 (KBRQ) for any edge detection.
// Interrupt starts DISABLED — only enabled in states that need it.
//
static void setupExternalInterrupts() {
    // ----- INT0 (KBRQ) — Any Edge -----
    
    EXT_INT_CONTROL |= (1 << INT0_MODE_BIT0);   // 01 = any edge
    EXT_INT_CONTROL &= ~(1 << INT0_MODE_BIT1);
    
    // Clear any pending interrupt flag from pin setup transients
    EXT_INT_FLAG |= (1 << INT0_FLAG);
    
    // Start with interrupt DISABLED
    // Will be enabled when entering TS_IDLE
    disableKBRQInterrupt();
}

// ==============================================
// State Transition Helper
// ==============================================
//
// Records timestamp for timeout tracking, Updates state, Updates debug pins
// Called by pollTransfer() on every state change.
//
static inline void transitionTo(Transfer *ts, TransferState newState) {
    ts->stateEnteredAt = micros();
    ts->state = newState;
    
    // State change toggle
    DEBUG_ONLINE_PORT ^= (1 << DEBUG_ONLINE_BIT);
    
    // SI path active
    if (newState >= TS_SI_SYN && newState <= TS_SI_FIN) {
        DEBUG_SI_PORT |= (1 << DEBUG_SI_BIT);
    } else {
        DEBUG_SI_PORT &= ~(1 << DEBUG_SI_BIT);
    }
    
    // SO path active
    if (newState >= TS_SO_SYN && newState <= TS_SO_FIN) {
        DEBUG_SO_PORT |= (1 << DEBUG_SO_BIT);
    } else {
        DEBUG_SO_PORT &= ~(1 << DEBUG_SO_BIT);
    }
}


// ==============================================
// Timer1 Compare Match ISR — Clocks One Bit
// ==============================================
//
// Called on every compare match. Timer stops at match,
// we toggle SCK manually, then restart timer if needed.
//
// This "one-shot" approach prevents extra pulses if ISR is delayed.
//
// Timeline for one byte transfer:
//
// ISR #  | Phase   | Action
// -------|---------|------------------------------------------
//    1   | FALLING | set bit 7 on SI, SCK HIGH→LOW, restart
//    2   | RISING  | read bit 7 from SO, SCK LOW→HIGH, restart
//    3   | FALLING | set bit 6 on SI, SCK HIGH→LOW, restart
//    4   | RISING  | read bit 6 from SO, SCK LOW→HIGH, restart
//   ...  | ...     | ...
//   15   | FALLING | set bit 0 on SI, SCK HIGH→LOW, restart
//   16   | RISING  | read bit 0 from SO, SCK LOW→HIGH, bitIndex=8, restart
//   17   | FALLING | bitIndex>=8, stopBitTransfer, return (SCK stays HIGH)
//
ISR(TIMER1_COMPA_vect) {
    // TIMER_COUNTER reached TIMER_COMPARE_VALUE so we stop and reset the TIMER
    stopTransferTimer();

    // Transfer complete — all 8 bits done, last bit held for one half-period
    if (bitIndex >= 8) {
        stopBitTransfer();
        return;
    }

    if (clockPhase) {
        // ===================
        // FALLING EDGE
        // ===================
        
        // Action: SCK and SI with minimal cycles between transitions.
        // The compare is done before toggling to keep SCK-SI timing tight.
        if (dataOut & (0x80 >> bitIndex)) {
            setSCKLow();
            setSIHigh();
        } else {
            setSCKLow();
            setSILow();
        }

    } else {
        // ===================
        // RISING EDGE
        // ===================

        // Action: Pull SCK HIGH and Read bit from SO
        setSCKHigh();
        if (isSOHigh()) {
            dataIn |= (0x80 >> bitIndex);
        } // If SO is LOW, bit stays 0 (dataIn was initialized to 0)

        // Move to next bit
        bitIndex++;
    }

    // Toggle Phase-Flag
    clockPhase = !clockPhase;
    
    // Restart timer for next phase
    startTransferTimer();
}


// ==============================================
// External Interrupt Service Routine
// ==============================================
//
// INT0: KBRQ Any Edge
//
// Only enabled in states that need edge detection:
//   - TS_IDLE: looking for rising edge (typewriter wants to send)
//   - TS_SO_FIN: looking for falling edge (transfer complete)
//
ISR(INT0_vect) {
    DEBUG_ISR_PORT ^= (1 << DEBUG_ISR_BIT);

    if (isKBRQHigh()) {
        // Rising edge — typewriter requests to send
        kbrqRising = true;
    } else {
        // Falling edge — typewriter released KBRQ
        kbrqFalling = true;
    }
}

// ==============================================
// Public: Initialize Transfer Layer
// ==============================================
//
// Sets up timer, configures KBRQ interrupt, and prepares
// for TS_IDLE state.
//
// Call with interrupts disabled. The KBRQ interrupt is
// enabled at the end, ready to detect typewriter requests
// once global interrupts are enabled.
//
void transferInit(Transfer *ts) {
    ts->state = TS_IDLE;
    ts->stateEnteredAt = 0;
    ts->lastWasSI = true;
    ts->receivedByte = 0;
    
    setupTransferTimer();
    setupExternalInterrupts();
    
    // Clear flags and enable KBRQ interrupt for TS_IDLE
    clearKBRQFlags();
    siPending = false;
    transferComplete = false;
    enableKBRQInterrupt();
}

// ==============================================
// Public: Check Typewriter Online
// ==============================================
//
// TODO: This will be replaced with a dedicated 5V power
// detection pin for reliable online/offline detection.
//
// Current implementation checks signal states as a proxy:
//   - KBRQ LOW: typewriter not requesting to send
//   - SO LOW: typewriter not mid-transmission
//
// This is unreliable because pull-ups can make signals
// appear valid even when typewriter is disconnected.
//
bool isTypewriterOnline() {
    return isKBRQLow() && isSOLow();
}

// ==============================================
// Public: Request SI Transfer
// ==============================================
//
// Protocol layer calls this to send a byte to typewriter.
// Only valid when transfer state is TS_IDLE.
// Pulls READY LOW and enters SI path.
//
bool transferQueueSI(Transfer *ts, uint8_t byte) {
    if (siPending) return false;  // Previous byte not yet picked up (should not happen)
    siPendingByte = byte;
    siPending = true;
    return true;
}


// ==============================================
// Public: Transfer State Machine
// ==============================================
//
// Runs the single-byte transfer handshake.
// Call every loop iteration.
//
// Returns TransferStatus so the protocol layer knows
// what's happening without touching hardware directly.
//
// Two paths:
//   SI: transferQueueSI() → SYN → TRANSFER → BUSY → FIN → IDLE (SI_DONE)
//   SO: KBRQ rises → SYN → ACK → TRANSFER → BUSY → FIN → IDLE (SO_DONE)
//
// Interrupt enable/disable points:
//   - TS_IDLE: KBRQ interrupt enabled, waiting for rising edge
//   - Leaving TS_IDLE: disable KBRQ interrupt
//   - TS_SO_BUSY → TS_SO_FIN: enable KBRQ interrupt (waiting for falling edge)
//   - TS_SI_FIN → TS_IDLE: enable KBRQ interrupt
//   - TS_SO_FIN → TS_IDLE: keep KBRQ interrupt enabled
//
TransferStatus pollTransfer(Transfer *ts) {
    
    uint32_t now = micros();
    
    switch (ts->state) {
        
        // ==========================================
        // IDLE — Ready for next transfer
        // ==========================================
        //
        // Entry: After SI_FIN or SO_FIN completes, or on init
        // Exit:  KBRQ rises (SO path) or siPending set (SI path)
        //
        // KBRQ interrupt is ENABLED in this state.
        //
        case TS_IDLE:
        
            // ----- TRANSITION: Typewriter requests to send -----
            // Guard: KBRQ rose (ISR set flag)
            // Action: Disable interrupt, enter SO synchronization
            if (kbrqRising) {
                // Disable and clear KBRQ Interrupt
                disableKBRQInterrupt();
                clearKBRQFlags();
                // Transition State
                transitionTo(ts, TS_SO_SYN);
                return TS_STATUS_SO_BUSY;
            }
            
            // ----- TRANSITION: Protocol layer queued a byte -----
            // Guard: siPending flag set
            // Action: Disable interrupt, pull READY LOW, enter SI synchronization
            if (siPending) {
                // Clear SI-Pending flag and set dataOut
                siPending = false;
                dataOut = siPendingByte;
                // Disable and clear KBRQ Interrupt
                disableKBRQInterrupt();
                clearKBRQFlags();
                // Set READY LOW
                setREADYLow();
                // Transition State
                transitionTo(ts, TS_SI_SYN);
                return TS_STATUS_SI_BUSY;
            }
            
            // No transition
            return TS_STATUS_IDLE;
        
        // ==========================================
        // SI Path — Interface (Arduino) Initiates
        // ==========================================
        
        // ------------------------------------------
        // TS_SI_SYN — Synchronization delay
        // ------------------------------------------
        //
        // Entry: READY just pulled LOW, byte ready in dataOut
        // Exit:  ~30µs elapsed
        //
        case TS_SI_SYN:
        
            // ----- TRANSITION: Sync delay elapsed -----
            // Guard: 30µs since READY went LOW
            // Action: Start 8-bit transfer via timer ISR
            if (now - ts->stateEnteredAt >= 30) {
                startBitTransfer(dataOut);
                // Transition State
                transitionTo(ts, TS_SI_TRANSFER);
                return TS_STATUS_SI_BUSY;
            }
            
            // No transition
            return TS_STATUS_SI_BUSY;
        
        // ------------------------------------------
        // TS_SI_TRANSFER — 8-bit transfer in progress
        // ------------------------------------------
        //
        // Entry: Timer ISR clocking bits out on SI
        // Exit:  transferComplete flag set by ISR
        //
        case TS_SI_TRANSFER:
        
            // ----- TRANSITION: Transfer complete -----
            // Guard: transferComplete flag set
            // Action: Enter KBACK wait state
            if (transferComplete) {
                // Transition State
                transitionTo(ts, TS_SI_BUSY);
                return TS_STATUS_SI_BUSY;
            }
            
            // No transition
            return TS_STATUS_SI_BUSY;
        
        // ------------------------------------------
        // TS_SI_BUSY — Waiting for typewriter acknowledgment
        // ------------------------------------------
        //
        // Entry: 8-bit transfer complete
        // Exit:  KBACK rises (100µs–5s) or 5s timeout
        //
        case TS_SI_BUSY:
        
            // ----- TRANSITION: Typewriter acknowledged -----
            // Guard: KBACK is HIGH (polled directly)
            // Action: Enter finalization delay
            if (isKBACKHigh()) {
                // Transition State
                transitionTo(ts, TS_SI_FIN);
                return TS_STATUS_SI_BUSY;
            }
            
            // ----- TRANSITION: Timeout -----
            // Guard: 5s elapsed without KBACK
            // Action: Enter error state, wait for protocol layer to reset
            if (now - ts->stateEnteredAt >= 5000000) {
                disableKBRQInterrupt();
                clearKBRQFlags();
                transitionTo(ts, TS_ERROR);
                return TS_STATUS_ERROR;
            }
            
            // No transition
            return TS_STATUS_SI_BUSY;
        
        // ------------------------------------------
        // TS_SI_FIN — Finalization delay
        // ------------------------------------------
        //
        // Entry: KBACK rose
        // Exit:  ~40µs elapsed → SI_DONE (one-shot)
        //
        case TS_SI_FIN:
        
            // ----- TRANSITION: Finalization complete -----
            // Guard: 40µs since KBACK rose
            // Action: Enable interrupt, release READY, signal SI_DONE
            if (now - ts->stateEnteredAt >= 40) {
                ts->lastWasSI = true;
                // Enable interrupt BEFORE releasing READY to catch any SO request
                clearKBRQFlags();
                enableKBRQInterrupt();
                // Release Ready
                setREADYHigh();
                // Transition State
                transitionTo(ts, TS_IDLE);
                return TS_STATUS_SI_DONE;
            }
            
            // No transition
            return TS_STATUS_SI_BUSY;
        
        // ==========================================
        // SO Path — Typewriter Initiates
        // ==========================================
        
        // ------------------------------------------
        // TS_SO_SYN — Synchronization delay
        // ------------------------------------------
        //
        // Entry: KBRQ rose
        // Exit:  ~100µs elapsed
        //
        case TS_SO_SYN:
        
            // ----- TRANSITION: Sync delay elapsed -----
            // Guard: 100µs since KBRQ rose
            // Action: Pull READY LOW to acknowledge
            if (now - ts->stateEnteredAt >= 100) {
                setREADYLow();
                // Transition State
                transitionTo(ts, TS_SO_ACK);
                return TS_STATUS_SO_BUSY;
            }
            
            // No transition
            return TS_STATUS_SO_BUSY;
        
        // ------------------------------------------
        // TS_SO_ACK — Acknowledge delay
        // ------------------------------------------
        //
        // Entry: READY just pulled LOW
        // Exit:  ~200µs elapsed
        //
        // KBRQ interrupt is DISABLED.
        // Note: Pulling READY LOW forces KBRQ LOW (per protocol spec).
        // We don't care about this expected edge.
        //
        case TS_SO_ACK:
        
            // ----- TRANSITION: Ack delay elapsed -----
            // Guard: 200µs since READY went LOW
            // Action: Start transfer (DEL on direction change, else 0xFF)
            if (now - ts->stateEnteredAt >= 200) {
                if (ts->lastWasSI) {
                    startBitTransfer(0x7F);
                } else {
                    startBitTransfer(0xFF);
                }
                // Transition State
                transitionTo(ts, TS_SO_TRANSFER);
                return TS_STATUS_SO_BUSY;
            }
            
            // No transition
            return TS_STATUS_SO_BUSY;
        
        // ------------------------------------------
        // TS_SO_TRANSFER — 8-bit transfer in progress
        // ------------------------------------------
        //
        // Entry: Timer ISR clocking bits in from SO
        // Exit:  transferComplete flag set by ISR
        //
        case TS_SO_TRANSFER:
        
            // ----- TRANSITION: Transfer complete -----
            // Guard: transferComplete flag set
            // Action: Store received byte, enter post-transfer delay
            if (transferComplete) {
                ts->receivedByte = dataIn;
                // Transition State
                transitionTo(ts, TS_SO_BUSY);
                return TS_STATUS_SO_BUSY;
            }
            
            // No transition
            return TS_STATUS_SO_BUSY;
        
        // ------------------------------------------
        // TS_SO_BUSY — Post-transfer delay
        // ------------------------------------------
        //
        // Entry: 8-bit transfer complete, byte stored
        // Exit:  ~240µs elapsed
        //
        case TS_SO_BUSY:
        
            // ----- TRANSITION: Post-transfer delay elapsed -----
            // Guard: 240µs since transfer completed
            // Action: Enable interrupt, release READY, wait for KBRQ to fall
            if (now - ts->stateEnteredAt >= 240) {
                // Enable interrupt and clear flags BEFORE releasing READY.
                // Per protocol: KBRQ rises briefly with READY, then typewriter
                // pulls it LOW again. We need to catch that falling edge.
                clearKBRQFlags();
                enableKBRQInterrupt();
                // Set READY HIGH: This may set kbrqRising, we clear it in TS_SO_FIN
                setREADYHigh();
                // Transition State
                transitionTo(ts, TS_SO_FIN);
                return TS_STATUS_SO_BUSY;
            }
            
            // No transition
            return TS_STATUS_SO_BUSY;
        
        // ------------------------------------------
        // TS_SO_FIN — Wait for KBRQ release
        // ------------------------------------------
        //
        // Entry: READY released
        // Exit:  KBRQ falls → SO_DONE (one-shot), or 100ms timeout
        //
        // KBRQ interrupt is ENABLED, waiting for falling edge.
        // Per protocol: KBRQ rises briefly when READY goes HIGH,
        // then typewriter pulls it LOW to signal transfer complete.
        //
        case TS_SO_FIN:
        
            // ----- TRANSITION: Typewriter set KBRQ LOW-----
            // Guard: KBRQ fell (ISR set flag) AND KBRQ is currently LOW
            // Action: Clear flags, signal SO_DONE (keep interrupt enabled for IDLE)
            if (kbrqFalling && isKBRQLow()) {
                // Clear ISR-Flags including kbrqRising wich may were set during TS_SO_BUSY
                clearKBRQFlags();
                ts->lastWasSI = false;
                // Keep interrupt enabled for TS_IDLE
                // Transition State
                transitionTo(ts, TS_IDLE);
                return TS_STATUS_SO_DONE;
            }
            
            // ----- TRANSITION: Timeout -----
            // Guard: 100ms elapsed with KBRQ still HIGH
            // Action: Enter error state, wait for protocol layer to reset
            if (now - ts->stateEnteredAt >= 100000) {
                disableKBRQInterrupt();
                clearKBRQFlags();
                transitionTo(ts, TS_ERROR);
                return TS_STATUS_ERROR;
            }
            
            // No transition
            return TS_STATUS_SO_BUSY;

            
        // ==========================================
        // ERROR — Waiting for protocol layer to reset
        // ==========================================
        //
        // Entry: Timeout in SI_BUSY or SO_FIN
        // Exit:  Protocol layer calls transferInit()
        //
        // Hardware state is frozen. Protocol layer decides
        // when and how to recover.
        //
        case TS_ERROR:
            return TS_STATUS_ERROR;

        
        // ------------------------------------------
        // DEFAULT — Should never happen
        // ------------------------------------------
        default:
            clearKBRQFlags();
            enableKBRQInterrupt();
            transitionTo(ts, TS_IDLE);
            return TS_STATUS_IDLE;
    }
}