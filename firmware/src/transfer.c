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
#include "transfer.h"
#include "hardware.h"

#include <avr/interrupt.h>

// For micros() — remove when porting away from Arduino
#include <Arduino.h>

// ==============================================
// Internal Helper Functions
// ==============================================

// ----- External Interrupt Control -----

static inline void disableExternalInterrupts() {
    EXT_INT_MASK &= ~((1 << INT0_ENABLE) | (1 << INT1_ENABLE));
}

static inline void enableExternalInterrupts() {
    EXT_INT_MASK |= (1 << INT0_ENABLE) | (1 << INT1_ENABLE);
}

// ----- Transfer Timer Control -----

static inline void startTransferTimer() {
    TIMER_CONTROL_B |= (1 << TIMER_PRESCALE_1);
}

static inline void stopTransferTimer() {
    TIMER_CONTROL_B &= ~(1 << TIMER_PRESCALE_1);
    TIMER_COUNTER = 0;
}

// ==============================================
// ISR Flags — Set by ISRs, read by state machine
// ==============================================

static volatile bool kbrqRising = false;        // Flag: KBRQ just went HIGH
static volatile bool kbrqFalling = false;       // Flag: KBRQ just went LOW
static volatile uint32_t kbrqLowSince = 0;      // micros() when KBRQ went LOW (0 if HIGH)
static volatile bool kbackRising = false;        // Flag: KBACK just went HIGH

// ==============================================
// Atomic Read Helper
// ==============================================
//
// kbrqLowSince is 32-bit, AVR reads it in 4 steps.
// INT0 ISR can modify it between reads, causing corrupted data.
// Disable external interrupts during read to prevent this.
//
static inline uint32_t readKbrqLowSince() {
    disableExternalInterrupts();
    uint32_t val = kbrqLowSince;
    enableExternalInterrupts();
    return val;
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
static volatile bool transferComplete = false;   // Flag: 8-bit transfer finished
static volatile bool clockPhase = true;          // SCK phase: true = HIGH (idle), false = LOW

// ==============================================
// Start a Bit Transfer
// ==============================================
//
// Resets bit-level state and starts the timer.
// After this, the timer ISR takes over and clocks 8 bits.
//
// noInterrupts()
//     │
//     ↓ Reset transfer state
//     ↓ Disable external interrupts
//     ↓ Start timer
//     │
// interrupts()
//     │
//     ↓ Timer counts to TIMER_COMPARE_VALUE
//     ↓ ISR fires, stops timer
//     ↓ ISR pulls SCK LOW (falling edge)
//     ↓ ISR sets bit 7 on SI
//     ↓ ISR restarts timer
//
static void startBitTransfer(uint8_t byteToSend) {
    noInterrupts();
    
    dataOut = byteToSend;
    dataIn = 0;
    bitIndex = 0;
    transferComplete = false;
    
    disableExternalInterrupts();
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
    // setSIHigh(); // Not protocol but easier to read on scope
    enableExternalInterrupts();
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
// Configures INT0 (KBRQ) for any edge and INT1 (KBACK) for rising edge.
//
static void setupExternalInterrupts() {
    // ----- INT0 (KBRQ) — Any Edge -----
    
    EXT_INT_CONTROL |= (1 << INT0_MODE_BIT0);   // 01 = any edge
    EXT_INT_CONTROL &= ~(1 << INT0_MODE_BIT1);
    
    // ----- INT1 (KBACK) — Rising Edge -----
    
    EXT_INT_CONTROL |= (1 << INT1_MODE_BIT1) | (1 << INT1_MODE_BIT0);  // 11 = rising

    // Clear any pending interrupt flags from pin setup transients
    EIFR |= (1 << INTF0) | (1 << INTF1);
    
    // ----- Enable Both Interrupts -----
    
    enableExternalInterrupts();
    
    // ----- Initialize KBRQ state -----
    // If KBRQ is already LOW, start tracking how long it's been LOW
    
    if (isKBRQLow()) {
        kbrqLowSince = micros();
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
// ISR #  | Action
// -------|------------------------------------------
//    1   | SCK HIGH→LOW, set bit 7 on SI, restart
//    2   | SCK LOW→HIGH, read bit 7 from SO, restart
//    3   | SCK HIGH→LOW, set bit 6 on SI, restart
//    4   | SCK LOW→HIGH, read bit 6 from SO, restart
//   ...  | ...
//   15   | SCK HIGH→LOW, set bit 0 on SI, restart
//   16   | SCK LOW→HIGH, read bit 0 from SO, DONE
//
ISR(TIMER1_COMPA_vect) {
    // TIMER_COUNTER reached TIMER_COMPARE_VALUE so we stop and reset the TIMER
    stopTransferTimer();
    
    if (clockPhase) {
        
        // ===================
        // FALLING EDGE
        // ===================
        // Action: Pull SCK low and toggle Phase-Flag
        setSCKLow();
        clockPhase = false;
        
        // Action: Set outgoing bit on SI
        if (dataOut & (0x80 >> bitIndex)) {
            setSIHigh();
        } else {
            setSILow();
        }
        
        // Start Timer for low phase
        startTransferTimer();
    } else {
        // ===================
        // RISING EDGE
        // ===================
        // Action: Pull SCK HIGH and toggle Phase-Flag
        setSCKHigh();
        clockPhase = true;
        
        // Read bit from SO
        if (isSOHigh()) {
            dataIn |= (0x80 >> bitIndex);
        } // If SO is LOW, bit stays 0 (dataIn was initialized to 0)
        
        // Move to next bit
        bitIndex++;
        
        if (bitIndex < 8) {
            // More bits to transfer, restart timer
            startTransferTimer();
        } else {
            // Done! Keep timer stopped
            stopBitTransfer();
        }
    }
}

// ==============================================
// External Interrupt Service Routines
// ==============================================
//
// These just set flags. The state machine in pollTransfer()
// checks the flags and handles the logic.
//

// ----- INT0: KBRQ Any Edge -----
//
// HIGH: Typewriter wants to send a byte
// LOW:  Normal idle state
//
ISR(INT0_vect) {
    DEBUG_ISR_PORT ^= (1 << DEBUG_ISR_BIT);

    if (isKBRQHigh()) {
        // KBRQ is HIGH — rising edge                                                                                                                                                                                                                                                                                                                                                              
        kbrqRising = true;
        kbrqLowSince = 0;
    } else {
        // KBRQ is LOW — falling edge
        kbrqFalling = true;
        kbrqLowSince = micros();
    }
}

// ----- INT1: KBACK Rising Edge -----
//
// Typewriter finished processing (SI path only)
//
ISR(INT1_vect) {
    kbackRising = true;
}

// ==============================================
// Public: Initialize Transfer Layer
// ==============================================

void transferInit(Transfer *ts) {
    ts->state = TS_IDLE;
    ts->stateEnteredAt = 0;
    ts->lastWasSI = true;
    ts->receivedByte = 0;
    
    setupTransferTimer();
    setupExternalInterrupts();
}

// ==============================================
// Public: Clear ISR Flags
// ==============================================
//
// Resets all ISR-set flags to their default state.
// Call after hardware settle time to discard any
// spurious edges from pin configuration transients
// or typewriter power-on noise.
//
// Must be called after transferInit() and after
// any delay used for hardware settling.
//
void transferClearFlags() {
    noInterrupts();
    kbrqRising = false;
    kbrqFalling = false;
    kbackRising = false;
    siPending = false;

    if (isKBRQLow()) {
        kbrqLowSince = micros();
    }else{
        kbrqLowSince = 0;
    }
    interrupts();
}

// ==============================================
// Public: Check Typewriter Online
// ==============================================
//
// Typewriter is considered online when KBRQ has been
// stable LOW for at least 100ms. Uses ISR data internally,
// protocol layer doesn't need to touch any hardware lines.
//
bool isTypewriterOnline() {
    uint32_t lowSince = readKbrqLowSince();
    return (lowSince != 0 
            && (micros() - lowSince >= 1500000)
            && isKBRQLow()
            && isSOLow());
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
//   SI: transferStartSI() → SYN → TRANSFER → BUSY → FIN → IDLE (SI_DONE)
//   SO: KBRQ rises → SYN → ACK → TRANSFER → BUSY → FIN → IDLE (SO_DONE)
//
TransferStatus pollTransfer(Transfer *ts) {
    
    uint32_t now = micros();
    TransferState previousState = ts->state;
    TransferStatus status = TS_STATUS_IDLE;
    
    switch (ts->state) {
        
        // ==========================================
        // IDLE — Ready for next transfer
        // ==========================================
        //
        // Entry: After SI_FIN or SO_FIN completes, or on init
        // Exit:  KBRQ rises (SO path) or siPending set (SI path)
        //
        case TS_IDLE:
        
            // ----- TRANSITION: Typewriter requests to send -----
            // Guard: KBRQ rose
            // Action: Clear flag, enter SO synchronization
            if (kbrqRising) {
                kbrqRising = false;
                ts->stateEnteredAt = now;
                ts->state = TS_SO_SYN;
                status = TS_STATUS_SO_BUSY;
                break;
            }
            
            // ----- TRANSITION: Protocol layer queued a byte -----
            // Guard: siPending flag set
            // Action: Pull READY LOW, enter SI synchronization
            if (siPending) {
                siPending = false;
                dataOut = siPendingByte;
                setREADYLow();
                ts->stateEnteredAt = now;
                ts->state = TS_SI_SYN;
                status = TS_STATUS_SI_BUSY;
                break;
            }
            
            // No transition
            status = TS_STATUS_IDLE;
            break;
        
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
                kbackRising = false;
                startBitTransfer(dataOut);
                ts->stateEnteredAt = now;
                ts->state = TS_SI_TRANSFER;
                status = TS_STATUS_SI_BUSY;
                break;
            }
            
            // No transition
            status = TS_STATUS_SI_BUSY;
            break;
        
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
                ts->stateEnteredAt = now;
                ts->state = TS_SI_BUSY;
                status = TS_STATUS_SI_BUSY;
                break;
            }
            
            // No transition
            status = TS_STATUS_SI_BUSY;
            break;
        
        // ------------------------------------------
        // TS_SI_BUSY — Waiting for typewriter acknowledgment
        // ------------------------------------------
        //
        // Entry: 8-bit transfer complete
        // Exit:  KBACK rises (100µs–5s) or 5s timeout
        //
        case TS_SI_BUSY:
        
            // ----- TRANSITION: Typewriter acknowledged -----
            // Guard: KBACK rose
            // Action: Enter finalization delay
            if (kbackRising) {
                kbackRising = false;
                ts->stateEnteredAt = now;
                ts->state = TS_SI_FIN;
                status = TS_STATUS_SI_BUSY;
                break;
            }
            
            // ----- TRANSITION: Timeout -----
            // Guard: 5s elapsed without KBACK
            // Action: Release READY, return to idle
            if (now - ts->stateEnteredAt >= 5000000) {
                kbackRising = false;
                kbrqRising = false;
                kbrqFalling = false;
                setREADYHigh();
                ts->state = TS_IDLE;
                status = TS_STATUS_TIMEOUT;
                break;
            }
            
            // No transition
            status = TS_STATUS_SI_BUSY;
            break;
        
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
            // Action: Release READY, clear flags, signal SI_DONE
            if (now - ts->stateEnteredAt >= 40) {
                setREADYHigh();
                ts->lastWasSI = true;
                kbrqRising = false;
                kbrqFalling = false;
                ts->state = TS_IDLE;
                status = TS_STATUS_SI_DONE;
                break;
            }
            
            // No transition
            status = TS_STATUS_SI_BUSY;
            break;
        
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
                ts->stateEnteredAt = now;
                ts->state = TS_SO_ACK;
                status = TS_STATUS_SO_BUSY;
                break;
            }
            
            // No transition
            status = TS_STATUS_SO_BUSY;
            break;
        
        // ------------------------------------------
        // TS_SO_ACK — Acknowledge delay
        // ------------------------------------------
        //
        // Entry: READY just pulled LOW
        // Exit:  ~200µs elapsed
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
                ts->stateEnteredAt = now;
                ts->state = TS_SO_TRANSFER;
                status = TS_STATUS_SO_BUSY;
                break;
            }
            
            // No transition
            status = TS_STATUS_SO_BUSY;
            break;
        
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
                ts->stateEnteredAt = now;
                ts->state = TS_SO_BUSY;
                status = TS_STATUS_SO_BUSY;
                break;
            }
            
            // No transition
            status = TS_STATUS_SO_BUSY;
            break;
        
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
            // Action: Release READY, wait for KBRQ to fall
            if (now - ts->stateEnteredAt >= 240) {
                setREADYHigh();
                ts->stateEnteredAt = now;
                ts->state = TS_SO_FIN;
                status = TS_STATUS_SO_BUSY;
                break;
            }
            
            // No transition
            status = TS_STATUS_SO_BUSY;
            break;
        
        // ------------------------------------------
        // TS_SO_FIN — Wait for KBRQ release
        // ------------------------------------------
        //
        // Entry: READY released
        // Exit:  KBRQ falls → SO_DONE (one-shot), or 100ms timeout
        //
        case TS_SO_FIN:
        
            // ----- TRANSITION: Typewriter released KBRQ -----
            // Guard: KBRQ fell
            // Action: Clear flags, signal SO_DONE
            if (kbrqFalling) {
                kbrqFalling = false;
                kbrqRising = false;
                ts->lastWasSI = false;
                ts->state = TS_IDLE;
                status = TS_STATUS_SO_DONE;
                break;
            }
            
            // ----- TRANSITION: Timeout -----
            // Guard: 100ms elapsed with KBRQ still HIGH
            // Action: Clear flags, return to idle
            if (isKBRQHigh() && (now - ts->stateEnteredAt >= 100000)) {
                kbrqRising = false;
                kbrqFalling = false;
                ts->state = TS_IDLE;
                status = TS_STATUS_TIMEOUT;
                break;
            }
            
            // No transition
            status = TS_STATUS_SO_BUSY;
            break;
        
        // ------------------------------------------
        // DEFAULT — Should never happen
        // ------------------------------------------
        default:
            ts->state = TS_IDLE;
            status = TS_STATUS_IDLE;
            break;
    }
    
    // ==============================================
    // Debug Pin Updates
    // ==============================================
    
    // State change toggle
    if (ts->state != previousState) {
        DEBUG_ONLINE_PORT ^= (1 << DEBUG_ONLINE_BIT);
    }
    
    // SI path active
    if (ts->state >= TS_SI_SYN && ts->state <= TS_SI_FIN) {
        DEBUG_SI_PORT |= (1 << DEBUG_SI_BIT);
    } else {
        DEBUG_SI_PORT &= ~(1 << DEBUG_SI_BIT);
    }
    
    // SO path active
    if (ts->state >= TS_SO_SYN && ts->state <= TS_SO_FIN) {
        DEBUG_SO_PORT |= (1 << DEBUG_SO_BIT);
    } else {
        DEBUG_SO_PORT &= ~(1 << DEBUG_SO_BIT);
    }
    
    return status;
}
