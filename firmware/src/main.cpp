// ==============================================
// Standard includes
// ==============================================
#include <stdint.h>
#include <stdbool.h>

// ==============================================
// Hardware specific includes
// ==============================================
#include <avr/io.h>
#include <avr/interrupt.h>

// ==============================================
// Debug only (remove when porting)
// ==============================================
#include <Arduino.h>  // Only for Serial debugging


// ==============================================
// Pin Definitions
// Change for different hardware
// ==============================================

// SCK Clock Pin (must be Timer1 OC1A pin on ATmega328P = pin 9)
#define PIN_SCK         9       // Arduino pin number (for pinMode)
#define SCK_PORT        PORTB   // Port register for writing
#define SCK_PIN         PINB    // Pin register for reading
#define SCK_DDR         DDRB    // Data direction register
#define SCK_BIT         1       // Bit position within port (pin 9 = PORTB bit 1)

// SI Signal Typewriter In — data we send to other device
#define PIN_SI          4       // Arduino pin number (for pinMode)
#define SI_PORT         PORTD   // Port register for writing
#define SI_PIN          PIND    // Pin register for reading
#define SI_DDR          DDRD    // Data direction register
#define SI_BIT          4       // Bit position within port (pin 4 = PORTD bit 4)

// SO Signal Typewriter Out — data we receive from other device
#define PIN_SO          5       // Arduino pin number (for pinMode)
#define SO_PORT         PORTD   // Port register for writing
#define SO_PIN          PIND    // Pin register for reading
#define SO_DDR          DDRD    // Data direction register
#define SO_BIT          5       // Bit position within port (pin 5 = PORTD bit 5)

// READY for Transmission — Interface (Arduino) pulls LOW to Signal Transmission
#define PIN_READY       6       // Arduino pin number (for pinMode)
#define READY_PORT      PORTD   // Port register for writing
#define READY_PIN       PIND    // Pin register for reading
#define READY_DDR       DDRD    // Data direction register
#define READY_BIT       6       // Bit position (pin 6 = PORTD bit 6)

// KBRQ  Keyboard Request — Typewriter pulls HIGH to request transmission
// Uses external interrupts INT0
#define PIN_KBRQ        2       // Arduino pin number (INT0)
#define KBRQ_PORT       PORTD   // Port register for writing (pull-up)
#define KBRQ_PIN        PIND    // Pin register for reading
#define KBRQ_DDR        DDRD    // Data direction register
#define KBRQ_BIT        2       // Bit position (pin 2 = PORTD bit 2)

// KBACK Keyboard Acknowledge, typewriter resets HIGH when done processing Trasmission
// Uses external interrupts INT1
#define PIN_KBACK       3       // Arduino pin number (INT1)
#define KBACK_PORT      PORTD   // Port register for writing (pull-up)
#define KBACK_PIN       PIND    // Pin register for reading
#define KBACK_DDR       DDRD    // Data direction register
#define KBACK_BIT       3       // Bit position (pin 3 = PORTD bit 3)

// ==============================================
// Debug Pin Definitions
// ==============================================

#define DEBUG_STATE_PORT    PORTB
#define DEBUG_STATE_DDR     DDRB
#define DEBUG_STATE_BIT     0       // Pin 8 — toggles on state change

#define DEBUG_FLAG_PORT     PORTB
#define DEBUG_FLAG_DDR      DDRB
#define DEBUG_FLAG_BIT      4       // Pin 12 — HIGH when kbrqRising is true

#define DEBUG_SI_PORT       PORTB
#define DEBUG_SI_DDR        DDRB
#define DEBUG_SI_BIT        5       // Pin 13 — HIGH during SI path

#define DEBUG_SO_PORT       PORTB
#define DEBUG_SO_DDR        DDRB
#define DEBUG_SO_BIT        2       // Pin 10 — HIGH during SO path

// ==============================================
// External Interrupt Definitions
// ==============================================
//
// INT0 (pin 2) — KBRQ rising edge
// INT1 (pin 3) — KBACK rising edge
//
// Change for different hardware
//

// Interrupt control registers (ATmega328P)
#define EXT_INT_CONTROL     EICRA   // External Interrupt Control Register A
#define EXT_INT_MASK        EIMSK   // External Interrupt Mask Register

// INT0 — KBRQ
#define INT0_ENABLE         INT0    // Bit in EIMSK to enable INT0
#define INT0_MODE_BIT0      ISC00   // Interrupt sense control bit 0
#define INT0_MODE_BIT1      ISC01   // Interrupt sense control bit 1

// INT1 — KBACK
#define INT1_ENABLE         INT1    // Bit in EIMSK to enable INT1
#define INT1_MODE_BIT0      ISC10   // Interrupt sense control bit 0
#define INT1_MODE_BIT1      ISC11   // Interrupt sense control bit 1

// Mode settings (for reference)
// 00 = low level
// 01 = any edge
// 10 = falling edge
// 11 = rising edge

// ==============================================
// Timer Definitions — Change for different hardware
// ==============================================

// Timer1 registers (ATmega328P)
#define TIMER_CONTROL_A     TCCR1A  // Timer Control Register A
#define TIMER_CONTROL_B     TCCR1B  // Timer Control Register B
#define TIMER_COUNTER       TCNT1   // The actual counter value
#define TIMER_COMPARE       OCR1A   // Compare match target value
#define TIMER_INT_MASK      TIMSK1  // Interrupt mask register

// Timer1 bits (ATmega328P)
#define TIMER_CTC_BIT       WGM12   // CTC mode enable bit
#define TIMER_TOGGLE_BIT    COM1A0  // Toggle output on compare match
#define TIMER_PRESCALE_1    CS10    // Prescaler = 1 (also starts timer)
#define TIMER_INT_ENABLE    OCIE1A  // Compare match interrupt enable

// Timer compare value for ~78kHz
// Formula: f_signal = 16MHz / (2 × prescaler × (OCR + 1))
// 78kHz ≈ 16MHz / (2 × 1 × 103) → OCR = 102
#define TIMER_COMPARE_VALUE 102


// ==============================================
// Handshake State Machine
// ==============================================
//
// Two paths through the state machine:
// - SI path: Interface (Arduino) initiates transmission
// - SO path: Typewriter initiates transmission
//
// Only one path active at a time (READY LOW blocks KBRQ)
//

enum HandshakeState {
    // ----- Typewriter Off / Startup -----
    
    HS_TW_OFF,          // Wait for KBRQ to be LOW for 100ms (typewriter ready)
    
    // ----- Idle State -----
    
    HS_IDLE,            // READY=HIGH, waiting for KBRQ or data in siBuffer
    
    // ----- SI Path: We Initiate -----
    
    HS_SI_SYN,          // Pull READY LOW, wait ~40µs
    HS_SI_TRANSFER,     // Clock 8 bits via startTransfer()
    HS_SI_BUSY,         // Wait for KBACK to rise (typewriter processing, 100µs-500ms)
    HS_SI_FIN,          // Wait ~40µs, then pull READY HIGH
    
    // ----- SO Path: Typewriter Initiates -----
    
    HS_SO_SYN,          // KBRQ rose, wait ~150µs before responding
    HS_SO_ACK,          // Pull READY LOW, wait ~200µs
    HS_SO_TRANSFER,     // Clock 8 bits via startTransfer()
    HS_SO_BUSY,         // Wait ~200µs before releasing READY
    HS_SO_FIN           // READY HIGH, wait ~10µs for KBRQ pulse to end
};


// ==============================================
// Handshake State Variables
// ==============================================

volatile HandshakeState hsState = HS_TW_OFF;    // Current state
volatile uint32_t hsStateEnteredAt = 0;         // micros() when state was entered

// ----- Flags and State Tracking set by ISRs -----

volatile bool kbrqRising = false;           // Flag: KBRQ just went HIGH
volatile bool kbrqFalling = false;          // Flag: KBRQ just went LOW
volatile uint32_t kbrqLowSince = 0;         // micros() when KBRQ went LOW (0 if HIGH)

volatile bool kbackRising = false;          // Set by INT1 ISR when KBACK rises


// ==============================================
// Brother Serial Protocol - Clock and Data Transfer
// ==============================================
//
// Protocol Overview:
// - Clock (SCK) idles HIGH
// - Data is SET on falling edge (by both devices)
// - Data is READ on rising edge (by both devices)
// - 8 bits per transfer, MSB first
//
// Timing diagram for one byte:
//
// SCK:  ‾‾‾\___/‾‾‾\___/‾‾‾\___/‾‾‾\___/‾‾‾\___/‾‾‾\___/‾‾‾\___/‾‾‾\___/‾‾‾
//          ↓   ↑   ↓   ↑   ↓   ↑   ↓   ↑   ↓   ↑   ↓   ↑   ↓   ↑   ↓   ↑
//          S7  R7  S6  R6  S5  R5  S4  R4  S3  R3  S2  R2  S1  R1  S0  R0
//          │   │                                                       │
//          │   │                                                       └── STOP
//          │   └── Read bit 7 from other device
//          └── Set bit 7 for other device
//
// S = Set outgoing bit on SI
// R = Read incoming bit from SO


// ==============================================
// Ring Buffers for SI and SO
// ==============================================
//
// siBuffer: Bytes waiting to be sent to typewriter
// soBuffer: Bytes received from typewriter
//
// Simple ring buffer: head = write position, tail = read position
// Buffer is empty when head == tail
// Buffer is full when (head + 1) % size == tail
//

#define BUFFER_SIZE 128

volatile uint8_t siBuffer[BUFFER_SIZE];
volatile uint8_t siHead = 0;    // Write position
volatile uint8_t siTail = 0;    // Read position

volatile uint8_t soBuffer[BUFFER_SIZE];
volatile uint8_t soHead = 0;    // Write position
volatile uint8_t soTail = 0;    // Read position


// ==============================================
// Transfer state variables
// ==============================================
// volatile = compiler must always read actual value
//            because ISR can change these at any moment

volatile uint8_t dataOut = 0;           // Byte we're sending
volatile uint8_t dataIn = 0;            // Byte we're receiving
volatile uint8_t bitIndex = 0;          // Current bit position (0-7)
volatile bool transferComplete = false; // Flag: transfer finished?

// When switching from SI path to SO path, interface sends DEL (0x7F)
// For consecutive SO transmissions, interface sends 0xFF
volatile bool lastTransmissionWasSI = true;  // true = Interface (Arduino) sent last


// ==============================================
// Function prototypes
// ==============================================
void setupPins();
void setupExternalInterrupts();
void setupTimer();
void startTransfer(uint8_t byteToSend);
void stopTransfer();


void setup() {
    // Debug output — remove when porting
    Serial.begin(9600);
    Serial.println("Ready. Send a byte to transmit.");

    noInterrupts();
    
    // Set up pin output direction, pullup/pulldown and initial state
    setupPins();

    // Set up clock, counter and compare match interrupt
    setupTimer();

    interrupts();
    
    // Let hardware settle after pins configured
    delay(100);
    
    noInterrupts();

    // Set up external Interrupt for Transmission Handshake
    setupExternalInterrupts();

    interrupts();
}


// ==============================================
// Pin Configuration
// ==============================================
//
void setupPins(){
    // ==========================================
    // Pin direction setup using DDR registers
    // ==========================================
    // DDR = Data Direction Register
    // 1 = output, 0 = input
    
    SCK_DDR |= (1 << SCK_BIT);   // SCK as output
    SI_DDR  |= (1 << SI_BIT);    // SI as output
    SO_DDR  &= ~(1 << SO_BIT);   // SO as input
    READY_DDR |= (1 << READY_BIT);    // READY as output
    KBRQ_DDR  &= ~(1 << KBRQ_BIT);    // KBRQ as input
    KBACK_DDR &= ~(1 << KBACK_BIT);   // KBACK as input
    
    
    // ==========================================
    // Initial pin states & Enable internal pull-ups
    // ==========================================
    
    SCK_PORT |= (1 << SCK_BIT);  // SCK HIGH (idle state)
    SI_PORT  |= (1 << SI_BIT);   // SI HIGH (pull-up / idle)
    SO_PORT  |= (1 << SO_BIT);   // SO pull-up enabled
    READY_PORT |= (1 << READY_BIT);   // READY HIGH (idle)
    KBRQ_PORT  |= (1 << KBRQ_BIT);    // KBRQ pull-up enabled
    KBACK_PORT |= (1 << KBACK_BIT);   // KBACK pull-up enabled


    // ==========================================
    // Debug Pin direction setup and initial states
    // ==========================================

    // Debug pins as outputs
    DEBUG_STATE_DDR |= (1 << DEBUG_STATE_BIT);  // Pin 8
    DEBUG_FLAG_DDR  |= (1 << DEBUG_FLAG_BIT);   // Pin 12
    DEBUG_SI_DDR    |= (1 << DEBUG_SI_BIT);     // Pin 13
    DEBUG_SO_DDR    |= (1 << DEBUG_SO_BIT);     // Pin 10

    // Initial states LOW
    DEBUG_STATE_PORT &= ~(1 << DEBUG_STATE_BIT);
    DEBUG_FLAG_PORT  &= ~(1 << DEBUG_FLAG_BIT);
    DEBUG_SI_PORT    &= ~(1 << DEBUG_SI_BIT);
    DEBUG_SO_PORT    &= ~(1 << DEBUG_SO_BIT);
}


// ==============================================
// External Interrupt Setup
// ==============================================
//
// Configures INT0 and INT1 for rising edge detection
//

void setupExternalInterrupts() {

    // ----- INT0 (KBRQ) — Any Edge -----
    
    EXT_INT_CONTROL |= (1 << INT0_MODE_BIT0);   // 01 = any edge
    EXT_INT_CONTROL &= ~(1 << INT0_MODE_BIT1);

    // ----- INT1 (KBACK) — Rising Edge -----
    
    EXT_INT_CONTROL |= (1 << INT1_MODE_BIT1) | (1 << INT1_MODE_BIT0);
    
    // ----- Enable Both Interrupts -----
    
    EXT_INT_MASK |= (1 << INT0_ENABLE) | (1 << INT1_ENABLE);
    
}


// ==============================================
// Timer Configuration
// ==============================================
//
// Timer1 is a 16-bit counter that counts up:
// In CTC mode, we set a target (OCR1A). When counter hits target:
//   1. Counter resets to 0
//   2. Pin toggles
//   3. Interrupt fires
//
// Frequency calculation:
//   f_signal = 16MHz / (2 × prescaler × (OCR1A + 1))
//   78kHz = 16MHz / (2 × 1 × 103)
//   OCR1A = 102
//
void setupTimer() {

    // Connect timer to pin in toggle mode
    TIMER_CONTROL_A = (1 << TIMER_TOGGLE_BIT);
    
    // CTC mode, stopped (no prescaler)
    TIMER_CONTROL_B = (1 << TIMER_CTC_BIT);
    
    // Compare value for ~78kHz
    TIMER_COMPARE = TIMER_COMPARE_VALUE;
    
    // Reset counter
    TIMER_COUNTER = 0;
    
    // Enable compare match interrupt
    TIMER_INT_MASK = (1 << TIMER_INT_ENABLE);
    
}


// ==============================================
// Pin State Helpers
// ==============================================
//
// Inline functions for clean, readable pin state checks.
// Compiler optimizes these to direct port reads.
//

// ----- SCK (Clock) -----

inline bool isSCKHigh() {
    return SCK_PIN & (1 << SCK_BIT);
}

inline bool isSCKLow() {
    return !(SCK_PIN & (1 << SCK_BIT));
}

// ----- SI (Signal In — we send to typewriter) -----

inline bool isSIHigh() {
    return SI_PIN & (1 << SI_BIT);
}

inline bool isSILow() {
    return !(SI_PIN & (1 << SI_BIT));
}

inline void setSIHigh() {
    SI_PORT |= (1 << SI_BIT);
}

inline void setSILow() {
    SI_PORT &= ~(1 << SI_BIT);
}

// ----- SO (Signal Out — we receive from typewriter) -----

inline bool isSOHigh() {
    return SO_PIN & (1 << SO_BIT);
}

inline bool isSOLow() {
    return !(SO_PIN & (1 << SO_BIT));
}

// ----- READY (we control) -----

inline bool isREADYHigh() {
    return READY_PIN & (1 << READY_BIT);
}

inline bool isREADYLow() {
    return !(READY_PIN & (1 << READY_BIT));
}

inline void setREADYHigh() {
    READY_PORT |= (1 << READY_BIT);
}

inline void setREADYLow() {
    READY_PORT &= ~(1 << READY_BIT);
}

// ----- KBRQ (Keyboard Request — typewriter pulls HIGH to send) -----

inline bool isKBRQHigh() {
    return KBRQ_PIN & (1 << KBRQ_BIT);
}

inline bool isKBRQLow() {
    return !(KBRQ_PIN & (1 << KBRQ_BIT));
}

// ----- KBACK (Keyboard Acknowledge — typewriter resets HIGH when done) -----

inline bool isKBACKHigh() {
    return KBACK_PIN & (1 << KBACK_BIT);
}

inline bool isKBACKLow() {
    return !(KBACK_PIN & (1 << KBACK_BIT));
}


// ==============================================
// Ring Buffer Functions
// ==============================================

// ----- SI Buffer (outgoing to typewriter) -----

bool siBufferEmpty() {
    return siHead == siTail;
}

bool siBufferFull() {
    return ((siHead + 1) % BUFFER_SIZE) == siTail;
}

bool siBufferPush(uint8_t byte) {
    if (siBufferFull()) return false;
    siBuffer[siHead] = byte;
    siHead = (siHead + 1) % BUFFER_SIZE;
    return true;
}

bool siBufferPop(uint8_t *byte) {
    if (siBufferEmpty()) return false;
    *byte = siBuffer[siTail];
    siTail = (siTail + 1) % BUFFER_SIZE;
    return true;
}

// ----- SO Buffer (incoming from typewriter) -----

bool soBufferEmpty() {
    return soHead == soTail;
}

bool soBufferFull() {
    return ((soHead + 1) % BUFFER_SIZE) == soTail;
}

bool soBufferPush(uint8_t byte) {
    if (soBufferFull()) return false;
    soBuffer[soHead] = byte;
    soHead = (soHead + 1) % BUFFER_SIZE;
    return true;
}

bool soBufferPop(uint8_t *byte) {
    if (soBufferEmpty()) return false;
    *byte = soBuffer[soTail];
    soTail = (soTail + 1) % BUFFER_SIZE;
    return true;
}


// ==============================================
// Start a Transfer
// ==============================================
//
// What happens when we start:
//
// noInterrupts()
//     │
//     ↓ Set counter to compare value (102)
//     ↓ Start timer
//     ↓ First tick: counter matches immediately
//     ↓ SCK toggles HIGH → LOW (falling edge)
//     ↓ ISR is queued (can't run yet, interrupts disabled)
//     │
// interrupts()
//     │
//     ↓ Queued ISR fires immediately
//     ↓ We're at falling edge → Set bit 7 on SI
//
void startTransfer(uint8_t byteToSend) {
    noInterrupts();
    
    // Reset transfer state
    dataOut = byteToSend;
    dataIn = 0;
    bitIndex = 0;
    transferComplete = false;
    
    // Set counter to compare value — triggers immediately
    TIMER_COUNTER = TIMER_COMPARE_VALUE;
    
    // Start timer (prescaler 1)
    // TIMER_CONTROL_A already set in setupTimer()
    TIMER_CONTROL_B |= (1 << TIMER_PRESCALE_1);
    
    interrupts();
}


// ==============================================
// Stop a Transfer
// ==============================================
//
// Called from ISR after all 8 bits are transferred.
// At this point:
//   - SCK just went HIGH (rising edge)
//   - Clock is in correct idle state (HIGH)
//
void stopTransfer() {
    // Stop timer — clear prescaler bit
    TIMER_CONTROL_B &= ~(1 << TIMER_PRESCALE_1);
    
    transferComplete = true;
}



// ==============================================
// External Interrupt Service Routines
// ==============================================
//
// These just set flags. The state machine in pollHandshake()
// checks the flags and handles the logic.
//
//

// ----- INT0: KBRQ Any Edge -----
//
// HIGH Typewriter wants to send a byte
// OR
// HIGH > 100ms Typewriter OFF or disconnected
//

ISR(INT0_vect) {
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
// Interrupt Service Routine
// ==============================================
//
// Called automatically by hardware on every compare match.
// That means: every time SCK toggles (both edges).
//
// Falling edge (SCK is LOW after toggle):
//   → Set outgoing bit on SI
//
// Rising edge (SCK is HIGH after toggle):
//   → Read incoming bit from SO
//   → Move to next bit
//   → Stop if all 8 bits done
//
// Timeline for one byte transfer:
//
// ISR #  | SCK after | bitIndex | Action
// -------|-----------|----------|---------------------------
//    1   |   LOW     |    0     | Set bit 7 on SI
//    2   |   HIGH    |    0→1   | Read bit 7 from SO
//    3   |   LOW     |    1     | Set bit 6 on SI
//    4   |   HIGH    |    1→2   | Read bit 6 from SO
//   ...  |   ...     |   ...    | ...
//   15   |   LOW     |    7     | Set bit 0 on SI
//   16   |   HIGH    |    7→8   | Read bit 0 from SO, STOP
//
ISR(TIMER1_COMPA_vect) {
    
    // Check SCK state to determine edge type
    if (isSCKHigh()) {
        
        // ===================
        // RISING EDGE
        // SCK is HIGH
        // ===================
        // Action: Read incoming bit from SO

        // Last bit? Stop timer FIRST
        if (bitIndex == 7) {
            TIMER_CONTROL_B &= ~(1 << TIMER_PRESCALE_1);
            transferComplete = true;
        }
        
        if (isSOHigh()) {
            // SO is HIGH — set this bit in dataIn
            // 0x80 >> 0 = 0b10000000 (bit 7)
            // 0x80 >> 1 = 0b01000000 (bit 6)
            // etc.
            dataIn |= (0x80 >> bitIndex);
        }// If SO is LOW, bit stays 0 (dataIn was initialized to 0)
        
        // Move to next bit
        bitIndex++;
        
        if(bitIndex == 8){
            // Reset SI HIGH for dormant period after transmission
            setSIHigh();
        }

        
    } else {
        
        // ===================
        // FALLING EDGE
        // SCK is LOW
        // ===================
        // Action: Set outgoing bit on SI
        
        // Check if current bit in dataOut is 1 or 0
        // 0x80 >> 0 = 0b10000000 (bit 7)
        // 0x80 >> 1 = 0b01000000 (bit 6)
        // etc.
        if (dataOut & (0x80 >> bitIndex)) {
            setSIHigh();
        } else {
            setSILow();
        }
    }
}

// ==============================================
// Transmission Handshake State Machine
// ==============================================
//
// Handles both SI (Arduino sends) and SO (typewriter sends) paths.
//
// Non-blocking: checks conditions and advances state when ready.
//

void pollHandshake() {
    
    uint32_t now = micros();
    HandshakeState previousState = hsState; // Track last state for debug Pin
    
    switch (hsState) {

        // ==========================================
        // TYPEWRITER OFF — Waiting for Typewriter ON
        // ==========================================

        case HS_TW_OFF:
            // Wait for KBRQ to be stable LOW for 100ms
            if (kbrqLowSince != 0 && (now - kbrqLowSince >= 100000)) {
                // Reset KBRQ FLags before entering HS_IDLE (defensiv)
                kbrqRising = false;
                kbrqFalling = false;

                // Change state to HS_IDLE
                hsStateEnteredAt = now;
                hsState = HS_IDLE;
            }
            break;
        
        // ==========================================
        // IDLE — Waiting for something to happen
        // ==========================================
        
        case HS_IDLE:
            
            // Priority: Typewriter request first (KBRQ)
            if (kbrqRising) {
                // KBRQ rose Typewriter wants to send
                kbrqRising = false;

                // Enter SO-Path
                // Change state to HS_SO_SYN
                hsStateEnteredAt = now;
                hsState = HS_SO_SYN;
            }
            // Otherwise: Check if we have data to send
            else if (!siBufferEmpty()) {
                // There is data to send - signal that interface (Arduino) will send
                setREADYLow();      // READY LOW

                // Enter SI-Path
                // Change state to HS_SI_SYN
                hsStateEnteredAt = now;
                hsState = HS_SI_SYN;
            }
            break;
        
        // ==========================================
        // SI Path — Interface (Arduino) Initiates
        // ==========================================
        
        case HS_SI_SYN:
            // Wait ~30µs after pulling READY LOW
            if (now - hsStateEnteredAt >= 30) {
                // ~30µs after pulling READY LOW
                // Get Byte from SI-Buffer
                uint8_t byteToSend;
                siBufferPop(&byteToSend);
                // Clear flag BEFORE transfer (defensiv)
                kbackRising = false;
                // Start Transfer
                startTransfer(byteToSend);

                // Change state to HS_SI_TRANSFER
                hsStateEnteredAt = now;
                hsState = HS_SI_TRANSFER;
            }
            break;
        
        case HS_SI_TRANSFER:
            // Wait for 8-bit transfer to complete
            if (transferComplete) {
                // Transfer is complete

                // Change state to HS_SI_BUSY
                hsStateEnteredAt = now;
                hsState = HS_SI_BUSY;
            }
            break;
        
        case HS_SI_BUSY:
            // Wait for KBACK to rise (typewriter done processing)
            // Can take 100µs to 500ms
            if (kbackRising) {
                // Typewriter done processing
                kbackRising = false; // Clear Flag

                // Change state to HS_SI_FIN
                hsStateEnteredAt = now;
                hsState = HS_SI_FIN;
            }
            break;
        
        case HS_SI_FIN:
            // Wait ~40µs before releasing READY
            if (now - hsStateEnteredAt >= 40) {
                // ~40µs since Typewriter done processing
                setREADYHigh();  // READY HIGH
                // Set Transmission direction flag
                lastTransmissionWasSI = true;   // Interface sent last
                // Reset KBRQ FLags before entering HS_IDLE (defensiv)
                kbrqRising = false;
                kbrqFalling = false;

                // Change state to HS_IDLE
                hsStateEnteredAt = now;
                hsState = HS_IDLE;
            }
            break;
        
        // ==========================================
        // SO Path — Typewriter Initiates
        // ==========================================
        
        case HS_SO_SYN:
            // Wait ~100µs after KBRQ rose
            if (now - hsStateEnteredAt >= 100) {
                // ~100µs since Typewriter Request
                setREADYLow();    // READY LOW

                // Change state to HS_SO_ACK
                hsStateEnteredAt = now;
                hsState = HS_SO_ACK;
            }
            break;
        
        case HS_SO_ACK:
            // Wait ~200µs after pulling READY LOW
            if (now - hsStateEnteredAt >= 200) {
                // ~200µs after pulling acknowledging typewriter request
                // Check who initiated last Transmission and start transfer
                if (lastTransmissionWasSI) {
                    startTransfer(0x7F);  // Direction changed, send DEL
                } else {
                    startTransfer(0xFF);  // Consecutive SO Transmission, keep SI HIGH
                }

                // Change state to HS_SO_TRANSFER
                hsStateEnteredAt = now;
                hsState = HS_SO_TRANSFER;
            }
            break;
        
        case HS_SO_TRANSFER:
            // Wait for 8-bit transfer to complete
            if (transferComplete) {
                // Transfer complete
                soBufferPush(dataIn);  // Store received byte

                // Change state to HS_SO_BUSY
                hsStateEnteredAt = now;
                hsState = HS_SO_BUSY;
            }
            break;
        
        case HS_SO_BUSY:
            // Wait ~240µs before releasing READY
            if (now - hsStateEnteredAt >= 240) {
                // ~240µs since transmission complete
                // Set Ready high to signal end of transmission to typewriter
                setREADYHigh();  // READY HIGH 
                // This will Trigger the kbrqRising flag as READY has forced it LOW.
                // We clear it and kbrqFalling in HS_SO_FIN
                
                // Change state to HS_SO_FIN
                hsStateEnteredAt = now;
                hsState = HS_SO_FIN;
            }
            break;
        
        case HS_SO_FIN:
            // Wait for KBRQ to go LOW
            if (kbrqFalling) {
                // Reset KBRQ FLags before entering HS_IDLE
                kbrqFalling = false;
                kbrqRising = false;
                // Set Transmission direction flag
                lastTransmissionWasSI = false;

                // Change State to HS_IDLE
                hsStateEnteredAt = now;
                hsState = HS_IDLE;
            }
            // Timeout: KBRQ still HIGH after 100ms - typewriter offline
            else if (isKBRQHigh() && (now - hsStateEnteredAt >= 100000)) {
                // Reset KBRQ FLags before entering HS_TW_OFF (defensiv)
                kbrqRising = false;
                kbrqFalling = false;

                // Change state to HS_TW_OFF
                hsStateEnteredAt = now;
                hsState = HS_TW_OFF;
            }
            break;
    }

    // Debug: toggle pin on any state change
    if (hsState != previousState) {
        DEBUG_STATE_PORT ^= (1 << DEBUG_STATE_BIT);
    }
    
    // Debug: show kbrqRising flag state
    if (kbrqRising) {
        DEBUG_FLAG_PORT |= (1 << DEBUG_FLAG_BIT);
    } else {
        DEBUG_FLAG_PORT &= ~(1 << DEBUG_FLAG_BIT);
    }

    // Debug: show current path
    if (hsState >= HS_SI_SYN && hsState <= HS_SI_FIN) {
        DEBUG_SI_PORT |= (1 << DEBUG_SI_BIT);
        DEBUG_SO_PORT &= ~(1 << DEBUG_SO_BIT);
    } 
    else if (hsState >= HS_SO_SYN && hsState <= HS_SO_FIN) {
        DEBUG_SI_PORT &= ~(1 << DEBUG_SI_BIT);
        DEBUG_SO_PORT |= (1 << DEBUG_SO_BIT);
    }
    else {
        DEBUG_SI_PORT &= ~(1 << DEBUG_SI_BIT);
        DEBUG_SO_PORT &= ~(1 << DEBUG_SO_BIT);
    }
}


// ==============================================
// Main Loop
// ==============================================
void loop() {
    pollHandshake();
    
    // Echo anything received from typewriter to Serial
    uint8_t byte;
    if (soBufferPop(&byte)) {
        Serial.print("SO: 0x");
        Serial.println(byte, HEX);
    }
    
    // Send anything received from Serial to typewriter
    if (Serial.available()) {
        byte = Serial.read();
        siBufferPush(byte);
        Serial.print("SI: 0x");
        Serial.println(byte, HEX);
    }
}