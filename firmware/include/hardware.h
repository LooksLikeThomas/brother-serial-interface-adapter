// ==============================================
// hardware.h — Pin Definitions, Helpers, and Setup
// ==============================================
//
// All hardware-specific definitions for ATmega328P.
// Change these when porting to different hardware.
//
#ifndef HARDWARE_H
#define HARDWARE_H

#include <stdint.h>
#include <stdbool.h>
#include <avr/io.h>

#ifdef __cplusplus
extern "C" {
#endif

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

#define DEBUG_ONLINE_PORT   PORTB
#define DEBUG_ONLINE_DDR    DDRB
#define DEBUG_ONLINE_BIT    0       // Pin 8 — toggles on state change

#define DEBUG_ISR_PORT      PORTB
#define DEBUG_ISR_DDR       DDRB
#define DEBUG_ISR_BIT       4       // Pin 12 — toggles on every INT0 fire

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
// INT0 (pin 2) — KBRQ any edge
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
#define TIMER_CONTROL_A     TCCR1A  // Pin output mode (set to 0, we toggle manually)
#define TIMER_CONTROL_B     TCCR1B  // Timer mode and prescaler (starts/stops timer)
#define TIMER_COUNTER       TCNT1   // The actual counter value
#define TIMER_COMPARE       OCR1A   // Compare match target value
#define TIMER_INT_MASK      TIMSK1  // Interrupt enable register

// Timer1 bits (ATmega328P)
#define TIMER_CTC_BIT       WGM12   // CTC mode enable bit
#define TIMER_PRESCALE_1    CS10    // Prescaler = 1, also starts timer
#define TIMER_INT_ENABLE    OCIE1A  // Compare match interrupt enable

// Timer compare value for ~6.4µs half-period (~78kHz clock)
//
// Base formula: t_half = (OCR + 1) / 16MHz
//               6.4µs = 103 / 16MHz → OCR = 102
//
// However, the ISR adds overhead before restarting the timer:
//   - Interrupt response: ~4 cycles
//   - Register push: ~20 cycles
//   - Stop timer, toggle SCK, set SI, restart timer: ~15 cycles
//   Total: ~39 cycles = ~2.4µs
//
// To achieve 6.4µs half-period:
//   Timer portion = 6.4µs - 2.4µs = 4µs
//   OCR1A = 4µs / 62.5ns = 64
//
// Value tuned based on oscilloscope measurements.
#define TIMER_COMPARE_VALUE 48

// ==============================================
// Pin State Helpers
// ==============================================
//
// Inline functions for clean, readable pin state checks.
// Compiler optimizes these to direct port reads.
//

// ----- SCK (Clock) -----
static inline bool isSCKHigh() {
    return SCK_PIN & (1 << SCK_BIT);
}
static inline bool isSCKLow() {
    return !(SCK_PIN & (1 << SCK_BIT));
}
static inline void setSCKHigh() {
    SCK_PORT |= (1 << SCK_BIT);
}
static inline void setSCKLow() {
    SCK_PORT &= ~(1 << SCK_BIT);
}

// ----- SI (Signal In — we send to typewriter) -----
static inline bool isSIHigh() {
    return SI_PIN & (1 << SI_BIT);
}
static inline bool isSILow() {
    return !(SI_PIN & (1 << SI_BIT));
}
static inline void setSIHigh() {
    SI_PORT |= (1 << SI_BIT);
}
static inline void setSILow() {
    SI_PORT &= ~(1 << SI_BIT);
}

// ----- SO (Signal Out — we receive from typewriter) -----
static inline bool isSOHigh() {
    return SO_PIN & (1 << SO_BIT);
}
static inline bool isSOLow() {
    return !(SO_PIN & (1 << SO_BIT));
}

// ----- READY (we control) -----
static inline bool isREADYHigh() {
    return READY_PIN & (1 << READY_BIT);
}
static inline bool isREADYLow() {
    return !(READY_PIN & (1 << READY_BIT));
}
static inline void setREADYHigh() {
    READY_PORT |= (1 << READY_BIT);
}
static inline void setREADYLow() {
    READY_PORT &= ~(1 << READY_BIT);
}

// ----- KBRQ (Keyboard Request — typewriter pulls HIGH to send) -----
static inline bool isKBRQHigh() {
    return KBRQ_PIN & (1 << KBRQ_BIT);
}
static inline bool isKBRQLow() {
    return !(KBRQ_PIN & (1 << KBRQ_BIT));
}

// ----- KBACK (Keyboard Acknowledge — typewriter resets HIGH when done) -----
static inline bool isKBACKHigh() {
    return KBACK_PIN & (1 << KBACK_BIT);
}
static inline bool isKBACKLow() {
    return !(KBACK_PIN & (1 << KBACK_BIT));
}

// ==============================================
// Pin Configuration
// ==============================================
//
// Sets pin directions (DDR), initial states, pull-ups,
// and debug pin configuration.
//
static inline void setupPins() {
    // ==========================================
    // Pin direction setup using DDR registers
    // ==========================================
    // DDR = Data Direction Register
    // 1 = output, 0 = input
    
    SCK_DDR |= (1 << SCK_BIT);        // SCK as output
    SI_DDR  |= (1 << SI_BIT);         // SI as output
    SO_DDR  &= ~(1 << SO_BIT);        // SO as input
    READY_DDR |= (1 << READY_BIT);    // READY as output
    KBRQ_DDR  &= ~(1 << KBRQ_BIT);   // KBRQ as input
    KBACK_DDR &= ~(1 << KBACK_BIT);   // KBACK as input
    
    // ==========================================
    // Initial pin states & Enable internal pull-ups
    // ==========================================
    
    SCK_PORT |= (1 << SCK_BIT);       // SCK HIGH (idle state)
    SI_PORT  |= (1 << SI_BIT);        // SI HIGH (pull-up / idle)
    SO_PORT  |= (1 << SO_BIT);        // SO pull-up enabled
    READY_PORT |= (1 << READY_BIT);   // READY HIGH (idle)
    KBRQ_PORT  |= (1 << KBRQ_BIT);   // KBRQ pull-up enabled
    KBACK_PORT |= (1 << KBACK_BIT);   // KBACK pull-up enabled
    
    // ==========================================
    // Debug Pin direction setup and initial states
    // ==========================================
    
    // Debug pins as outputs
    DEBUG_ONLINE_DDR |= (1 << DEBUG_ONLINE_BIT);    // Pin 8
    DEBUG_ISR_DDR    |= (1 << DEBUG_ISR_BIT);       // Pin 12
    DEBUG_SI_DDR     |= (1 << DEBUG_SI_BIT);        // Pin 13
    DEBUG_SO_DDR     |= (1 << DEBUG_SO_BIT);        // Pin 10
    
    // Initial states LOW
    DEBUG_ONLINE_PORT &= ~(1 << DEBUG_ONLINE_BIT);
    DEBUG_ISR_PORT    &= ~(1 << DEBUG_ISR_BIT);
    DEBUG_SI_PORT     &= ~(1 << DEBUG_SI_BIT);
    DEBUG_SO_PORT     &= ~(1 << DEBUG_SO_BIT);
}

#ifdef __cplusplus
}
#endif

#endif // HARDWARE_H
