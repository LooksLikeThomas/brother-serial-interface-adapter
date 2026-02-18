// ==============================================
// translate.h — Bus ↔ Serial Translation Module
// ==============================================
//
// Pure translation functions with no internal state.
// All state (keyboard ID, code key state, etc.) is managed 
// by the protocol layer and passed in as parameters.
//
// The translation module acts as a stateless lookup engine
// for converting typewriter bus bytes (petal positions) 
// into ASCII serial bytes.
//
#ifndef TRANSLATE_H
#define TRANSLATE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================
// Translation Result
// ==============================================
//
// Returned by all translation functions to allow for
// multi-byte serial sequences (like ESC+Y).
//
typedef struct {
    uint8_t bytes[4];
    uint8_t len;          // 0 = swallowed, 1–4 = output bytes
} TranslateResult;


// ==============================================
// Helper: Build a TranslateResult
// ==============================================

static inline TranslateResult result0(void) {
    return (TranslateResult){ .bytes = {0, 0}, .len = 0 };
}

static inline TranslateResult result1(uint8_t b) {
    return (TranslateResult){ .bytes = {b, 0}, .len = 1 };
}

static inline TranslateResult result2(uint8_t b0, uint8_t b1) {
    return (TranslateResult){ .bytes = {b0, b1}, .len = 2 };
}

static inline TranslateResult result3(uint8_t b0, uint8_t b1, uint8_t b2) {
    return (TranslateResult){ .bytes = {b0, b1, b2, 0}, .len = 3 };
}

static inline TranslateResult result4(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
    return (TranslateResult){ .bytes = {b0, b1, b2, b3}, .len = 4 };
}


// ==============================================
// Reverse Translation (Typewriter → Serial)
// ==============================================

// Translate a bus byte received from the typewriter keyboard
// into serial byte(s) for the standard (Normal/Shifted) path.
//
// Handles keyboard-dependent national variant positions, 
// Shifted number row mappings, and identity mappings for letters.
//
// Parameters:
//   busByte  — the petal position or function byte from the bus
//   keyboard — KEYBOARD_KB1, KEYBOARD_KB2, or KEYBOARD_KB3
//
TranslateResult translateNormalBusToSerial(uint8_t busByte, uint8_t keyboard);

// Translate a bus byte received from the typewriter keyboard
// into serial byte(s) for the "Code" modifier path.
//
// Handles the "Ctrl convention" (busByte & 0x1F) for letters
// and the fixed keyboard-independent mapping for the number row.
//
// Parameters:
//   busByte  — the petal position or function byte from the bus
//   keyboard — KEYBOARD_KB1, KEYBOARD_KB2, or KEYBOARD_KB3
//
TranslateResult translateCodeBusToSerial(uint8_t busByte, uint8_t keyboard);


// ==============================================
// Forward Translation (Serial → Typewriter Bus)
// ==============================================

// Translate a serial byte from the PC into bus byte(s)
// for the typewriter.
//
// Handles printable characters (0x20–0x7E) including
// keyboard-dependent remapping, Code key simulation
// (0x88/0x89 framing), and dead key compensation (trailing 0x00).
// Also handles simple control codes (BS, LF, BEL).
//
// CR, CR+LF combining, and ESC sequences are NOT handled here —
// they require protocol-level state (pitch, lookahead) and are
// handled in protocol.c.
//
// Parameters:
//   serialByte — the ASCII byte received from the PC
//   keyboard   — KEYBOARD_KB1, KEYBOARD_KB2, or KEYBOARD_KB3
//
TranslateResult translateSerialToBus(uint8_t serialByte, uint8_t keyboard);

// ==============================================
// Utility Functions
// ==============================================

// Check if a bus byte represents a key that is handled internally
// by the typewriter and should not generate serial output.
// (e.g., Relocate, Correction, Half-space)
//
bool isNonTranslatingKey(uint8_t busByte);

// Return the horizontal travel delta for a serial byte:
//   +1  printable (0x20-0x7E) — carriage moves right
//   -1  BS (0x08)             — carriage moves left
//    0  all other bytes       — no horizontal movement
//
int8_t serialHtDelta(uint8_t serialByte);

#ifdef __cplusplus
}
#endif

#endif // TRANSLATE_H