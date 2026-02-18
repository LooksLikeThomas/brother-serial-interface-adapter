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
    uint8_t bytes[2];
    uint8_t len;          // 0 = swallowed, 1–2 = output bytes
} TranslateResult;

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
// Utility Functions
// ==============================================

// Check if a bus byte represents a key that is handled internally
// by the typewriter and should not generate serial output.
// (e.g., Relocate, Correction, Half-space)
//
bool isNonTranslatingKey(uint8_t busByte);

#ifdef __cplusplus
}
#endif

#endif // TRANSLATE_H