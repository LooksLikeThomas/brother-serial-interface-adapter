// ==============================================
// utf8.h — UTF-8 Multi-byte Sequence Handling
// ==============================================
//
// Provides helper functions to detect and measure UTF-8
// sequences. This allows the interface to safely 
// handle or discard non-ASCII characters.
//
// ==============================================
// utf8.h — UTF-8 Multi-byte Sequence Handling
// ==============================================
//
#ifndef UTF8_H
#define UTF8_H

#include <stdint.h>
#include <stdbool.h>
#include "translate.h" // Für TranslateResult

#ifdef __cplusplus
extern "C" {
#endif


// ASCII Filter
// 
// Returns true if the byte is a standard 1-byte ASCII
// character. Returns false if it is part of a UTF-8
// multi-byte sequence or an extended character.
//
static inline bool isPureAscii(uint8_t data) {
    // In UTF-8, all multi-byte components have the 
    // highest bit (0x80) set.
    return (data < 0x80);
}

// ==============================================
// UTF-8 Sequence Length Helper
// ==============================================
//
// Determines the length of a UTF-8 sequence based on the Lead Byte.
// Returns:
//   0       — Sequence is incomplete (need more bytes)
//   1       — Not a multi-byte sequence (standard ASCII 0-127)
//   2, 3, 4 — Length of the sequence if fully available
//
static inline uint8_t getUtf8Length(const uint8_t *buf, uint8_t avail) {
    uint8_t lead = buf[0];
    if (lead < 0x80) return 1;

    uint8_t expected;
    if ((lead & 0xE0) == 0xC0)      expected = 2; // 110xxxxx
    else if ((lead & 0xF0) == 0xE0) expected = 3; // 1110xxxx
    else if ((lead & 0xF8) == 0xF0) expected = 4; // 1111xxxx
    else return 1; // Malformed resync

    if (avail < expected) return 0;
    return expected;
}

// ==============================================
// UTF-8 Flattening (Transliteration)
// ==============================================
//
// Attempts to map a multi-byte UTF-8 sequence to a 
// TranslateResult. This allows mapping complex characters
// (like German umlauts) to single ASCII characters or 
// overstrike sequences (e.g., 'u' + Backspace + '"').
//
// Parameters:
//   buf  — Pointer to the start of the UTF-8 sequence
//   len  — Length of the sequence (identified by getUtf8Length)
//
// Returns:
//   TranslateResult — length 0 if no mapping exists (swallow),
//                     otherwise contains the ASCII bytes to process.
//
TranslateResult flattenUtf8(const uint8_t *buf, uint8_t len);

#ifdef __cplusplus
}
#endif

#endif // UTF8_H