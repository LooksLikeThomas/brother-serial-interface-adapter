// ==============================================
// escape.h — ESC Sequence Identification
// ==============================================
//
// Stateless identification of ESC sequences from the
// serial input stream. The protocol layer peeks ahead
// into siBuffer and passes the bytes here for classification.
//
// This module only identifies sequences — all state changes
// and bus byte generation are handled by protocol.c.
//
#ifndef ESCAPE_H
#define ESCAPE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================
// ESC Sequence Identifiers
// ==============================================
//
// Ordering matters: non-parametric (2-byte) sequences
// MUST come before parametric (3-byte) sequences.
// escConsumeLen() relies on this boundary.
//

typedef enum {

    // --- Parser status (not a valid sequence) ---
    ESC_NEED_MORE,          // Not enough bytes to decide yet
    ESC_UNKNOWN,            // Not a recognized ESC sequence

    // --- Non-parametric: 2 bytes (ESC + cmd) ---
    ESC_AUTO_LF_ON,         // ESC "    Auto line feed on
    ESC_AUTO_LF_OFF,        // ESC #    Auto line feed off
    ESC_UNDERLINE_ON,       // ESC E    Enable underline
    ESC_UNDERLINE_OFF,      // ESC R    Disable underline
    ESC_CLEAR_FORMAT,       // ESC X    Clear all formatting
    ESC_SET_LEFT_MARGIN,    // ESC 9    Set left margin at current column
    ESC_SET_RIGHT_MARGIN,   // ESC 0    Set right margin at current column
    ESC_CLEAR_MARGINS,      // ESC B    Clear margins
    ESC_SET_HT,             // ESC 1    Set HT at current position
    ESC_CLEAR_HT,           // ESC 8    Clear HT at current position
    ESC_CLEAR_ALL_HT_VT,   // ESC 2    Clear all HT and VT
    ESC_RESET_DEFAULTS,     // ESC S    Reset to DIP switch defaults
    ESC_PRINT_Y,            // ESC Y    Print glyph at 0x20 position
    ESC_PRINT_Z,            // ESC Z    Print glyph at 0x7F position
    ESC_SET_VT,             // ESC -    Set VT at current position
    ESC_CLEAR_TOP_BOT,      // ESC C    Clear top/bottom margins
    ESC_BOLD_ON,            // ESC O    Bold print (CX only)
    ESC_SHADOW_ON,          // ESC W    Shadow print (CX only)
    ESC_DBLSTRIKE_ON,       // ESC F    Double-strike (CX only)
    ESC_CLEAR_BOLD_ETC,     // ESC &    Clear bold/shadow/double-strike
    ESC_BACKWARD_ON,        // ESC /    Auto backward print on
    ESC_BACKWARD_OFF,       // ESC \    Auto backward print off
    ESC_SET_TOP_MARGIN,     // ESC T    Set top margin (CX only)
    ESC_SET_BOT_MARGIN,     // ESC L    Set bottom margin (CX only)
    ESC_HALF_LF_DOWN,       // ESC U    Half LF down 1/12" (CX only)
    ESC_HALF_LF_UP,         // ESC D    Reverse half LF 1/12" (CX only)
    ESC_REVERSE_LF,         // ESC LF   Reverse LF (CX only)

    // --- Parametric: 3 bytes (ESC + cmd + n) ---
    ESC_ABS_HT,             // ESC HT n   Absolute horizontal tab
    ESC_ABS_VT,             // ESC VT n   Absolute vertical tab
    ESC_SET_PAGE_LENGTH,    // ESC FF n   Set page length
    ESC_SET_PITCH,          // ESC US n   Set HMI / character pitch
    ESC_SET_VMI,            // ESC RS n   Set VMI / line spacing (CX only)
    ESC_RESET_PRINTER,      // ESC CR P   Reset printer

    // --- ANSI CSI Sequences ---
    ESC_CSI,            // Valid CSI sequence, but not one we care about
    ESC_CSI_SGR_SET,    // CSI sequence enabling formatting (Underline = 4)
    ESC_CSI_SGR_RES,    // CSI sequence clearing formatting (Reset = 0 or 24)
} EscapeSeq;

// ==============================================
// Identification
// ==============================================
//
// Pass peeked bytes from siBuffer (starting with 0x1B).
// available = number of bytes peeked (1–3).
//
// Returns:
//   ESC_NEED_MORE  — not enough bytes yet, wait
//   ESC_UNKNOWN    — not a recognized sequence
//   (other)        — identified sequence
//
EscapeSeq escIdentify(const uint8_t *buf, uint8_t available);

// ==============================================
// Consume Length
// ==============================================
//
// Returns number of bytes to discard from siBuffer
// after a successful identification.
//
static inline uint8_t escConsumeLen(EscapeSeq seq) {
    return (seq >= ESC_ABS_HT) ? 3 : 2;
}

// ==============================================
// ANSI CSI Sequence Parser
// ==============================================
//
// Scans the provided buffer for a complete Control 
// Sequence Introducer (ESC [ ...) and returns its 
// total length in bytes.
//
// Returns:
//   0       — not a CSI sequence, or sequence incomplete
//   (other) — total length of the sequence in bytes
//
uint8_t escCsiConsumeLen(const uint8_t *buf, uint8_t available);

#ifdef __cplusplus
}
#endif

#endif // ESCAPE_H