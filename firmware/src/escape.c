// ==============================================
// escape.c — ESC Sequence Identification
// ==============================================
//
// Stateless classifier. Peeks at up to 3 bytes from
// the serial input and returns the identified sequence.
//
// All state changes and bus output are handled by
// the protocol layer — this module only identifies.
//
#include "escape.h"

#include <stdbool.h>

// ==============================================
// Parametric Command Check
// ==============================================
//
// Returns true if the command byte following ESC
// requires a third (parameter) byte.
//
static bool isParametric(uint8_t cmd) {
    switch (cmd) {
        case 0x09:  // HT
        case 0x0B:  // VT
        case 0x0C:  // FF
        case 0x0D:  // CR
        case 0x1E:  // RS
        case 0x1F:  // US
            return true;
        default:
            return false;
    }
}

// ==============================================
// ESC Sequence Identification
// ==============================================
//
// Pass peeked bytes from siBuffer (starting with 0x1B).
// available = number of bytes actually peeked (1–3).
//
// Returns:
//   ESC_NEED_MORE  — not enough bytes to identify yet
//   ESC_UNKNOWN    — not a recognized/implemented sequence
//   (other)        — identified sequence
//
// The caller is responsible for:
//   - Only calling when buf[0] == 0x1B
//   - Consuming escConsumeLen(seq) bytes on a match
//   - Handling ESC_UNKNOWN (discard ESC + command byte)
//
EscapeSeq escIdentify(const uint8_t *buf, uint8_t available) {

    if (buf == 0 || available == 0) return ESC_UNKNOWN;

    // Need at least ESC + command byte
    if (available < 2) return ESC_NEED_MORE;

    uint8_t cmd = buf[1];

    // Parametric sequences need a third byte
    if (isParametric(cmd)) {
        if (available < 3) return ESC_NEED_MORE;
    }

    switch (cmd) {

        // --- Non-parametric (2 bytes) ---
        case '"':   return ESC_AUTO_LF_ON;
        case '#':   return ESC_AUTO_LF_OFF;
        case 'E':   return ESC_UNDERLINE_ON;
        case 'R':   return ESC_UNDERLINE_OFF;
        case 'X':   return ESC_CLEAR_FORMAT;
        case '9':   return ESC_SET_LEFT_MARGIN;
        case '0':   return ESC_SET_RIGHT_MARGIN;
        case 'B':   return ESC_CLEAR_MARGINS;
        case '1':   return ESC_SET_HT;
        case '8':   return ESC_CLEAR_HT;
        case '2':   return ESC_CLEAR_ALL_HT_VT;
        case 'S':   return ESC_RESET_DEFAULTS;
        case 'Y':   return ESC_PRINT_Y;
        case 'Z':   return ESC_PRINT_Z;
        case '-':   return ESC_SET_VT;
        case 'C':   return ESC_CLEAR_TOP_BOT;
        case 'O':   return ESC_BOLD_ON;
        case 'W':   return ESC_SHADOW_ON;
        case 'F':   return ESC_DBLSTRIKE_ON;
        case '&':   return ESC_CLEAR_BOLD_ETC;
        case '/':   return ESC_BACKWARD_ON;
        case '\\':  return ESC_BACKWARD_OFF;
        case 'T':   return ESC_SET_TOP_MARGIN;
        case 'L':   return ESC_SET_BOT_MARGIN;
        case 'U':   return ESC_HALF_LF_DOWN;
        case 'D':   return ESC_HALF_LF_UP;
        case 0x0A:  return ESC_REVERSE_LF;

        // --- Parametric (3 bytes) ---
        case 0x09:  return ESC_ABS_HT;
        case 0x0B:  return ESC_ABS_VT;
        case 0x0C:  return ESC_SET_PAGE_LENGTH;
        case 0x0D:  return ESC_RESET_PRINTER;
        case 0x1E:  return ESC_SET_VMI;
        case 0x1F:  return ESC_SET_PITCH;

        default:    return ESC_UNKNOWN;
    }
}