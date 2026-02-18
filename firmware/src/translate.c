// ==============================================
// translate.c — Bus → Serial Reverse Translation
// ==============================================
//
// Pure translation functions. No static state.
//
// Reverse lookup tables are derived from protocol captures
// taken on 2026-02-12, 2026-02-13, and 2026-02-17.
//
#include "translate.h"
#include "protocol.h"

// ==============================================
// Non-Translating Keys
// ==============================================

bool isNonTranslatingKey(uint8_t busByte) {
    switch (busByte) {
        case 0x08:  // Half-space / micro-step
        case 0x09:  // Index / express
        case 0x0B:  // Unknown function key
        case 0x0C:  // Correction / erase
        case 0x1D:  // Relocate
            return true;
        default:
            return false;
    }
}

// ==============================================
// Keyboard-Dependent Lookup Helper
// ==============================================

static inline TranslateResult kbSelect(uint8_t keyboard,
                                        uint8_t kb1, uint8_t kb2, uint8_t kb3) {
    switch (keyboard) {
        case KEYBOARD_KB1: return result1(kb1);
        case KEYBOARD_KB2: return result1(kb2);
        case KEYBOARD_KB3: return result1(kb3);
        default:           return result0();
    }
}

// ==============================================
// Forward Translation (Serial → Typewriter Bus)
// ==============================================
//
// Switch on serial byte for compiler-optimized dispatch.
// Keyboard-dependent entries use kbSelect().
// Multi-byte outputs use Code key simulation (0x88/0x89 framing)
// and/or dead key compensation (trailing 0x00).
//
// Control codes handled here are limited to simple 1:1 mappings
// (BS, LF, BEL). CR, CR+LF combining, and ESC sequences require
// protocol-level state and are handled in protocol.c.
//
// Space is mapped to 0x00 (AX20 carriage advance).
// CE650 would need 0xB1 instead — requires device-type awareness.
//
TranslateResult translateSerialToBus(uint8_t serialByte, uint8_t keyboard) {

    // --- Control codes ---
    switch (serialByte) {
        case 0x07: return result1(0xF5);        // BEL
        case 0x08: return result1(0x03);        // BS
        case 0x0A: return result1(0x9F);        // LF
        // CR and CR+LF combining handled in protocol.c
        // DC1, DC3 handled in protocol.c
        default: break;
    }

    // --- Printable characters (0x20-0x7E) ---
    if (serialByte < 0x20 || serialByte > 0x7E)
        return result0();  // swallow unknown control codes for now

    switch (serialByte) {

        // --- Characters identical across all keyboards ---
        // Digits 0-9: identity
        // Letters a-x, A-X: identity (except those listed below)

        // --- Y/Z swap (QWERTZ on KB1/KB2) ---
        case 0x59: return kbSelect(keyboard, 0x5A, 0x5A, 0x59); // Y
        case 0x5A: return kbSelect(keyboard, 0x59, 0x59, 0x5A); // Z
        case 0x79: return kbSelect(keyboard, 0x7A, 0x7A, 0x79); // y
        case 0x7A: return kbSelect(keyboard, 0x79, 0x79, 0x7A); // z

        // --- KB3 letter/symbol remaps ---
        case 0x40: // @
            if (keyboard == KEYBOARD_KB3) return result2(0x66, 0x00);
            return kbSelect(keyboard, 0x23, 0x2D, 0);
        case 0x5B: // [
            if (keyboard == KEYBOARD_KB1) return result1(0x22);
            if (keyboard == KEYBOARD_KB2) return result1(0x7B);
            return result1(0x6A);
        case 0x5C: // backslash
            if (keyboard == KEYBOARD_KB1) return result1(0x3A);
            if (keyboard == KEYBOARD_KB2) return result3(0x88, 0x58, 0x89);
            return result2(0x5C, 0x00);
        case 0x5D: // ]
            if (keyboard == KEYBOARD_KB1) return result1(0x7B);
            if (keyboard == KEYBOARD_KB2) return result1(0x3A);
            return result2(0x2F, 0x00);
        case 0x5E: // ^
            if (keyboard == KEYBOARD_KB1) return result3(0x88, 0x57, 0x89);
            if (keyboard == KEYBOARD_KB2) return result2(0x23, 0x00);
            return result1(0x25);
        case 0x5F: // _
            return kbSelect(keyboard, 0x3F, 0x3F, 0x2E);

        case 0x21: // !
            if (keyboard == KEYBOARD_KB3) return result3(0x88, 0x56, 0x89);
            return result1(0x21);
        case 0x22: // "
            if (keyboard == KEYBOARD_KB3) return result2(0x22, 0x00);
            return result1(0x40);
        case 0x23: // #
            if (keyboard == KEYBOARD_KB1) return result1(0x5C);
            if (keyboard == KEYBOARD_KB2) return result1(0x27);
            return result2(0x76, 0x00);
        case 0x24: // $
            return kbSelect(keyboard, 0x24, 0x24, 0x21);
        case 0x25: // %
            return kbSelect(keyboard, 0x25, 0x25, 0x28);
        case 0x26: // &
            return kbSelect(keyboard, 0x20, 0x20, 0x23);
        case 0x27: // '
            if (keyboard == KEYBOARD_KB3) return result2(0x27, 0x00);
            return result1(0x60);
        case 0x28: // (
            return kbSelect(keyboard, 0x2A, 0x2A, 0x29);
        case 0x29: // )
            return kbSelect(keyboard, 0x28, 0x28, 0x2C);
        case 0x2A: // *
            return kbSelect(keyboard, 0x5B, 0x5B, 0x24);
        case 0x2B: // +
            if (keyboard == KEYBOARD_KB3) return result2(0x2B, 0x00);
            return result1(0x5D);
        case 0x2C: // ,
            if (keyboard == KEYBOARD_KB3) return result4(0x88, 0x5A, 0x89, 0x00);
            return result1(0x2C);
        case 0x2D: // -
            return kbSelect(keyboard, 0x2F, 0x2F, 0x3F);
        case 0x2E: // .
            if (keyboard == KEYBOARD_KB3) return result4(0x88, 0x58, 0x89, 0x00);
            return result1(0x2E);
        case 0x2F: // /
            if (keyboard == KEYBOARD_KB3) return result2(0x2D, 0x00);
            return result1(0x26);

        case 0x3A: // :
            if (keyboard == KEYBOARD_KB3) return result2(0x3A, 0x00);
            return result1(0x3E);
        case 0x3B: // ;
            if (keyboard == KEYBOARD_KB3) return result2(0x3B, 0x00);
            return result1(0x3C);
        case 0x3C: // 
            if (keyboard == KEYBOARD_KB1) return result3(0x88, 0x58, 0x89);
            if (keyboard == KEYBOARD_KB2) return result1(0x22);
            return result3(0x88, 0x59, 0x89);
        case 0x3D: // =
            if (keyboard == KEYBOARD_KB3) return result2(0x3D, 0x00);
            return result1(0x29);
        case 0x3E: // >
            if (keyboard == KEYBOARD_KB1) return result3(0x88, 0x56, 0x89);
            if (keyboard == KEYBOARD_KB2) return result1(0x3B);
            return result3(0x88, 0x55, 0x89);
        case 0x3F: // ?
            if (keyboard == KEYBOARD_KB3) return result2(0x5F, 0x00);
            return result1(0x5F);

        case 0x60: // `
            if (keyboard == KEYBOARD_KB3) return result2(0x60, 0x00);
            return result2(0x2B, 0x00);

        case 0x66: // f
            if (keyboard == KEYBOARD_KB3) return result1(0x4A);
            return result1(0x66);
        case 0x6A: // j
            if (keyboard == KEYBOARD_KB3) return result1(0x7C);
            return result1(0x6A);
        case 0x76: // v
            if (keyboard == KEYBOARD_KB3) return result1(0x40);
            return result1(0x76);

        case 0x7B: // {
            return kbSelect(keyboard, 0x27, 0x5C, 0x5B);
        case 0x7C: // |
            if (keyboard == KEYBOARD_KB1) return result1(0x3B);
            if (keyboard == KEYBOARD_KB2) return result3(0x88, 0x56, 0x89);
            return result1(0x5D);
        case 0x7D: // }
            if (keyboard == KEYBOARD_KB1) return result1(0x7C);
            if (keyboard == KEYBOARD_KB2) return result3(0x88, 0x57, 0x89);
            return result1(0x26);
        case 0x7E: // ~
            if (keyboard == KEYBOARD_KB1) return result1(0x2D);
            if (keyboard == KEYBOARD_KB2) return result2(0x3D, 0x00);
            return result1(0x2A);

        case 0x4A: // J
            if (keyboard == KEYBOARD_KB3) return result1(0x7B);
            return result1(0x4A);

        // --- Identity: digits, common letters, space ---
        default: return result1(serialByte);
    }
}

// ==============================================
// Code+Key Path
// ==============================================
//
// Code+letter: letter & 0x1F (standard Ctrl convention)
// Code+specials and Code+1..6: keyboard-independent
// Code+7/8/9/0: keyboard-dependent (KB3 not yet captured)
//
// Code+M (bus 0x6D) is intercepted by the protocol layer
// before reaching this function (auto-LF handling).
//
TranslateResult translateCodeBusToSerial(uint8_t busByte, uint8_t keyboard) {

    // Code+letter (0x61-0x7A): Ctrl convention
    if (busByte >= 0x61 && busByte <= 0x7A)
        return result1(busByte & 0x1F);

    // Code+Specials & Number Row
    switch (busByte) {
        case 0x00: return result1(0x20);            // Code+SP
        case 0x01: return result1(0x09);            // Code+TAB
        case 0x02: return result1(0x0D);            // Code+CR
        case 0x03: return result0();                // Code+BS (Swallowed)
        case 0x04: return result1(0x7F);            // Code+DEL
        case 0x05: return result1(0x1B);            // Code+Esc
        case 0x0D: return result1(0x1E);            // Code+4 -> RS
        case 0x30:                                  // Code+0
            if (keyboard == KEYBOARD_KB1) return result1(0x3E); // >
            if (keyboard == KEYBOARD_KB2) return result1(0x7C); // |
            break;
        case 0x31: return result1(0x1B);            // Code+1 -> ESC
        case 0x35: return result1(0x1F);            // Code+5 -> US
        case 0x36: return result0();                // Code+6 -> None
        case 0x37:                                  // Code+7
            if (keyboard == KEYBOARD_KB1) return result1(0x5E); // ^
            if (keyboard == KEYBOARD_KB2) return result1(0x7D); // }
            break;
        case 0x38:                                  // Code+8
            if (keyboard == KEYBOARD_KB1) return result2(0x1B, 0x59); // ESC+Y
            if (keyboard == KEYBOARD_KB2) return result2(0x1B, 0x5A); // ESC+Z
            break;
        case 0x39:                                  // Code+9
            if (keyboard == KEYBOARD_KB1) return result1(0x3C); // < 
            if (keyboard == KEYBOARD_KB2) return result1(0x5C); /* \ */
            break;
        case 0x8A: return result1(0x1C);            // Code+2 -> FS
        case 0x98: return result1(0x1D);            // Code+3 -> GS
        default:   break;
    }

    return result0();
}

// ==============================================
// Normal/Shifted Path
// ==============================================
//
// Switch on bus byte for compiler-optimized dispatch.
// Keyboard-dependent entries use kbSelect().
// Multi-byte outputs (ESC+Y, ESC+Z) are handled inline.
//
// Bus byte values correspond to daisy wheel petal positions.
// ISO 646 national variant positions differ across keyboards;
// common ASCII positions are identity-mapped (default case).
//

TranslateResult translateNormalBusToSerial(uint8_t busByte, uint8_t keyboard) {
    switch (busByte) {

        // --- Shifted number row ---
        case 0x20: // Shift+6
            if (keyboard == KEYBOARD_KB3) return result2(0x1B, 0x59);
            return kbSelect(keyboard, 0x26, 0x26, 0x26);
        case 0x21: return kbSelect(keyboard, 0x21, 0x21, 0x24); // Shift+1
        case 0x23: return kbSelect(keyboard, 0x40, 0x5E, 0x26); // Shift+3
        case 0x24: return kbSelect(keyboard, 0x24, 0x24, 0x2A); // Shift+4
        case 0x25: return kbSelect(keyboard, 0x25, 0x25, 0x5E); // Shift+5
        case 0x40: return kbSelect(keyboard, 0x22, 0x22, 0x76); // Shift+2
        case 0x26: return kbSelect(keyboard, 0x2F, 0x2F, 0x7D); // Shift+7
        case 0x2A: return kbSelect(keyboard, 0x28, 0x28, 0x7E); // Shift+8
        case 0x28: return kbSelect(keyboard, 0x29, 0x29, 0x25); // Shift+9
        case 0x29: return kbSelect(keyboard, 0x3D, 0x3D, 0x28); // Shift+0

        // --- Punctuation (normal) ---
        case 0x2C: return kbSelect(keyboard, 0x2C, 0x2C, 0x29); // Comma
        case 0x2D: return kbSelect(keyboard, 0x7E, 0x40, 0x2F); // ISO 646 national
        case 0x2E: return kbSelect(keyboard, 0x2E, 0x2E, 0x5F); // Period
        case 0x2F: return kbSelect(keyboard, 0x2D, 0x2D, 0x5D); // Minus

        // --- Punctuation (shifted) ---
        case 0x3C: return kbSelect(keyboard, 0x3B, 0x3B, 0x29); // Shift+Comma
        case 0x3E: return kbSelect(keyboard, 0x3A, 0x3A, 0x5F); // Shift+Period
        case 0x3F: return kbSelect(keyboard, 0x5F, 0x5F, 0x2D); // Shift+Minus
        case 0x5F: return result1(0x3F);                        // Shift+ISO 646 national

        // --- ISO 646 national variant positions (normal) ---
        case 0x27: return kbSelect(keyboard, 0x7B, 0x23, 0x27);
        case 0x3B: return kbSelect(keyboard, 0x7C, 0x3E, 0x3B);
        case 0x3D:
            if (keyboard == KEYBOARD_KB1) return result2(0x1B, 0x5A);
            if (keyboard == KEYBOARD_KB2) return result1(0x7E);
            if (keyboard == KEYBOARD_KB3) return result1(0x3D);
        case 0x5C: return kbSelect(keyboard, 0x23, 0x7B, 0x5C);
        case 0x5D: return kbSelect(keyboard, 0x2B, 0x2B, 0x7C);
        case 0x7C:
            if (keyboard == KEYBOARD_KB1) return result1(0x7D);
            if (keyboard == KEYBOARD_KB2) return result2(0x1B, 0x59);
            if (keyboard == KEYBOARD_KB3) return result1(0x6A);

        // --- ISO 646 national variant positions (shifted) ---
        case 0x22: return kbSelect(keyboard, 0x5B, 0x3C, 0x22);
        case 0x2B: return kbSelect(keyboard, 0x60, 0x60, 0x2B);
        case 0x3A: return kbSelect(keyboard, 0x5C, 0x5D, 0x3A);
        case 0x5B: return kbSelect(keyboard, 0x2A, 0x2A, 0x7B);
        case 0x60: return kbSelect(keyboard, 0x27, 0x27, 0x60);
        case 0x7B: return kbSelect(keyboard, 0x5D, 0x5B, 0x4A);

        // --- Y/Z swap (QWERTZ on KB1/KB2, QWERTY on KB3) ---
        case 0x79: return kbSelect(keyboard, 0x7A, 0x7A, 0x79);
        case 0x7A: return kbSelect(keyboard, 0x79, 0x79, 0x7A);
        case 0x59: return kbSelect(keyboard, 0x5A, 0x5A, 0x59);
        case 0x5A: return kbSelect(keyboard, 0x59, 0x59, 0x5A);

        // --- KB3 letter/symbol remaps ---
        case 0x66: return kbSelect(keyboard, 0x66, 0x66, 0x40);
        case 0x6A: return kbSelect(keyboard, 0x6A, 0x6A, 0x5B);
        case 0x76: return kbSelect(keyboard, 0x76, 0x76, 0x23);
        case 0x4A: return kbSelect(keyboard, 0x4A, 0x4A, 0x66);

        // --- Identity: digits, common letters, standard ASCII ---
        default: return result1(busByte);
    }
}
