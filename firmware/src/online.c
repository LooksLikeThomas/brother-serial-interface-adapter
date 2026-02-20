// ==============================================
// online.c — PS_ONLINE Serial/Bus Byte Handling
// ==============================================
//
// Extracted from protocol.c to keep the state machine
// focused on transitions and transfer layer coordination.
//
// Two entry points, called by protocol.c during PS_ONLINE:
//
//   onlineHandleSI() — Serial → Bus path
//     Peeks at siBuffer, classifies the next byte(s),
//     and loads the appropriate bus sequence into BSB.
//     Handles control codes (CR, LF, FF, BS, HT),
//     ESC sequences, auto-wrap, and printable translation.
//
//   onlineHandleSO() — Bus → Serial path
//     Reads the byte received from the typewriter,
//     tracks Code key state, filters non-translating keys,
//     and pushes translated serial byte(s) to soBuffer.
//
// Neither function interacts with the transfer layer
// directly. Bus bytes are loaded into BSB for protocol.c
// to drain one per transfer cycle.
//
#include "online.h"
#include "escape.h"
#include "translate.h"
#include "buffers.h"
#include "config.h"
#include "debug.h"

// ==============================================
// Repositioning Helper
// ==============================================
//
// Emits a horizontal repositioning sequence into ps->bsb.
// Always sends 0x8B first (disables underline during movement).
// After movement, re-enables underline with 0x8A if underline
// is currently active.
//
// Positive delta: 0x00 advance steps.
// Negative delta: 0x03 backspace steps.
// Zero delta: no-op.
//
// Does not update ps->column — caller is responsible.
//
static void bsbAddMove(Protocol *ps, int16_t delta) {
    if (delta == 0) return;
    bsbAddByte(&ps->bsb, 0x8B);
    if (delta > 0) bsbAddRepeat(&ps->bsb, 0x00, (uint8_t)delta);
    if (delta < 0) bsbAddRepeat(&ps->bsb, 0x03, (uint8_t)(-delta));
    if (ps->underlineEnabled) bsbAddByte(&ps->bsb, 0x8A);
}

// ==============================================
// Carriage Return Helper
// ==============================================
//
// Appends a standalone CR sequence to ps->bsb:
//   [0x9E, pitchByte] + advance to left margin
//
// Updates ps->column = ps->leftMargin.
//
static void bsbAddCR(Protocol *ps) {
    bsbAddByte2(&ps->bsb, 0x9E, ps->pitchByte);
    bsbAddMove(ps, (int16_t)(ps->leftMargin - 1));
    ps->column = ps->leftMargin;
}

// ==============================================
// CR+LF Helper
// ==============================================
//
// Appends a combined CR+LF sequence to ps->bsb:
//   [0x02, pitchByte] + advance to left margin
//
// Updates ps->column = ps->leftMargin and increments ps->line.
//
static void bsbAddCRLF(Protocol *ps) {
    bsbAddByte2(&ps->bsb, 0x02, ps->pitchByte);
    bsbAddMove(ps, (int16_t)(ps->leftMargin - 1));
    ps->column = ps->leftMargin;
    ps->line++;
}

// ==============================================
// ESC Sequence Action Handler
// ==============================================
//
// Applies the identified ESC sequence to protocol state.
// Called after escIdentify() returns a valid sequence
// and the bytes have been consumed from siBuffer.
//
// Sequences fall into three categories:
//   - State-only: update protocol fields (margins, flags)
//   - Bus-only: load byte(s) into BSB (underline on/off)
//   - Both: update state and emit bus bytes (pitch, reset)
//
// Unimplemented/CX-only sequences fall through to default
// and are logged but otherwise ignored.
//
static ProtocolStatus handleEscAction(Protocol *ps, EscapeSeq seq, uint8_t param) {

    switch (seq) {

        // ----- Auto Line Feed -----
        // ESC " enables, ESC # disables.
        // When enabled, CR automatically appends LF (bus 0x02 instead of 0x9E).
        // The flag is checked during CR handling in onlineHandleSI().
        //
        case ESC_AUTO_LF_ON:
            ps->autoLfEnabled = true;
            DBG_EVENT("ESC: AUTO LF ON");
            break;

        case ESC_AUTO_LF_OFF:
            ps->autoLfEnabled = false;
            DBG_EVENT("ESC: AUTO LF OFF");
            break;

        // ----- Underline Control -----
        // ESC E enables (0x8A), ESC R disables (0x8B).
        // ESC X clears all formatting — on AX20 this is just
        // underline off (0x8B), same as ESC R.
        //
        case ESC_UNDERLINE_ON:
            bsbClear(&ps->bsb);
            bsbAddByte(&ps->bsb, 0x8A);
            ps->underlineEnabled = true;
            DBG_EVENT("ESC: UNDERLINE ON");
            break;

        case ESC_UNDERLINE_OFF:
            bsbClear(&ps->bsb);
            bsbAddByte(&ps->bsb, 0x8B);
            ps->underlineEnabled = false;
            DBG_EVENT("ESC: UNDERLINE OFF");
            break;

        case ESC_CLEAR_FORMAT:
            bsbClear(&ps->bsb);
            bsbAddByte(&ps->bsb, 0x8B);
            ps->underlineEnabled = false;
            DBG_EVENT("ESC: CLEAR FORMAT");
            break;

        // ----- Margin Control -----
        // ESC 9 sets left margin at current column.
        // ESC 0 sets right margin at current column.
        // ESC B clears both margins to defaults.
        //
        // Margins are stored as 1-based column numbers.
        // The right margin default depends on the active pitch
        // (wider pitch = fewer columns per line).
        //
        // The manual specifies a minimum distance of 24/120"
        // between margins — not enforced here.
        //
        case ESC_SET_LEFT_MARGIN:
            ps->leftMargin = ps->column;
            DBG_EVENT_HEX("ESC: LEFT MARGIN", ps->leftMargin);
            break;

        case ESC_SET_RIGHT_MARGIN:
            ps->rightMargin = ps->column;
            DBG_EVENT_HEX("ESC: RIGHT MARGIN", ps->rightMargin);
            break;

        case ESC_CLEAR_MARGINS:
            ps->leftMargin = 1;
            ps->rightMargin = rightMarginForPitch(ps->pitchByte);
            DBG_EVENT("ESC: CLEAR MARGINS");
            break;

        // ----- Horizontal Tab Stops -----
        // ESC 1 sets a tab stop at the current column.
        // ESC 8 clears the tab stop at the current column.
        // ESC 2 clears all HT and VT stops.
        //
        // Tab stops are stored in the TabStops bitmask and
        // used by the HT (0x09) handler in onlineHandleSI().
        //
        case ESC_SET_HT:
            tabSet(&ps->tabs, ps->column);
            DBG_EVENT_HEX("ESC: SET HT", ps->column);
            break;

        case ESC_CLEAR_HT:
            tabClear(&ps->tabs, ps->column);
            DBG_EVENT_HEX("ESC: CLEAR HT", ps->column);
            break;

        case ESC_CLEAR_ALL_HT_VT:
            tabClearAll(&ps->tabs);
            DBG_EVENT("ESC: CLEAR ALL HT/VT");
            break;

        // ----- Character Pitch (HMI) -----
        // ESC US n sets the horizontal motion index.
        // Valid values: 0xB1 (10cpi), 0xB2 (12cpi), 0xB3 (15cpi).
        //
        // Changing pitch affects:
        //   - The HMI byte appended to CR/CR+LF sequences
        //   - The physical width of each 0x00 advance step
        //   - The right margin (fewer columns at wider pitch)
        //
        // The pitch byte is emitted to the bus to reconfigure
        // the typewriter's step size immediately.
        //
        case ESC_SET_PITCH:
            if (param == 0xB1 || param == 0xB2 || param == 0xB3) {
                ps->pitchByte = param;
                ps->rightMargin = rightMarginForPitch(param);

                if (ps->column != 1) {
                    DBG_EVENT_HEX("ESC: SET PITCH DEFERRED (NOT AT COL 1)", param);
                    break;
                }

                bsbClear(&ps->bsb);
                bsbAddByte(&ps->bsb, param);
                DBG_EVENT_HEX("ESC: SET PITCH", param);
            } else {
                DBG_EVENT_HEX("ESC: INVALID PITCH", param);
            }
            break;

        // ----- Absolute Horizontal Tab -----
        // ESC HT n moves the carriage directly to column n.
        // Movement range = (n-1) × HMI from the left end of platen.
        //
        // Per the manual, n = 1–126 (excluding NUL and DEL).
        // Margins are ignored — absolute HT can move past them.
        //
        // Forward movement uses 0x00 advance steps.
        // Backward movement uses 0x03 backspace steps.
        // bsbAddMove() disables underline during movement and
        // re-enables it afterward if underline was active.
        //
        // No-op if already at the target column.
        //
        case ESC_ABS_HT: {
            uint8_t target = param;

            if (target == 0 || target == 0x7F) {
                DBG_EVENT_HEX("ESC: ABS HT INVALID", target);
                break;
            }

            if (target == ps->column) {
                DBG_EVENT_HEX("ESC: ABS HT NO-OP", target);
                break;
            }

            bsbClear(&ps->bsb);
            bsbAddMove(ps, (int16_t)target - (int16_t)ps->column);
            ps->column = target;
            DBG_EVENT_HEX("ESC: ABS HT TO", target);
            break;
        }

        // ----- Reset to DIP Switch Defaults -----
        // ESC S resets all session state to power-on defaults:
        //   - Pitch reverts to configured default
        //   - Margins revert to full width
        //   - Tab stops reinitialised
        //   - Auto-LF reverts to configured default
        //   - Carriage position reset
        //
        // Emits CR to home the carriage, then repositions to
        // the saved column so the physical position is preserved.
        //
        case ESC_RESET_DEFAULTS: {
            uint8_t savedColumn = ps->column;
            ps->autoLfEnabled = AUTO_LINE_FEED;
            ps->pitchByte = PITCH_BYTE;
            ps->leftMargin = 1;
            ps->rightMargin = rightMarginForPitch(ps->pitchByte);
            tabInit(&ps->tabs, TAB_EVERY_N, ps->rightMargin);
            ps->underlineEnabled = false;
            ps->line = 1;

            bsbClear(&ps->bsb);
            bsbAddCR(ps);
            bsbAddMove(ps, (int16_t)(savedColumn - 1));
            ps->column = savedColumn;
            DBG_EVENT("ESC: RESET DEFAULTS");
            break;
        }

        // ----- Wheel Probes -----
        // ESC+Y prints the physical glyph at the space (0x20) wheel
        // position using Code key simulation. ESC+Z prints the glyph
        // at the DEL (0x7F) position. These allow probing which
        // physical daisy wheel is installed.
        //
        case ESC_PRINT_Y:
            bsbClear(&ps->bsb);
            bsbAddByte(&ps->bsb, 0x88);
            bsbAddByte(&ps->bsb, 0x55);
            bsbAddByte(&ps->bsb, 0x89);
            ps->column++;
            DBG_EVENT("ESC: PRINT Y (SPACE GLYPH)");
            break;

        case ESC_PRINT_Z:
            bsbClear(&ps->bsb);
            bsbAddByte(&ps->bsb, 0x3D);
            bsbAddByte(&ps->bsb, 0x00);
            ps->column++;
            DBG_EVENT("ESC: PRINT Z (DEL GLYPH)");
            break;

        // ----- Printer Reset -----
        // ESC+CR+P performs a full printer reset.
        // The full IF60 sequence is [0xF4, <P>, 0x8B, 0xFD, 0x7F]
        // with the typewriter responding after 0xFD. This simplified
        // version emits [0xF4, <P>, 0x8B] and resets internal state.
        //
        // The 0xFD/0x7F handshake is omitted — it requires a state
        // machine transition to wait for the typewriter response,
        // which cannot be done from within handleEscAction(). If the
        // handshake is needed, this must be promoted to a protocol-level
        // state (like PS_SELECT).
        //
        case ESC_RESET_PRINTER:
            ps->autoLfEnabled = AUTO_LINE_FEED;
            ps->pitchByte = PITCH_BYTE;
            ps->leftMargin = 1;
            ps->rightMargin = rightMarginForPitch(ps->pitchByte);
            tabInit(&ps->tabs, TAB_EVERY_N, ps->rightMargin);
            ps->column = 1;
            ps->line = 1;

            bsbClear(&ps->bsb);
            bsbAddByte2(&ps->bsb, 0xF4, ps->pitchByte);
            bsbAddByte(&ps->bsb, 0x8B);
            DBG_EVENT("ESC: RESET PRINTER");
            break;

        // ----- Not Implemented -----
        // CX-only features (VMI, bold, shadow, double-strike,
        // sub/superscript, reverse LF, backward printing,
        // top/bottom margins) are recognised but not acted upon.
        //
        default:
            DBG_EVENT_HEX("ESC: NOT IMPL", (uint8_t)seq);
            break;
    }

    return PS_STATUS_ONLINE;
}

// ==============================================
// Serial Input Handler
// ==============================================
//
// Processes the next byte(s) from siBuffer and loads
// the resulting bus sequence into ps->bsb.
//
// The check order matters:
//   1. DC3 (deselect trigger) — must be caught before translation
//   2. CR — peek-ahead to coalesce CR+LF before LF is seen
//   3. ESC sequences — consume 2–3 bytes via peek-ahead
//   4. Control codes (LF, FF, BS, HT) — each with
//      specific bus sequences and state side-effects
//   5. Auto-wrap — triggers before the character is consumed
//   6. Printable characters — translated via translateSerialToBus()
//
// Characters that translateSerialToBus() doesn't recognise
// (unsupported control codes, etc.) are silently swallowed.
//
ProtocolStatus onlineHandleSI(Protocol *ps) {

    uint8_t si_byte = siBufferPeek();

    // ----- DC3: Signal DESELECT to protocol layer -----
    // The actual state transition is handled by protocol.c;
    // we just consume the byte and signal via return value.
    //
    if (si_byte == 0x13) {
        siBufferPop(&si_byte);
        DBG_EVENT("DC3 RECEIVED - START DESELECT");
        return PS_STATUS_DESELECTING;
    }

    // ----- Carriage Return (0x0D) -----
    // Don't consume yet — peek ahead to decide how to handle.
    // If no second byte is available, return and wait.
    //
    if (si_byte == 0x0D) {
        if (siBufferCount() < 2) {
            return PS_STATUS_ONLINE;  // Wait for next byte
        }

        siBufferPop(&si_byte);
        uint8_t next = siBufferPeek();

        bsbClear(&ps->bsb);

        if (next == 0x0A) {
            // CR+LF: coalesce
            uint8_t lf;
            siBufferPop(&lf);
            bsbAddCRLF(ps);
            DBG_EVENT("CR+LF");
        } else if (ps->autoLfEnabled) {
            // Auto-LF: CR becomes CR+LF
            bsbAddCRLF(ps);
            DBG_EVENT("CR+AUTO-LF");
        } else {
            // Standalone CR
            bsbAddCR(ps);
            DBG_EVENT("CR");
        }

        return PS_STATUS_ONLINE;
    }

    // ----- ESC Sequence Handling -----
    // Peek up to 3 bytes for the escape module to classify.
    // If not enough bytes are available yet (ESC_NEED_MORE),
    // return without consuming — the bytes will still be
    // there next time we're called.
    //
    if (si_byte == 0x1B) {
        uint8_t peekBuf[3];
        uint8_t avail = siBufferPeekN(peekBuf, 3);
        EscapeSeq seq = escIdentify(peekBuf, avail);

        if (seq == ESC_NEED_MORE) {
            return PS_STATUS_ONLINE;
        }

        uint8_t len = escConsumeLen(seq);
        siBufferDiscard(len);

        if (seq == ESC_UNKNOWN) {
            DBG_EVENT_HEX("ESC UNKNOWN CMD", peekBuf[1]);
            return PS_STATUS_ONLINE;
        }

        uint8_t param = (len == 3) ? peekBuf[2] : 0;
        return handleEscAction(ps, seq, param);
    }

    // ----- Line Feed (0x0A) -----
    // Vertical movement only — carriage stays at current column.
    // Bus byte 0x9F advances paper one line.
    //
    // When AUTO_CARRIAGE_RETURN is enabled, standalone LF is
    // promoted to CR+LF (bus 0x02) for Unix/Linux compatibility.
    // Explicit CR+LF pairs never reach here — the CR lookahead
    // coalesces them before the LF is processed.
    //
    if (si_byte == 0x0A) {
        siBufferPop(&si_byte);
        bsbClear(&ps->bsb);
        if (AUTO_CARRIAGE_RETURN) {
            bsbAddCRLF(ps);
            DBG_EVENT("LF+AUTO-CR");
        } else {
            bsbAddByte(&ps->bsb, 0x9F);
            ps->line++;
            DBG_EVENT("LF");
        }
        return PS_STATUS_ONLINE;
    }

    // ----- Form Feed (0x0C) -----
    // Advances to the top of the next page by emitting
    // 0x9F (line feed) for each remaining line on the
    // current page, then resets the line counter.
    //
    if (si_byte == 0x0C) {
        siBufferPop(&si_byte);
        uint8_t remaining = (ps->line < LINES_PER_PAGE) ? LINES_PER_PAGE - ps->line : 0;
        bsbClear(&ps->bsb);
        bsbAddRepeat(&ps->bsb, 0x9F, remaining);
        ps->line = 1;
        DBG_EVENT_HEX("FF LINES", remaining);
        return PS_STATUS_ONLINE;
    }

    // ----- Auto line wrap at right margin -----
    // When a printable character would be printed past the
    // right margin, insert a CR+LF before it. The character
    // stays in siBuffer and will be translated on the next call
    // after the wrap sequence has been drained from BSB.
    //
    if (si_byte >= 0x20 && si_byte <= 0x7E && ps->column > ps->rightMargin) {

        // Remove spaces from left start of line
        if(si_byte == 0x20){
            siBufferPop(&si_byte);
            DBG_EVENT("SP SWALLOWED (AT RMARGIN)");
        }

        bsbClear(&ps->bsb);
        bsbAddCRLF(ps);
        DBG_EVENT("AUTO WRAP");
        return PS_STATUS_ONLINE;
    }

    // ----- Backspace at left margin: swallow -----
    // Prevents the carriage from moving left of the margin.
    //
    if (si_byte == 0x08 && ps->column <= ps->leftMargin) {
        siBufferPop(&si_byte);
        DBG_EVENT("BS SWALLOWED (AT MARGIN)");
        return PS_STATUS_ONLINE;
    }

    // ----- Horizontal Tab (0x09) -----
    // Advances to the next set tab stop within the right margin.
    // No-op if no tab stop exists ahead of the current column.
    //
    // Emits 0x8B (disable underline defensively) followed by
    // 0x00 × N advance steps to reach the tab stop column.
    //
    if (si_byte == 0x09) {
        siBufferPop(&si_byte);
        uint8_t target = tabNextStop(&ps->tabs, ps->column, ps->rightMargin);
        if (target == 0) {
            DBG_EVENT("HT NO STOP");
            return PS_STATUS_ONLINE;
        }
        uint8_t delta = target - ps->column;
        bsbClear(&ps->bsb);
        bsbAddByte(&ps->bsb, 0x8B);
        bsbAddRepeat(&ps->bsb, 0x00, delta);
        ps->column = target;
        DBG_EVENT_HEX("HT TO COL", target);
        return PS_STATUS_ONLINE;
    }

    // ----- Printable Characters and Remaining Codes -----
    // Consume the byte and translate via translateSerialToBus().
    // The translation module handles keyboard-dependent remapping,
    // Code key simulation (0x88/0x89 framing), and dead key
    // compensation (trailing 0x00).
    //
    // Characters that produce no output (r.len == 0) are
    // silently swallowed — these are unsupported control codes
    // or bytes outside the printable range.
    //
    // Column is updated by the serial byte's horizontal delta:
    //   +1 for printable characters, -1 for BS, 0 for everything else.
    //
    siBufferPop(&si_byte);
    TranslateResult r = translateSerialToBus(si_byte, ps->keyboardID);

    if (r.len == 0) {
        DBG_EVENT_HEX("SI SWALLOWED", si_byte);
        return PS_STATUS_ONLINE;
    }

    bsbClear(&ps->bsb);
    bsbAddResult(&ps->bsb, r);
    ps->column += serialHtDelta(si_byte);
    DBG_CHAR_EVENT_HEX_CHAR("SI TRANSLATED", si_byte);
    return PS_STATUS_ONLINE;
}

// ==============================================
// Bus Input Handler
// ==============================================
//
// Processes a byte received from the typewriter and
// pushes the translated serial byte(s) to soBuffer.
//
// The Code key (0x88 down, 0x89 up) is tracked as a
// modifier state — bytes received between 0x88 and 0x89
// are translated through the Code path, which produces
// ASCII control codes (letter & 0x1F) or keyboard-dependent
// extra characters.
//
// Non-translating keys (Correction, Relocate, Half-space,
// etc.) are filtered — these are typewriter-internal
// functions with no serial equivalent.
//
// Code+M is intercepted before general translation because
// it produces CR (+ optional LF) rather than the standard
// Ctrl+M control code, to match the original IF60 behaviour
// of treating Code+Return as a newline entry point.
//
ProtocolStatus onlineHandleSO(Protocol *ps) {

    uint8_t b = ps->ts.receivedByte;
    DBG_CHAR_EVENT_HEX("SO RECEIVED", b);

    // ----- Code Key Modifier Tracking -----
    // 0x88 = Code key pressed, 0x89 = Code key released.
    // All bus bytes between these markers are translated
    // through the Code path instead of the normal path.
    //
    if (b == 0x88) { ps->codePressed = true;  return PS_STATUS_ONLINE; }
    if (b == 0x89) { ps->codePressed = false; return PS_STATUS_ONLINE; }

    // ----- Non-Translating Keys -----
    // Typewriter-internal functions (Correction, Relocate,
    // Half-space, Index) that have no serial equivalent.
    //
    if (isNonTranslatingKey(b)) return PS_STATUS_ONLINE;

    // ----- Code+M: Newline -----
    // Special case: Code+Return sends CR (+ LF if auto-LF
    // is enabled) rather than the Ctrl+M control code.
    //
    if (ps->codePressed && b == 0x6D) {
        soBufferPush(0x0D);
        if (ps->autoLfEnabled) soBufferPush(0x0A);
        return PS_STATUS_ONLINE;
    }

    // ----- Translation Dispatch -----
    // Code path: letter & 0x1F for control codes, plus
    // keyboard-dependent Code+number row mappings.
    // Normal path: keyboard-dependent character translation
    // including Y/Z swap and ISO 646 variant positions.
    //
    TranslateResult r;
    if (ps->codePressed) {
        r = translateCodeBusToSerial(b, ps->keyboardID);
    } else {
        r = translateNormalBusToSerial(b, ps->keyboardID);
    }

    // ----- Push to Serial Out Buffer -----
    // Most keys produce one byte; a few produce two
    // (e.g. ESC+Y wheel probe, Code+M newline).
    //
    for (uint8_t i = 0; i < r.len; i++) {
        soBufferPush(r.bytes[i]);
        DBG_CHAR_EVENT_HEX_CHAR("SO MAPPED", r.bytes[i]);
    }

    return PS_STATUS_ONLINE;
}