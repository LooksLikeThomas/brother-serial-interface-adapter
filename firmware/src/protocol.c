// ==============================================
// protocol.c — Protocol Sequence Implementation
// ==============================================
//
// Handles multi-byte protocol sequences by calling into
// the transfer layer for individual byte transfers.
//
// Flow:
//   PS_OFFLINE → PS_STARTUP_INIT → PS_STARTUP_RESPONSE → PS_STANDBY
//   PS_STANDBY → PS_SELECT → PS_ONLINE
//   PS_ONLINE → PS_DESELECT → PS_STANDBY
//
// In PS_ONLINE, the protocol layer acts as a pass-through:
//   - Pops bytes from siBuffer and hands them to transferQueueSI()
//   - Pushes received bytes (from transfer layer) into soBuffer
//
// On timeout at any stage, returns to PS_OFFLINE.
//
#include "protocol.h"
#include "transfer.h"
#include "buffers.h"
#include "hardware.h"
#include "translate.h"
#include "debug.h"

// For micros() — remove when porting away from Arduino
#include <Arduino.h>

// ==============================================
// State Transition Helper
// ==============================================
//
// Encapsulates the common transition logic:
//   - Records timestamp for timeout tracking
//   - Updates state
//
// Called by pollProtocol() on every state change.
//
static inline void transitionTo(Protocol *ps, ProtocolState newState) {
    ProtocolState oldState = ps->state;
    ps->stateEnteredAt = micros();
    ps->state = newState;

    // Log the transition with the status that triggered it
    DBG_TRANSITION(oldState, newState, ps->lastTsStatus);
}

// ==============================================
// Session State Reset
// ==============================================
//
// Resets all protocol state that is only valid while
// connected to a typewriter. Called by protocolInit()
// and transitionToOffline().
//
// Does NOT touch: state, stateEnteredAt, or Transfer struct.
//
static void protocolReset(Protocol *ps) {
    ps->deviceType = 0;
    ps->selectState = SEL_IDLE;
    ps->deselectState = DESEL_IDLE;
    ps->codePressed = false;
    ps->autoLfEnabled = AUTO_LINE_FEED;
    ps->keyboardID = KEYBOARD_KB1;
    ps->fwdLen = 0;
    ps->fwdIdx = 0;
}

// ==============================================
// Transition to Offline
// ==============================================
//
// Centralised transition to PS_OFFLINE.
// Tears down the transfer layer, resets all session
// state, and transitions to PS_OFFLINE.
//
// Called from every error/timeout/power-loss path.
//
static void transitionToOffline(Protocol *ps) {
    transferDeinit(&ps->ts);
    protocolReset(ps);
    transitionTo(ps, PS_OFFLINE);
}

// ==============================================
// Public: Initialize Protocol Layer
// ==============================================

void protocolInit(Protocol *ps) {
    ps->state = PS_OFFLINE;
    ps->stateEnteredAt = 0;

    // Transfer layer not initialized yet
    ps->ts.state = TS_NOT_INIT;
    ps->ts.stateEnteredAt = 0;
    ps->ts.lastWasSI = true;
    ps->ts.receivedByte = 0;

    // Session state defaults
    protocolReset(ps);
}

// ==============================================
// Public: Protocol State Machine
// ==============================================
//
// Called every loop iteration after pollTransfer().
// Uses TransferStatus to know when transfers complete,
// then decides what to do next.
//
// The protocol layer never touches hardware pins directly.
// All hardware interaction goes through the transfer layer.
//
ProtocolStatus pollProtocol(Protocol *ps) {

    uint32_t now = micros();
    Transfer *ts = &ps->ts;

    // ==========================================
    // Poll Transfer Layer
    // ==========================================
    TransferStatus status = pollTransfer(ts);
    ps->lastTsStatus = status;

    // ==========================================
    // Debug Flush (when safe)
    // ==========================================
    DBG_FLUSH_IF_SAFE(status);

    // ==========================================
    // Global Error Handling
    // ==========================================
    //
    // Transfer layer error can happen in any active state.
    // Reset transfer layer and return to offline.
    //
    if (status == TS_STATUS_ERROR) {
        DBG_ERROR("TS_STATUS_ERROR");
        transitionToOffline(ps);
        return PS_STATUS_OFFLINE;
    }

    // ==========================================
    // Global Power Check
    // ==========================================
    //
    // If typewriter loses power, return to offline.
    // Grace Period of 100us to prevent fast switching between States
    // Skip check if already offline.
    //
    if (ps->state != PS_OFFLINE && now - ps->stateEnteredAt >= 100 && !isTypewriterPowered()) {
        DBG_EVENT("TW Power OFF");
        transitionToOffline(ps);
        return PS_STATUS_OFFLINE;
    }
    
    switch (ps->state) {
        
        // ==========================================
        // PS_OFFLINE — Waiting for Typewriter
        // ==========================================
        //
        // Entry: Init, or after error/disconnect
        // Exit:  Typewriter power detected
        //
        case PS_OFFLINE:
        
            // ----- TRANSITION: Typewriter powered on -----
            // Guard: isTypewriterPowered() returns true
            // Action: Begin settle wait
            if (isTypewriterPowered()) {
                DBG_EVENT("TW Power ON");
                transitionTo(ps, PS_STARTUP_SETTLE);
                return PS_STATUS_STARTUP;
            }
            
            // No transition
            return PS_STATUS_OFFLINE;
        
        // ==========================================
        // PS_STARTUP_SETTLE — Wait for hardware settle
        // ==========================================
        //
        // Entry: Typewriter power detected
        // Exit:  1.5s elapsed, or power lost
        //
        case PS_STARTUP_SETTLE:
            
            // ----- TRANSITION: Settle time elapsed -----
            // Guard: 1.5s since power detected
            // Action: Initialize transfer layer, begin startup sequence
            if (now - ps->stateEnteredAt >= 1500000) {
                transferInit(ts);
                transitionTo(ps, PS_STARTUP_INIT);
                return PS_STATUS_STARTUP;
            }
            
            // No transition
            return PS_STATUS_STARTUP;
        
        // ==========================================
        // PS_STARTUP_INIT — Send presence announcement
        // ==========================================
        //
        // Entry: Typewriter just came online
        // Exit:  0xFE sent successfully, or timeout
        //
        case PS_STARTUP_INIT:
        
            // ----- TRANSITION: Transfer layer ready -----
            // Guard: Transfer idle
            // Action: Queue 0xFE (interface announces presence)
            if (status == TS_STATUS_IDLE) {
                DBG_EVENT("IF REQUEST 0xFE");
                transferQueueSI(ts, 0xFE);
                return PS_STATUS_STARTUP;
            }
            
            // ----- TRANSITION: 0xFE sent successfully -----
            // Guard: SI transfer complete
            // Action: Wait for device type response
            if (status == TS_STATUS_SI_DONE) {
                transitionTo(ps, PS_STARTUP_RESPONSE);
                return PS_STATUS_STARTUP;
            }
            
            // ----- TRANSITION: Transfer failed -----
            // Guard: Timeout from transfer layer
            // Action: Return to offline state
            if (status == TS_STATUS_TIMEOUT) {
                DBG_EVENT("STARTUP REQUEST TIMEOUT");
                transitionToOffline(ps);
                return PS_STATUS_OFFLINE;
            }
            
            // No transition (SI_BUSY — transfer in progress)
            return PS_STATUS_STARTUP;
        
        // ==========================================
        // PS_STARTUP_RESPONSE — Wait for device type
        // ==========================================
        //
        // Entry: 0xFE sent, awaiting typewriter response
        // Exit:  Device type received (e.g. 0x30), or timeout
        //
        case PS_STARTUP_RESPONSE:
        
            // ----- TRANSITION: Device type received -----
            // Guard: SO transfer complete
            // Action: Store device type, enter standby
            if (status == TS_STATUS_SO_DONE) {
                DBG_EVENT_HEX("TW RESPONSE", ts->receivedByte);
                ps->deviceType = ts->receivedByte;
                transitionTo(ps, PS_STANDBY);
                return PS_STATUS_STANDBY;
            }
            
            // ----- TRANSITION: Typewriter Transfer Timeout -----
            // Guard: Timeout from transfer layer
            // Action: Return to offline state
            if (status == TS_STATUS_TIMEOUT) {
                DBG_EVENT("STARTUP TRANSFER TIMEOUT");
                transitionToOffline(ps);
                return PS_STATUS_OFFLINE;
            }

            // ----- TRANSITION: Typewriter Response Timeout -----
            // Guard: 1s since Startup Request
            // Action: Return to offline state
            if (now - ps->stateEnteredAt >= 1000000) {
                DBG_EVENT("STARTUP RESPONSE TIMEOUT");
                transitionToOffline(ps);
                return PS_STATUS_OFFLINE;
            }
            
            // No transition (waiting for SO transfer)
            return PS_STATUS_STARTUP;
        
        // ==========================================
        // PS_STANDBY — Connected, waiting for SELECT
        // ==========================================
        //
        // Entry: Startup complete, or DESELECT complete
        // Exit:  SELECT trigger received (DC1 or typewriter request)
        //
        case PS_STANDBY:

            // ----- TRANSITION: Typewriter Transfer Timeout -----
            // Guard: Timeout from transfer layer
            // Action: Return to offline state
            if (status == TS_STATUS_TIMEOUT) {
                DBG_EVENT("STANDBY TRANSFER TIMEOUT");
                transitionToOffline(ps);
                return PS_STATUS_OFFLINE;
            }
            
            // Check if PC send something
            if (!siBufferEmpty()) {
                uint8_t si_byte = siBufferPeek();
                
                // Discard unexpected bytes
                if(si_byte != 0x11){
                    siBufferPop(&si_byte);
                    DBG_EVENT_HEX("STANDBY: DISCARDED BYTE", si_byte);
                } 
                
                // ----- TRANSITION: DC1 received from PC -----
                // Guard: siBuffer has DC1 (0x11)
                // Action: Consume DC1, begin SELECT sequence
                if (si_byte == 0x11) {
                    siBufferPop(&si_byte);
                    DBG_EVENT("DC1 RECEIVED - START SELECT");
                    ps->selectState = SEL_QUEUE_MODE;
                    transitionTo(ps, PS_SELECT);
                    return PS_STATUS_SELECTING;
                }
            }
            
            // No transition (waiting for DC1)
            return PS_STATUS_STANDBY;
        
        // ==============================================
        // PS_SELECT — Mode selection sequence
        // ==============================================
        //
        // Entry: DC1 received from PC
        // Exit:  SELECT sequence complete → ONLINE, or timeout → OFFLINE
        //
        // Sequence (AX20):
        //   1. Send 0xF9 (terminal mode)
        //   2. Send 0xFD
        //   3. Wait for 0x04 (EOT) from typewriter
        //   4. Send 0xF4 (reset margins + newline)
        //   5. Send 0xB1 (pitch) ?
        //   6. Send 0xB1 (pitch again) ?
        //   → ONLINE
        //
        case PS_SELECT:

            // ----- Global: Transfer Timeout -----
            if (status == TS_STATUS_TIMEOUT) {
                DBG_EVENT("SELECT TRANSFER TIMEOUT");
                transitionToOffline(ps);
                return PS_STATUS_OFFLINE;
            }

            switch (ps->selectState) {
                
                // ----- Send Mode Byte -----
                case SEL_QUEUE_MODE:
                    if (status == TS_STATUS_IDLE) {
                        DBG_EVENT_HEX("SELECT: SEND MODE", MODE_BYTE);
                        transferQueueSI(ts, MODE_BYTE);
                        ps->selectState = SEL_WAIT_MODE;
                    }
                    break;
                    
                case SEL_WAIT_MODE:
                    if (status == TS_STATUS_SI_DONE) {
                        ps->selectState = SEL_QUEUE_FD;
                    }
                    break;
                
                // ----- Send 0xFD ----- 
                // TODO: Find out if static or dynamic
                // Still in question what this byte does.
                // Typewriter will respond after so probably
                // some kind of Query.
                case SEL_QUEUE_FD:
                    if (status == TS_STATUS_IDLE) {
                        DBG_EVENT_HEX("SELECT: QUEUE", 0xFD);
                        transferQueueSI(ts, 0xFD);
                        ps->selectState = SEL_WAIT_FD;
                    }
                    break;
                    
                case SEL_WAIT_FD:
                    if (status == TS_STATUS_SI_DONE) {
                        ps->selectState = SEL_WAIT_KBBYTE;
                    }
                    break;
                
                // ----- Wait for Keyboard-Byte from Typewriter -----
                case SEL_WAIT_KBBYTE:
                    if (status == TS_STATUS_SO_DONE) {
                        switch (ts->receivedByte){
                            case KEYBOARD_KB1:
                                ps->keyboardID = KEYBOARD_KB1;
                                DBG_EVENT_HEX("SELECT: KBCONFIG LOCAL (KB1)", KEYBOARD_KB1);
                                break;
                            case KEYBOARD_KB2:
                                ps->keyboardID = KEYBOARD_KB2;
                                DBG_EVENT_HEX("SELECT: KBCONFIG INTERNATIONAL (KB2)", KEYBOARD_KB2);
                                break;
                            case KEYBOARD_KB3:
                                ps->keyboardID = KEYBOARD_KB3;
                                DBG_EVENT_HEX("SELECT: KBCONFIG SYMBOL (KB3)", KEYBOARD_KB3);
                                break;
                            default:
                                ps->keyboardID = KEYBOARD_KB1;
                                DBG_EVENT_HEX("SELECT: KBCONFIG ERROR, DEFAULTING KB1", ts->receivedByte);
                                break;
                        }
                        ps->selectState = SEL_QUEUE_F4;
                    }
                    break;
                
                // ----- Send 0xF4 (Reset margins + newline) -----
                case SEL_QUEUE_F4:
                    if (status == TS_STATUS_IDLE) {
                        DBG_EVENT_HEX("SELECT: QUEUE", 0xF4);
                        transferQueueSI(ts, 0xF4);
                        ps->selectState = SEL_WAIT_F4;
                    }
                    break;
                    
                case SEL_WAIT_F4:
                    if (status == TS_STATUS_SI_DONE) {
                        ps->selectState = SEL_QUEUE_PITCH1;
                    }
                    break;
                
                // ----- Send Pitch (0xB1 = 10 CPI) -----
                case SEL_QUEUE_PITCH1:
                    if (status == TS_STATUS_IDLE) {
                        DBG_EVENT_HEX("SELECT: QUEUE PITCH1", 0xB1);
                        transferQueueSI(ts, 0xB1);
                        ps->selectState = SEL_WAIT_PITCH1;
                    }
                    break;
                    
                case SEL_WAIT_PITCH1:
                    if (status == TS_STATUS_SI_DONE) {
                        ps->selectState = SEL_QUEUE_PITCH2;
                    }
                    break;
                
                // ----- Send Pitch Again (0xB1) -----
                case SEL_QUEUE_PITCH2:
                    if (status == TS_STATUS_IDLE) {
                        DBG_EVENT_HEX("SELECT: QUEUE PITCH2", 0xB1);
                        transferQueueSI(ts, 0xB1);
                        ps->selectState = SEL_WAIT_PITCH2;
                    }
                    break;
                    
                case SEL_WAIT_PITCH2:
                    if (status == TS_STATUS_SI_DONE) {
                        ps->selectState = SEL_COMPLETE;
                    }
                    break;
                
                // ----- TRANSITION: SELECT sequence complete -----
                // Guard: reached SEL_COMPLETE Sub-State
                // Action: Enter Operation
                case SEL_COMPLETE:
                    DBG_EVENT("SELECT COMPLETE");
                    ps->selectState = SEL_IDLE;
                    transitionTo(ps, PS_ONLINE);
                    return PS_STATUS_ONLINE;
                
                // ----- Invalid State -----
                default:
                    DBG_ERROR("SELECT: INVALID STATE");
                    ps->selectState = SEL_IDLE;
                    transitionTo(ps, PS_STANDBY);
                    break;
            }
            
            return PS_STATUS_SELECTING;
        
        // ==================================================
        // PS_ONLINE — Bidirectional/Unidirectional data mode
        // ==================================================
        //
        // Entry: SELECT sequence complete
        // Exit:  DESELECT trigger (DC3) or timeout
        //
        // Acts as a bridge between buffers and transfer layer:
        //   - siBuffer → transfer layer → typewriter
        //   - typewriter → transfer layer → soBuffer
        //
        case PS_ONLINE:
            // TODO: TRANSITION: DESELECT trigger received
            // Guard: DC3 from PC or typewriter DESELECT request
            // Action: Begin DESELECT sequence
            
            // ----- TRANSITION: Typewriter Timeout -----
            // Guard: Timeout from transfer layer
            // Action: Return to offline state
            if (status == TS_STATUS_TIMEOUT) {
                DBG_EVENT("TRANSFER TIMEOUT");
                transitionToOffline(ps);
                return PS_STATUS_OFFLINE;
            }
            
            // ----- ACTION: Drain forward buffer -----
            // Guard: Transfer layer idle AND forward buffer has pending bytes
            // 
            // Multi-byte translations (e.g. Code key simulation [0x88, pos, 0x89])
            // are loaded into fwdBuf and sent one byte per transfer cycle.
            // This block drains those bytes before translating new serial input.
            //
            // Safe to return early: if a typewriter byte arrives, status will be
            // TS_STATUS_SO_DONE (not IDLE), so this block is skipped and the
            // SO_DONE handler below processes the received byte.
            //
            if (status == TS_STATUS_IDLE && fwdBufHasData(ps)) {
                uint8_t b = ps->fwdBuf[ps->fwdIdx];
                if (transferQueueSI(ts, b)) {
                    ps->fwdIdx++;
                    DBG_EVENT_HEX("SI QUEUED", b);
                }
                return PS_STATUS_ONLINE;
            }

            // ----- ACTION: Translate next serial byte -----
            // Guard: Transfer layer idle AND forward buffer fully drained AND siBuffer has data
            //
            // Peeks at the next serial byte to check for protocol control (DC3),
            // then translates printable/control bytes into bus byte(s) via
            // translateSerialToBus(). The result is loaded into the forward
            // buffer; the first byte is sent immediately, the rest will be
            // drained by the block above on subsequent loop iterations.
            //
            if (status == TS_STATUS_IDLE && !fwdBufHasData(ps) && !siBufferEmpty()) {
                uint8_t si_byte = siBufferPeek();

                // DC3 → begin DESELECT (consumed but not translated)
                if (si_byte == 0x13) {
                    siBufferPop(&si_byte);
                    DBG_EVENT("DC3 RECEIVED - START DESELECT");
                    ps->deselectState = DESEL_QUEUE;
                    transitionTo(ps, PS_DESELECT);
                    return PS_STATUS_DESELECTING;
                }

                // Consume and translate
                siBufferPop(&si_byte);
                TranslateResult r = translateSerialToBus(si_byte, ps->keyboardID);

                // Unknown or swallowed byte (e.g. unsupported control codes)
                if (r.len == 0) {
                    DBG_EVENT_HEX("SI SWALLOWED", si_byte);
                    return PS_STATUS_ONLINE;
                }

                // Load into forward buffer — will be drained starting next cycle
                fwdBufLoadResult(ps, &r);
                DBG_EVENT_HEX_CHAR("SI TRANSLATED", si_byte);
                return PS_STATUS_ONLINE;
            }
            
            // ----- ACTION: Handle received byte -----
            // Guard: SO transfer complete
            // Action: Track Code key state, translate, push to soBuffer
            if (status == TS_STATUS_SO_DONE) {
                uint8_t b = ts->receivedByte;
                DBG_EVENT_HEX("SO RECEIVED", b);

                // Code Modifier Tracking
                if (b == 0x88) { ps->codePressed = true;  return PS_STATUS_ONLINE; }
                if (b == 0x89) { ps->codePressed = false; return PS_STATUS_ONLINE; }

                // Non-Translating Keys (Correction, Relocate, etc.)
                if (isNonTranslatingKey(b)) return PS_STATUS_ONLINE;

                // Code+M: CR with optional auto-LF
                if (ps->codePressed && b == 0x6D) {
                    soBufferPush(0x0D);
                    if (ps->autoLfEnabled) soBufferPush(0x0A);
                    return PS_STATUS_ONLINE;
                }

                // Translation Dispatch
                TranslateResult r;
                if (ps->codePressed) {
                    r = translateCodeBusToSerial(b, ps->keyboardID);
                } else {
                    r = translateNormalBusToSerial(b, ps->keyboardID);
                }

                // Push Results to Serial Out Buffer
                for (uint8_t i = 0; i < r.len; i++) {
                    soBufferPush(r.bytes[i]);
                    DBG_EVENT_HEX_CHAR("SO MAPPED", r.bytes[i]);
                }
                return PS_STATUS_ONLINE;
            }
            
            // No transition (idle or transfer in progress)
            return PS_STATUS_ONLINE;
        
        // ==============================================
        // PS_DESELECT — Disconnect sequence
        // ==============================================
        //
        // Entry: DC3 received from PC while ONLINE
        // Exit:  DESELECT byte sent → STANDBY, or timeout → OFFLINE
        //
        // Sequence depends on device type:
        //   0x30: Send 0xF8
        //   other: TODO
        //
        case PS_DESELECT:

            // ----- Global: Transfer Timeout -----
            if (status == TS_STATUS_TIMEOUT) {
                DBG_EVENT("DESELECT TRANSFER TIMEOUT");
                transitionToOffline(ps);
                return PS_STATUS_OFFLINE;
            }

            switch (ps->deselectState) {
                
                // Send deselect byte based on device type
                case DESEL_QUEUE:
                    if (status == TS_STATUS_IDLE) {
                        if (ps->deviceType == 0x30) {
                            DBG_EVENT_HEX("DESELECT: QUEUE", 0xF8);
                            transferQueueSI(ts, 0xF8);
                            ps->deselectState = DESEL_WAIT;
                        } else {
                            // Unknown device type
                            DBG_EVENT_HEX("DESELECT: UNKNOWN DEVICE", ps->deviceType);
                            transferQueueSI(ts, 0xF8);
                            ps->deselectState = DESEL_WAIT;
                        }
                    }
                    break;
                    
                // Wait for SI Transfer to Complete
                case DESEL_WAIT:
                    if (status == TS_STATUS_SI_DONE) {
                        ps->deselectState = DESEL_COMPLETE;
                    }
                    break;
                
                // ----- TRANSITION: DESELECT complete -----
                // Guard: reached DESEL_COMPLETE Substate
                // Action: Return to STANDBY
                case DESEL_COMPLETE:
                    DBG_EVENT("DESELECT COMPLETE");
                    ps->deselectState = DESEL_IDLE;
                    transitionTo(ps, PS_STANDBY);
                    return PS_STATUS_STANDBY;
                
                // ----- Invalid State -----
                default:
                    DBG_ERROR("DESELECT: INVALID STATE");
                    ps->deselectState = DESEL_QUEUE;
                    break;
            }
            
            // No transition (deselecting in progress)
            return PS_STATUS_DESELECTING;
        
        // ------------------------------------------
        // DEFAULT — Should never happen
        // ------------------------------------------
        default:
            transitionToOffline(ps);
            return PS_STATUS_OFFLINE;
    }
}