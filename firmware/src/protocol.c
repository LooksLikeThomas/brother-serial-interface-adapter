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
// Public: Initialize Protocol Layer
// ==============================================

void protocolInit(Protocol *ps) {
    ps->state = PS_OFFLINE;
    ps->stateEnteredAt = 0;
    ps->deviceType = 0;
    ps->selectStep = 0;

    // Transfer layer not initialized yet
    ps->ts.state = TS_NOT_INIT;
    ps->ts.stateEnteredAt = 0;
    ps->ts.lastWasSI = true;
    ps->ts.receivedByte = 0;
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
        transferDeinit(ts);
        transitionTo(ps, PS_OFFLINE);
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
        transferDeinit(ts);
        transitionTo(ps, PS_OFFLINE);
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
                transferDeinit(ts);
                transitionTo(ps, PS_OFFLINE);
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
                transferDeinit(ts);
                DBG_EVENT("STARTUP TRANSFER TIMEOUT");
                transitionTo(ps, PS_OFFLINE);
                return PS_STATUS_OFFLINE;
            }

            // ----- TRANSITION: Typewriter Response Timeout -----
            // Guard: 1s since Startup Request
            // Action: Return to offline state
            if (now - ps->stateEnteredAt >= 1000000) {
                transferDeinit(ts);
                DBG_EVENT("STARTUP RESPONSE TIMEOUT");
                transitionTo(ps, PS_OFFLINE);
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
            
            // TODO: TRANSITION: SELECT trigger received
            // Guard: DC1 from PC or typewriter SELECT request
            // Action: Begin SELECT sequence
            
            transitionTo(ps, PS_SELECT);
            return PS_STATUS_STANDBY;
        
        // ==========================================
        // PS_SELECT — Mode selection sequence
        // ==========================================
        //
        // Entry: SELECT trigger received
        // Exit:  SELECT sequence complete, or timeout
        //
        case PS_SELECT:
        
            // ----- TRANSITION: Typewriter Timeout -----
            // Guard: Timeout from transfer layer
            // Action: Return to offline state
            if (status == TS_STATUS_TIMEOUT) {
                DBG_EVENT("SELECT TRANSFER TIMEOUT");
                transitionTo(ps, PS_OFFLINE);
                return PS_STATUS_OFFLINE;
            }
            
            // TODO: Implement SELECT sequence using selectStep
            // For now, go directly to online
            ps->selectStep = 0;
            transitionTo(ps, PS_ONLINE);
            return PS_STATUS_ONLINE;
        
        // ==========================================
        // PS_ONLINE — Bidirectional data mode
        // ==========================================
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
                transitionTo(ps, PS_OFFLINE);
                return PS_STATUS_OFFLINE;
            }
            
            // ----- ACTION: Send next byte from buffer -----
            // Guard: Idle and siBuffer has data
            // Action: Queue byte to transfer layer
            if (status == TS_STATUS_IDLE && !siBufferEmpty()) {
                uint8_t si_byte = siBufferPeek();
                if (transferQueueSI(ts, si_byte)) {
                    DBG_EVENT_HEX("SI QUEUED", si_byte);
                    siBufferPop(&si_byte);
                }else{
                    DBG_ERROR("QUEUED BYTE COLLISION");
                }
                return PS_STATUS_ONLINE;
            }
            
            // ----- ACTION: Store received byte -----
            // Guard: SO transfer complete
            // Action: Push byte to soBuffer
            if (status == TS_STATUS_SO_DONE) {
                soBufferPush(ts->receivedByte);
                DBG_EVENT_HEX("SO RECEIVED", ts->receivedByte);
                return PS_STATUS_ONLINE;
            }
            
            // No transition (idle or transfer in progress)
            return PS_STATUS_ONLINE;
        
        // ==========================================
        // PS_DESELECT — Disconnect sequence
        // ==========================================
        //
        // Entry: DESELECT trigger received
        // Exit:  DESELECT sequence complete → STANDBY, or timeout
        //
        case PS_DESELECT:
        
            // ----- TRANSITION: Typewriter went offline -----
            // Guard: Timeout from transfer layer
            // Action: Return to offline state
            if (status == TS_STATUS_TIMEOUT) {
                DBG_EVENT("TRANSFER TIMEOUT");
                transitionTo(ps, PS_OFFLINE);
                return PS_STATUS_OFFLINE;
            }
            
            // TODO: Implement DESELECT sequence
            // For now, go directly to standby
            transitionTo(ps, PS_STANDBY);
            return PS_STATUS_STANDBY;
        
        // ------------------------------------------
        // DEFAULT — Should never happen
        // ------------------------------------------
        default:
            transitionTo(ps, PS_OFFLINE);
            return PS_STATUS_OFFLINE;
    }
}