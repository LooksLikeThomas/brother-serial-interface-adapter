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
    ps->stateEnteredAt = micros();
    ps->state = newState;
}

// ==============================================
// Public: Initialize Protocol Layer
// ==============================================

void protocolInit(Protocol *ps) {
    ps->state = PS_OFFLINE;
    ps->stateEnteredAt = 0;
    ps->deviceType = 0;
    ps->selectStep = 0;
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
ProtocolStatus pollProtocol(Protocol *ps, TransferStatus status, Transfer *ts) {
    
    switch (ps->state) {
        
        // ==========================================
        // PS_OFFLINE — Waiting for Typewriter
        // ==========================================
        //
        // Entry: Init, or after timeout/disconnect
        // Exit:  Typewriter comes online (KBRQ stable LOW)
        //
        case PS_OFFLINE:
        
            // ----- TRANSITION: Typewriter detected -----
            // Guard: isTypewriterOnline() returns true
            // Action: Begin startup sequence
            if (isTypewriterOnline()) {
                transitionTo(ps, PS_STARTUP_INIT);
                return PS_STATUS_STARTUP;
            }
            
            // No transition
            return PS_STATUS_OFFLINE;
        
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
                ps->deviceType = ts->receivedByte;
                transitionTo(ps, PS_STANDBY);
                return PS_STATUS_STANDBY;
            }
            
            // ----- TRANSITION: No response -----
            // Guard: Timeout from transfer layer
            // Action: Return to offline state
            if (status == TS_STATUS_TIMEOUT) {
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
        
            // ----- TRANSITION: Typewriter went offline -----
            // Guard: Timeout from transfer layer
            // Action: Return to offline state
            if (status == TS_STATUS_TIMEOUT) {
                transitionTo(ps, PS_OFFLINE);
                return PS_STATUS_OFFLINE;
            }
            
            // ----- TRANSITION: Silent disconnect -----
            // Guard: Idle and typewriter no longer online
            // Action: Return to offline state
            if (status == TS_STATUS_IDLE && !isTypewriterOnline()) {
                transitionTo(ps, PS_OFFLINE);
                return PS_STATUS_OFFLINE;
            }
            
            // TODO: TRANSITION: SELECT trigger received
            // Guard: DC1 from PC or typewriter SELECT request
            // Action: Begin SELECT sequence
            
            // No transition
            return PS_STATUS_STANDBY;
        
        // ==========================================
        // PS_SELECT — Mode selection sequence
        // ==========================================
        //
        // Entry: SELECT trigger received
        // Exit:  SELECT sequence complete, or timeout
        //
        case PS_SELECT:
        
            // ----- TRANSITION: Typewriter went offline -----
            // Guard: Timeout from transfer layer
            // Action: Return to offline state
            if (status == TS_STATUS_TIMEOUT) {
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
        
            // ----- TRANSITION: Typewriter went offline -----
            // Guard: Timeout from transfer layer
            // Action: Return to offline state
            if (status == TS_STATUS_TIMEOUT) {
                transitionTo(ps, PS_OFFLINE);
                return PS_STATUS_OFFLINE;
            }
            
            // ----- TRANSITION: Silent disconnect -----
            // Guard: Idle and typewriter no longer online
            // Action: Return to offline state
            if (status == TS_STATUS_IDLE && !isTypewriterOnline()) {
                transitionTo(ps, PS_OFFLINE);
                return PS_STATUS_OFFLINE;
            }
            
            // TODO: TRANSITION: DESELECT trigger received
            // Guard: DC3 from PC or typewriter DESELECT request
            // Action: Begin DESELECT sequence
            
            // ----- ACTION: Send next byte from buffer -----
            // Guard: Idle and siBuffer has data
            // Action: Queue byte to transfer layer
            if (status == TS_STATUS_IDLE && !siBufferEmpty()) {
                uint8_t byte = siBufferPeek();
                if (transferQueueSI(ts, byte)) {
                    siBufferPop(&byte);
                }
                return PS_STATUS_ONLINE;
            }
            
            // ----- ACTION: Store received byte -----
            // Guard: SO transfer complete
            // Action: Push byte to soBuffer
            if (status == TS_STATUS_SO_DONE) {
                soBufferPush(ts->receivedByte);
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