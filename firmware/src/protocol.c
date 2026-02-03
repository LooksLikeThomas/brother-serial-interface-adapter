// ==============================================
// protocol.c — Protocol Sequence Implementation
// ==============================================
//
// Handles multi-byte protocol sequences by calling into
// the transfer layer for individual byte transfers.
//
// Flow:
//   PS_TW_OFF → PS_STARTUP_INIT → PS_STARTUP_RESPONSE → PS_SELECT → PS_READY
//
// In PS_READY, the protocol layer acts as a pass-through:
//   - Pops bytes from siBuffer and hands them to transferStartSI()
//   - Pushes received bytes (from transfer layer) into soBuffer
//
// On timeout at any stage, returns to PS_TW_OFF.
//
#include "protocol.h"
#include "transfer.h"
#include "buffers.h"

// For micros() — remove when porting away from Arduino
#include <Arduino.h>

// ==============================================
// Public: Initialize Protocol Layer
// ==============================================

void protocolInit(Protocol *ps) {
    ps->state = PS_TW_OFF;
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
void pollProtocol(Protocol *ps, TransferStatus status, Transfer *ts) {
    
    uint32_t now = micros();
    
    switch (ps->state) {
        
        // ==========================================
        // TYPEWRITER OFF — Waiting for Typewriter
        // ==========================================
        //
        // Polls isTypewriterOnline() which checks if KBRQ
        // has been stable LOW for 100ms (handled by transfer layer).
        //
        case PS_TW_OFF:
            if (isTypewriterOnline()) {
                ps->stateEnteredAt = now;
                ps->state = PS_STARTUP_INIT;
            }
            break;
        
        // ==========================================
        // STARTUP: Send 0xFE (Interface announces presence)
        // ==========================================
        //
        // Wait for transfer layer to be idle, then send 0xFE.
        // After send completes, wait for typewriter response.
        //
        case PS_STARTUP_INIT:
            if (status == TS_STATUS_IDLE) {
                transferQueueSI(ts, 0xFE);
            }else if(status == TS_STATUS_SI_BUSY){ 
                // 0xFE Transfer in progress
            }else if (status == TS_STATUS_SI_DONE) {
                // 0xFE sent successfully, wait for device type response
                ps->stateEnteredAt = now;
                ps->state = PS_STARTUP_RESPONSE;
            }
            else if (status == TS_STATUS_TIMEOUT) {
                // Typewriter didn't respond, go back to off
                ps->stateEnteredAt = now;
                ps->state = PS_TW_OFF;
            }
            break;
        
        // ==========================================
        // STARTUP: Wait for Device Type Response
        // ==========================================
        //
        // Typewriter should respond with device type byte (e.g. 0x30).
        // This comes as an SO transfer initiated by the typewriter.
        //
        case PS_STARTUP_RESPONSE:
            if (status == TS_STATUS_SO_DONE) {
                // Store device type for later use (SELECT, DESELECT)
                ps->deviceType = ts->receivedByte;
                // Startup complete, move to SELECT
                ps->selectStep = 0;
                ps->stateEnteredAt = now;
                ps->state = PS_SELECT;
            }
            else if (status == TS_STATUS_TIMEOUT) {
                ps->stateEnteredAt = now;
                ps->state = PS_TW_OFF;
            }
            break;
        
        // ==========================================
        // SELECT — Mode Selection Sequence
        // ==========================================
        //
        // Placeholder: Will sequence through SELECT bytes
        // using selectStep counter. For now, skip to READY.
        //
        case PS_SELECT:
            // TODO: Implement SELECT sequence
            // For now, go directly to normal operation
            ps->stateEnteredAt = now;
            ps->state = PS_READY;
            break;
        
        // ==========================================
        // READY — Normal Operation (Pass-Through)
        // ==========================================
        //
        // In this state, the protocol layer acts as a bridge:
        //   - If transfer is idle and siBuffer has data → send next byte
        //   - If SO transfer completed → push received byte to soBuffer
        //   - On timeout → typewriter went offline
        //
        case PS_READY:
            if (status == TS_STATUS_IDLE && !siBufferEmpty()) {
                // Peek first, only consume on successful queue
                uint8_t byte = siBufferPeek();
                if (transferQueueSI(ts, byte)) {
                    siBufferPop(&byte);
                }
            }
            else if (status == TS_STATUS_SO_DONE) {
                // Store received byte in SO buffer for application layer
                soBufferPush(ts->receivedByte);
            }
            else if (status == TS_STATUS_TIMEOUT) {
                // Typewriter went offline
                ps->stateEnteredAt = now;
                ps->state = PS_TW_OFF;
            }else if (status == TS_STATUS_IDLE && !isTypewriterOnline()) {
                // Detect silent disconnection while idle
                ps->stateEnteredAt = now;
                ps->state = PS_TW_OFF;
            }
            break;
        
        // ==========================================
        // DESELECT — Disconnect Sequence
        // ==========================================
        //
        // Placeholder: Will handle device-specific disconnect.
        //
        case PS_DESELECT:
            // TODO: Implement DESELECT sequence
            break;
    }
}
