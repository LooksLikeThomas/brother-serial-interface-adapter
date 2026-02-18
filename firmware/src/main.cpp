// ==============================================
// Brother Serial Interface Firmware — Main Application
// ==============================================
//
// Brother AX-Series Serial Interface
//
// Architecture:
//   hardware.h    — Pin definitions, helpers, setup
//   buffers.h/c   — Ring buffers for SI (outgoing) and SO (incoming)
//   transfer.h/c  — Single-byte transfer handshake (SI and SO paths)
//   protocol.h/c  — Multi-byte protocol sequences (STARTUP, SELECT, etc.)
//   this file     — Setup, main loop, Serial bridge
//
// Data flow:
//   Serial RX → siBuffer → protocol layer → transfer layer → typewriter
//   typewriter → transfer layer → protocol layer → soBuffer → Serial TX
//
#include <Arduino.h>
#include "hardware.h"
#include "buffers.h"
#include "transfer.h"
#include "protocol.h"
#include "debug.h"

// ==============================================
// Global State
// ==============================================

#if FLOWCONTROL_ENABLED
    static bool flowStopped = false;
#endif

Protocol ps;    // Protocol layer state

// ==============================================
// Setup
// ==============================================

void setup() {
    // Set up pins 
    setupPins();

    // wait for everything to settle
    delay(100);

    // Initialize Serial for data transfer (and debug if enabled)
    Serial.begin(SERIAL_BAUD);

    // Initialize debug system
    DBG_INIT();
    
    // Initialize protocol layer (state machine)
    // Transfer layer will be initialized when typewriter powers on
    protocolInit(&ps);
    
    DBG_EVENT("SETUP COMPLETE");
}

// ==============================================
// Helper Methods
// ==============================================
// Checks buffer levels and manages Software Flow Control (XON/XOFF).
// Must be called every loop iteration to prevent deadlocks.
//
#if FLOWCONTROL_ENABLED
void manageFlowControl() {
    uint8_t bufLevel = siBufferCount();

    if (!flowStopped && bufLevel >= FLOW_HIGH_WATER) {
        Serial.write(0x13); // XOFF
        flowStopped = true;
        DBG_EVENT_HEX("FLOW: XOFF SENT, BUFFER AT", bufLevel);
    } 
    else if (flowStopped && bufLevel <= FLOW_LOW_WATER) {
        Serial.write(0x11); // XON
        flowStopped = false;
        DBG_EVENT_HEX("FLOW: XON SENT, BUFFER AT", bufLevel);
    }
}
#endif

// ==============================================
// Main Loop
// ==============================================
//
// Each iteration:
//   1. Run transfer state machine (handles byte-level handshake)
//   2. Run protocol state machine (handles sequences and pass-through)
//   3. Bridge SO buffer to Serial output
//   4. Bridge Serial input to SI buffer
//
// IMPORTANT: pollProtocol must run in the same iteration
// as pollTransfer — done statuses are one-shot.
// See TransferStatus contract in transfer.h.
//
void loop() {
    // Empty UART Buffer and push to SI-Puffer
    uint8_t readCount = 0;
    
    while (Serial.available() && readCount < 64) {
        uint8_t serial_byte = Serial.read();
        
        // Try to push to buffer
        if (!siBufferPush(serial_byte)) {
            DBG_ERROR("SI BUFFER FULL - BYTE DROPPED");
        } else {
            readCount++;
        }
    }
    
    if (readCount > 0) {
        #if FLOWCONTROL_ENABLED
            manageFlowControl();
        #endif
        
        DBG_EVENT_HEX("SERIAL BATCH READ", readCount);
    }

    // Run state machines
    pollProtocol(&ps);
        
    // Echo anything received from typewriter to Serial
    uint8_t so_byte;
    if (soBufferPop(&so_byte)) {

        if (DEBUG_ENABLED){
            DBG_EVENT_HEX_CHAR("SERIAL SEND", so_byte);
        }else{
            Serial.write(so_byte);
        }
    }
}