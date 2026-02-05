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

static bool flowStopped = false;

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
    Serial.begin(115200);

    // Initialize debug system
    DBG_INIT();
    
    // Initialize protocol layer (state machine)
    // Transfer layer will be initialized when typewriter powers on
    protocolInit(&ps);
    
    DBG_EVENT("SETUP COMPLETE");
}

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
    // Run state machines
    pollProtocol(&ps);
    
     // Flow control
    uint8_t bufLevel = siBufferCount();
    if (!flowStopped && bufLevel >= FLOW_HIGH_WATER) {
        Serial.write(XOFF);
        flowStopped = true;
        DBG_EVENT_HEX("FLOW: XOFF SENT, BUFFER AT", bufLevel);
    } else if (flowStopped && bufLevel <= FLOW_LOW_WATER) {
        Serial.write(XON);
        flowStopped = false;
        DBG_EVENT_HEX("FLOW: XON SENT, BUFFER AT", bufLevel);
    }
    
    // Echo anything received from typewriter to Serial
    uint8_t so_byte;
    if (soBufferPop(&so_byte)) {

        if (DEBUG_ENABLED){
            DBG_EVENT_HEX("SERIAL SEND", so_byte);
        }else{
            Serial.print(so_byte);
        }
    }
    
    // Send anything received from Serial to typewriter
    if (Serial.available()) {
        uint8_t serial_byte = Serial.read();

        DBG_EVENT_HEX("SERIAL RECEIVED", serial_byte);
        
        siBufferPush(serial_byte);
    }
}