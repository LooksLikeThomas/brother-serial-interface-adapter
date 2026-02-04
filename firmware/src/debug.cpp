// ==============================================
// debug.cpp — Debug Logging Implementation
// ==============================================
//
// Buffered debug output that won't interfere with
// timing-critical transfer operations.
//
// Messages are accumulated in a ring buffer and only
// sent over Serial when the transfer layer is idle.
//
// Note: This is a .cpp file to access Arduino's Serial class.
//
#include "debug.h"

#if DEBUG_ENABLED

#include <Arduino.h>
#include <string.h>

// Include these for the enum definitions (used in name lookup tables)
#include "transfer.h"
#include "protocol.h"

// ==============================================
// Buffer Configuration
// ==============================================

#define DEBUG_BUFFER_SIZE 256

static char debugBuffer[DEBUG_BUFFER_SIZE];
static volatile uint16_t bufferHead = 0;  // Write position
static volatile bool overflowOccurred = false;
static volatile uint32_t lastLogTime = 0;  // micros() of last log entry

// Overflow message appended on flush if overflow occurred
static const char OVERFLOW_MSG[] = "[OVERFLOW]\n";

// ==============================================
// State and Status Name Tables
// ==============================================

static const char* getProtocolStateName(ProtocolState state) {
    switch (state) {
        case PS_OFFLINE:          return "PS_OFFLINE";
        case PS_STARTUP_SETTLE:   return "PS_STARTUP_SETTLE";
        case PS_STARTUP_INIT:     return "PS_STARTUP_INIT";
        case PS_STARTUP_RESPONSE: return "PS_STARTUP_RESPONSE";
        case PS_STANDBY:          return "PS_STANDBY";
        case PS_SELECT:           return "PS_SELECT";
        case PS_ONLINE:           return "PS_ONLINE";
        case PS_DESELECT:         return "PS_DESELECT";
        default:                  return "PS_UNKNOWN";
    }
}

static const char* getTransferStatusName(TransferStatus status) {
    switch (status) {
        case TS_STATUS_NOT_INIT:  return "TS_NOT_INIT";
        case TS_STATUS_IDLE:      return "TS_IDLE";
        case TS_STATUS_SI_BUSY:   return "TS_SI_BUSY";
        case TS_STATUS_SI_DONE:   return "TS_SI_DONE";
        case TS_STATUS_SO_BUSY:   return "TS_SO_BUSY";
        case TS_STATUS_SO_DONE:   return "TS_SO_DONE";
        case TS_STATUS_TIMEOUT:   return "TS_TIMEOUT";
        case TS_STATUS_ERROR:     return "TS_ERROR";
        default:                  return "TS_UNKNOWN";
    }
}

// ==============================================
// Buffer Management
// ==============================================

// Append a string to the buffer, set overflow flag if full
static void bufferAppend(const char* str) {
    uint16_t len = strlen(str);
    
    // Check if we have room (leave space for potential overflow message later)
    if (bufferHead + len >= DEBUG_BUFFER_SIZE - sizeof(OVERFLOW_MSG)) {
        overflowOccurred = true;
        return;
    }
    
    // Copy string to buffer
    memcpy(&debugBuffer[bufferHead], str, len);
    bufferHead += len;
}

// Append a number to the buffer (for timestamps)
static void bufferAppendNumber(uint32_t num) {
    char numStr[12];  // Max 10 digits + sign + null
    ltoa(num, numStr, 10);
    bufferAppend(numStr);
}

// Append timestamp prefix: "+1234us\t"
// Shows microseconds since last log entry
static void bufferAppendTimestamp(void) {
    uint32_t now = micros();
    uint32_t delta = now - lastLogTime;
    lastLogTime = now;
    
    bufferAppend("+");
    bufferAppendNumber(delta);
    if(delta >= 10000){
        bufferAppend("us\t");
    }else{
        bufferAppend("us\t\t");
    }
}

// ==============================================
// Public API
// ==============================================

void debugInit(void) {
    Serial.begin(115200);
    bufferHead = 0;
    overflowOccurred = false;
    lastLogTime = micros();  // Initialize timestamp baseline
    
    // Wait a moment for Serial to initialize
    delay(10);
    
    // Send startup message directly (we're in setup, no timing concerns)
    Serial.println(F("=== DEBUG INIT ==="));
}

void debugTransition(int from, int to, int status) {
    // Format: "+1234us\tPS_FROM -> PS_TO\t(TS_STATUS)\n"
    bufferAppendTimestamp();
    bufferAppend(getProtocolStateName((ProtocolState)from));
    bufferAppend(" -> ");
    bufferAppend(getProtocolStateName((ProtocolState)to));
    bufferAppend("\t(");
    bufferAppend(getTransferStatusName((TransferStatus)status));
    bufferAppend(")\n");
}

void debugEvent(const char* event) {
    // Format: "+1234us [EVENT]\n"
    bufferAppendTimestamp();
    bufferAppend("[");
    bufferAppend(event);
    bufferAppend("]\n");
}

bool debugCanFlush(int status) {
    // Safe to flush when transfer layer is not actively transferring
    // TS_STATUS_NOT_INIT = 0
    // TS_STATUS_IDLE = 1
    // TS_STATUS_SI_BUSY = 2  (not safe)
    // TS_STATUS_SI_DONE = 3
    // TS_STATUS_SO_BUSY = 4  (not safe)
    // TS_STATUS_SO_DONE = 5
    // TS_STATUS_TIMEOUT = 6
    // TS_STATUS_ERROR = 7
    
    return (status != TS_STATUS_SI_BUSY && status != TS_STATUS_SO_BUSY);
}

void debugFlush(void) {
    // Nothing to flush
    if (bufferHead == 0 && !overflowOccurred) {
        return;
    }
    
    // Send buffered content
    if (bufferHead > 0) {
        // Null-terminate for safety
        debugBuffer[bufferHead] = '\0';
        Serial.print(debugBuffer);
        bufferHead = 0;
    }
    
    // Append overflow message if overflow occurred
    if (overflowOccurred) {
        Serial.print(OVERFLOW_MSG);
        overflowOccurred = false;
    }
}

#endif // DEBUG_ENABLED