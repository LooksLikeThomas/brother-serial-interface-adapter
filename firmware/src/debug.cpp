// ==============================================
// debug.cpp — Debug Logging Implementation
// ==============================================
#include "debug.h"

#if DEBUG_ENABLED

#include <Arduino.h>
#include <string.h>
#include "transfer.h"
#include "protocol.h"

// ==============================================
// Buffer Configuration
// ==============================================

#define DEBUG_BUFFER_SIZE 512
#define DEBUG_FLUSH_CHUNK_SIZE 16

static char debugBuffer[DEBUG_BUFFER_SIZE];
static volatile uint16_t bufferHead = 0;  // Write position
static volatile uint16_t bufferTail = 0;  // Read position
static volatile bool overflowOccurred = false;
static volatile uint32_t lastLogTime = 0;

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

static void bufferAppend(const char* str) {
    uint16_t len = strlen(str);
    uint16_t available = DEBUG_BUFFER_SIZE - bufferHead - 1; // Reserve space for overflow msg
    
    if (len > available - sizeof(OVERFLOW_MSG)) {
        overflowOccurred = true;
        return;
    }
    memcpy(&debugBuffer[bufferHead], str, len);
    bufferHead += len;
}

static void bufferAppendNumber(uint32_t num) {
    char numStr[12];  // Max 10 digits + sign + null
    ltoa(num, numStr, 10);
    bufferAppend(numStr);
}

static void bufferAppendTimestamp(void) {
    uint32_t now = micros();
    uint32_t delta = now - lastLogTime;
    lastLogTime = now;
    
    bufferAppend("+");
    bufferAppendNumber(delta);
    bufferAppend("us");
    
    // Align the [TYPE] column
    if (delta < 10000) {
        bufferAppend("\t\t");
    } else {
        bufferAppend("\t");
    }
}

// ==============================================
// Public API
// ==============================================

void debugInit(void) {
    bufferHead = 0;
    bufferTail = 0;
    overflowOccurred = false;
    lastLogTime = micros();  // Initialize timestamp baseline
    
    delay(10);
    Serial.println(F("=== DEBUG START ==="));
    Serial.println(F("DIFF\t\t[TYPE] DETAILS\t\t\t\t(STATUS)"));
    Serial.println(F("-----------------------------------------------------------------"));
}

void debugTransition(int from, int to, int status) {
    bufferAppendTimestamp();
    bufferAppend("[TRANS] ");

    const char* nameFrom = getProtocolStateName((ProtocolState)from);
    const char* nameTo   = getProtocolStateName((ProtocolState)to);

    bufferAppend(nameFrom);
    bufferAppend(" -> ");
    bufferAppend(nameTo);

    // Calculate length to align the (STATUS) column
    // " -> " is 4 chars
    uint16_t currentLen = strlen(nameFrom) + 4 + strlen(nameTo);

    // Dynamic padding: target column start at char 40+
    if (currentLen < 16) {
        bufferAppend("\t\t\t\t");
    } else if (currentLen < 24) {
        bufferAppend("\t\t\t");
    } else if (currentLen < 32) {
        bufferAppend("\t\t");
    } else {
        bufferAppend("\t");
    }

    bufferAppend("(");
    bufferAppend(getTransferStatusName((TransferStatus)status));
    bufferAppend(")\n");
}

void debugEvent(const char* event) {
    bufferAppendTimestamp();
    bufferAppend("[EVENT] ");
    bufferAppend(event);
    bufferAppend("\n");
}

void debugEventHex(const char* event, uint8_t val) {
    bufferAppendTimestamp();
    bufferAppend("[EVENT] ");
    bufferAppend(event);

    // Calculate length of the label to determine padding
    uint16_t len = strlen(event);

    // Dynamic padding logic
    // This aligns the HEX value to the same column regardless of label length
    // Adjust thresholds (16, 24) if your labels get longer
    if (len < 8) {
        bufferAppend("\t\t\t");
    } else if (len < 16) {
        bufferAppend("\t\t");
    } else if (len < 24) {
        bufferAppend("\t");
    } else {
        bufferAppend(" "); // Just a space if the label is very long
    }

    bufferAppend(byte_to_hex_str(val));
    bufferAppend("\n");
}

void debugError(const char* error) {
    bufferAppendTimestamp();
    bufferAppend("[ERROR] !!");
    bufferAppend(error);
    bufferAppend("\n");
}

bool debugCanFlush(int status) {
    return (status != TS_STATUS_SI_BUSY && status != TS_STATUS_SO_BUSY);
}

void debugFlush(void) {
    // If buffer is empty and no overflow, do nothing
    if (bufferHead == bufferTail && !overflowOccurred) {
        return;
    }
    
    // Check how much space is available in the Hardware UART buffer
    int available = Serial.availableForWrite();
    
    // If hardware is full, return immediately to keep main loop running
    if (available == 0) {
        return; 
    }
    
    // Send overflow message first if it occurred and buffer is drained
    if (overflowOccurred && bufferHead == bufferTail) {
        Serial.print(OVERFLOW_MSG);
        overflowOccurred = false;
        return;
    }
    
    // Calculate the largest block we can send right now
    int pending = bufferHead - bufferTail;
    int chunk = (pending < available) ? pending : available;
    
    // Limit to 64 bytes per call to avoid blocking
    if (chunk > DEBUG_FLUSH_CHUNK_SIZE) {
        chunk = DEBUG_FLUSH_CHUNK_SIZE;
    }
    
    // BLOCK WRITE: Efficiently transfer the chunk
    Serial.write((uint8_t*)&debugBuffer[bufferTail], chunk);
    
    // Advance the tail
    bufferTail += chunk;
    
    // Reset pointers only when completely drained
    if (bufferTail == bufferHead) {
        bufferHead = 0;
        bufferTail = 0;
    }
}

const char* byte_to_hex_str(unsigned char val) {
    static char buf[5];
    static const char hex_chars[] = "0123456789ABCDEF";
    
    buf[0] = '0';
    buf[1] = 'x';
    buf[2] = hex_chars[(val >> 4) & 0x0F];
    buf[3] = hex_chars[val & 0x0F];
    buf[4] = '\0';
    
    return buf;
}

#endif // DEBUG_ENABLED