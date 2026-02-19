// ==============================================
// buffers.c — Ring Buffer Implementation
// ==============================================
//
// Simple ring buffer: head = write position, tail = read position
// Buffer is empty when head == tail
// Buffer is full when (head + 1) % size == tail
//
#include "buffers.h"

#if USE_EEPROM_BUFFER
    #include "eeprombuffer.h"
#endif

// ----- SI Buffer Storage (outgoing to typewriter) -----

static volatile uint8_t siBuffer[SI_RAMBUFFER_SIZE];
static volatile uint16_t siHead = 0;    // Write position
static volatile uint16_t siTail = 0;    // Read position

// ----- SO Buffer Storage (incoming from typewriter) -----

static volatile uint8_t soBuffer[SO_BUFFER_SIZE];
static volatile uint8_t soHead = 0;    // Write position
static volatile uint8_t soTail = 0;    // Read position

// ==============================================
// SI Buffer Functions
// ==============================================

bool siBufferEmpty() {
    #if USE_EEPROM_BUFFER
        return (siHead == siTail) && eeBufferEmpty();
    #else
        return siHead == siTail;
    #endif
}

bool siBufferFull() {
    bool ramFull = (((siHead + 1) % SI_RAMBUFFER_SIZE) == siTail);

    #if USE_EEPROM_BUFFER
        return ramFull && eeBufferFull();
    #else
        return ramFull;
    #endif
}

uint16_t siBufferCount(void) {
    uint16_t ramCount = (siHead - siTail + SI_RAMBUFFER_SIZE) % SI_RAMBUFFER_SIZE;

    #if USE_EEPROM_BUFFER
        return ramCount + eeBufferCount(); 
    #else
        return ramCount;
    #endif
}

bool siBufferPush(uint8_t byte) {
    bool ramFull = (((siHead + 1) % SI_RAMBUFFER_SIZE) == siTail);

    #if USE_EEPROM_BUFFER
        // SPILLOVER LOGIC: If EEPROM has data, or RAM is full, push to EEPROM
        if (!eeBufferEmpty() || ramFull) {
            return eeBufferPush(byte);
        }
    #else
        if (ramFull) return false;
    #endif

    // Normal RAM push
    siBuffer[siHead] = byte;
    siHead = (siHead + 1) % SI_RAMBUFFER_SIZE;
    return true;
}

bool siBufferPop(uint8_t *byte) {
    if (siBufferEmpty()) return false;
    *byte = siBuffer[siTail];
    siTail = (siTail + 1) % SI_RAMBUFFER_SIZE;

    #if USE_EEPROM_BUFFER
        // Siphon data from EEPROM back into RAM
        if (!eeBufferEmpty()) {
            uint8_t eeByte;
            if (eeBufferPop(&eeByte)) {
                siBuffer[siHead] = eeByte;
                siHead = (siHead + 1) % SI_RAMBUFFER_SIZE;
            }
        }
    #endif

    return true;
}

uint8_t siBufferPeek() {
    return siBuffer[siTail];
}

// SI Buffer Multi-Byte Helpers

uint8_t siBufferPeekN(uint8_t *buf, uint8_t n) {
    uint16_t ramCount = (siHead - siTail + SI_RAMBUFFER_SIZE) % SI_RAMBUFFER_SIZE;
    if (n > ramCount) n = ramCount;
    uint8_t pos = siTail;
    for (uint8_t i = 0; i < n; i++) {
        buf[i] = siBuffer[pos];
        pos = (pos + 1) % SI_RAMBUFFER_SIZE;
    }
    return n;
}

void siBufferDiscard(uint8_t n) {
    uint8_t dummy;
    for (uint8_t i = 0; i < n; i++) {
        if (!siBufferPop(&dummy)) {
            break;
        }
    }
}

// ==============================================
// SO Buffer Functions
// ==============================================

bool soBufferEmpty() {
    return soHead == soTail;
}

bool soBufferFull() {
    return ((soHead + 1) % SO_BUFFER_SIZE) == soTail;
}

bool soBufferPush(uint8_t byte) {
    if (soBufferFull()) return false;
    soBuffer[soHead] = byte;
    soHead = (soHead + 1) % SO_BUFFER_SIZE;
    return true;
}

bool soBufferPop(uint8_t *byte) {
    if (soBufferEmpty()) return false;
    *byte = soBuffer[soTail];
    soTail = (soTail + 1) % SO_BUFFER_SIZE;
    return true;
}

uint8_t soBufferPeek() {
    return soBuffer[soTail];
}
