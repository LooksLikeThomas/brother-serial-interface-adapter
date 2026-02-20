// ==============================================
// sibuffer.c — SI Ring Buffer (outgoing to typewriter)
// ==============================================
//
// Plain ring buffer, no compression.
// Compiled only when USE_COMPRESSION is 0.
//

#include "config.h"

#if !USE_COMPRESSION

#include "buffers.h"
#include "debug.h"

static volatile uint8_t siBuffer[SI_BUFFER_SIZE];
static volatile uint16_t siHead = 0;
static volatile uint16_t siTail = 0;

void buffersInit(void) {
    siHead = 0;
    siTail = 0;
    DBG_EVENT("SI BUFFER: PLAIN MODE");
}

bool siBufferEmpty(void) {
    return siHead == siTail;
}

bool siBufferFull(void) {
    return ((siHead + 1) % SI_BUFFER_SIZE) == siTail;
}

uint16_t siBufferCount(void) {
    return (siHead - siTail + SI_BUFFER_SIZE) % SI_BUFFER_SIZE;
}

bool siBufferPush(uint8_t byte) {
    if (siBufferFull()) return false;
    siBuffer[siHead] = byte;
    siHead = (siHead + 1) % SI_BUFFER_SIZE;
    return true;
}

bool siBufferPop(uint8_t *byte) {
    if (siBufferEmpty()) return false;
    *byte = siBuffer[siTail];
    siTail = (siTail + 1) % SI_BUFFER_SIZE;
    return true;
}

uint8_t siBufferPeek(void) {
    return siBuffer[siTail];
}

uint8_t siBufferPeekN(uint8_t *buf, uint8_t n) {
    uint16_t count = siBufferCount();
    if (n > count) n = count;
    uint16_t pos = siTail;
    for (uint8_t i = 0; i < n; i++) {
        buf[i] = siBuffer[pos];
        pos = (pos + 1) % SI_BUFFER_SIZE;
    }
    return n;
}

void siBufferDiscard(uint8_t n) {
    uint8_t dummy;
    for (uint8_t i = 0; i < n; i++) {
        if (!siBufferPop(&dummy)) break;
    }
}

#endif // !USE_COMPRESSION