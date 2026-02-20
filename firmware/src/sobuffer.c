// ==============================================
// sobuffer.c — SO Ring Buffer (incoming from typewriter)
// ==============================================

#include "buffers.h"
#include "config.h"
#include "debug.h"

static volatile uint8_t soBuffer[SO_BUFFER_SIZE];
static volatile uint8_t soHead = 0;
static volatile uint8_t soTail = 0;

bool soBufferEmpty(void) {
    return soHead == soTail;
}

bool soBufferFull(void) {
    return ((soHead + 1) % SO_BUFFER_SIZE) == soTail;
}

bool soBufferPush(uint8_t byte) {
    if (soBufferFull()) {
        DBG_ERROR("SO BUFFER FULL - BYTE DROPPED");
        return false;
    }
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

uint8_t soBufferPeek(void) {
    return soBuffer[soTail];
}