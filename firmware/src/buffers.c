// ==============================================
// buffers.c — Ring Buffer Implementation
// ==============================================
//
// Simple ring buffer: head = write position, tail = read position
// Buffer is empty when head == tail
// Buffer is full when (head + 1) % size == tail
//
#include "buffers.h"
#include "config.h"

// ----- SI Buffer Storage (outgoing to typewriter) -----

static volatile uint8_t siBuffer[SI_BUFFER_SIZE];
static volatile uint8_t siHead = 0;    // Write position
static volatile uint8_t siTail = 0;    // Read position

// ----- SO Buffer Storage (incoming from typewriter) -----

static volatile uint8_t soBuffer[SO_BUFFER_SIZE];
static volatile uint8_t soHead = 0;    // Write position
static volatile uint8_t soTail = 0;    // Read position

// ==============================================
// SI Buffer Functions
// ==============================================

bool siBufferEmpty() {
    return siHead == siTail;
}

bool siBufferFull() {
    return ((siHead + 1) % SI_BUFFER_SIZE) == siTail;
}

uint8_t siBufferCount(void) {
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

uint8_t siBufferPeek() {
    return siBuffer[siTail];
}

// SI Buffer Multi-Byte Helpers

uint8_t siBufferPeekN(uint8_t *buf, uint8_t n) {
    uint8_t count = siBufferCount();
    if (n > count) n = count;
    uint8_t pos = siTail;
    for (uint8_t i = 0; i < n; i++) {
        buf[i] = siBuffer[pos];
        pos = (pos + 1) % SI_BUFFER_SIZE;
    }
    return n;
}

void siBufferDiscard(uint8_t n) {
    uint8_t count = siBufferCount();
    if (n > count) n = count;
    siTail = (siTail + n) % SI_BUFFER_SIZE;
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
