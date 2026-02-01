// ==============================================
// buffers.c — Ring Buffer Implementation
// ==============================================
//
// Simple ring buffer: head = write position, tail = read position
// Buffer is empty when head == tail
// Buffer is full when (head + 1) % size == tail
//
#include "buffers.h"

#define BUFFER_SIZE 128

// ----- SI Buffer Storage (outgoing to typewriter) -----

static volatile uint8_t siBuffer[BUFFER_SIZE];
static volatile uint8_t siHead = 0;    // Write position
static volatile uint8_t siTail = 0;    // Read position

// ----- SO Buffer Storage (incoming from typewriter) -----

static volatile uint8_t soBuffer[BUFFER_SIZE];
static volatile uint8_t soHead = 0;    // Write position
static volatile uint8_t soTail = 0;    // Read position

// ==============================================
// SI Buffer Functions
// ==============================================

bool siBufferEmpty() {
    return siHead == siTail;
}

bool siBufferFull() {
    return ((siHead + 1) % BUFFER_SIZE) == siTail;
}

bool siBufferPush(uint8_t byte) {
    if (siBufferFull()) return false;
    siBuffer[siHead] = byte;
    siHead = (siHead + 1) % BUFFER_SIZE;
    return true;
}

bool siBufferPop(uint8_t *byte) {
    if (siBufferEmpty()) return false;
    *byte = siBuffer[siTail];
    siTail = (siTail + 1) % BUFFER_SIZE;
    return true;
}

// ==============================================
// SO Buffer Functions
// ==============================================

bool soBufferEmpty() {
    return soHead == soTail;
}

bool soBufferFull() {
    return ((soHead + 1) % BUFFER_SIZE) == soTail;
}

bool soBufferPush(uint8_t byte) {
    if (soBufferFull()) return false;
    soBuffer[soHead] = byte;
    soHead = (soHead + 1) % BUFFER_SIZE;
    return true;
}

bool soBufferPop(uint8_t *byte) {
    if (soBufferEmpty()) return false;
    *byte = soBuffer[soTail];
    soTail = (soTail + 1) % BUFFER_SIZE;
    return true;
}
