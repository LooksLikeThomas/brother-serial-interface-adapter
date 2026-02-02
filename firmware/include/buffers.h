// ==============================================
// buffers.h — Ring Buffer Interface
// ==============================================
//
// Two ring buffers for serial communication:
//   siBuffer: Bytes waiting to be sent to typewriter (protocol layer pushes)
//   soBuffer: Bytes received from typewriter (protocol layer pops)
//
// Internal storage (arrays, head/tail pointers) is hidden in buffers.c.
//
#ifndef BUFFERS_H
#define BUFFERS_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ----- SI Buffer (outgoing to typewriter) -----

bool siBufferEmpty();
bool siBufferFull();
bool siBufferPush(uint8_t byte);
bool siBufferPop(uint8_t *byte);
uint8_t siBufferPeek();

// ----- SO Buffer (incoming from typewriter) -----

bool soBufferEmpty();
bool soBufferFull();
bool soBufferPush(uint8_t byte);
bool soBufferPop(uint8_t *byte);
uint8_t soBufferPeek();

#ifdef __cplusplus
}
#endif

#endif // BUFFERS_H
