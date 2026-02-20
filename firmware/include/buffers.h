// ==============================================
// buffers.h — Ring Buffer Interface
// ==============================================
//
// SI buffer: implemented in sibuffer.c or sicompressedbuffer.c
//            depending on USE_COMPRESSION in config.h.
// SO buffer: implemented in sobuffer.c (always uncompressed).
//
#ifndef BUFFERS_H
#define BUFFERS_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void buffersInit(void);

// ----- SI Buffer (outgoing to typewriter) -----

bool     siBufferEmpty(void);
uint16_t siBufferCount(void);
bool     siBufferFull(void);
bool     siBufferPush(uint8_t byte);
bool     siBufferPop(uint8_t *byte);
uint8_t  siBufferPeek(void);
uint8_t  siBufferPeekN(uint8_t *buf, uint8_t n);
void     siBufferDiscard(uint8_t n);

// ----- SO Buffer (incoming from typewriter) -----

bool    soBufferEmpty(void);
bool    soBufferFull(void);
bool    soBufferPush(uint8_t byte);
bool    soBufferPop(uint8_t *byte);
uint8_t soBufferPeek(void);

#ifdef __cplusplus
}
#endif

#endif // BUFFERS_H