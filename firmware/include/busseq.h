// ==============================================
// busseq.h — Bus Sequence Buffer
// ==============================================
//
// Encodes a sequence of bus bytes to emit, using run-length
// encoding (RLE) for repeated bytes (e.g., margin repositioning).
//
// Builder functions (bsbAdd*) construct the sequence.
// Consumer functions (bsbHasData, bsbNext) drain it one byte at a time.
//
#ifndef BUSSEQ_H
#define BUSSEQ_H

#include <stdint.h>
#include <stdbool.h>
#include "translate.h"

#ifdef __cplusplus
extern "C" {
#endif

// RLE marker byte — never appears in the bus protocol
#define BSB_REPEAT  0xFF

// Max encoded buffer length
#define BSB_MAX     12

// ==============================================
// Bus Sequence Buffer
// ==============================================

typedef struct {
    uint8_t buf[BSB_MAX];
    uint8_t len;        // encoded byte count written to buf[]
    uint8_t idx;        // read cursor into buf[]
    uint8_t bytesLeft;  // actual bus bytes remaining to emit
} BusSequenceBuffer;

// ==============================================
// Consumer Helpers
// ==============================================

// Returns true if there are bus bytes remaining to emit
static inline bool bsbHasData(const BusSequenceBuffer *s) {
    return s->bytesLeft > 0;
}

// Returns the next bus byte and advances internal state.
// When idx lands on BSB_REPEAT: buf[idx+1] is the remaining count,
// buf[idx+2] is the byte to emit. Decrements count in place;
// when count reaches zero, advances idx past the 3-byte RLE triplet.
// Always decrements bytesLeft.
static inline uint8_t bsbNext(BusSequenceBuffer *s) {
    uint8_t b = s->buf[s->idx];
    s->bytesLeft--;
    if (b != BSB_REPEAT) {
        s->idx++;
        return b;
    }
    uint8_t byte = s->buf[s->idx + 2];
    if (--s->buf[s->idx + 1] == 0) {
        s->idx += 3;
    }
    return byte;
}

// ==============================================
// Builder Helpers
// ==============================================

// Reset the buffer to empty — must be called before adding bytes
static inline void bsbClear(BusSequenceBuffer *s) {
    s->len = 0;
    s->idx = 0;
    s->bytesLeft = 0;
}

// Append a single literal byte
static inline void bsbAddByte(BusSequenceBuffer *s, uint8_t b) {
    s->buf[s->len++] = b;
    s->bytesLeft++;
}

// Append two literal bytes
static inline void bsbAddByte2(BusSequenceBuffer *s, uint8_t a, uint8_t b) {
    s->buf[s->len++] = a;
    s->buf[s->len++] = b;
    s->bytesLeft += 2;
}

// Append three literal bytes
static inline void bsbAddByte3(BusSequenceBuffer *s, uint8_t a, uint8_t b, uint8_t c) {
    s->buf[s->len++] = a;
    s->buf[s->len++] = b;
    s->buf[s->len++] = c;
    s->bytesLeft += 3;
}

// Append an RLE-encoded repeated byte.
// Encodes as [BSB_REPEAT, count, byte]. If count is 0, the triplet
// is written but adds nothing to bytesLeft — harmless, avoids a branch
// at the call site.
static inline void bsbAddRepeat(BusSequenceBuffer *s, uint8_t b, uint8_t count) {
    if (count == 0) return;
    s->buf[s->len++] = BSB_REPEAT;
    s->buf[s->len++] = count;
    s->buf[s->len++] = b;
    s->bytesLeft += count;
}

// Append all bytes from a TranslateResult
static inline void bsbAddResult(BusSequenceBuffer *s, TranslateResult r) {
    for (uint8_t i = 0; i < r.len; i++) {
        s->buf[s->len++] = r.bytes[i];
    }
    s->bytesLeft += r.len;
}

#ifdef __cplusplus
}
#endif

#endif // BUSSEQ_H
