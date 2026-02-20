// ==============================================
// sicompressedbuffer.c — Compressed SI Buffer
// ==============================================
//
// Drop-in replacement for sibuffer.c.
// Compiled only when USE_COMPRESSION is 1.
//
// Self-contained: owns its own ring buffer, peek buffer,
// and heatshrink encoder/decoder state. No external coupling.
//
// Data flow:
//
//   siBufferPush(byte):
//     raw byte → encoder.sink → encoder.poll → ring
//
//   siBufferPop(byte):
//     peekBuf ← decoder.poll ← decoder.sink ← ring
//     (auto-flushes encoder when ring runs empty)
//

#include "config.h"

#if USE_COMPRESSION

#include "buffers.h"
#include "debug.h"
#include "heatshrink/heatshrink_encoder.h"
#include "heatshrink/heatshrink_decoder.h"

// ==============================================
// Ring buffer (compressed storage)
// ==============================================

static volatile uint8_t ring[SI_BUFFER_SIZE];
static volatile uint16_t rHead = 0;
static volatile uint16_t rTail = 0;

static inline bool     rEmpty(void) { return rHead == rTail; }
static inline bool     rFull(void)  { return ((rHead + 1) % SI_BUFFER_SIZE) == rTail; }
static inline uint16_t rCount(void) { return (rHead - rTail + SI_BUFFER_SIZE) % SI_BUFFER_SIZE; }

static inline bool rPush(uint8_t b) {
    if (rFull()) return false;
    ring[rHead] = b;
    rHead = (rHead + 1) % SI_BUFFER_SIZE;
    return true;
}

static inline bool rPop(uint8_t *b) {
    if (rEmpty()) return false;
    *b = ring[rTail];
    rTail = (rTail + 1) % SI_BUFFER_SIZE;
    return true;
}

// ==============================================
// Peek buffer (decoded output)
// ==============================================

static uint8_t pb[PEEK_BUF_SIZE];
static uint8_t pbHead = 0;
static uint8_t pbTail = 0;

static inline uint8_t pbCount(void) { return (pbHead - pbTail + PEEK_BUF_SIZE) % PEEK_BUF_SIZE; }
static inline bool    pbFull(void)  { return ((pbHead + 1) % PEEK_BUF_SIZE) == pbTail; }

static inline void pbPush(uint8_t b) {
    pb[pbHead] = b;
    pbHead = (pbHead + 1) % PEEK_BUF_SIZE;
}

static inline uint8_t pbPop(void) {
    uint8_t b = pb[pbTail];
    pbTail = (pbTail + 1) % PEEK_BUF_SIZE;
    return b;
}

// ==============================================
// Heatshrink state
// ==============================================

static heatshrink_encoder enc;
static heatshrink_decoder dec;
static bool encFinished;
static bool encActive;      // true once a byte has been sunk since last reset

// ==============================================
// Internal plumbing
// ==============================================

// Poll compressed bytes out of encoder into ring.
static void encToRing(void) {
    uint8_t out;
    size_t n;
    do {
        n = 0;
        heatshrink_encoder_poll(&enc, &out, 1, &n);
        if (n > 0 && !rPush(out)) break;
    } while (n > 0);
}

// Flush encoder: finish → drain → mark finished.
static void encFlush(void) {
    DBG_EVENT("COMPRESS: FLUSH");
    HSE_finish_res r;
    do {
        r = heatshrink_encoder_finish(&enc);
        encToRing();
    } while (r == HSER_FINISH_MORE);
    encFinished = true;
}

// Fill peek buffer from decoder ← ring ← encoder chain.
static void fillPb(void) {
    while (!pbFull()) {
        // 1) Poll decoder for decoded bytes
        uint8_t out;
        size_t n = 0;
        heatshrink_decoder_poll(&dec, &out, 1, &n);
        if (n > 0) { pbPush(out); continue; }

        // 2) Feed decoder from ring
        if (!rEmpty()) {
            uint8_t c;
            rPop(&c);
            size_t s = 0;
            heatshrink_decoder_sink(&dec, &c, 1, &s);
            continue;
        }

        // 3) Ring empty — already flushed? Then we're done.
        if (encFinished) {
            DBG_EVENT("COMPRESS: STREAM RESET");
            heatshrink_encoder_reset(&enc);
            heatshrink_decoder_reset(&dec);
            encFinished = false;
            encActive = false;
            break;
        }

        // 4) Encoder never received data? Nothing to flush.
        if (!encActive) break;

        // 5) Flush encoder to squeeze out remaining bytes
        encFlush();
        heatshrink_decoder_finish(&dec);

        // Ring may have data now — loop back
        if (!rEmpty()) continue;

        // 6) One more decoder poll after finish
        n = 0;
        heatshrink_decoder_poll(&dec, &out, 1, &n);
        if (n > 0) { pbPush(out); continue; }

        // Truly empty — reset for next stream
        DBG_EVENT("COMPRESS: STREAM RESET");
        heatshrink_encoder_reset(&enc);
        heatshrink_decoder_reset(&dec);
        encFinished = false;
        encActive = false;
        break;
    }
}

// ==============================================
// Public API (same signatures as sibuffer.c)
// ==============================================

void buffersInit(void) {
    rHead = 0;
    rTail = 0;
    pbHead = 0;
    pbTail = 0;
    heatshrink_encoder_reset(&enc);
    heatshrink_decoder_reset(&dec);
    encFinished = false;
    encActive = false;
    DBG_EVENT("COMPRESS: INIT");
}

bool siBufferEmpty(void) {
    if (pbCount() > 0) return false;
    fillPb();
    return pbCount() == 0;
}

bool siBufferFull(void) {
    return rFull();
}

uint16_t siBufferCount(void) {
    fillPb();
    return pbCount() + rCount();
}

bool siBufferPush(uint8_t byte) {
    if (encFinished) {
        heatshrink_encoder_reset(&enc);
        heatshrink_decoder_reset(&dec);
        encFinished = false;
        encActive = false;
    }

    encActive = true;
    size_t sunk = 0;
    heatshrink_encoder_sink(&enc, &byte, 1, &sunk);
    if (sunk == 0) {
        DBG_EVENT("COMPRESS: ENCODER STALL");
        encToRing();
        sunk = 0;
        heatshrink_encoder_sink(&enc, &byte, 1, &sunk);
        if (sunk == 0) {
            DBG_ERROR("COMPRESS: BYTE DROPPED");
            return false;
        }
    }
    encToRing();
    return true;
}

bool siBufferPop(uint8_t *byte) {
    fillPb();
    if (pbCount() == 0) return false;
    *byte = pbPop();
    return true;
}

uint8_t siBufferPeek(void) {
    fillPb();
    return pb[pbTail];
}

uint8_t siBufferPeekN(uint8_t *buf, uint8_t n) {
    fillPb();
    uint8_t avail = pbCount();
    if (n > avail) n = avail;
    uint8_t pos = pbTail;
    for (uint8_t i = 0; i < n; i++) {
        buf[i] = pb[pos];
        pos = (pos + 1) % PEEK_BUF_SIZE;
    }
    return n;
}

void siBufferDiscard(uint8_t n) {
    uint8_t dummy;
    for (uint8_t i = 0; i < n; i++) {
        if (!siBufferPop(&dummy)) break;
    }
}

#endif // USE_COMPRESSION