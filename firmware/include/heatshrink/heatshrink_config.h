#ifndef HEATSHRINK_CONFIG_H
#define HEATSHRINK_CONFIG_H

#include "config.h"

/* Should functionality assuming dynamic allocation be used? */
#ifndef HEATSHRINK_DYNAMIC_ALLOC
#define HEATSHRINK_DYNAMIC_ALLOC 0
#endif

#if HEATSHRINK_DYNAMIC_ALLOC
    /* Optional replacement of malloc/free */
    #define HEATSHRINK_MALLOC(SZ) malloc(SZ)
    #define HEATSHRINK_FREE(P, SZ) free(P)
#else
    /* Required parameters for static configuration */
    #define HEATSHRINK_STATIC_INPUT_BUFFER_SIZE 4
    #if USE_COMPRESSION
        #define HEATSHRINK_STATIC_WINDOW_BITS HS_WINDOW_SZ2
        #define HEATSHRINK_STATIC_LOOKAHEAD_BITS HS_LOOKAHEAD_SZ2
    #else
        /* Dummy values — heatshrink code won't be used but needs to compile */
        #define HEATSHRINK_STATIC_WINDOW_BITS 8
        #define HEATSHRINK_STATIC_LOOKAHEAD_BITS 4
    #endif
#endif

/* Turn on logging for debugging. */
#define HEATSHRINK_DEBUGGING_LOGS 0

/* Use indexing for faster compression. (This requires additional space.) */
#define HEATSHRINK_USE_INDEX 0

#endif
