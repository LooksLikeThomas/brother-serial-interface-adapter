// ==============================================
// keymap.c — Keyboard Mapping Implementation
// ==============================================
//
// German keyboard with ASCII wheel mapping.
// Bidirectional translation between PC and typewriter bytes.
//
// Only non-passthrough mappings are stored to save memory.
// Unmapped bytes pass through unchanged.
//
#include "keymap.h"

typedef struct {
    uint8_t pc;
    uint8_t tw;
} KeymapPair;

// ==============================================
// Mapping Table
// ==============================================
//
// Bidirectional mappings - only entries that differ from passthrough.
// Format: { PC byte, Typewriter byte }
//

static const KeymapPair mappings[] = {
    // PC,  TW      // PC char <-> TW char
    { 34,  64 },    // " <-> @
    { 35,  92 },    // # <-> \ 
    { 38,  32 },    // & <-> space
    { 39,  96 },    // ' <-> `
    { 40,  42 },    // ( <-> *
    { 41,  40 },    // ) <-> (
    { 42,  91 },    // * <-> [
    { 43,  93 },    // + <-> ]
    { 45,  47 },    // - <-> /
    { 47,  38 },    // / <-> &
    { 58,  62 },    // : <-> >
    { 59,  60 },    // ; <-> 
    { 61,  41 },    // = <-> )
    { 63,  95 },    // ? <-> _
    { 64,  35 },    // @ <-> #
    { 89,  90 },    // Y <-> Z
    { 90,  89 },    // Z <-> Y
    { 91,  34 },    // [ <-> "
    { 92,  58 },    // \ <-> :
    { 93, 123 },    // ] <-> {
    { 95,  63 },    // _ <-> ?
    { 96,  43 },    // ` <-> +
    { 121, 122 },   // y <-> z
    { 122, 121 },   // z <-> y
    { 123,  39 },   // { <-> '
    { 124,  59 },   // | <-> ;
    { 125, 124 },   // } <-> |
    { 126,  45 },   // ~ <-> -
};

#define MAPPING_COUNT (sizeof(mappings) / sizeof(mappings[0]))

// ==============================================
// Public API
// ==============================================

uint8_t keymapToTypewriter(uint8_t pc) {
    for (uint8_t i = 0; i < MAPPING_COUNT; i++) {
        if (mappings[i].pc == pc) {
            return mappings[i].tw;
        }
    }
    return pc;
}

uint8_t keymapToPC(uint8_t tw) {
    for (uint8_t i = 0; i < MAPPING_COUNT; i++) {
        if (mappings[i].tw == tw) {
            return mappings[i].pc;
        }
    }
    return tw;
}