// ==============================================
// keymap.h — Keyboard Mapping
// ==============================================
//
// Bidirectional mapping between PC and typewriter bytes.
// Currently hardcoded for German keyboard with ASCII wheel.
//
#ifndef KEYMAP_H
#define KEYMAP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Translate PC byte to typewriter byte
uint8_t keymapToTypewriter(uint8_t pcByte);

// Translate typewriter byte to PC byte
uint8_t keymapToPC(uint8_t twByte);

#ifdef __cplusplus
}
#endif

#endif // KEYMAP_H