// ==============================================
// eeprombuffer.h
// ==============================================
#ifndef EEPROMBUFFER_H
#define EEPROMBUFFER_H

#include "config.h"

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// We only define these functions if compiling for the Arduino Uno
#ifdef __AVR_ATmega328P__

void eeBufferInit(void);
bool eeBufferEmpty(void);
bool eeBufferFull(void);
uint16_t eeBufferCount(void);
bool eeBufferPush(uint8_t byte);
bool eeBufferPop(uint8_t *byte);

#endif // __AVR_ATmega328P__

#ifdef __cplusplus
}
#endif

#endif // EEPROMBUFFER_H