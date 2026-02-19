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

#include "config.h"

#include <stdint.h>
#include <stdbool.h>

// EEPROM Usable flag
#define USE_EEPROM_BUFFER (CONF_ENABLE_EEPROM && __AVR_ATmega328P__)

// Total logical capacity for SI
#define SI_BUFFER_SIZE (SI_RAMBUFFER_SIZE + (USE_EEPROM_BUFFER * SI_EEPROMBUFFER_SIZE))

#ifdef __cplusplus
extern "C" {
#endif

// ----- SI Buffer (outgoing to typewriter) -----

bool siBufferEmpty();
uint16_t siBufferCount(void);
bool siBufferFull();
bool siBufferPush(uint8_t byte);
bool siBufferPop(uint8_t *byte);
uint8_t siBufferPeek();
uint8_t siBufferPeekN(uint8_t *buf, uint8_t n);
void siBufferDiscard(uint8_t n);

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
