// ==============================================
// eeprombuffer.c
// ==============================================
#include "eeprombuffer.h"

#ifdef __AVR_ATmega328P__
#include <avr/eeprom.h>

static volatile uint16_t eeHead = 0;
static volatile uint16_t eeTail = 0;

void eeBufferInit(void) {
    eeHead = 0;
    eeTail = 0;
}

bool eeBufferEmpty(void) {
    return eeHead == eeTail;
}

bool eeBufferFull(void) {
    return ((eeHead + 1) % SI_EEPROMBUFFER_SIZE) == eeTail;
}

uint16_t eeBufferCount(void) {
    return (eeHead - eeTail + SI_EEPROMBUFFER_SIZE) % SI_EEPROMBUFFER_SIZE;
}

bool eeBufferPush(uint8_t byte) {
    if (eeBufferFull()) return false;
    
    // Cast the integer address to a pointer for the AVR library
    uint8_t* address = (uint8_t*)eeHead;
    
    // update_byte checks the value first and skips writing if it matches
    eeprom_update_byte(address, byte);
    
    eeHead = (eeHead + 1) % SI_EEPROMBUFFER_SIZE;
    return true;
}

bool eeBufferPop(uint8_t *byte) {
    if (eeBufferEmpty()) return false;
    
    uint8_t* address = (uint8_t*)eeTail;
    *byte = eeprom_read_byte(address);
    
    eeTail = (eeTail + 1) % SI_EEPROMBUFFER_SIZE;
    return true;
}

#endif // __AVR_ATmega328P__