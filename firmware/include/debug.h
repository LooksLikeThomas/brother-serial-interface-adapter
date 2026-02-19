// ==============================================
// debug.h — Debug Logging Interface
// ==============================================
//
// Buffered debug logging for protocol layer debugging.
// Messages are buffered during execution and flushed
// only when the transfer layer is idle.
//
// Set DEBUG_ENABLED to false in config.h to completely 
// remove all debugging code from the build.
//
#ifndef DEBUG_H
#define DEBUG_H

#include <config.h>

#if DEBUG_ENABLED

#include <stdint.h>
#include <stdbool.h>
#include <avr/pgmspace.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize debug system (call in setup, initializes Serial)
void debugInit(void);

// Log a state transition with the transfer status that triggered it
// Parameters are int to avoid include dependencies - cast from enums is safe
void debugTransition(int from, int to, int status);

// Log a simple event (e.g., "POWER_DETECTED")
void debugEvent(const char* event);

// Log a simple event and a byte as hex (e.g., "RESPONSE: 0x55")
void debugEventHex(const char* event, uint8_t val);

// Log a simple event and a byte as hex including the char (e.g., "RESPONSE: 0x55 'U'")
void debugEventHexChar(const char* event, uint8_t val);

// Log an Error (e.g., "TRANSFER_ERROR")
void debugError(const char* error);

// Log current SI buffer fill level (e.g., "128 / 1536 bytes")
void debugBufferStatus(uint16_t currentCount);

// Flush buffer to Serial (only call when transfer layer is idle)
void debugFlush(void);

// Buffer is about to overflow
bool debugBufferBusy(void);

// Check if flush is safe based on transfer status
// Parameter is int to avoid include dependencies
bool debugCanFlush(int status);

const char* byte_to_hex_str(unsigned char val);

#ifdef __cplusplus
}
#endif

// Macros for easy enable/disable
#define DBG_INIT()                      debugInit()
#define DBG_TRANSITION(from, to, st)    debugTransition((int)(from), (int)(to), (int)(st))
#define DBG_EVENT(evt)               do { static const char _s[] PROGMEM = evt; debugEvent(_s); } while(0)
#define DBG_EVENT_HEX(evt, val)      do { static const char _s[] PROGMEM = evt; debugEventHex(_s, val); } while(0)
#define DBG_EVENT_HEX_CHAR(evt, val) do { static const char _s[] PROGMEM = evt; debugEventHexChar(_s, val); } while(0)
#define DBG_ERROR(err)               do { static const char _s[] PROGMEM = err; debugError(_s); } while(0)
#define DBG_BUFFER_STATUS(count)        debugBufferStatus(count)
#define DBG_FLUSH_IF_SAFE(status)       do { if (debugCanFlush((int)(status))) debugFlush(); } while(0)
#define DBG_BUFFER_BUSY()               debugBufferBusy()
#define DBG_B_TO_HEX(val)               byte_to_hex_str((unsigned)(char)(val))

#else

// No-op macros when debugging is disabled
#define DBG_INIT()                      ((void)0)
#define DBG_TRANSITION(from, to, st)    ((void)0)
#define DBG_EVENT(evt)                  ((void)0)
#define DBG_EVENT_HEX(msg, val)         ((void)0)
#define DBG_EVENT_HEX_CHAR(evt, val)    ((void)0)
#define DBG_ERROR(error)                ((void)0)
#define DBG_BUFFER_STATUS(count)        ((void)0)
#define DBG_FLUSH_IF_SAFE(status)       ((void)0)
#define DBG_BUFFER_BUSY()               (false)
#define DBG_B_TO_HEX(val)               ((void)0)

#endif // DEBUG_ENABLED

#endif // DEBUG_H