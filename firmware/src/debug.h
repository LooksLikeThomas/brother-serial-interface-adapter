// ==============================================
// debug.h — Debug Logging Interface
// ==============================================
//
// Buffered debug logging for protocol layer debugging.
// Messages are buffered during execution and flushed
// only when the transfer layer is idle.
//
// Set DEBUG_ENABLED to 0 to completely remove all
// debugging code from the build.
//
#ifndef DEBUG_H
#define DEBUG_H

#define DEBUG_ENABLED 1  // Set to 0 to disable all debugging

#if DEBUG_ENABLED

#include <stdint.h>
#include <stdbool.h>

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

// Flush buffer to Serial (only call when transfer layer is idle)
void debugFlush(void);

// Check if flush is safe based on transfer status
// Parameter is int to avoid include dependencies
bool debugCanFlush(int status);

#ifdef __cplusplus
}
#endif

// Macros for easy enable/disable
#define DBG_INIT()                      debugInit()
#define DBG_TRANSITION(from, to, st)    debugTransition((int)(from), (int)(to), (int)(st))
#define DBG_EVENT(evt)                  debugEvent(evt)
#define DBG_FLUSH_IF_SAFE(status)       do { if (debugCanFlush((int)(status))) debugFlush(); } while(0)

#else

// No-op macros when debugging is disabled
#define DBG_INIT()                      ((void)0)
#define DBG_TRANSITION(from, to, st)    ((void)0)
#define DBG_EVENT(evt)                  ((void)0)
#define DBG_FLUSH_IF_SAFE(status)       ((void)0)

#endif // DEBUG_ENABLED

#endif // DEBUG_H