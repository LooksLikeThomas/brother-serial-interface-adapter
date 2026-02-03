// ==============================================
// protocol.h — Protocol Sequence Layer
// ==============================================
//
// Handles multi-byte protocol sequences:
//   STARTUP:   Send 0xFE, receive device type
//   SELECT:    Mode selection sequence
//   ONLINE:    Normal operation — pass-through between buffers and transfer layer
//   DESELECT:  Disconnect sequence
//
// Uses the transfer layer for individual byte transfers.
// Owns the SI and SO buffers for normal operation.
//
#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>
#include <stdbool.h>
#include "transfer.h"

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================
// Protocol State Machine
// ==============================================

typedef enum {
    PS_OFFLINE,             // Waiting for typewriter to come online
    PS_STARTUP_INIT,        // Sending 0xFE to announce presence
    PS_STARTUP_RESPONSE,    // Waiting for device type byte (0x30)
    PS_STANDBY,             // Connected, waiting for SELECT trigger
    PS_SELECT,              // Running SELECT sequence
    PS_ONLINE,              // Normal operation, bidirectional data
    PS_DESELECT             // Running DESELECT sequence
} ProtocolState;

// ==============================================
// Protocol Status
// ==============================================
//
// Returned by pollProtocol() so the application layer can
// observe the protocol state without accessing internals.
//
typedef enum {
    PS_STATUS_OFFLINE,      // Typewriter not connected / powered off
    PS_STATUS_STARTUP,      // Running startup handshake
    PS_STATUS_STANDBY,      // Connected, idle, waiting for SELECT trigger
    PS_STATUS_SELECTING,    // Running SELECT sequence
    PS_STATUS_ONLINE,       // Bidirectional data mode
    PS_STATUS_DESELECTING   // Running DESELECT sequence
} ProtocolStatus;

// ==============================================
// Protocol Struct
// ==============================================

typedef struct {
    ProtocolState state;        // Current protocol state
    uint32_t stateEnteredAt;    // micros() when state was entered
    uint8_t deviceType;         // Device type from startup response (e.g. 0x30)
    uint8_t selectStep;         // Sub-step counter for SELECT sequence
} Protocol;

// ==============================================
// Public API
// ==============================================

// Initialize protocol layer — sets initial state to PS_OFFLINE
void protocolInit(Protocol *ps);

// Run the protocol state machine — call every loop iteration
// Takes the transfer status from pollTransfer() and the transfer struct
// Returns current protocol status for the application layer
ProtocolStatus pollProtocol(Protocol *ps, TransferStatus status, Transfer *ts);

#ifdef __cplusplus
}
#endif

#endif // PROTOCOL_H