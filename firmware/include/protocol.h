// ==============================================
// protocol.h — Protocol Sequence Layer
// ==============================================
//
// Handles multi-byte protocol sequences:
//   STARTUP:  Send 0xFE, receive device type
//   SELECT:   Mode selection sequence (placeholder)
//   READY:    Normal operation — pass-through between buffers and transfer layer
//   DESELECT: Disconnect sequence (placeholder)
//
// Uses the transfer layer for individual byte transfers.
// Owns the SI and SO buffers for normal operation.
//
#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>
#include <stdbool.h>
#include "transfer.h"

// ==============================================
// Protocol State Machine
// ==============================================

typedef enum {
    PS_TW_OFF,              // Waiting for typewriter to come online
    PS_STARTUP_INIT,        // Sending 0xFE to announce presence
    PS_STARTUP_RESPONSE,    // Waiting for device type byte (0x30)
    PS_SELECT,              // Running SELECT sequence (placeholder)
    PS_READY,               // Normal operation, pass-through
    PS_DESELECT             // Running DESELECT sequence (placeholder)
} ProtocolState;

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

// Initialize protocol layer — sets initial state to PS_TW_OFF
void protocolInit(Protocol *ps);

// Run the protocol state machine — call every loop iteration
// Takes the transfer status from pollTransfer() and the transfer struct
void pollProtocol(Protocol *ps, TransferStatus status, Transfer *ts);

#endif // PROTOCOL_H
