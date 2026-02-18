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
#include "translate.h"
#include "busseq.h"
#include "tabstops.h"

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================
// Keyboard Identifiers
// ==============================================
//
// Values match the keyboard ID byte returned by the
// typewriter in the SELECT response (byte 3).
//
#define KEYBOARD_KB1  0x04  // Local layout
#define KEYBOARD_KB2  0x24  // International layout
#define KEYBOARD_KB3  0x44  // Symbol layout

// ==============================================
// Protocol State Machine
// ==============================================

typedef enum {
    PS_OFFLINE,             // Waiting for typewriter to come online
    PS_STARTUP_SETTLE,      // Typewriter powered on, waiting 1.5s for settle
    PS_STARTUP_INIT,        // Sending 0xFE to announce presence
    PS_STARTUP_RESPONSE,    // Waiting for device type byte (0x30)
    PS_STANDBY,             // Connected, waiting for SELECT trigger
    PS_SELECT,              // Running SELECT sequence
    PS_ONLINE,              // Normal operation, bidirectional data
    PS_DESELECT             // Running DESELECT sequence
} ProtocolState;

// PS_SELECT substates
typedef enum {
    SEL_IDLE,           // Not in SELECT sequence
    SEL_QUEUE_MODE,     // Queue 0xF9 (terminal) or 0xF8 (typewriter)
    SEL_WAIT_MODE,      // Wait for SI_DONE
    SEL_QUEUE_FD,       // Queue 0xFD
    SEL_WAIT_FD,        // Wait for SI_DONE
    SEL_WAIT_KBBYTE,    // Wait for SO Keyboard config (0x04/0x24/0x44)
    SEL_QUEUE_F4,       // Queue 0xF4
    SEL_WAIT_F4,        // Wait for SI_DONE
    SEL_QUEUE_PITCH1,   // Queue 0xB1
    SEL_WAIT_PITCH1,    // Wait for SI_DONE
    SEL_QUEUE_PITCH2,   // Queue 0xB1
    SEL_WAIT_PITCH2,    // Wait for SI_DONE
    SEL_COMPLETE        // Transition to ONLINE
} SelectState;

// PS_DESELECT substates
typedef enum {
    DESEL_IDLE,     // Not in DESELECT sequence
    DESEL_QUEUE,    // Queue deselect byte
    DESEL_WAIT,     // Wait for SI_DONE
    DESEL_COMPLETE  // Transition to STANDBY
} DeselectState;

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
    SelectState selectState;    // Substate for SELECT sequence
    DeselectState deselectState;// Substate for DESELECT sequence
    Transfer ts;                // Transfer layer state (owned by protocol)
    TransferStatus lastTsStatus;// Last status from pollTransfer() (for debugging)

    // Session State
    uint8_t keyboardID;         // From SELECT response byte 3 (0x04/0x24/0x44)
    uint8_t pitchByte;          // Current HMI byte: 0xB1 (10cpi), 0xB2 (12cpi), 0xB3 (15cpi)
    bool autoLfEnabled;         // Set via config or ESC Sequence. Adds LF after CR

    // Reverse path state (typewriter → serial)
    bool codePressed;           // True between bus 0x88 (Code down) and 0x89 (Code up)
    bool swallowNextLf;         // For autoLF

    // Forward path sequence buffer (serial → bus)
    BusSequenceBuffer bsb;      // Pending bus bytes to emit

    // Carriage position tracking (1-based)
    uint8_t column;             // Current column (1-based, starts at leftMargin)
    uint8_t line;               // Current line on page (1-based)
    uint8_t leftMargin;         // Left margin column (default 1)
    uint8_t rightMargin;        // Right margin column (pitch-dependent)
    TabStops tabs;              // Horizontal tab stop positions (bitmask)
} Protocol;

// ==============================================
// Public API
// ==============================================

// Initialize protocol layer — sets initial state to PS_OFFLINE
void protocolInit(Protocol *ps);

// Run the protocol state machine — call every loop iteration
// Returns current protocol status for the application layer
ProtocolStatus pollProtocol(Protocol *ps);

#ifdef __cplusplus
}
#endif

#endif // PROTOCOL_H