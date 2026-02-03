// ==============================================
// transfer.h — Single-Byte Transfer Layer
// ==============================================
//
// Handles the low-level handshake for transferring one byte
// between interface (Arduino) and typewriter.
//
// Two paths:
//   SI path: Interface sends a byte to typewriter (started by protocol layer)
//   SO path: Typewriter sends a byte to interface (started automatically by KBRQ)
//
// The protocol layer controls this module through:
//   transferStartSI()  — request sending a byte
//   pollTransfer()     — run the state machine, get status back
//
// Received bytes are stored in Transfer.receivedByte.
// The protocol layer reads them when status is TS_STATUS_SO_DONE.
//
#ifndef TRANSFER_H
#define TRANSFER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================
// Transfer State Machine
// ==============================================
//
// Internal states for the single-byte handshake.
// The protocol layer does not need to read these directly,
// it uses TransferStatus instead.
//
typedef enum {
    // ----- Idle -----
    
    TS_IDLE,            // Ready for next transfer
    
    // ----- SI Path: Interface Initiates -----
    
    TS_SI_SYN,          // READY pulled LOW, waiting ~30µs
    TS_SI_TRANSFER,     // Clocking 8 bits via timer ISR
    TS_SI_BUSY,         // Waiting for KBACK to rise (100µs to 5s)
    TS_SI_FIN,          // Waiting ~40µs, then release READY
    
    // ----- SO Path: Typewriter Initiates -----
    
    TS_SO_SYN,          // KBRQ rose, waiting ~100µs
    TS_SO_ACK,          // READY pulled LOW, waiting ~200µs
    TS_SO_TRANSFER,     // Clocking 8 bits via timer ISR
    TS_SO_BUSY,         // Waiting ~240µs before releasing READY
    TS_SO_FIN,          // Waiting for KBRQ to fall back

    // ----- Error -----
    
    TS_ERROR            // Timeout or error, waiting for protocol layer to reset
    
} TransferState;

// ==============================================
// Transfer Status
// ==============================================
//
// Returned by pollTransfer() so the protocol layer can
// observe what the transfer layer is doing without
// accessing internal state directly.
//
// CONTRACT: TS_STATUS_SI_DONE, TS_STATUS_SO_DONE, and
// TS_STATUS_TIMEOUT are returned exactly once — on the
// pollTransfer() call where the transition completes.
// The next call returns TS_STATUS_IDLE.
//
// The protocol layer MUST consume these in the same loop
// iteration. pollProtocol() is always called directly
// after pollTransfer() with the returned status, so this
// is guaranteed by the main loop structure:
//
//   TransferStatus status = pollTransfer(&ts);
//   pollProtocol(&ps, status, &ts);
//
// If this loop structure changes (e.g. multiple consumers
// or deferred processing), a sticky status mechanism
// would be needed.
//
typedef enum {
    TS_STATUS_IDLE,         // Nothing happening, ready for SI request
    TS_STATUS_SI_BUSY,      // SI path in progress
    TS_STATUS_SI_DONE,      // SI path completed successfully
    TS_STATUS_SO_BUSY,      // SO path in progress (typewriter initiated)
    TS_STATUS_SO_DONE,      // SO path completed, byte available in receivedByte
    TS_STATUS_TIMEOUT,      // Typewriter not responding (5s KBACK or 100ms KBRQ)
    TS_STATUS_ERROR         // Error/timeout, persistent until transferInit() called
} TransferStatus;

// ==============================================
// Transfer Struct
// ==============================================
//
// Groups all transfer state into one struct.
// Protocol layer receives this from pollTransfer().
//
typedef struct {
    TransferState state;        // Current internal state
    uint32_t stateEnteredAt;    // micros() when state was entered
    bool lastWasSI;             // Direction tracking for DEL/0xFF on SO path
    uint8_t receivedByte;       // Last byte received via SO path
} Transfer;

// ==============================================
// Public API
// ==============================================

// Clear all ISR flags — call after hardware settle to discard startup noise
void transferClearFlags();

// Initialize transfer layer — sets up timer, external interrupts, initial state
void transferInit(Transfer *ts);

// Run the transfer state machine — call every loop iteration
// Returns current status for the protocol layer to act on
TransferStatus pollTransfer(Transfer *ts);

// Request an SI transfer — only call when status is TS_STATUS_IDLE
bool transferQueueSI(Transfer *ts, uint8_t byte);

// Check if typewriter is online (KBRQ stable LOW for 100ms)
// Uses ISR data internally, protocol layer doesn't touch hardware
bool isTypewriterOnline();

#ifdef __cplusplus
}
#endif

#endif // TRANSFER_H
