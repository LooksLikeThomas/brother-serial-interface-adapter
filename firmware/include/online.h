// ==============================================
// online.h — PS_ONLINE Serial/Bus Byte Handling
// ==============================================
//
// Handles byte-level processing during PS_ONLINE state:
//   - Serial → bus: control codes, ESC sequences, translation
//   - Bus → serial: Code key tracking, reverse translation
//
// Called by protocol.c when the transfer layer is idle
// and data is available. Does not interact with the
// transfer layer directly — loads results into BSB
// for protocol.c to drain.
//
// State transitions are signalled via return value,
// never performed directly.
//
#ifndef ONLINE_H
#define ONLINE_H

#include "protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================
// Serial Input Handler
// ==============================================
//
// Process the next byte(s) from siBuffer.
//
// Called when:
//   - Transfer layer is idle
//   - BSB is fully drained
//   - siBuffer has data
//
// May consume 0–3 bytes from siBuffer.
// May load bus bytes into ps->bsb.
// May update session state (column, line, margins,
// tabs, pitch, flags).
//
// Returns:
//   PS_STATUS_ONLINE       — continue normal operation
//   PS_STATUS_DESELECTING  — DC3 received, caller handles
//                            transition to PS_DESELECT
//
ProtocolStatus onlineHandleSI(Protocol *ps);

// ==============================================
// Bus Input Handler
// ==============================================
//
// Process a byte received from the typewriter.
//
// Called when:
//   - Transfer status is TS_STATUS_SO_DONE
//
// Reads ps->ts.receivedByte.
// May push bytes to soBuffer.
// May update Code key state.
//
// Returns:
//   PS_STATUS_ONLINE — always
//
ProtocolStatus onlineHandleSO(Protocol *ps);

#ifdef __cplusplus
}
#endif

#endif // ONLINE_H