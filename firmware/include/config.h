// ==============================================
// config.h — Global System Configuration
// ==============================================
//
// Static configuration values included by other parts
// of the program. Adjust these settings to match the
// desired operating mode and hardware setup.
//
#ifndef CONFIG_H
#define CONFIG_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================
// Debugging
// ==============================================

// Debug Logging Enable
// Controls the inclusion of debug code at compile time.
//
// false = Disable debugging (Production build; removes logging overhead)
// true  = Enable debugging (Writes buffered debug messages to the Serial port)
#define DEBUG_ENABLED 1


// ==============================================
// Buffers
// ==============================================

// Buffer size for SI (outgoing to typewriter)
#define SI_BUFFER_SIZE 256

// Buffer size for SO (incoming from typewriter)
#define SO_BUFFER_SIZE 16

// ==============================================
// Software Flow Control (XON/XOFF)
// ==============================================
//
// When the SI buffer fills up, the interface sends XOFF to tell
// the PC to stop sending. When the buffer drains, it sends XON
// to resume. The PC's terminal program must have XON/XOFF enabled.
//
// HIGH_WATER: Send XOFF when buffer reaches this level
// LOW_WATER:  Send XON when buffer drops to this level
//

#define FLOWCONTROL_ENABLED 1
#define FLOW_HIGH_WATER (SI_BUFFER_SIZE * 3 / 4)    // Send XOFF above this
#define FLOW_LOW_WATER  (SI_BUFFER_SIZE / 4)        // Send XON below this

// ==============================================
// Operation Modes
// ==============================================

// System Operation Mode
// #define MODE_BYTE 0xF8   // Printer Mode  (PC --> Typewriter)
#define MODE_BYTE 0xF9      // Terminal Mode (PC <-> Typewriter)

// ASCII Wheel Selection
// Controls how country/region switches are interpreted based on the wheel used.
//
// true  = ASCII Wheel (Switch UP)
//         The typewriter switch selection for a given country is ignored.
//         Convenient if an ASCII Wheel is used in printer or terminal mode.
//
// false = Non-ASCII Wheel (Switch DOWN)
//         The typewriter switch selection for a given country is taken into account.
//         Convenient if a wheel other than the ASCII wheel is used.
static const bool ASCII_WHEEL = true; // TODO: NOT IMPLEMENTED

// Paper Length Configuration
// false = 11 inch paper (Standard US Letter)
// true  = 12 inch paper (Roughly DIN A4)
static const bool PAPER_LENGTH = false; // TODO: NOT IMPLEMENTED

// ==============================================
// Printer Behavior Settings
// ==============================================

// Auto Skip Perforation
// Controls handling of continuous paper perforations.
//
// false = Non-auto skip mode (Ignores continuous paper perforations)
// true  = Auto skip mode (Skips continuous paper perforations)
static const bool AUTO_SKIP_PERFORATION = false; // TODO: NOT IMPLEMENTED

// DC-1/DC-3 Control Codes
// Controls XON/XOFF flow control interpretation.
// false = DC-1/DC-3 control disabled
// true  = DC-1/DC-3 control enabled
static const bool DC_CONTROL = true; // TODO: NOT IMPLEMENTED

// Automatic Line Feed
// Adds an implicit Line Feed (LF) after every Carriage Return (CR).
// Since most computers send both CR+LF for a new line, enabling this
// causes two LFs to occur, resulting in double spacing.
//
// false = Auto line feed off (Standard single spacing)
// true  = Auto line feed on (Double spacing)
static const bool AUTO_LINE_FEED = false;

// Pre-fill Horizontal Tab Stops
// Sets a tab stop every N columns on startup (column 1 + N, 1 + 2N, ...).
// 0 = no pre-fill (manual setup only via future ESC+1/ESC+8 sequences).
#define TAB_EVERY_N  8

// Printing Pitch (Characters Per Inch)
// Controls horizontal character density.
// #define PITCH_BYTE 0xB2  // 12cpi
// #define PITCH_BYTE 0xB3  // 15cpi
#define PITCH_BYTE 0xB1     // 10cpi (default)

static inline uint8_t rightMarginForPitch(uint8_t pitch) {
    switch (pitch) {
        case 0xB2: return 96;   // 12 cpi × 8"
        case 0xB3: return 120;  // 15 cpi × 8"
        default:   return 80;   // 10 cpi × 8"
    }
}

// Line Pitch (Line Spacing)
// Controls vertical line spacing density.
// Options: 1.0, 1.5, 2.0, 3.0
static const float LINE_PITCH = 1.0; // TODO: NOT IMPLEMENTED

// ==============================================
// Serial Interface Configuration
// ==============================================

// Serial Data Bits
// false = 7-bit data
// true  = 8-bit data
static const bool SERIAL_N = true; // TODO: NOT IMPLEMENTED

// Serial Parity
// false = Even parity
// true  = Odd parity
static const bool SERIAL_PARITY = false; // TODO: NOT IMPLEMENTED

// Serial Baud Rate
// Speed of the serial connection in bits per second.
// Maximum supported rate: 112500
static const uint32_t SERIAL_BAUD = 9600;

#ifdef __cplusplus
}
#endif

#endif // CONFIG_H