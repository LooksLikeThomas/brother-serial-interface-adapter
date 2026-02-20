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
// Debug 
//
// 0 = Disable debugging (Production build; removes logging overhead)
// 1 = Enable debugging (Writes buffered debug messages to the Serial port)
#define DEBUG_ENABLED 0

// Set to 1 to log per-character SI/SO events (high volume).
// Set to 0 to suppress character-level logging while keeping all other debug output.
#define DEBUG_CHAR_VERBOSE 0

//Debug Buffer config
//
// Size of the debug log ring buffer in bytes                                  
#define DEBUG_BUFFER_SIZE 128
// Max bytes flushed to Serial per flush call (prevents blocking the main loop)
#define DEBUG_FLUSH_CHUNK_SIZE 64

// ==============================================
// Buffers
// ==============================================

// Physical memory allocations
#define SI_BUFFER_SIZE (1024 - (DEBUG_ENABLED * 512))

// Buffer size for SO (incoming from typewriter)
#define SO_BUFFER_SIZE 16

// ==============================================
// Compression Configuration
// ==============================================
//
// Set to 1 to enable heatshrink compression on the SI buffer.
// Set to 0 to compile out all compression code entirely.
//
#define USE_COMPRESSION 1

#if USE_COMPRESSION

// Heatshrink parameters (must also match heatshrink_config.h)
#define HS_WINDOW_SZ2 6
#define HS_LOOKAHEAD_SZ2 4

// Peek buffer: must be > your max peek-ahead (3 byte escape seq)
#define PEEK_BUF_SIZE           8

#endif // USE_COMPRESSION

// ==============================================
// Operation Modes
// ==============================================

// System Operation Mode
// #define MODE_BYTE 0xF8   // Printer Mode  (PC --> Typewriter)
#define MODE_BYTE 0xF9      // Terminal Mode (PC <-> Typewriter)

// Automaticly Start Selecting typewriter when powered on
#define AUTO_SELECT 1

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
// Paper length in inches (11 = US Letter, 12 = A4/US Legal)
#define PAPER_LENGTH  11

// Vertical Line Spacing (VMI)
// 1 = single spacing (6 lpi), 2 = double spacing (3 lpi)
#define VMI  1

// Usable lines per page: paper × 6 lpi ÷ VMI, minus 2-line top and bottom margins
#define LINES_PER_PAGE  (PAPER_LENGTH * 6 / VMI - 4)

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
static const bool AUTO_LINE_FEED = true;

// Automatic Carriage Return
// Treats standalone LF as CR+LF for Unix/Linux compatibility.
// Also adds +LF to CR coming from Typewriter
// Does not affect explicit CR+LF pairs (handled by CR lookahead).
//
// false = Standalone LF moves paper only (typewriter standard)
// true  = Standalone LF also returns carriage (Unix compatibility)
static const bool AUTO_CARRIAGE_RETURN = true;

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
        case 0xB2: return 84;   // 12 cpi × 7"
        case 0xB3: return 105;  // 15 cpi × 7"
        default:   return 70;   // 10 cpi × 7"
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
static const bool SERIAL_N = true;

// Serial Parity
// 0 = No parity (N)
// 1 = Even parity (E)
// 2 = Odd parity (O)
static const uint8_t SERIAL_PARITY = 0;

// Serial Baud Rate
// Speed of the serial connection in bits per second.
// Maximum supported rate: 115200
static const uint32_t SERIAL_BAUD = 9600;

// Software Flow Control (XON/XOFF)
//
// When the SI buffer fills up, the interface sends XOFF to tell
// the PC to stop sending. When the buffer drains, it sends XON
// to resume. The PC's terminal program must have XON/XOFF enabled.
//
// HIGH_WATER: Send XOFF when buffer reaches this level
// LOW_WATER:  Send XON when buffer drops to this level
//

#define FLOWCONTROL_ENABLED 1
#define FLOW_HIGH_WATER (SI_BUFFER_SIZE * 2 / 4)    // Send XOFF above this
#define FLOW_LOW_WATER  (SI_BUFFER_SIZE / 4)        // Send XON below this


#ifdef __cplusplus
}
#endif

#endif // CONFIG_H