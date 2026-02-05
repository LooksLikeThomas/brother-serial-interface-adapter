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
// Operation Modes
// ==============================================

// System Operation Mode
// false = Printer Mode
// true  = Terminal Mode
static const bool MODE = true; // TODO: NOT IMPLEMENTED

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
static const bool AUTO_LINE_FEED = false; // TODO: NOT IMPLEMENTED

// Printing Pitch (Characters Per Inch)
// Controls horizontal character density.
// Options: 6, 10, 12, 15
static const int PRINTING_PITCH = 10; // TODO: NOT IMPLEMENTED

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
#define SERIAL_BAUD 9600 // TODO: NOT IMPLEMENTED

#ifdef __cplusplus
}
#endif

#endif // CONFIG_H