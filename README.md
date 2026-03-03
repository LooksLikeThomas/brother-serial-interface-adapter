```
              /|
             /+|
 _________  /++|_____    _____  _    _ ____  _         __
|++++++++/||###|:::::|  |  __ `| |  | |  _ `| |  __   / /
|+++++++/#||###|:::::|  | |  | | |__| | |_) | | /  | / /
|++++++/##||##/::::::|  | |  | |  __  |  _ -| |/ | |/ /
|+++++|###||#/:::::::|  | |__| | |  | | |_) | | /| | /
|=====|###||/========|  |_____/|_|  |_|____/|__/ |__/
      |::/
      |:/      Brother Serial Interface Adapter
      |/
```

A USB-Aapter for vintage Brother typewriters — a functional replacement for the
original IF-60 interface box, built on a fully reverse-engineered proprietary protocol.

**Status:** Functional prototype — protocol fully reverse-engineered, firmware operational.

---

## Overview

In the 1980s, Brother offered the IF-60 as a hardware interface that connected compatible
electronic typewriters to PCs via RS-232 or Centronics, turning them into daisy-wheel
printers or interactive TTY terminals. The IF-60 translated between the PC's standard
interfaces and the proprietary six-signal "Brother Serial Interface System" bus on the
typewriter connector.

This project replaces the IF-60 with an Arduino-Uno. The typewriter
appears as a virtual COM port on any modern PC — no drivers, no IF-60 required.

The reverse engineering effort covered the full stack:

- The six-signal bus and its SPI-like handshake timing (≈78 kHz, MSB-first)
- All bus command bytes and their conditions (character output, cursor movement,
  pitch changes, underline control, SELECT/DESELECT sequences)
- The complete character encoding including CR/LF coalescing, tab expansion, pitch
  management, and right-margin auto-wrap behaviour

Reference documentation: [`protocol.md`](protocol.md)

---

## Compatible Typewriters

The adapter is compatible with typewriters that carry a 12-pin "Brother Serial Interface
System" connector. Two series were tested:

| Series | Connector label | Tested | Other known-compatible models |
|---|---|---|---|
| AX (JP-12X, JP-18) | — | AX-20 ✓ | AX-45, COMPACTRONIC 350, EM-31 |
| CX (JP-16X, JP-16XX) | — | CE-650 ✓ | Professional 400/440/90/80, EM-401, EM-411 |

Typewriters carrying an IF-50 connector may be compatible with unidirectional printer mode from Arduino to typewriter but is **UNTESTED**.

CX-series models are recognised at power-on and will operate in AX-compatibility
mode (plain ASCII printing), but their extended features — bold, shadow print,
half-line spacing (sub/superscript), and vertical positioning — are **not
implemented**. The CX-series is therefore untested and unsupported beyond basic
printing.

---

## Hardware Requirements

- **Arduino Uno** (ATmega328P) — the only currently supported platform. Timer1 and INT0
  are used for timing-critical functions and cannot be remapped without firmware changes.
- **Custom 12-pin connector** — the Brother Serial Interface connector is not a standard
  part. A 3D-printable FreeCAD model is provided in [`hardware/connector-3d-models/`](hardware/connector-3d-models/).
- **USB-A to USB-B cable** — standard Arduino programming cable, doubles as the PC
  connection in normal use.
- No level shifter is required. Both the Arduino and the typewriter bus operate at 5 V TTL.
  Signal lines are held in High-Z when idle to prevent back-feeding when the typewriter
  is switched off.

---

## Wiring

### Typewriter Connector Pinout

![Connector pin assignment](img/InterfaceConnectorPinAssignment.svg)

| Pin | Signal | Idle Voltage |
|-----|--------|-------------|
| 1   | GND (reference) | 0 V |
| 2   | +5 V (typewriter supply) | +5 V |
| 3   | SI (Serial In — to typewriter) | 5 V |
| 4   | SO (Serial Out — from typewriter) | 5 V |
| 5   | SCK (Serial Clock) | 5 V |
| 6   | GND | 0 V |
| 7   | KBACK (Keyboard Acknowledge) | 0 V |
| 8   | READY | 5 V |
| 9   | KBRQ (Keyboard Request) | 0 V |
| 10  | GND (frame ground) | 0 V |
| 11  | +30 V (motor supply) | 0 V* |
| 12  | +30 V (motor return) | 0 V* |

\* Pins 11–12 are bridged by the IF-60; no voltage was measured on the CE-650.
Other models may use these pins.

### Arduino Wiring

Connect the typewriter's 12-pin connector to the Arduino as follows:

| Bus signal | Arduino pin | Notes |
|---|---|---|
| SCK | 9 | Timer1 OC1A — **cannot be changed** |
| SI | 4 | Data to typewriter |
| SO | 5 | Data from typewriter |
| READY | 6 | Interface pulls LOW to signal transmission |
| KBRQ | 2 | INT0 — **cannot be changed** |
| KBACK | 3 | Polled directly, no interrupt |
| POWER | 7 | Typewriter 5 V rail via voltage divider (5 V → Arduino-safe level) |
| GND | GND | Common ground |

The remaining pins on the typewriter connector carry the typewriter's own power supply
rails and are not connected to the Arduino.

---

## Building and Flashing

The firmware uses [PlatformIO](https://platformio.org/). With PlatformIO installed:

```bash
cd firmware
pio run -e uno              # compile
pio run -e uno -t upload    # compile and flash
```

---

## Configuration

All compile-time options are in [`firmware/include/config.h`](firmware/include/config.h).
The most relevant settings:

| Option | Default | Description |
|---|---|---|
| `SERIAL_BAUD` | `9600` | Baud rate of the USB serial port |
| `AUTO_SELECT` | `1` | Automatically assert SELECT on power-on |
| `AUTO_CARRIAGE_RETURN` | `true` | Treat standalone LF as CR+LF (Unix compatibility) |
| `COAL_SGR_TO_UNDERLINE` | `true` | Map incoming ANSI SGR sequences to underline |
| `SYSTEM_LOCALE` | `LOCALE_INTL_REF` | Character set of the installed daisy wheel (`LOCALE_GERMAN`, `LOCALE_INTL_REF`) |
| `USE_COMPRESSION` | `0` | Enable Heatshrink compression on the SI buffer |
| `PITCH_BYTE` | `0xB1` | Default printing pitch: `0xB1` = 10 cpi, `0xB2` = 12 cpi, `0xB3` = 15 cpi |
| `TAB_EVERY_N` | `8` | Pre-fill tab stops every N columns at startup |

---

## Usage

1. Connect the typewriter via the 12-pin connector and the Arduino via USB.
2. Open any serial terminal at the configured baud rate (default 9600, 8N1):
   ```
   screen /dev/ttyUSB0 9600,cs8,ixon,-crtscts
   ```
   `ixon` enables XON/XOFF software flow control (required — the firmware sends
   XOFF when the buffer fills); `-crtscts` disables hardware flow control; `cs8` = 8N1.
3. Switch on the typewriter. The adapter detects power and initialises automatically.
4. If `AUTO_SELECT` is enabled, the typewriter enters online mode after a short delay.
   Otherwise, send DC1 (`0x11`) to assert SELECT manually.
5. Text sent to the serial port is printed on the typewriter. The Return key on the
   typewriter sends a newline back to the terminal.

**Character set:** The firmware expects plain ASCII input. Multibyte UTF-8 sequences
are forwarded to a transliteration engine that is **not yet complete** — unknown
sequences are silently dropped. Characters outside the installed wheel's repertoire
are either approximated via overstrike sequences (if `OVERSTRIKE_MISSING_ASCII`
is enabled) or substituted with the closest available glyph.

**ANSI formatting:** Bold, italic, and other SGR attributes sent by terminal emulators
are coalesced to the typewriter's underline mechanism when `COAL_SGR_TO_UNDERLINE` is
enabled.

**Flow control:** Software XON/XOFF is active by default. The terminal program must have
XON/XOFF enabled (hence the `ixon` flag above) if the SI buffer fills faster than the
typewriter can print.

### Escape Sequences

**Implemented and working:**

- `ESC+E` / `ESC+R` / `ESC+X` — underline on / off / clear formatting
- `ESC+"` / `ESC+#` — auto line-feed on / off
- `ESC+9` / `ESC+0` / `ESC+B` — set left margin / set right margin / clear margins
- `ESC+1` / `ESC+8` / `ESC+2` — set tab stop / clear tab stop / clear all tabs
- `ESC+US+n` — set character pitch (0x1F + 0xB1/0xB2/0xB3); only takes effect at column 1
- `ESC+HT+n` — absolute horizontal tab to column n
- `ESC+S` — full reset to `config.h` defaults
- `ESC+Y` / `ESC+Z` — print daisy-wheel position 0x20 / 0x7F (wheel probe)
- `ESC+CR+P` — printer reset (simplified; emits 0xF4 + pitch + 0x8B)
- `ESC+[ SGR` — ANSI formatting coalesced to underline (when `COAL_SGR_TO_UNDERLINE=true`)

**Not yet implemented (silently swallowed):**

- `ESC+O` / `ESC+W` / `ESC+F` / `ESC+&` — bold / shadow / double-strike (CX-series only)
- `ESC+U` / `ESC+D` / `ESC+LF` — half line-feed up/down, reverse LF (CX-series only)
- `ESC+T` / `ESC+L` / `ESC+C` — top/bottom/clear vertical margins (CX-series only)
- `ESC+/` / `ESC+\` — auto backward print (CX-series only)
- `ESC+VT+n` / `ESC+RS+n` — absolute vertical tab / set VMI (CX-series only)

---

## Known Limitations & Future Work

**Buffer capacity.**
The IF-60 has an 8 kB buffer; the Arduino Uno adapter defaults to 1024 bytes (further
reduced to 512 in debug mode). XON/XOFF flow control (`ixon`) compensates in
interactive use, but programs that send data without honouring XOFF (e.g. plain
`cat > /dev/ttyUSB0`) will overflow the buffer. A platform switch to ESP32
(520 KB SRAM, but requires 5 V level shifters) would eliminate this constraint.

**Architecture / refactoring.**
The SELECT/DESELECT handling is implemented as a separate 13-state machine. In hindsight it could be unified with the `online.c` BusSequenceBuffer path used for regular printing. Similarly, `translate.c` (stateless character tables) and `utf8.c`
(UTF-8 transliteration) solve related problems but are separate modules;
merging them into a single input-normalisation layer would simplify the code.

**UTF-8 transliteration is incomplete.**
The UTF-8 engine covers common German and West-European characters but is not
exhaustive. Unknown multibyte sequences are silently dropped. Contributions of
additional character mappings for other locales and daisy-wheel variants are
welcome — see `firmware/src/utf8.c` and `firmware/include/translate.h`.

**Untested models.**
Only the AX-20 and CE-650 have been tested. Around two dozen compatible models
exist; compatibility with AX-25, AX-30, EM-31, CE-60, Professional 80/90/400/440
is plausible given the shared bus architecture but unconfirmed.

**Protocol specification is observation-based.**
The entire protocol spec was derived from a single IF-60 unit by black-box signal
analysis; there is no official Brother documentation of the bus protocol. Edge
cases (e.g. ESC+CR+P reset, timing tolerances) may differ across IF-60 firmware
versions.

**Contributing.**
If you own a compatible typewriter model not listed above, or if you have a
daisy wheel with a different character layout and want to contribute keyboard
mappings, please open an issue or pull request.

---

## Protocol Reference

The bus uses a synchronous, master-driven protocol with six signals (SCK, SI, SO, READY,
KBRQ, KBACK). The clock runs at approximately 78 kHz; bytes are transferred MSB-first in
a SPI-like fashion. The command set is proprietary and fully documented.

Full specification: [`protocol.md`](protocol.md)

---

## Project Structure

```
brother-serial-interface-adapter/
├── firmware/                  # PlatformIO C/C++ firmware (ATmega328P)
│   ├── src/                   # Protocol, transfer, online, translate, utf8, ...
│   └── include/               # Header files and config.h
├── protocol-analysis/         # Analysis tools, signal captures, test logs
│   ├── analysis_lib/          # Python package: decoder and visualisation
│   ├── signal_captures/       # .vcd and .pkl oscilloscope captures (AX-20, CE-650)
│   └── *.ipynb                # Jupyter notebooks for analysis and testing
├── hardware/                  # 3D model of the proprietary connector (FreeCAD)
└── docs/                      # Reference documents (manuals, patents)
```

---

## Academic Context

This project is a student thesis (T3\_3101) at the
[Baden-Württemberg Cooperative State University (DHBW Mannheim)](https://www.dhbw-mannheim.de/),
conducted by [Thomas Henseler](https://www.linkedin.com/in/thomas-henseler-a101172b8/)
and supervised by [Prof. Dr.-Ing. Johannes Bauer](https://www.mannheim.dhbw.de/profile/bauer).

---

## License

The firmware, hardware designs, and analysis tools in this project are open source under
the **MIT License**. See the [LICENSE](LICENSE) file for details.

**Third-Party Copyrights**
Original Brother service manuals and documentation referenced in this project remain the
copyright of Brother Industries, Ltd. They are not covered by the MIT license of this
project.

**Trademark Acknowledgement**
Brother, Brother Serial Interface System, IF-60, IF-50, and all related product names are
trademarks of Brother Industries, Ltd. This project is an independent educational
endeavour and is not affiliated with Brother Industries, Ltd.