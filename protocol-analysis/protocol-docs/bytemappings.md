# Brother IF60 Serial Interface — Protocol Analysis

## 1. Overview

The Brother IF60 is a serial interface module that connects to Brother electronic typewriters (tested: AX20, CE650). It sits between a host PC (RS-232C serial) and the typewriter's internal bus, acting as a **bidirectional protocol translator**. The IF60 accepts standard ASCII serial data from the PC and converts it into proprietary bus commands for the typewriter. It also reads typewriter responses from the bus and can relay status information back to the PC.

The IF60 manual describes two typewriter series: **CX** (higher-end, e.g., CE650) and **AX** (basic, e.g., AX20). Many ESC sequences are CX-only — the interface swallows these commands when connected to an AX-series machine.

This document summarises findings from protocol captures taken on 2026-02-12 and 2026-02-13 across multiple test sessions, varying the typewriter model, keyboard switch, DIP switch settings, pitch selection, and power-on/select/deselect sequencing.

### Architecture

```
  PC (RS-232C)  ←→  IF60 Interface  ←→  Typewriter Internal Bus
   ASCII bytes       translates          proprietary bus bytes
                     both directions
```

The IF60 is not a simple pass-through. It maintains:
- A **character mapping table** selected by the keyboard switch and DIP settings
- Knowledge of the **connected typewriter model** (CX or AX series), generating different bus sequences for different models
- Awareness of **dead keys** on the installed daisy wheel, automatically adding carriage advance commands when printing from serial (where combining keystrokes don't apply)
- **Pitch-aware spacing**, inserting the correct HMI byte (0xB1/0xB2/0xB3) in carriage returns, newlines, and other movement commands
- **Flow control** via DC1/DC3 (XON/XOFF) for the serial side
- Internal storage of **margins, tab stops, page length**, and other formatting state

### Test Matrix (2026-02-12)

| File | Typewriter | Keyboard | DIP 1-2 (Mode) | DIP 1-3 (Region) | DIP 1-4 (Wheel) | DIP Hex |
|------|-----------|----------|-----------------|-------------------|------------------|---------|
| 163042 | AX20 | 1 (Local) | UP (Terminal) | UP | DOWN (Non-ASCII) | 0x3FF8 |
| 171307 | AX20 | 1 (Local) | DOWN (Printer) | UP | DOWN (Non-ASCII) | 0x3FFA |
| 172524 | AX20 | 1 (Local) | UP (Terminal) | DOWN | DOWN (Non-ASCII) | 0x3FFC |
| 173845 | AX20 | 2 (International) | UP (Terminal) | UP | DOWN (Non-ASCII) | 0x3FF8 |
| 174940 | AX20 | 2 (International) | UP (Terminal) | DOWN | DOWN (Non-ASCII) | 0x3FFC |
| 180042 | AX20 | 3 (Symbol) | UP (Terminal) | UP | DOWN (Non-ASCII) | 0x3FF8 |
| 181406 | AX20 | 3 (Symbol) | UP (Terminal) | DOWN | DOWN (Non-ASCII) | 0x3FFC |
| 182619 | AX20 | 1 (Local) | UP (Terminal) | UP | UP (ASCII) | 0x3FF0 |
| 184019 | AX20 | 2 (International) | UP (Terminal) | UP | UP (ASCII) | 0x3FF0 |
| 193344 | CE650 | 1 (Local) | UP (Terminal) | UP | DOWN (Non-ASCII) | 0x3FF8 |

### Test Matrix (2026-02-13) — Pitch and DIP Sweep

| Time | Typewriter | Keyboard | Pitch | DIP Variation | Notes |
|------|-----------|----------|-------|---------------|-------|
| 10:53:00 | AX20 | 1 | 12cpi | Default | Full control code scan |
| 10:59:46 | AX20 | 1 | 15cpi | Default | Full control code scan |
| 11:23:17 | AX20 | 1 | 15cpi | Default | Pitch left at 15 from prior test |
| 11:23:28–12:11:52 | AX20 | 1 | 10cpi | Various DIPs | Power-on/SELECT/DESELECT sweep |

### Test Matrix (2026-02-13) — Reverse Mapping (Typewriter → Serial)

| Time | Typewriter | Keyboard | Pitch | Notes |
|------|-----------|----------|-------|-------|
| 13:01:59 | AX20 | 1 (Local) | 10cpi | All keys: normal, Shift, Code |
| 13:28:05 | AX20 | 2 (International) | 10cpi | All keys: normal, Shift, Code (without reselect — invalid) |
| 13:46:32 | AX20 | 2 (International) | 10cpi | All keys: normal, Shift (after proper reselect) |
| 13:56:51 | AX20 | 3 (Symbol) | 10cpi | All keys: normal, Shift |

### Test Matrix (2026-02-17) — Code+Key Keyboard Comparison

| Time | Typewriter | Keyboard | Pitch | Notes |
|------|-----------|----------|-------|-------|
| 17:17:15 | AX20 | 1 (Local) | 10cpi | All Code+keys |
| 17:20:51 | AX20 | 2 (International) | 10cpi | All Code+keys |


---

## 2. DIP Switch Configuration

The IF60 has 14 DIP switches in two banks of 6 and one bank of 3 (for baud rate). These are active settings — the interface reads them and changes its translation behaviour accordingly.

### DIP Switch Map

| Switch | UP | DOWN |
|--------|-----|------|
| **1-1** | RS-232C interface | CDCC interface |
| **1-2** | Terminal mode | Printer mode |
| **1-3** | (see §3 below) | (see §3 below) |
| **1-4** | ASCII Wheel installed | Non-ASCII Wheel installed |
| **1-5** | 12-inch paper | 11-inch paper |
| **1-6** | Auto skip perforation | No auto skip |
| **2-1** | Local echo (half-duplex) | No echo (full-duplex) |
| **2-2** | DC1/DC3 serial select/deselect disabled | DC1/DC3 serial select/deselect enabled |
| **2-3** | Auto line feed off | Double spacing |
| **2-4** | 7-bit data | 8-bit data |
| **2-5** | Even parity | Odd parity |
| **2-6 to 2-8** | Baud rate selection (111 = 9600) | |


---

## 3. DIP 1-3: "USA Users Can Ignore"

### Observed Behaviour

DIP 1-3 affects **only control codes** in the range 0x03–0x06 and 0x15. It has **zero effect** on printable character mappings (0x20–0x7F), ESC sequences, or any other behaviour. This was confirmed across all three keyboard settings.

| Control Code | Name | DIP 1-3 UP | DIP 1-3 DOWN |
|-------------|------|------------|--------------|
| 0x03 | ETX | `[0x00]` | None (swallowed) |
| 0x04 | EOT | `[0x00]` | None (swallowed) |
| 0x05 | ENQ | `[0x00]` | None (swallowed) |
| 0x06 | ACK | `[0x00]` | None (swallowed) |
| 0x15 | NAK | `[0x00]`* | None (swallowed) |

\* NAK returns `[0x23]` in Printer mode (DIP 1-2=DOWN) for unknown reasons.

### Interpretation

When UP, the interface forwards these otherwise-unused control codes to the bus as NUL bytes (0x00 = carriage advance). When DOWN, they are silently discarded.

The label "USA users can ignore" suggests this relates to an accessory sold only in the US market — possibly a spell checker module that plugged into the same bus and used ETX/EOT/ENQ/ACK/NAK as part of its communication protocol. US users would leave this UP (the default) to maintain compatibility; non-US users (who would never have this accessory) could set it DOWN to discard these codes cleanly.


---

## 4. Keyboard Switch

The typewriter has a physical 3-position keyboard switch that selects which character mapping table the IF60 uses.

### Keyboard Positions

| Position | Role | Y/Z Swap | Dead Keys | Notes |
|----------|------|----------|-----------|-------|
| **KB1** (Local) | National/local layout | Yes (QWERTZ) | 1 (`` ` ``) | Paired to the local daisy wheel |
| **KB2** (International) | International layout | Yes (QWERTZ) | 3 (`` ^ ` ~ ``) | Classic diacritical combiners |
| **KB3** (Symbol) | Symbol layout | No (QWERTY) | 15 | Many punctuation marks are dead keys |

### Dead Key Handling

On a physical keyboard, dead keys print a character but do not advance the carriage — the user then types a base letter, which overprints to form an accented character (e.g., ´ + e = é). When driven from serial, there is no "next keystroke" to combine with, and the typewriter has no bus-level mechanism to disable the dead key behaviour. The IF60 works around this by appending `0x00` (carriage advance) after every dead key character, forcing the carriage to move forward automatically.

**KB3 dead keys** (trailing 0x00 in output):
`"`, `#`, `'`, `+`, `,`, `.`, `/`, `:`, `;`, `=`, `?`, `@`, `\`, `]`, `` ` ``

Some characters require the Code modifier to access on the daisy wheel (see "Code Key Simulation" below). When such a character is also a dead key, the output combines both mechanisms independently: `[0x88, <pos>, 0x89, 0x00]` — the `0x88/0x89` framing accesses the Code-layer character and the trailing `0x00` compensates for the dead key. These two features are completely independent: `0x88/0x89` framing occurs whenever a character needs the Code modifier (whether or not it is a dead key), and the trailing `0x00` occurs whenever a character is a dead key (whether or not it requires Code access).

### Code Key Simulation (0x88/0x89)

The typewriter's Code key is a third modifier (alongside Shift) that accesses additional characters printed in green on the keycaps. These are extra glyphs on the daisy wheel that are not reachable through normal or shifted keypresses.

On the physical bus, 0x88 represents the Code key being pressed down and 0x89 represents the Code key being released. When a physical user presses Code+A, the typewriter sends `[0x88, 0x61, 0x89]` on the bus — Code down, the 'a' wheel position, Code up. Multiple keypresses can occur between a single 0x88/0x89 pair if the Code key is held while pressing several keys.

When the IF60 needs to print a character that resides on a Code-accessible wheel position (in the forward serial→bus direction), it simulates a Code keypress by sending `[0x88, <pos>, 0x89]` on the bus. The typewriter interprets this exactly as if someone physically pressed Code and struck that key.

Characters that use Code key simulation in forward translation include positions like `<` on KB1 (`[0x88, 0x58, 0x89]`), `>` on KB1 (`[0x88, 0x56, 0x89]`), and various KB3 punctuation characters. Whether a character requires Code access depends on the physical daisy wheel layout and is independent of whether the character is a dead key.

In the reverse direction (typewriter→serial, §16), 0x88 and 0x89 appear as literal Code key press/release events, and the IF60 uses the enclosed bus bytes to determine which control code or character to send on the serial side.

### DC1 SELECT Keyboard Identification

The DC1 (SELECT) response includes a keyboard identification byte in the third position of the typewriter status response:

| Keyboard | ID Byte | Binary |
|----------|---------|--------|
| KB1 | 0x04 | 00000100 |
| KB2 | 0x24 | 00100100 |
| KB3 | 0x44 | 01000100 |

Bit 2 is always set. KB2 additionally sets bit 5, KB3 sets bit 6.

### Keyboard Switch Latching

The IF60 reads the keyboard switch position during the DC1 SELECT handshake and latches it. Changing the physical switch without re-selecting does **not** change the translation tables — the interface continues using the previously selected keyboard's mappings in both directions. This was confirmed experimentally on 2026-02-13: after switching from KB1 to KB2 without reselecting, the reverse mapping produced KB1 serial output. A DC3 DESELECT → DC1 SELECT cycle was required to activate the KB2 tables.

### Character Mapping Differences Across Keyboards (Forward: Serial → Bus)

Letters A–X and digits 0–9 map identically across all keyboards. The differences concentrate on punctuation and the ISO 646 national variant positions (0x23, 0x40, 0x5B–0x5E, 0x60, 0x7B–0x7E).

**Positions that differ (DIP 1-3=UP, DIP 1-4=DOWN):**

| Input | ASCII | KB1 Output | KB2 Output | KB3 Output |
|-------|-------|------------|------------|------------|
| 0x21 | `!` | `[0x21]` | `[0x21]` | `[0x88,0x56,0x89]` |
| 0x22 | `"` | `[0x40]` | `[0x40]` | `[0x22,0x00]` |
| 0x23 | `#` | `[0x5C]` | `[0x27]` | `[0x76,0x00]` |
| 0x24 | `$` | `[0x24]` | `[0x24]` | `[0x21]` |
| 0x25 | `%` | `[0x25]` | `[0x25]` | `[0x28]` |
| 0x26 | `&` | `[0x20]` | `[0x20]` | `[0x23]` |
| 0x27 | `'` | `[0x60]` | `[0x60]` | `[0x27,0x00]` |
| 0x28 | `(` | `[0x2A]` | `[0x2A]` | `[0x29]` |
| 0x29 | `)` | `[0x28]` | `[0x28]` | `[0x2C]` |
| 0x2A | `*` | `[0x5B]` | `[0x5B]` | `[0x24]` |
| 0x2B | `+` | `[0x5D]` | `[0x5D]` | `[0x2B,0x00]` |
| 0x2C | `,` | `[0x2C]` | `[0x2C]` | `[0x88,0x5A,0x89,0x00]` |
| 0x2D | `-` | `[0x2F]` | `[0x2F]` | `[0x3F]` |
| 0x2E | `.` | `[0x2E]` | `[0x2E]` | `[0x88,0x58,0x89,0x00]` |
| 0x2F | `/` | `[0x26]` | `[0x26]` | `[0x2D,0x00]` |
| 0x3A | `:` | `[0x3E]` | `[0x3E]` | `[0x3A,0x00]` |
| 0x3B | `;` | `[0x3C]` | `[0x3C]` | `[0x3B,0x00]` |
| 0x3C | `<` | `[0x88,0x58,0x89]` | `[0x22]` | `[0x88,0x59,0x89]` |
| 0x3D | `=` | `[0x29]` | `[0x29]` | `[0x3D,0x00]` |
| 0x3E | `>` | `[0x88,0x56,0x89]` | `[0x3B]` | `[0x88,0x55,0x89]` |
| 0x3F | `?` | `[0x5F]` | `[0x5F]` | `[0x5F,0x00]` |
| 0x40 | `@` | `[0x23]` | `[0x2D]` | `[0x66,0x00]` |
| 0x4A | `J` | `[0x4A]` | `[0x4A]` | `[0x7B]` |
| 0x59 | `Y` | `[0x5A]` | `[0x5A]` | `[0x59]` |
| 0x5A | `Z` | `[0x59]` | `[0x59]` | `[0x5A]` |
| 0x5B | `[` | `[0x22]` | `[0x7B]` | `[0x6A]` |
| 0x5C | `\` | `[0x3A]` | `[0x88,0x58,0x89]` | `[0x5C,0x00]` |
| 0x5D | `]` | `[0x7B]` | `[0x3A]` | `[0x2F,0x00]` |
| 0x5E | `^` | `[0x88,0x57,0x89]` | `[0x23,0x00]` | `[0x25]` |
| 0x5F | `_` | `[0x3F]` | `[0x3F]` | `[0x2E]` |
| 0x60 | `` ` `` | `[0x2B,0x00]` | `[0x2B,0x00]` | `[0x60,0x00]` |
| 0x66 | `f` | `[0x66]` | `[0x66]` | `[0x4A]` |
| 0x6A | `j` | `[0x6A]` | `[0x6A]` | `[0x7C]` |
| 0x76 | `v` | `[0x76]` | `[0x76]` | `[0x40]` |
| 0x79 | `y` | `[0x7A]` | `[0x7A]` | `[0x79]` |
| 0x7A | `z` | `[0x79]` | `[0x79]` | `[0x7A]` |
| 0x7B | `{` | `[0x27]` | `[0x5C]` | `[0x5B]` |
| 0x7C | `\|` | `[0x3B]` | `[0x88,0x56,0x89]` | `[0x5D]` |
| 0x7D | `}` | `[0x7C]` | `[0x88,0x57,0x89]` | `[0x26]` |
| 0x7E | `~` | `[0x2D]` | `[0x3D,0x00]` | `[0x2A]` |

**Note:** Entries shown as `[0x88, <pos>, 0x89]` use Code key simulation to access characters on the daisy wheel that require the Code modifier. Entries with a trailing `0x00` indicate dead key characters where the IF60 appends a carriage advance to compensate for the typewriter's non-advancing dead key behaviour. These two mechanisms are independent.


---

## 5. DIP 1-4: ASCII Wheel / Non-ASCII Wheel

### Observed Behaviour

DIP 1-4 tells the interface which physical daisy wheel is installed. When set to UP (ASCII Wheel), the interface remaps **KB2 (International)** to use nearly the same translation table as **KB1 (Local)**. This makes sense: if the ASCII wheel is installed (same wheel as the local layout), the international keyboard's national variant characters aren't physically present on the wheel, so the interface substitutes the ASCII/local mappings instead.

**KB1 is completely unaffected** by this switch — its table stays the same regardless.

### KB2 Positions That Change (DIP 1-4 UP vs DOWN)

| Input | ASCII | KB2 + ASCII Wheel (UP) | KB2 + Non-ASCII Wheel (DOWN) |
|-------|-------|------------------------|------------------------------|
| 0x23 | `#` | `[0x5C]` (= KB1) | `[0x27]` |
| 0x3C | `<` | `[0x88,0x58,0x89]` (= KB1) | `[0x22]` |
| 0x3E | `>` | `[0x88,0x56,0x89]` (= KB1) | `[0x3B]` |
| 0x40 | `@` | `[0x23,0x00]` (≈ KB1) | `[0x2D]` |
| 0x5B | `[` | `[0x22]` (= KB1) | `[0x7B]` |
| 0x5C | `\` | `[0x3A]` (= KB1) | `[0x88,0x58,0x89]` |
| 0x5D | `]` | `[0x7B]` (= KB1) | `[0x3A]` |
| 0x5E | `^` | `[0x88,0x57,0x89]` (= KB1) | `[0x23,0x00]` |
| 0x7B | `{` | `[0x27]` (= KB1) | `[0x5C]` |
| 0x7C | `\|` | `[0x3B]` (= KB1) | `[0x88,0x56,0x89]` |
| 0x7D | `}` | `[0x7C,0x00]` (≈ KB1) | `[0x88,0x57,0x89]` |
| 0x7E | `~` | `[0x2D]` (= KB1) | `[0x3D,0x00]` |

With DIP 1-4=UP, KB1 and KB2 become nearly identical. The only remaining differences are at 0x40 (`@`) and 0x7D (`}`), where KB2 adds a trailing `0x00` — these positions are physically dead keys on KB2's keyboard even when remapped to KB1's output characters, so the IF60 still appends the carriage advance compensation.

In summary: when the ASCII wheel is installed, setting DIP 1-4=UP effectively tells the interface "ignore the KB1/KB2 distinction — they should both use the ASCII mapping."


---

## 6. DIP 1-2: Terminal Mode / Printer Mode

Only one Printer mode capture exists (KB1, DIP 1-3=UP). The control code and printable character tables are identical to Terminal mode with one exception:

| Code | Terminal Mode | Printer Mode |
|------|-------------|--------------|
| 0x15 (NAK) | `[0x00]` | `[0x23]` |

The NAK response changes from a NUL byte to `[0x23]`. This may relate to status reporting in printer mode, or to the spell checker accessory protocol (which would operate in printer mode context).

### Diablo 630 Compatibility

The IF60 manual describes Printer Mode as making the typewriter function as a Diablo-compatible printer. The Diablo 630 was the dominant daisy wheel printer of the 1980s, and its command language became a de facto standard — Diablo emulation was an expected feature on competing daisy wheel printers and even early laser printers. The IF60's ESC sequences share substantial overlap with the Diablo 630 command set: ESC+HT+n (absolute horizontal tab), ESC+LF (reverse line feed), ESC+VT+n (absolute vertical tab), ESC+FF+n (set lines per page), ESC+CR+P (remote reset), auto backward printing (ESC+/ and ESC+\\), bold/shadow print, underscore, and margin control are all present in both protocols.

In Printer Mode, the IF60 does not relay typewriter keyboard data back over serial — data flows one direction (host → IF60 → typewriter) with only protocol responses (ACK, NAK, DC1/DC3) sent back to the host. Per the manual, **Auto Backward Print is the default in Printer Mode** (ESC+/ sets, ESC+\\ clears). When auto backward print is active, the typewriter performs logic seeking (bidirectional printing). Certain ESC sequences (marked with † in §12) cause a carriage return to the left margin and reset to forward printing.


---

## 7. Pitch Selection and Pitch-Aware Bus Commands

The IF60 interface has a pitch selection setting (separate from DIP switches, controlled by the typewriter's PITCH select key) that determines the character spacing. This setting has a **pervasive effect** on bus output — the interface inserts the appropriate HMI byte into every movement command that needs to be pitch-aware.

### Pitch → HMI Byte Mapping

Per the manual, HMI = (n - 1) × 1/120 inch, where n is the parameter value:

| Pitch Setting | CPI | HMI Parameter (n) | Physical Spacing | HMI Byte |
|--------------|-----|-------------------|-----------------|----------|
| 10 (default) | 10 cpi | 13 | 12/120 = 1/10 inch | 0xB1 |
| 12 | 12 cpi | 11 | 10/120 = 1/12 inch | 0xB2 |
| 15 | 15 cpi | 9 | 8/120 = 1/15 inch | 0xB3 |

**Important:** 0xB1, 0xB2, and 0xB3 are **mode-setting bytes** — they configure the typewriter's step size for all subsequent movement, not movement commands in themselves. When the IF60 sends `[0x9E, <P>]` for CR, the pitch byte tells the typewriter mechanism "each step is now this wide." After that, every 0x00 the typewriter receives moves the carriage by that pitch-dependent amount. The pitch byte is a configuration command, not a motion command.

**Pitch changes require the carriage to be at column 1.** The IF60 only emits the pitch byte on the bus when `column == 1`. If ESC+US+n is sent while the carriage is at any other position — including the left margin — the command is silently swallowed. The internal pitch state is presumably updated regardless, as the next CR returns the carriage to column 1 and includes the pitch byte, which applies the new pitch at that point. This was confirmed by testing ESC+US+11 (12cpi) at column 17 with a left margin set there: the command produced no bus output. The same command at column 1 produced `[0xB2]` as expected.

### Commands Affected by Pitch

The pitch HMI byte (denoted `<P>` below) appears in these positions:

| Command | 10cpi Output | 12cpi Output | 15cpi Output | Pattern |
|---------|-------------|-------------|-------------|---------|
| DC1 SELECT | `[0xF9,0xFD,0x7F,0xF4,0xB1,0xB1]` | `[0xF9,0xFD,0x7F,0xF4,0xB1,0xB2]` | `[0xF9,0xFD,0x7F,0xF4,0xB1,0xB3]` | `[0xF9,0xFD,0x7F,0xF4,0xB1,<P>]` |
| CR (0x0D) | `[0x9E,0xB1]` | `[0x9E,0xB2]` | `[0x9E,0xB3]` | `[0x9E,<P>]` |
| CR+LF (0x0D,0x0A) | `[0x02,0xB1]` | `[0x02,0xB2]` | `[0x02,0xB3]` | `[0x02,<P>]` |

In the SELECT sequence, the **first** `0xB1` is fixed (a "set to default 10cpi" command), and the **second** byte is the active pitch setting. This means SELECT always resets the typewriter's pitch to 10cpi first, then immediately switches to the requested pitch. Per the manual, ESC+S resets HMI to the pitch specified by the PITCH select key.

The pitch byte appears in CR, CR+LF, and SELECT sequences to reassert the pitch mode at line boundaries and during initialization — a defensive "known good state" pattern ensuring the typewriter remains synchronized with the IF60's pitch setting.

### Commands NOT Affected by Pitch

The following commands produce the **same output** regardless of pitch setting:

| Command | Output (all pitches) | Notes |
|---------|---------------------|-------|
| LF (0x0A) | `[0x9F]` | Vertical movement, pitch-independent |
| LF+LF (0x0A,0x0A) | `[0x9F,0x9F]` | Multiple LFs |
| BS (0x08) | `[0x03]` | Backspace is one position regardless |
| BEL (0x07) | `[0xF5]` | Sound, not movement |
| DC3 DESELECT | `[0xF8]` | Power-off, no movement |
| ESC+VT (absolute VT) | `[0x9F × N]` | Vertical, pitch-independent |

ESC+HT absolute positioning sends `[0x8B, 0x00 × N]` where N = (n - 1). The 0x00 bytes are standard advance steps whose physical width depends on the current pitch. The IF60 sends the same number of bytes regardless of pitch, but the resulting physical position changes because each step covers a different distance at different pitch settings.

### Margin Repositioning and Pitch

After setting a left margin with ESC+9, the CR command repositions the carriage to the margin. The repositioning sequence sends 0x8B (disable underline, defensively) followed by 0x00 advance bytes to step to the margin column:

```
CR with left margin at col 17 (underline off):
  [0x9E, <P>, 0x8B, 0x00 × 16]
   └ CR   └ pitch  └ underline  └ 16 advance steps
                    └ off

CR with left margin at col 17 (underline on):
  [0x9E, <P>, 0x8B, 0x00 × 16, 0x8A]
   └ CR   └ pitch  └ underline  └ 16 advance steps  └ underline
                    └ off                             └ restored
```

The `0x8B` / `0x8A` pair brackets all horizontal repositioning to prevent drawing an underline during carriage movement. The `0x8B` (disable underline) is always emitted before movement. The `0x8A` (enable underline) is only emitted after movement if underline was active before the move. This same bracket pattern appears in all horizontal repositioning: CR with left margin, ESC+HT absolute movement, ESC+S reset repositioning, and auto-wrap at right margin.

The pitch byte in the CR command sets the step width before the repositioning movement. At 12cpi, the same CR with left margin at col 17 produces:

```
[0x9E, 0xB2, 0x8B, 0x00 × 16]
```

The byte count (16 advances) is identical — only the pitch byte changes. No trailing pitch byte is emitted after the repositioning. The pitch is set once at the start and applies to all subsequent `0x00` steps.

Margins are stored as a **column number** (character position), not as an absolute physical distance. The 0x00 bytes in the repositioning sequence are the same pitch-dependent steps as all other movement — each 0x00 advances one position at the current pitch. The physical position of the margin therefore changes with the active pitch. Per the manual, left margin is set at the present position (ESC+9), and the minimum distance between left and right margins is 24/120 inch.

### Implications

The pitch setting is **not purely cosmetic** — it fundamentally changes the bus protocol for every horizontal movement. Any software emulating the IF60 must track the current pitch setting and substitute the correct HMI byte in all affected sequences. The 0x00 byte always means "advance one position at the current pitch" — the number of 0x00 bytes in a sequence stays the same regardless of pitch, but the physical distance each 0x00 moves the carriage depends on the active pitch mode.


---

## 8. Complete Control Code Map (AX20)

The 2026-02-13 captures provide a complete scan of all 32 control codes (0x00–0x1F) on the AX20. Codes shown at 10cpi pitch unless otherwise noted; pitch-dependent bytes are marked with `<P>`. The CX/AX columns indicate support per the IF60 manual.

| Code | Name | AX20 Bus Output | CX | AX | Notes |
|------|------|-----------------|----|----|-------|
| 0x00 | NUL | None | | | Swallowed — no bus output |
| 0x01 | SOH | None | | | Swallowed |
| 0x02 | STX | None | | | Swallowed |
| 0x03 | ETX | `[0x00]` | | | Carriage advance (DIP 1-3=UP only) |
| 0x04 | EOT | `[0x00]` | | | Carriage advance (DIP 1-3=UP only) |
| 0x05 | ENQ | `[0x00]` | | | Carriage advance (DIP 1-3=UP only) |
| 0x06 | ACK | `[0x00]` | | | Carriage advance (DIP 1-3=UP only) |
| 0x07 | BEL | `[0xF5]` | O | O | Acoustic alarm ~2 sec |
| 0x08 | BS | `[0x03]` | O | O | Backspace one character |
| 0x09 | HT | — | O | O | Move to next HT position (if set) |
| 0x0A | LF | `[0x9F]` | O | O | Line feed; carriage stays at current column |
| 0x0B | VT | — | O | O | Feed to next VT position (if set; does not return to left margin) |
| 0x0C | FF | `[0x9F × N]` | O | O | Form feed (see §9 for full behaviour) |
| 0x0D | CR | `[0x9E, <P>]` | O | O | Carriage return + pitch byte |
| 0x0E | SO | None | | | Swallowed |
| 0x0F | SI | None | | | Swallowed |
| 0x10 | DLE | None | | | Swallowed |
| 0x11 | DC1 | SELECT sequence | O | O | Select — puts IF60 in active state |
| 0x12 | DC2 | None | | | Swallowed (no function observed) |
| 0x13 | DC3 | `[0xF8]` | O | O | Deselect — puts IF60 in inactive state |
| 0x14 | DC4 | None | | | Swallowed |
| 0x15 | NAK | `[0x00]` | | | Carriage advance (DIP 1-3=UP only)* |
| 0x16 | SYN | None | | | Swallowed |
| 0x17 | ETB | None | | | Swallowed |
| 0x18 | CAN | None | | | Swallowed |
| 0x19 | EM | None | | | Swallowed |
| 0x1A | SUB | None | | | Swallowed |
| 0x1B | ESC | None | O | O | Escape prefix — see §12 |
| 0x1C | FS | None | | | Swallowed |
| 0x1D | GS | None | | | Swallowed |
| 0x1E | RS | None | | | Swallowed |
| 0x1F | US | None | | | Swallowed |

\* NAK returns `[0x23]` in Printer mode (DIP 1-2=DOWN).


---

## 9. Carriage Return, Line Feed, and Form Feed Interaction

The IF60 uses a CR lookahead mechanism: when CR is received, it is held pending until the next byte arrives. The following byte determines how the CR is emitted. This single mechanism explains CR+LF coalescing, LF+CR swallowing, and triple CR deduplication.

### CR+LF Combinations (AX20, <P> = pitch byte)

| Input Sequence | Bus Output | Interpretation |
|---------------|-----------|----------------|
| `0x0D` (CR alone, more data follows) | `[0x9E, <P>]` | CR held pending, flushed when next non-LF byte arrives |
| `0x0D` (CR alone, no more data) | (swallowed) | CR held pending indefinitely, never flushed |
| `0x0A` (LF alone) | `[0x9F]` | Immediate line feed, carriage stays at current column |
| `0x0D, 0x0A` (CR then LF) | `[0x02, <P>]` | CR held, LF arrives → coalesced into single bus command |
| `0x0A, 0x0D` (LF then CR, no more data) | `[0x9F]` | LF emits immediately, CR held pending, swallowed (no more data) |
| `0x0A, 0x0D, 0x20` (LF then CR then printable) | `[0x9F, 0x9E, <P>, 0x00]` | LF emits, CR held, printable flushes CR, then printable emits |
| `0x0A, 0x0A` (two LFs) | `[0x9F, 0x9F]` | Two immediate line feeds (no lookahead on LF) |
| `0x0D, 0x0D, 0x0D` (three CRs, no more data) | `[0x9E, <P>, 0x9E, <P>]` | Each CR flushes the previous; last CR swallowed (no more data) |
| `0x0D, 0x0D, 0x0D, 0x58` (three CRs then 'X') | `[0x9E, <P>, 0x9E, <P>, 0x9E, <P>, 0x58]` | Each CR flushes the previous; 'X' flushes the last; all three emitted |

### Form Feed Behaviour

Per the manual, FF (0x0C) performs three operations in sequence:
1. Prints one line of data from the buffer
2. Feeds the form one line if auto-LF is set by DIP switch or ESC sequence (effective even when CR alone is entered, as CR is always followed by LF in this mode)
3. Carriage return is effective even if print data is not received

On the AX20, the observed bus output for FF is `[0x9F × N]` — a stream of line feeds to advance to the top of the next page. The number of line feeds depends on the current position on the page and the page length setting (DIP 1-5: 12-inch or 11-inch paper, or set via ESC+FF+n).

On the CE650, FF uses a coarse-to-fine VMI stepping sequence: `[0xA3, 0x9F×N, 0xA2, 0x9F, 0xA0]` — widest spacing for the bulk, then medium, then finest for precise positioning at the top of the next page.

### Auto Line Feed (ESC+" and ESC+#)

Per the manual:
- **ESC+"** enables auto LF: when CR is received, the typewriter automatically performs a LF after it.
- **ESC+#** disables auto LF.

DIP 2-3 controls the default: DOWN = double spacing (auto LF enabled), UP = auto LF off.

### Key Observations

**CR lookahead mechanism:** The IF60 never emits CR immediately. When CR (0x0D) is received, it enters a pending state. The next byte determines the outcome: LF → coalesce to `0x02`; another CR → flush pending CR as `0x9E` and hold the new CR; any other byte → flush pending CR as `0x9E`. If no further byte arrives, the pending CR is silently discarded. This was confirmed by sending `[0x0A, 0x0D]` with a 10-second wait (CR swallowed), then `[0x0A, 0x0D, 0x20]` (CR flushed by the space), and `[0x0D, 0x0D, 0x0D, 0x58]` (all three CRs flushed by the trailing 'X'). The mechanism has no timeout — the CR remains pending indefinitely until resolved by the next byte or discarded at end of input.

**LF has no lookahead:** LF (0x0A) emits `[0x9F]` immediately and unconditionally. It does not inspect what follows.

**LF column persistence:** A standalone LF moves the paper vertically but leaves the carriage at its current horizontal position. This is consistent with the manual's description: "The subsequent data is over-printed in the same position as the carriage does not return to the left margin."


---

## 10. Power-On Initialization and SELECT/DESELECT Sequences

### 10.1 Power-On Handshake

The IF60 interface and typewriter perform a power-on initialization handshake automatically when power is applied.

#### AX20 Power-On

```
[  ] = [ 0xFE, 0x7F ] ([ 0x00, 0x30 ])
```

#### CE650 Power-On

```
[  ] = [ 0xFE, 0x7F ] ([ 0x00, 0x6A ])
```

The interface sends `[0xFE, 0x7F]` and the typewriter responds with a 2-byte identification code. The `0xFE` byte is exclusively a power-on initialization command — it does not appear anywhere else in the protocol. The `0x7F` is the same identification byte seen in DC1 SELECT and RESET sequences.

#### Model Identification

| Typewriter | Init Command | Response Byte 1 | Response Byte 2 |
|-----------|-------------|-----------------|-----------------|
| AX20 | `[0xFE, 0x7F]` | 0x00 | 0x30 |
| CE650 | `[0xFE, 0x7F]` | 0x00 | 0x6A |

The first byte is 0x00 for both models. The second byte serves as the model identifier — `0x30` for the AX20, `0x6A` for the CE650. This is how the IF60 determines whether to use CX-series or AX-series command translations.

#### Intermittent Capture

The power-on handshake is not always captured in test logs — it was present in approximately 77% of captures (10 of 13). This is a timing issue with the bus monitor start-up window, not a protocol inconsistency. The handshake is **not affected by any DIP switch setting**.

### 10.2 SELECT (DC1) and DESELECT (DC3)

Per the manual: DC1 (0x11) puts the IF60 in Select state; DC3 (0x13) puts it in Deselect state.

#### SELECT Sequence (AX20, pitch-dependent)

```
DC1: [0xF9, 0xFD, 0x7F, 0xF4, 0xB1, <P>] → response (0x00, 0x00, <KB>, 0x00, 0x00, 0x00)
```

The last byte reflects the current pitch setting (`0xB1` at 10cpi, `0xB2` at 12cpi, `0xB3` at 15cpi). The penultimate `0xB1` is always fixed — it resets the typewriter to 10cpi before applying the selected pitch.

#### DESELECT Sequence

```
DC3: [0xF8]
```

DESELECT is a single byte, unaffected by pitch or any DIP setting.

#### DC2 (0x12)

DC2 produces no bus output. It is swallowed by the interface.

#### DIP Switch Effects on SELECT/DESELECT

The following DIP switches were individually varied. **None affected the SELECT/DESELECT bus sequences:**

| DIP Switch | Setting Tested | Effect on SELECT/DESELECT |
|-----------|---------------|--------------------------|
| 1-2 (Mode) | DOWN (Printer) | No change |
| 1-5 (Paper) | UP (12-inch) | No change |
| 1-6 (Skip perf) | UP (Auto skip) | No change |
| 2-1 (Echo) | UP (Half-duplex) | No change |
| 2-2 (DC1/DC3) | UP (Disabled) | No change* |
| 2-3 (Auto LF) | UP (Auto LF off) | No change |

\* DIP 2-2 controls whether the IF60 accepts DC1 (0x11) and DC3 (0x13) bytes received from the serial side as SELECT/DESELECT triggers. When UP (disabled), the host cannot remotely select or deselect the interface by sending DC1/DC3 over serial. The bus-side SELECT/DESELECT sequences are unaffected because they are generated by the IF60 internally, not triggered by incoming serial bytes during this test. This switch does not control XON/XOFF flow control — it controls whether serial DC1/DC3 are interpreted as typewriter select/deselect commands.


---

## 11. Margin, Tab, and Page Formatting

### Setting Margins

Margins are set using ESC sequences that produce **no bus output** — they are stored internally by the IF60 interface:

| ESC Sequence | Function | Notes (from manual) |
|-------------|----------|---------------------|
| ESC+9 | Set left margin at current column | Absolute HT or BS can move past it; minimum 24/120" between margins |
| ESC+0 | Set right margin at current column | Absolute HT can move past it; minimum 24/120" between margins |
| ESC+B | Clear margins (return to defaults) | — |

### CR Behaviour with Left Margin

When a left margin is set, CR no longer returns to column 1. Instead, it returns to the left margin position:

```
CR with left margin at col 17:
  [0x9E, 0xB1, 0x8B, 0x00 × 16]
   └ CR to col 0
         └ pitch mode
               └ disable underline
                     └ 16 advance steps to col 17 (1-based)
```

The interface sends a full CR to column 0, then sends 0x8B (disable underline, defensively) followed by 0x00 advance bytes to step the carriage to the margin column. Each 0x00 is one position at the current pitch, so the physical distance to the margin depends on the active pitch.

### Automatic Line Wrapping at Right Margin

When characters are printed past the right margin, the interface automatically inserts a CR+LF:

```
Writing 4 spaces past the right margin:
  Input:  [0x20, 0x20, 0x20, 0x20]
  Output: [0x02, 0xB1, 0x00, 0x00, 0x00, 0x00]
           └ auto CR+LF
                  └ pitch advance
                        └ remaining spaces on new line
```

### Tab Stops (HT and VT)

Per the manual:

| ESC Sequence | Function | CX | AX |
|-------------|----------|----|----|
| ESC+1 | Set HT at current position (up to 10) | O | O |
| ESC+8 | Clear HT at current position only | O | O |
| ESC+2 | Clear all HT **and** all VT positions | O | O |
| ESC+- | Set VT at current position (up to 10) | O | |
| HT (0x09) | Move carriage to next HT position (no-op if none set) | O | O |
| VT (0x0B) | Feed paper to next VT position (no-op if none set; does not return to left margin) | O | O |

### Absolute Positioning (ESC+HT and ESC+VT)

Per the manual:
- **ESC+HT+n**: Absolute horizontal movement. Range = (n-1) × HMI. The n specifies 1–126 (excluding NUL and DEL). Moves directly from left end of platen — not stored as HT. Margins are ignored. Does not operate if position goes beyond right end of platen.
- **ESC+VT+n**: Absolute vertical movement (CX only). Range = (n-1) × VMI. Feeds from page top to set position — not stored as VT. Top and bottom margins are ignored. Does not operate beyond page length.

Observed bus output (AX20):

| Command | Input | Output | Notes |
|---------|-------|--------|-------|
| HT to col 17 | `[0x1B, 0x09, 0x11]` | `[0x8B, 0x00 × 16]` | 16 advance steps (column 1 is home, column 17 requires 16 steps) |
| HT to col 17 (already there) | `[0x1B, 0x09, 0x11]` | None | No movement needed |
| HT back to col 9 | `[0x1B, 0x09, 0x09]` | `[0x8B, 0x03 × 8]` | 8 backspaces |
| VT to line 3 | `[0x1B, 0x0B, 0x03]` | `[0x9F, 0x9F]` | 2 line feeds (from line 1) |
| VT to line 3 (already there) | `[0x1B, 0x0B, 0x03]` | None | No movement needed |
| VT to line 2 (upward) | `[0x1B, 0x0B, 0x02]` | None | AX20 cannot reverse paper feed |

The `0x8B` byte disables underline mode defensively to prevent drawing an underline during carriage movement. The subsequent `0x00` bytes are standard carriage advance steps — the same byte used for space on the AX20. The number of 0x00 bytes equals (n - 1), consistent with the manual's formula: movement range = (n - 1) × HMI. The physical distance of each 0x00 step depends on the currently active pitch — at 10cpi each step is 1/10 inch, at 12cpi each step is 1/12 inch, at 15cpi each step is 1/15 inch. The IF60 does not perform any pitch calculation; it simply sends (n - 1) advance bytes and the typewriter's mechanism moves by one HMI unit per step.

Tab stop positions use the same 0x00 advance bytes as regular character spacing — their physical width is determined by the current pitch setting, not by a separate fixed-width unit system. The same column number lands at a different physical position depending on the active pitch.

### Page Length and Page Margins

Per the manual:

| ESC Sequence | Function | CX | AX |
|-------------|----------|----|----|
| ESC+FF+n | Set page length = n × VMI lines; ESC+S resets to DIP switch default | O | O |
| ESC+T | Set top margin at current position | O | |
| ESC+L | Set bottom margin at current position | O | |
| ESC+C | Clear top and bottom margins | O | |

Page length is stored as an absolute position, so changing VMI changes the number of lines per page. Top margin causes automatic paper feed when the paper reaches page top by LF. Bottom margin causes automatic feed to next page top when reached by LF, Auto LF, or Half LF. When skip perforation is set (DIP 1-6), clearing margins reverts to a 1-inch margin.


---

## 12. Complete ESC Sequence Reference

The following table lists all documented ESC sequences with their observed bus output and CX/AX compatibility. Sequences marked with † cause a carriage return to the left margin and reset to forward printing when auto backward print is active.

| ESC Sequence | Function | AX20 Bus Output | CE650 Bus Output | CX | AX |
|-------------|----------|-----------------|------------------|----|----|
| ESC+HT+n | Absolute HT movement | `[0x8B, 0x00×N, 0x8A?]` or `[0x8B, 0x03×N, 0x8A?]` | `[0x9E, 0x8B, 0xB1, 0x00×N, 0xB1]` | O | O |
| † ESC+LF | Reverse paper feed (one VMI) | None (swallowed) | `[0x06, 0x06]` | O | |
| ESC+VT+n | Absolute VT movement | `[0x9F × N]` | — | O | |
| ESC+FF+n | Set page length | (internal) | — | O | O |
| ESC+CR+P | Reset printer | `[0xF4,0xB1,0x8B,0xFD,0x7F]` | `[0xA0,0xF4,0xB1,0x8D,0x8B,0xFD]` | O | O |
| ESC+RS+n | Set VMI (line spacing) | None (swallowed) | `[0xA0–0xA3]` | O | |
| ESC+US+n | Set HMI (character pitch) | `[0xB1–0xB3]`* | `[0xB1–0xB3]` | O | O |
| ESC+" | Auto LF ON | (internal) | — | O | O |
| † ESC+# | Auto LF OFF | (internal) | — | O | O |
| † ESC+& | Clear bold, shadow, double-strike | None (swallowed) | `[0x8D]` | O | O |
| ESC+- | Set VT at current position | (internal) | — | O | |
| † ESC+/ | Set auto backward print | (internal) | — | O | |
| ESC+0 | Set right margin | (internal) | — | O | O |
| ESC+1 | Set HT at current position | (internal) | — | O | O |
| ESC+2 | Clear all HT and VT | (internal) | — | O | O |
| ESC+8 | Clear current position HT | (internal) | — | O | O |
| ESC+9 | Set left margin | (internal) | — | O | O |
| ESC+B | Clear margins | (internal) | — | O | O |
| ESC+C | Clear top/bottom margins | (internal) | — | O | |
| † ESC+D | Reverse half LF (up 1/12") | None (swallowed) | `[0x06]` | O | |
| ESC+E | Enable underline | `[0x8A]` | `[0x8A]` | O | O |
| † ESC+F | Set double-strike print | None (swallowed) | `[0x8C]` | O | |
| ESC+L | Set bottom margin | (internal) | — | O | |
| † ESC+O | Set bold print | None (swallowed) | `[0x8C]` | O | |
| ESC+R | Disable underline | `[0x8B]` | 9 bytes (noisy) | O | O |
| ESC+S | Reset to DIP defaults | `[0x9E,<P>,0x8B,0x00×N,0x8A?]` | `[0xA0, 0xB1]` | O | O |
| † ESC+T | Set top margin | (internal) | — | O | |
| † ESC+U | Half LF down (1/12") | None (swallowed) | `[0x07, 0xA0]` | O | |
| † ESC+W | Set shadow print | None (swallowed) | `[0x8C]` | O | |
| ESC+X | Clear all formatting | `[0x8B]` | `[0x8B, 0x8D]` | O | O |
| ESC+Y | Print 0x20 glyph (space position) | varies by KB | varies | O | O |
| ESC+Z | Print 0x7F glyph (DEL position) | varies by KB | varies | O | O |
| † ESC+\\ | Clear auto backward print | (internal) | — | O | |

Notes:
- "(internal)" means the command is processed by the IF60 without producing bus output — the state is stored in the interface's internal memory.
- "None (swallowed)" means the command is silently discarded because the AX20 does not support the feature.
- AX-series machines do not support: VMI, reverse LF, half LF (sub/superscript), bold/shadow/double-strike, VT stops, top/bottom margins, or auto backward print.
- \* ESC+US+n only produces bus output when the carriage is at column 1. At any other position the command is silently swallowed. See §7.

**ESC+S repositioning detail:** ESC+S resets internal settings (pitch, margins, tabs, auto-LF) to DIP switch defaults, then repositions the carriage back to its pre-reset column. The bus output is a CR to column 1 (required for pitch change), followed by a standard horizontal repositioning to the saved column position. The underline bracket (`0x8B` before, `0x8A` after) is conditional — `0x8A` only appears if underline was active before the reset. Underline state is preserved across ESC+S; it is not reset to a DIP default.

```
ESC+S from col 7, underline off:
  [0x9E, <P>, 0x8B, 0x00 × 6]

ESC+S from col 7, underline on:
  [0x9E, <P>, 0x8B, 0x00 × 6, 0x8A]
```

The `0x00` count equals (column - 1), confirmed by testing at different carriage positions. The pitch byte `<P>` is the new (reset) default pitch.


---

## 13. Typewriter Model Differences: AX20 vs CE650

The IF60 generates different bus command sequences depending on which typewriter model is connected. The CE650 is a CX-series machine with more capabilities; the AX20 is an AX-series basic model. Commands the typewriter does not support are **swallowed by the interface** — they never reach the bus.

### Model Detection

The interface detects the typewriter model via the power-on handshake response byte: `0x30` = AX20 (AX series), `0x6A` = CE650 (CX series). See §10.1.

### Control Code Translation Differences

| Function | Input | AX20 Bus Output | CE650 Bus Output |
|----------|-------|-----------------|------------------|
| Space | 0x20 | `[0x00]` | `[0xB1]` |
| Backspace | 0x08 | `[0x03]` | `[0x03]` |
| Carriage Return | 0x0D | `[0x9E, <P>]` | `[0x9E, 0x8D, <P>]` |
| CR + LF | 0x0D, 0x0A | `[0x02, <P>]` | `[0x02, 0x8D, <P>]` |
| Line Feed | 0x0A | `[0x9F]` | `[0xA0, 0x9F, 0xA0]` |
| Form Feed | 0x0C | `[0x9F × 33]` | `[0xA3, 0x9F×21, 0xA2, 0x9F, 0xA0]` |
| Bell | 0x07 | `[0xF5]` | `[0xF5]` |
| DC1 SELECT | 0x11 | `[0xF9,0xFD,0x7F,0xF4,0xB1,<P>]` | `[0xF9,0xFD,0x7F,0xA0,0xF4,0xB1,0xF2,0xA0,<P>]` |
| DC3 DESELECT | 0x13 | `[0xF8]` | `[0xF2, 0xF8]` |

### Key Differences Explained

**Space:** The AX20 uses `0x00` as a one-step advance; the CE650 uses `0xB1` (the HMI mode byte). Both are pitch-dependent — the physical width of a 0x00 step depends on the current pitch mode set by the most recent HMI byte, just as 0xB1 explicitly sets 10cpi. The CE650 sends the explicit mode byte with each space rather than relying on the previously set pitch state.

**Carriage Return:** The CE650 inserts `0x8D` (clear enhanced print modes) into every CR sequence. The AX20 doesn't have enhanced print modes, so the interface doesn't send this byte.

**Line Feed:** The CE650 wraps every LF in VMI context bytes: `[0xA0, 0x9F, 0xA0]`. The AX20 has no variable line spacing, so it receives only `[0x9F]`.

**Form Feed:** The AX20 receives a simple stream of 0x9F line feeds. The CE650 receives a coarse-to-fine VMI stepping sequence for efficient page advance.

**Bold/Shadow/Double-Strike:** Swallowed for AX20 — the CE650 receives `[0x8C]` to enable them. The manual confirms ESC+O (bold), ESC+W (shadow), and ESC+F (double-strike) are CX-only. All three produce the same bus byte `0x8C`.

**VMI:** All four VMI settings are swallowed for the AX20. The manual confirms ESC+RS+n is CX-only. VMI = (n-1) × 1/48 inch; valid n values are 9, 13, 17, 25.

**Reverse Paper Feed / Sub-Superscript:** Swallowed for AX20. The manual confirms ESC+LF (reverse LF), ESC+U (half LF = 1/12"), and ESC+D (reverse half LF = 1/12") are CX-only.


---

## 14. Internal Bus Command Reference

### Byte Range Summary

| Range | Function |
|-------|----------|
| 0x00–0x07 | Low-level movement and micro-steps |
| 0x20–0x7E | Daisy wheel petal positions (print characters) |
| 0x7F | Unmapped character / identification byte |
| 0x88–0x89 | Dead-key print bracket |
| 0x8A–0x8D | Print formatting commands |
| 0x8E–0x8F | Shift lock toggle (typewriter → bus only) |
| 0x9E–0x9F | Carriage return / line feed |
| 0xA0–0xA3 | VMI (line spacing) modes [CE650/CX only] |
| 0xB1–0xB3 | HMI (character pitch) modes |
| 0xF2–0xF9 | System/protocol commands |
| 0xFD–0xFE | System identification/initialization |

### Complete Byte Definitions

#### System / Initialization

| Byte | Function | Models |
|------|----------|--------|
| **0xFE** | Power-on initialization command. Sent by the interface immediately after power-up, paired with `0x7F`. The typewriter responds with a 2-byte identification code (`0x30` = AX20, `0x6A` = CE650). Only appears during the power-on handshake. | Both |

#### Print Character Commands

| Byte | Function | Models |
|------|----------|--------|
| **0x20–0x7E** | Strike daisy wheel petal at this position. The byte value corresponds to a physical petal on the wheel, not to ASCII. The IF60 translates ASCII input to the correct wheel position. | Both |
| **0x88** | Code key down. In the bus→typewriter direction, engages the Code modifier to access additional wheel characters (green-printed keycap legends). In the typewriter→bus direction, indicates the physical Code key has been pressed. | Both |
| **0x89** | Code key up. Releases the Code modifier. Together: `[0x88, <pos>, 0x89]` simulates pressing Code + the key at wheel position `<pos>`. Multiple key position bytes can appear between a single 0x88/0x89 pair if Code is held while pressing several keys. When the accessed character is also a dead key, a trailing `0x00` is appended by the IF60 to force carriage advance: `[0x88, <pos>, 0x89, 0x00]`. | Both |
| **0x7F** | Unmapped character / identification byte. Used inside `[0x88, 0x7F, 0x89]` on CE650. Also appears in DC1/RESET/power-on. | Both |
| **0x8E** | Repeat last command. When sent to the typewriter, causes the previously received command to be repeated. Used in combination with other commands, e.g., sending `[0x09, 0x8E, 0x8F]` clears all tab stops by repeating the "remove tab" command. Note: the physical Shift Lock key does not produce any bus activity — the Shift and Shift Lock keys are handled entirely within the typewriter's keyboard mechanism and do not appear on the bus. | Both |
| **0x8F** | End repeat. Terminates the repeat mode initiated by 0x8E. | Both |

#### Horizontal Movement

| Byte | Function | Models |
|------|----------|--------|
| **0x00** | Advance carriage one position at the current pitch. Physical distance depends on the active pitch mode set by the most recent HMI byte (12/120" at 10cpi, 10/120" at 12cpi, 8/120" at 15cpi). Used as space on AX20, as dead-key advance on both models, and for absolute HT and margin repositioning steps. | Both |
| **0xB1** | Set pitch / advance one HMI unit at 10cpi (HMI=13, 1/10"). Used as space on CE650. Appears in CR, CR+LF, SELECT, and margin repositioning. | Both |
| **0xB2** | Set pitch / advance one HMI unit at 12cpi (HMI=11, 1/12"). | Both |
| **0xB3** | Set pitch / advance one HMI unit at 15cpi (HMI=9, 1/15"). | Both |
| **0x03** | Backspace one position. | Both |
| **0x04** | Destructive backspace (correction). Moves the carriage back one position and activates the typewriter's correction mechanism (e.g., lift-off tape) to erase the previous character. Compare with 0x03 which moves back without erasing. | Both |
| **0x9E** | Carriage return — move print head to left margin (or column 0). | Both |
| **0x14** | Carriage return variant. Functionally equivalent to 0x9E (returns carriage to column 1 / left margin). Exact distinction from 0x9E is unclear; may relate to margin handling or internal typewriter state. Not generated by the IF60 but recognised by the typewriter. | Both |
| **0x8B** | Disable underline mode. Also appears before sequences of 0x00 advance bytes used for absolute HT movement and margin repositioning — in these cases, it defensively disables underline to prevent drawing an underline across the page during carriage movement. The movement itself is performed by the subsequent 0x00 bytes, not by 0x8B. | Both |

#### Vertical Movement

| Byte | Function | Models |
|------|----------|--------|
| **0x9F** | Line feed — advance paper one line. Repeated for multiple lines in FF and VT. | Both |
| **0x02** | Combined line feed + carriage return. Used for CR+LF newline, auto-LF, and automatic line wrapping at right margin. | Both |
| **0x92** | Newline (CR+LF variant). Returns carriage to left margin and feeds one line. Functionally similar to 0x02 but may differ in margin or formatting behaviour. Not generated by the IF60 but recognised by the typewriter. | Both |
| **0x06** | Reverse paper feed (up) micro-step. Used for reverse LF (ESC+LF), superscript (ESC+D = reverse 1/12"). | CX only |
| **0x07** | Forward paper feed (down) micro-step. Used for subscript (ESC+U = 1/12" down). | CX only |
| **0xA0** | Set VMI to 9 (tightest: 8/48 = 1/6"). Also wraps LF operations on CE650. | CX only |
| **0xA1** | Set VMI to 13 (default: 12/48 = 1/4"). | CX only |
| **0xA2** | Set VMI to 17 (wider: 16/48 = 1/3"). Used mid-FF for step-down. | CX only |
| **0xA3** | Set VMI to 25 (widest: 24/48 = 1/2"). Used at start of FF for fast bulk feed. | CX only |

#### Print Formatting

| Byte | Function | Models |
|------|----------|--------|
| **0x8A** | Enable underline mode. | Both |
| **0x8B** | Disable underline mode. See horizontal movement entry above for use in repositioning sequences. | Both |
| **0x8C** | Enable enhanced print mode: bold (ESC+O), shadow (ESC+W), or double-strike (ESC+F). Same byte for all three. | CX only |
| **0x8D** | Clear all enhanced print modes. Sent as part of every CR on CE650, resetting formatting per line. Also cleared by ESC+&. | CX only |

#### System / Protocol

| Byte | Function | Models |
|------|----------|--------|
| **0xF4** | Initialise / power on typewriter print mechanism. Appears in DC1 (SELECT) and ESC+CR+P (RESET). | Both |
| **0xF5** | Ring bell / beeper (~2 seconds per manual). | Both |
| **0xF8** | Deselect / power down typewriter mechanism. Used in DC3 (DESELECT). | Both |
| **0xF9** | Begin select sequence. Always the first byte of DC1. | Both |
| **0xFD** | Unknown system command. Appears in SELECT, RESET, and power-on sequences, always near 0x7F. Possibly triggers a status/identification query. | Both |
| **0xFE** | Power-on initialization. Sent once at startup, paired with 0x7F. | Both |
| **0xF2** | Motor/mechanism control. Appears in DC1 (after init) and DC3 (before deselect) on CE650. | CX only |

### Composite Sequences

```
Power-On Handshake:
  AX20:  [0xFE, 0x7F] → response (0x00, 0x30)
  CE650: [0xFE, 0x7F] → response (0x00, 0x6A)

DC1 SELECT (power on, <P> = pitch byte):
  AX20:  [0xF9, 0xFD, 0x7F, 0xF4, 0xB1, <P>]
  CE650: [0xF9, 0xFD, 0x7F, 0xA0, 0xF4, 0xB1, 0xF2, 0xA0, <P>]

DC3 DESELECT (power off):
  AX20:  [0xF8]
  CE650: [0xF2, 0xF8]

Carriage Return (<P> = pitch byte):
  AX20:  [0x9E, <P>]
  CE650: [0x9E, 0x8D, <P>]

CR + LF (newline, <P> = pitch byte):
  AX20:  [0x02, <P>]
  CE650: [0x02, 0x8D, <P>]

Line Feed (standalone):
  AX20:  [0x9F]
  CE650: [0xA0, 0x9F, 0xA0]

Form Feed:
  AX20:  [0x9F × N]
  CE650: [0xA3, 0x9F × N, 0xA2, 0x9F, 0xA0]

HT forward (pitch-independent):
  AX20:  [0x8B, 0x00 × N]
  CE650: [0x9E, 0x8B, 0xB1, 0x00 × N, 0xB1]

HT backward:
  AX20:  [0x8B, 0x03 × N]
  CE650: [0x9E, 0x8B, 0xB1, 0x00 × N, 0xB1]  (CR + forward reposition)

Auto line wrap at right margin:
  AX20:  [0x02, <P>, 0x00 × remaining_chars]

Reset printer (ESC+CR+P):
  AX20:  [0xF4, 0xB1, 0x8B, 0xFD, 0x7F]
  CE650: [0xA0, 0xF4, 0xB1, 0x8D, 0x8B, 0xFD]

Reset to DIP defaults (ESC+S):
  AX20:  [0x9E, 0xB1, 0x8B, 0x00 × 6, 0x8A]
  CE650: [0xA0, 0xB1]
```


---

## 15. ESC+Y / ESC+Z: Wheel Position Probes

Per the manual, ESC+Y prints the character at the 0x20 (space) position of the daisy wheel, and ESC+Z prints the character at the 0x7F (DEL) position. These vary by keyboard and typewriter, confirming they reflect the physical wheel installed:

| Config | ESC+Y Output | ESC+Z Output |
|--------|-------------|-------------|
| AX20 KB1 | `[0x88, 0x55, 0x89]` | `[0x3D, 0x00]` |
| AX20 KB2 | `[0x7C, 0x00]` | `[0x88, 0x55, 0x89]` |
| AX20 KB3 | `[0x20]` | `[0x88, 0x57, 0x89]` |
| CE650 KB1 | `[0x4E]` | `[0x88, 0x7F, 0x89]` |

These outputs are **not affected** by DIP 1-3 (confirmed identical UP vs DOWN).


---

## 16. Reverse Mapping: Typewriter Keypress → Serial Output

### 16.1 Overview

The previous sections document the **forward** direction: ASCII bytes sent from the PC via serial, translated by the IF60 into bus commands for the typewriter. This section documents the **reverse** direction: physical keystrokes on the typewriter keyboard, captured as bus bytes, and translated by the IF60 into serial bytes sent to the PC.

Testing was performed on 2026-02-13 with the AX20 typewriter, DIP switches at 0x3FF8 (Terminal mode, Non-ASCII wheel), IF PITCH=10. Every physical key was pressed in three modes: normal, Shift held, and Code held. The keyboard switch was cycled through all three positions (KB1, KB2, KB3) with a full DC3 DESELECT → DC1 SELECT cycle between each switch change to ensure the interface latched the new keyboard setting.

### 16.2 Architecture

```
  Physical Key → Typewriter sends fixed bus byte(s) → IF60 translates → Serial byte(s) to PC
                 (independent of KB switch)            (KB-dependent, latched at SELECT)
```

The typewriter hardware always sends the **same bus bytes** for a given key regardless of the keyboard switch position. The keyboard switch only affects the IF60's reverse translation table, which is latched at SELECT time.

### 16.3 Bus Byte Patterns for Keystrokes

Three distinct bus patterns were observed for keystrokes:

| Pattern | Meaning | Example |
|---------|---------|---------|
| `( 0xNN )` | Normal keystroke — single wheel position byte | `( 0x61 )` = lowercase 'a' position |
| `( 0x88, 0xNN, 0x89 )` | Code+key — Code modifier key held (0x88 = down, 0x89 = up) around one or more key position bytes | `( 0x88, 0x61, 0x89 )` = Code+'a' |

**Note:** The physical Shift and Shift Lock keys do not produce any bus activity. Shift state changes are handled entirely within the typewriter's keyboard mechanism. The bytes 0x8E and 0x8F observed in some contexts are repeat/end-repeat commands (see §14), not Shift Lock events.

Shift does not produce its own bus framing — it changes which wheel position byte the typewriter sends (e.g., `0x61` for 'a' becomes `0x41` for Shift+'a').

### 16.4 Keys That Produce No Serial Output

Several keys generate bus activity but the IF60 does not translate them to serial. These are typewriter-internal function keys:

| Bus Byte | Likely Function |
|----------|----------------|
| 0x0B | Unknown function key (top-left corner) |
| 0x1D | Relocate |
| 0x0C | Correction / erase |
| 0x08 | Half-space or micro-step |
| 0x09 | Index / express key |

**Note:** The Shift Lock key does not produce any bus activity and therefore does not appear in this table. Shift state is handled internally by the typewriter's keyboard mechanism.

Code+key versions of non-translating keys also produce no serial output.

### 16.5 Special Key Mappings

| Physical Key | Bus (normal) | Serial | Bus (Shift) | Serial | Bus (Code) | Serial |
|-------------|-------------|--------|-------------|--------|------------|--------|
| ESC/Cancel | `0x05` | `0x1B` (ESC) | `0x05` | `0x1B` | `0x88,0x05,0x89` | `0x1B` |
| TAB | `0x01` | `0x09` (HT) | `0x01` | `0x09` | `0x88,0x01,0x89` | `0x09` |
| Return | `0x02` | `0x0D` (CR) | `0x02` | `0x0D` | `0x88,0x02,0x89` | `0x0D` |
| Backspace | `0x03` | `0x08` (BS) | `0x03` | `0x08` | `0x88,0x14,0x89` | None |
| Space | * | `0x20` | * | `0x20` | `0x88,0x00,0x89` | `0x20` |
| DEL/Correct | `0x04` | `0x08` (BS) | `0x04` | `0x08` | `0x88,0x04,0x89` | `0x7F` (DEL) |

\* Space bar bus capture was inconsistent due to timing; serial output was reliably `0x20` in all modes.

Note: Both Backspace and DEL/Correct send `0x08` (BS) in normal and Shift modes. Only Code+DEL sends true `0x7F` (DEL).

### 16.6 Code+Key = Control Codes

The Code key modifier produces ASCII control codes using the standard Ctrl convention: the serial output equals the key's lowercase letter value masked to 5 bits (`letter & 0x1F`). This mapping is consistent across all three keyboard positions — it uses a fixed algorithm rather than the keyboard-dependent translation table.

| Code+Key | Bus | Serial | ASCII Control |
|----------|-----|--------|---------------|
| Code+A | `0x88,0x61,0x89` | `0x01` | SOH |
| Code+B | `0x88,0x62,0x89` | `0x02` | STX |
| Code+C | `0x88,0x63,0x89` | `0x03` | ETX |
| Code+D | `0x88,0x64,0x89` | `0x04` | EOT |
| Code+E | `0x88,0x65,0x89` | `0x05` | ENQ |
| Code+F | `0x88,0x66,0x89` | `0x06` | ACK |
| Code+G | `0x88,0x67,0x89` | `0x07` | BEL |
| Code+H | `0x88,0x68,0x89` | `0x08` | BS |
| Code+I | `0x88,0x69,0x89` | `0x09` | HT |
| Code+J | `0x88,0x6A,0x89` | `0x0A` | LF |
| Code+K | `0x88,0x6B,0x89` | `0x0B` | VT |
| Code+L | `0x88,0x6C,0x89` | `0x0C` | FF |
| Code+M | `0x88,0x6D,0x89` | `0x0D, 0x0A` | CR+LF |
| Code+N | `0x88,0x6E,0x89` | `0x0E` | SO |
| Code+O | `0x88,0x6F,0x89` | `0x0F` | SI |
| Code+P | `0x88,0x70,0x89` | `0x10` | DLE |
| Code+Q | `0x88,0x71,0x89` | `0x11` | DC1 |
| Code+R | `0x88,0x72,0x89` | `0x12` | DC2 |
| Code+S | `0x88,0x73,0x89` | `0x13` | DC3 |
| Code+T | `0x88,0x74,0x89` | `0x14` | DC4 |
| Code+U | `0x88,0x75,0x89` | `0x15` | NAK |
| Code+V | `0x88,0x76,0x89` | `0x16` | SYN |
| Code+W | `0x88,0x77,0x89` | `0x17` | ETB |
| Code+X | `0x88,0x78,0x89` | `0x18` | CAN |
| Code+Y | `0x88,0x7A,0x89` | `0x19` | EM |
| Code+Z | `0x88,0x79,0x89` | `0x1A` | SUB |

Code+M is the only letter that produces two serial bytes (CR+LF) rather than a single control code. Code+Y and Code+Z follow the physical QWERTZ key positions, so Code+Y sends bus byte `0x7A` (Z petal) → serial `0x19`, and Code+Z sends bus byte `0x79` (Y petal) → serial `0x1A`.

### 16.7 Code+Number Row

| Code+Key | Bus | Serial | Notes |
|----------|-----|--------|-------|
| Code+1 | `0x88,0x31,0x89` | `0x1B` | ESC (same as ESC key) |
| Code+2 | `0x88,0x8A,0x89` | `0x1C` | FS |
| Code+3 | `0x88,0x98,0x89` | `0x1D` | GS |
| Code+4 | `0x88,0x0D,0x89` | `0x1E` | RS |
| Code+5 | `0x88,0x35,0x89` | `0x1F` | US |
| Code+6 | `0x88,0x36,0x89` | None | Gap — no serial output |

| Code+Key | Bus | KB1 Serial | KB2 Serial | KB3 Serial | Notes |
|----------|-----|-----------|-----------|-----------|-------|
| Code+7 | `0x88,0x37,0x89` | `0x5E` `^` | `0x7D` `}` | — | Extra character |
| Code+8 | `0x88,0x38,0x89` | `0x1B,0x59` ESC+Y | `0x1B,0x5A` ESC+Z | — | Wheel probe |
| Code+9 | `0x88,0x39,0x89` | `0x3C` `<` | `0x5C` `\` | — | Extra character |
| Code+0 | `0x88,0x30,0x89` | `0x3E` `>` | `0x7C` `\|` | — | Extra character |

Code+1 through Code+5 produce the remaining ASCII control codes (0x1B–0x1F) and are keyboard-independent. Code+6 is a gap (no output). Code+7 through Code+0 are **keyboard-dependent** — they produce the "extra characters" shown in green on the keycaps, which vary by country/keyboard layout. On KB1, Code+8 produces ESC+Y (wheel probe: print glyph at 0x20 position); on KB2, Code+8 produces ESC+Z (wheel probe: print glyph at 0x7F position). KB3 Code+7/8/9/0 have not yet been captured.

### 16.8 Complete Reverse Translation Tables

The following tables show the IF60's reverse translation for every bus byte that produces serial output. Bus bytes are the wheel petal positions sent by the typewriter; serial bytes are what the IF60 sends to the PC.

#### 16.8.1 Normal (Unshifted) Keys

| Bus | KB1 Serial | KB2 Serial | KB3 Serial | Physical Key |
|-----|-----------|-----------|-----------|-------------|
| 0x01 | 0x09 (HT) | 0x09 | 0x09 | TAB |
| 0x02 | 0x0D (CR) | 0x0D | 0x0D | Return |
| 0x03 | 0x08 (BS) | 0x08 | 0x08 | Backspace |
| 0x04 | 0x08 (BS) | 0x08 | 0x08 | DEL/Correct |
| 0x05 | 0x1B (ESC) | 0x1B | 0x1B | ESC |
| 0x2C | 0x2C `,` | 0x2C `,` | 0x29 `)` | Comma key |
| 0x2D | 0x7E `~` | 0x40 `@` | 0x2F `/` | ß key |
| 0x2E | 0x2E `.` | 0x2E `.` | 0x5F `_` | Period key |
| 0x2F | 0x2D `-` | 0x2D `-` | 0x5D `]` | Minus key |
| 0x30 | 0x30 `0` | 0x30 | 0x30 | 0 |
| 0x31 | 0x31 `1` | 0x31 | 0x31 | 1 |
| 0x32 | 0x32 `2` | 0x32 | 0x32 | 2 |
| 0x33 | 0x33 `3` | 0x33 | 0x33 | 3 |
| 0x34 | 0x34 `4` | 0x34 | 0x34 | 4 |
| 0x35 | 0x35 `5` | 0x35 | 0x35 | 5 |
| 0x36 | 0x36 `6` | 0x36 | 0x36 | 6 |
| 0x37 | 0x37 `7` | 0x37 | 0x37 | 7 |
| 0x38 | 0x38 `8` | 0x38 | 0x38 | 8 |
| 0x39 | 0x39 `9` | 0x39 | 0x39 | 9 |
| 0x27 | 0x7B `{` | 0x23 `#` | 0x27 `'` | Ä key |
| 0x3B | 0x7C `\|` | 0x3E `>` | 0x3B `;` | Ö key |
| 0x3D | 0x1B,0x5A | 0x7E `~` | 0x3D `=` | ´ key |
| 0x5C | 0x23 `#` | 0x7B `{` | 0x5C `\` | # key |
| 0x5D | 0x2B `+` | 0x2B `+` | 0x7C `\|` | + key |
| 0x7C | 0x7D `}` | 0x1B,0x59 | 0x6A `j` | Ü key |
| 0x61 | 0x61 `a` | 0x61 | 0x61 | A |
| 0x62 | 0x62 `b` | 0x62 | 0x62 | B |
| 0x63 | 0x63 `c` | 0x63 | 0x63 | C |
| 0x64 | 0x64 `d` | 0x64 | 0x64 | D |
| 0x65 | 0x65 `e` | 0x65 | 0x65 | E |
| 0x66 | 0x66 `f` | 0x66 | 0x40 `@` | F |
| 0x67 | 0x67 `g` | 0x67 | 0x67 | G |
| 0x68 | 0x68 `h` | 0x68 | 0x68 | H |
| 0x69 | 0x69 `i` | 0x69 | 0x69 | I |
| 0x6A | 0x6A `j` | 0x6A | 0x5B `[` | J |
| 0x6B | 0x6B `k` | 0x6B | 0x6B | K |
| 0x6C | 0x6C `l` | 0x6C | 0x6C | L |
| 0x6D | 0x6D `m` | 0x6D | 0x6D | M |
| 0x6E | 0x6E `n` | 0x6E | 0x6E | N |
| 0x6F | 0x6F `o` | 0x6F | 0x6F | O |
| 0x70 | 0x70 `p` | 0x70 | 0x70 | P |
| 0x71 | 0x71 `q` | 0x71 | 0x71 | Q |
| 0x72 | 0x72 `r` | 0x72 | 0x72 | R |
| 0x73 | 0x73 `s` | 0x73 | 0x73 | S |
| 0x74 | 0x74 `t` | 0x74 | 0x74 | T |
| 0x75 | 0x75 `u` | 0x75 | 0x75 | U |
| 0x76 | 0x76 `v` | 0x76 | 0x23 `#` | V |
| 0x77 | 0x77 `w` | 0x77 | 0x77 | W |
| 0x78 | 0x78 `x` | 0x78 | 0x78 | X |
| 0x79 | 0x7A `z` | 0x7A `z` | 0x79 `y` | Y key (QWERTZ) |
| 0x7A | 0x79 `y` | 0x79 `y` | 0x7A `z` | Z key (QWERTZ) |

#### 16.8.2 Shifted Keys

| Bus | KB1 Serial | KB2 Serial | KB3 Serial | Physical Key |
|-----|-----------|-----------|-----------|-------------|
| 0x20 | 0x26 `&` | 0x26 `&` | 0x1B,0x59 | Shift+6 |
| 0x21 | 0x21 `!` | 0x21 `!` | 0x24 `$` | Shift+1 |
| 0x22 | 0x5B `[` | 0x3C `<` | 0x22 `"` | Shift+Ä |
| 0x23 | 0x40 `@` | 0x5E `^` | 0x26 `&` | Shift+3 |
| 0x24 | 0x24 `$` | 0x24 `$` | 0x2A `*` | Shift+4 |
| 0x25 | 0x25 `%` | 0x25 `%` | 0x5E `^` | Shift+5 |
| 0x26 | 0x2F `/` | 0x2F `/` | 0x7D `}` | Shift+7 |
| 0x28 | 0x29 `)` | 0x29 `)` | 0x25 `%` | Shift+9 |
| 0x29 | 0x3D `=` | 0x3D `=` | 0x28 `(` | Shift+0 |
| 0x2A | 0x28 `(` | 0x28 `(` | 0x7E `~` | Shift+8 |
| 0x2B | 0x60 `` ` `` | 0x60 `` ` `` | 0x2B `+` | Shift+´ |
| 0x3A | 0x5C `\` | 0x5D `]` | 0x3A `:` | Shift+Ö |
| 0x3C | 0x3B `;` | 0x3B `;` | 0x29 `)` | Shift+, (KB3) |
| 0x3E | 0x3A `:` | 0x3A `:` | 0x5F `_` | Shift+. (KB3) |
| 0x3F | 0x5F `_` | 0x5F `_` | 0x2D `-` | Shift+- (KB3) |
| 0x40 | 0x22 `"` | 0x22 `"` | 0x76 `v` | Shift+2 |
| 0x41–0x58 | A–X | A–X | A–X | Shift+letters |
| 0x59 | 0x5A `Z` | 0x5A `Z` | 0x59 `Y` | Shift+Y |
| 0x5A | 0x59 `Y` | 0x59 `Y` | 0x5A `Z` | Shift+Z |
| 0x5B | 0x2A `*` | 0x2A `*` | 0x7B `{` | Shift++ |
| 0x5F | 0x3F `?` | 0x3F `?` | 0x3F `?` | Shift+ß |
| 0x60 | 0x27 `'` | 0x27 `'` | 0x60 `` ` `` | Shift+# |
| 0x7B | 0x5D `]` | 0x5B `[` | 0x4A `J` | Shift+Ü |
| 0x4A | 0x4A `J` | 0x4A `J` | 0x66 `f` | Shift+J (KB3 remaps) |

### 16.9 Analysis: How the Reverse Mappings Generalise

#### 16.9.1 Letters and Digits Are Identity-Mapped (with Y/Z Exception)

For all three keyboards, the 26 lowercase letters (bus 0x61–0x7A) and 26 uppercase letters (bus 0x41–0x5A) map to their identical ASCII values, with two exceptions:

- **KB1 and KB2** swap Y/Z: bus `0x79` ↔ serial `0x7A` and `0x7A` ↔ `0x79`. This implements the QWERTZ→ASCII conversion.
- **KB3** does **not** swap Y/Z. KB3 also remaps several letter positions (f↔@, j↔[, v↔#, J↔f) to symbols, consistent with the symbol wheel having non-letter glyphs at those petal positions.

Digits 0–9 (bus 0x30–0x39) are identity-mapped across all keyboards.

#### 16.9.2 KB1 and KB2 Differ Only at National Variant Positions

KB1 (Local) and KB2 (International) share identical mappings for all letters, digits, and common punctuation. They differ at exactly the ISO 646 national replacement character positions: the slots traditionally used for Ä, Ö, Ü, #, and a handful of bracket/symbol characters. This mirrors the forward direction behaviour.

#### 16.9.3 KB3 Is a Complete Symbol Remapping

KB3 (Symbol) uses a fundamentally different translation table. In addition to the letter→symbol remaps, the entire punctuation set is reorganised. Many bus positions that map to German special characters on KB1/KB2 map to their literal ASCII symbols on KB3 (e.g., bus `0x3B` → `;`, bus `0x27` → `'`, bus `0x5C` → `\`).

#### 16.9.4 The Reverse Mapping Is the Exact Inverse of the Forward Mapping

Cross-referencing the forward tables (§4) with the reverse tables confirms that the IF60 maintains a true bijective mapping. For every entry where the forward mapping says "serial byte X → bus byte Y", the reverse mapping says "bus byte Y → serial byte X". The same translation table is used in both directions, just read in reverse.

#### 16.9.5 Code Key Is Mostly KB-Independent

The Code+key → control code mapping is identical across all three keyboard positions for letters (`letter & 0x1F`) and for Code+1 through Code+6 (fixed control code assignments). However, **Code+7/8/9/0 are keyboard-dependent** — these four keys access the "extra characters" printed in green on the keycaps, which vary by country/keyboard layout (see §16.7). This was confirmed by the 2026-02-17 captures comparing KB1 and KB2 Code+key output.

#### 16.9.6 Multi-Byte Serial Responses

Most keys produce exactly one serial byte. The exceptions are:

| Key | Serial Output | Notes |
|-----|--------------|-------|
| Code+M | `0x0D, 0x0A` (CR+LF) | Newline entry |
| Code+8 (KB1) | `0x1B, 0x59` (ESC+Y) | Wheel probe — print glyph at 0x20 position |
| Code+8 (KB2) | `0x1B, 0x5A` (ESC+Z) | Wheel probe — print glyph at 0x7F position |
| KB1 ´ key | `0x1B, 0x5A` (ESC+Z) | Wheel probe — print glyph at DEL position |
| KB2 Ü key | `0x1B, 0x59` (ESC+Y) | Same wheel probe, different physical key on KB2 |
| KB3 Shift+6 | `0x1B, 0x59` (ESC+Y) | Same wheel probe, different physical key on KB3 |

The ESC+Y and ESC+Z sequences are the same escape commands documented in §15 for the forward direction. The interface makes these available from the keyboard so the user can probe the daisy wheel directly.

#### 16.9.7 Bus Bytes Are Hardware-Fixed

The typewriter always sends the same bus bytes for a given physical key regardless of the keyboard switch, DIP switches, or any other setting. The bus byte is determined by the physical key mechanism, not by any software configuration. Only the IF60's translation of those bytes into serial output changes based on the latched keyboard setting.


---

## 17. Typewriter Responses

The values in parentheses `()` in the log files represent actual responses from the typewriter back through the bus.

### Power-On Response

| Typewriter | Response |
|-----------|----------|
| AX20 | `(0x00, 0x30)` |
| CE650 | `(0x00, 0x6A)` |

### DC1 SELECT Typewriter Response

| Config | Response |
|--------|----------|
| AX20 KB1 | `(0x00, 0x00, 0x04, 0x00, 0x00, 0x00)` |
| AX20 KB2 | `(0x00, 0x00, 0x24, 0x00, 0x00, 0x00)` |
| AX20 KB3 | `(0x00, 0x00, 0x44, 0x00, 0x00, 0x00)` |
| CE650 KB1 | `(0xFF, 0xFF, 0x04, 0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)` |

The AX20 returns a 6-byte status with the keyboard ID in byte 3. The CE650 returns a 9-byte status with the keyboard ID in byte 3 and `0x7F` in byte 4.


---

## 18. Open Questions

- **0xFD**: Exact function unknown. Always appears near 0x7F in SELECT, RESET, and power-on. Likely a status query or handshake.
- **0xFE**: Confirmed as a power-on initialization command. Can it be sent at any time to re-identify the typewriter, or is it strictly a boot-time operation?
- **DIP 1-4 + KB3**: No test with KB3 and DIP 1-4=UP. Does the symbol layout also collapse to KB1 when the ASCII wheel is set, or does it remain independent?
- **Pitch and SELECT (CE650)**: Does the CE650 SELECT sequence also end with a pitch-dependent byte? Need a CE650 capture at non-default pitch.
- **CE650 printable characters**: The CE650 capture had noise issues. A clean recapture would confirm whether the CE650's wheel position mapping differs from the AX20's.
- **0xF2 (CE650 control)**: Is this specifically the paper feed motor? Does it have parameters, or is it purely on/off?
- **0x14 and 0x92**: These additional CR and newline bus bytes are recognized by the typewriter but not generated by the IF60. What are the exact behavioral differences from 0x9E and 0x02? Are they related to margin handling or formatting state?
- **0x8E/0x8F (repeat/end-repeat)**: What commands can be repeated? Is only the immediately preceding command eligible, or can 0x8E repeat a longer sequence? Are there limits to the repeat count?
- **Code+7/8/9/0 on KB3**: Not yet captured. Based on the pattern (KB1 and KB2 produce different ISO 646 variant characters), KB3 likely produces yet another set. Capture needed.