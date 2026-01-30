# Protocol Specification — Brother AX-Series Serial Interface

## Signal Overview

| Signal | Direction | Idle State | Description |
|--------|-----------|------------|-------------|
| SCK | Interface → Typewriter | HIGH | Serial clock, always controlled by interface |
| SI | Interface → Typewriter | Last bit sent | Serial data to typewriter |
| SO | Typewriter → Interface | LOW | Serial data from typewriter |
| READY | Interface → Typewriter | HIGH | Interface controls this, forces KBRQ LOW when LOW |
| KBRQ | Typewriter → Interface | LOW | Keyboard Request — typewriter pulls HIGH to request send |
| KBACK | Typewriter → Interface | Depends on last transmission | Keyboard Acknowledge — flip-flop controlled by clock and typewriter |

---

## KBACK Flip-Flop Behavior

- **First falling edge of SCK** → KBACK goes LOW immediately
- **Typewriter reset** → KBACK goes HIGH (typewriter finished processing)
- Typewriter can only RESET (set HIGH), never SET (pull LOW)

**KBACK State Rule:**
- After interface transmission → KBACK is HIGH (typewriter resets it when done processing and ready for next transmission)
- After typewriter transmission → KBACK stays LOW (no need to signal processing complete)

---

## READY and KBRQ Relationship
- KBRQ is pulled up by a resitor internally from the typewriter
- In the typewriter READY is directly wired to KBRQ 
- KBRQ can only be HIGH when: `READY == HIGH` AND typewriter does not pull it LOW
- When Interface pulls READY LOW → KBRQ is forced LOW regardless of typewriter
- When Interface pulls READY HIGH → KBRQ briefly rises then typewriter pulls LOW

---

## Transmission Type 1: Interface → Typewriter (SI Path)

### Sequence of Operations

1. **Interface pulls READY LOW**
2. **Wait ~30µs**
3. **Interface starts clock, transmits 8 bits on SI**
   - First falling edge of SCK forces KBACK LOW
   - Data set on falling edge, read on rising edge
   - MSB first
4. **Clock stops, SCK stays HIGH, SI stays at last bit level**
5. **Typewriter processes data (100µs to 500ms)**
   - Fast: ~100-250µs
   - Buffer full: up to 500ms (Buffer emptys when bytes are printed)
6. **Typewriter resets KBACK to HIGH** (signals processing complete and ready for next transmission)
7. **Wait ~40µs**
8. **Interface pulls READY HIGH** (signaling end of transmission)

---

## Transmission Type 2: Typewriter → Interface (SO Path)

### Sequence of Operations

1. **Typewriter prepares:**
   - Pulls KBRQ HIGH
   - Resets KBACK to HIGH
   - Pulls SO HIGH
2. **Wait ~100µs**
3. **Interface pulls READY LOW**
   - This forces KBRQ LOW
4. **Wait ~200µs**
5. **Interface starts clock, receives 8 bits on SO**
   - First falling edge of SCK forces KBACK LOW
   - Interface sends DEL (0x7F) on SI if last transmission was SI path
   - SI stays HIGH after DEL (0x7F) if last transmission was SO path (ignored by typewriter)
   - MSB first
6. **Clock stops**
7. **Typewriter pulls SO LOW** (cleanup)
8. **Wait ~240µs**
9. **Interface pulls READY HIGH** (signals processing complete and ready for next transmission)
10. **KBRQ rises with READY briefly**
11. **Typewriter pulls KBRQ LOW** (back to idle)

---

## DEL Byte Behavior (Direction Change Acknowledgment)

When switching from SI path to SO path, interface sends DEL (0x7F) to acknowledge direction change and maybe signal disregarding of 0xFF Bytes until next SI-Path:

| Last Transmission | Current Transmission | SI Line During SO Transfer |
|-------------------|---------------------|---------------------------|
| SI (Interface sent) | SO (Typewriter sends) | DEL (0x7F) |
| SO (Typewriter sent) | SO (Typewriter sends) | 0xFF (HIGH) |
| SO (Typewriter sent) | SI (Interface sends) | Normal data |

---

## Idle State Summary

| Signal | Idle State | Notes |
|--------|------------|-------|
| SCK | HIGH | |
| SI | Last bit sent | No defined idle, stays where it was |
| SO | LOW | |
| READY | HIGH | |
| KBRQ | LOW | |
| KBACK | HIGH after SI path | Typewriter resets after processing |
| KBACK | LOW after SO path | No reset needed |

---

## Clock Specification

| Parameter | Value |
|-----------|-------|
| Clock frequency | ~78kHz |
| Clock idle state | HIGH |
| Bits per transfer | 8 |
| Bit order | MSB first |
| Data set on | Falling edge |
| Data read on | Rising edge |

---

# Startup and SELECT Sequences

## 1. Power-On / Connection Sequence

| Step | Direction | Byte | Description |
|------|-----------|------|-------------|
| 1 | — | — | Power settles, noise |
| 2 | Interface → Typewriter | 0xFE | Interface announces presence |
| 3 | Typewriter → Interface | 0x30 | Device type response |
| 3 | Interface → Typewriter | 0x7F (DEL) | Sent synchronously (direction change ack) |

---

## 2. SELECT Sequence (Mode Selection) AX20

| Step | Direction | Byte | Description                             |
|------|-----------|------|-----------------------------------------|
| 1 | Interface → Typewriter | 0xF9 or 0xF8 | Mode select (terminal/typewriter)       |
| 2 | Interface → Typewriter | 0xFD | ?                                       |
| 3 | Typewriter → Interface | 0x04 | EOT (End of Transmission)               |
| 4 | Interface → Typewriter | 0xF4 | ? (Reset margins and new line)          |
| 5 | Interface → Typewriter | 0xB1 | ?                                       |
| 6 | Interface → Typewriter | 0xB1 | ? Pitch setting (repeated)              |
| 7 | Interface → Typewriter | 0x8B | Underline off (only if not at column 1) |
| 8 | Interface → Typewriter | 0x00 × N | Space to restore column position        |

---

## Known Command Bytes

| Byte | Name | Description |
|------|------|-------------|
| 0xFE | INIT | Interface announces presence |
| 0xFD | SELECT | Select/configure command |
| 0xF9 | TERMINAL MODE | Interface can read keyboard |
| 0xF8 | TYPEWRITER MODE | Interface cannot read keyboard |
| 0xF4 | RESET MARGINS + NEWLINE | Reset margins and new line |
| 0xB1 | PITCH 10 | 10 characters per inch |
| 0xB2 | PITCH 12 | 12 characters per inch |
| 0xB3 | PITCH 15 | 15 characters per inch |
| 0x8B | UNDERLINE OFF | Disable underline |
| 0x00 | SPACE | Move one column right |
| 0x04 | EOT | End of Transmission |
| 0x30 | DEVICE TYPE | Typewriter identification |
| 0x7F | DEL | Direction change acknowledgment |