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
| 2 | Interface → Typewriter | 0xFE | Power-on initialization |
| 3 | Typewriter → Interface | 0x00, \<ID\> | Model identification: AX20 = `0x00, 0x30`; CE650 = `0x00, 0x6A` |
| 4 | Interface → Typewriter | 0x7F (DEL) | Direction change acknowledgment (sent on SI during SO transfer) |

---

## 2. SELECT Sequence (Mode Selection) AX20

| Step | Direction | Byte | Description |
|------|-----------|------|-------------|
| 1 | Interface → Typewriter | 0xF9 | Begin select sequence |
| 2 | Interface → Typewriter | 0xFD | System identification query |
| 3 | Typewriter → Interface | 0x00, 0x00, \<KB\>, 0x00, 0x00, 0x00 | Status response; KB = keyboard ID (KB1=0x04, KB2=0x24, KB3=0x44) |
| 4 | Interface → Typewriter | 0x7F (DEL) | Direction change acknowledgment |
| 5 | Interface → Typewriter | 0xF4 | Initialise print mechanism |
| 6 | Interface → Typewriter | 0xB1 | Reset pitch to 10cpi (always fixed) |
| 7 | Interface → Typewriter | \<P\> | Set active pitch (0xB1=10cpi, 0xB2=12cpi, 0xB3=15cpi) |

---

## Known Command Bytes

| Byte | Name | Description |
|------|------|-------------|
| 0xFE | INIT | Power-on initialization command sent by interface at startup |
| 0xFD | STATUS QUERY | System command appearing in SELECT, RESET, and power-on sequences, always near 0x7F. Possibly triggers a status/identification query |
| 0xF9 | SELECT | Begin select sequence; always the first byte of DC1 |
| 0xF8 | DESELECT | Deselect / power down typewriter mechanism; used in DC3 |
| 0xF4 | INIT MECHANISM | Initialise / power on typewriter print mechanism; appears in DC1 SELECT and ESC+CR+P RESET |
| 0xB1 | PITCH 10 | Set pitch to 10 cpi (HMI mode byte) |
| 0xB2 | PITCH 12 | Set pitch to 12 cpi (HMI mode byte) |
| 0xB3 | PITCH 15 | Set pitch to 15 cpi (HMI mode byte) |
| 0x8B | UNDERLINE OFF | Disable underline mode |
| 0x8A | UNDERLINE ON | Enable underline mode |
| 0x00 | ADVANCE | Advance carriage one position at the current pitch |
| 0x04 | DESTRUCTIVE BS | Destructive backspace: move carriage back one position and erase previous character |
| 0x30 | AX20 ID | AX20 model identification byte (typewriter → interface, power-on response only) |
| 0x6A | CE650 ID | CE650 model identification byte (typewriter → interface, power-on response only) |
| 0x7F | DEL | Direction change acknowledgment; sent on SI during SO transfers after SI→SO direction change |