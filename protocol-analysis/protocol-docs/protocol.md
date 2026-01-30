# Protocol Specification — Brother AX-Series Serial Interface

## Signal Overview

| Signal | Direction | Idle State | Description |
|--------|-----------|------------|-------------|
| SCK | Interface → Typewriter | HIGH | Serial clock, always controlled by interface |
| SI | Interface → Typewriter | Last bit sent | Serial data to typewriter |
| SO | Typewriter → Interface | LOW | Serial data from typewriter |
| READY | Interface → Typewriter | HIGH | Interface controls this, also forces KBRQ LOW when LOW |
| KBRQ | Typewriter → Interface | LOW | Keyboard Request — typewriter pulls HIGH to request send |
| KBACK | Typewriter → Interface | Depends on last transmission | Keyboard Acknowledge — flip-flop controlled by clock and typewriter |

---

## KBACK Flip-Flop Behavior

```
                    ┌─────────────┐
SCK (falling) ──────┤ SET       Q ├────── KBACK (active LOW logic)
                    │             │
Typewriter Reset ───┤ RESET       │
                    └─────────────┘
```

- **First falling edge of SCK** → KBACK goes LOW immediately
- **Typewriter reset** → KBACK goes HIGH (typewriter finished processing)
- Typewriter can only RESET (set HIGH), never SET (pull LOW)

**KBACK State Rule:**
- After interface transmission → KBACK is HIGH (typewriter resets it when done processing)
- After typewriter transmission → KBACK stays LOW (no need to signal processing complete)

---

## READY and KBRQ Relationship

```
READY (from Interface) ────┬────────────────→ To Typewriter
                           │
                           ↓ (when LOW)
                      Forces KBRQ LOW
                           │
KBRQ (from Typewriter) ────┴────────────────→ To Interface
```

- KBRQ can only be HIGH when: `READY == HIGH` AND typewriter pulls it HIGH
- When Interface pulls READY LOW → KBRQ is forced LOW regardless of typewriter

---

## Transmission Type 1: Interface → Typewriter

### Timing Diagram

```
READY:  ‾‾‾‾\_____________________________________________/‾‾‾‾
            │                                             ↑
            │                                             │ 200µs after KBACK rises
            │ 20-30µs                                     │
            ↓                                             │
SCK:    ‾‾‾‾‾‾‾‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾‾‾‾‾‾‾‾‾‾‾│‾‾‾‾
                ↑                           ↑             │
                │ 8 bits                    │             │
                │                           │             │
KBACK:  ????????\___________________________│_____________/‾‾‾‾
                ↑                           │             ↑
                │ First falling edge        │             │
                  forces LOW                │    100µs-500ms later
                                            │    Typewriter resets to HIGH
                                      Clock ends
                                            
SI:     ‾‾‾‾‾‾‾‾╱D7╲╱D6╲╱D5╲╱D4╲╱D3╲╱D2╲╱D1╲╱D0╲══════════════
                                               ↑
                                               │ Stays at last bit level
```

### Sequence of Operations

1. **Interface pulls READY LOW**
2. **Wait 20-30µs**
3. **Interface starts clock, transmits 8 bits on SI**
   - First falling edge of SCK forces KBACK LOW
   - Data set on falling edge, read on rising edge
4. **Clock stops, SCK stays HIGH**
5. **Typewriter processes data (100µs to 500ms)**
   - Fast: ~100-250µs
   - Buffer full: up to 500ms
6. **Typewriter resets KBACK to HIGH** (signals processing complete)
7. **Wait 200µs**
8. **Interface pulls READY HIGH**

---

## Transmission Type 2: Typewriter → Interface

### Timing Diagram

```
KBRQ:   _____/‾‾‾‾‾‾\___________________________________/‾\_____
             ↑      ↓                                   ↑ ↑
             │      │ READY forces LOW                  │ │ 10µs pulse then
             │      │                                   │ │ typewriter pulls LOW
             │ 150µs                                    │ │
             │      ↓                                   │ │
READY:  ‾‾‾‾‾‾‾‾‾‾‾‾\___________________________________/‾‾‾‾‾‾
                    │                                   ↑
                    │ 200µs                             │ 200µs after clock stops
                    ↓                                   │
SCK:    ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾‾‾‾‾‾‾
                       ↑                             ↑
                       │ 8 bits                      │ Clock stops
                       │                             │
KBACK:  ????????‾‾‾‾‾‾‾\___________________________________________
                ↑      ↑
                │      │ First falling edge forces LOW
                │ Typewriter resets HIGH before transmission
                  (part of "want to send" signal)
                       
SO:     ________/‾‾‾‾‾‾╱D7╲╱D6╲╱D5╲╱D4╲╱D3╲╱D2╲╱D1╲╱D0╲\_________
                ↑                                      ↑
                │ Typewriter pulls HIGH                │ Typewriter pulls LOW
                  (preparation)                          (cleanup)

SI:     ════════╱        DEL (0x7F) or HIGH         ╲════════════
                ↑                                    ↑
                │ Only DEL on first typewriter       │
                  byte after interface sent
```

### Sequence of Operations

1. **Typewriter prepares:**
   - Pulls KBRQ HIGH
   - Resets KBACK to HIGH
   - Pulls SO HIGH
2. **Wait ~150µs**
3. **Interface pulls READY LOW**
   - This forces KBRQ LOW
4. **Wait ~200µs**
5. **Interface starts clock, receives 8 bits on SO**
   - First falling edge of SCK forces KBACK LOW
   - Interface sends DEL (0x7F) on SI if this is first typewriter byte after interface sent
   - Interface keeps SI HIGH if typewriter continues sending
6. **Clock stops**
7. **Typewriter pulls SO LOW** (cleanup)
8. **Wait ~200µs**
9. **Interface pulls READY HIGH**
10. **KBRQ rises with READY briefly (~10µs)**
11. **Typewriter pulls KBRQ LOW** (back to idle)

---

## DEL Byte Behavior (Edge Case Handling)

When switching from Interface→Typewriter to Typewriter→Interface:

| Previous Sender | Current Sender | SI Line During Receive |
|-----------------|----------------|------------------------|
| Interface | Typewriter (1st byte) | DEL (0x7F) |
| Typewriter | Typewriter (continuing) | HIGH |
| Typewriter | Interface | Normal data |

The DEL byte (0b01111111) may signal: "I acknowledge the direction change, ignore SI until I pull READY LOW again"

---

## Idle State Summary

| Signal | Idle State | Notes |
|--------|------------|-------|
| SCK | HIGH | |
| SI | Last bit | No defined idle, stays where it was |
| SO | LOW | |
| READY | HIGH | |
| KBRQ | LOW | |
| KBACK | HIGH after interface sent | Typewriter resets after processing |
| KBACK | LOW after typewriter sent | No reset needed |

---

## Timing Summary

| Event | Duration |
|-------|----------|
| READY LOW → Clock start (interface sending) | 20-30µs |
| Clock stop → KBACK HIGH (typewriter processing) | 100µs - 500ms |
| KBACK HIGH → READY HIGH | 200µs |
| KBRQ HIGH → READY LOW (interface response) | ~150µs |
| READY LOW → Clock start (typewriter sending) | ~200µs |
| Clock stop → READY HIGH | ~200µs |
| READY HIGH → KBRQ pulse end | ~10µs |

---

# Startup and SELECT Sequences

## 1. Power-On / Connection Sequence

| Step | Direction | Byte | Description |
|------|-----------|------|-------------|
| 1 | — | — | Power settles, noise |
| 2 | Interface → Typewriter | 0xFE | Interface announces presence |
| 3 | Typewriter → Interface | 0x30 | Device type response |
| 3 | Interface → Typewriter | 0x7F (DEL) | Sent synchronously (direction change ack) |

```
Interface:  ──── 0xFE ────────── DEL ────────
                              ↗ (simultaneous)
Typewriter: ────────────── 0x30 ─────────────
```

---

## 2. SELECT Sequence (Mode Selection)

| Step | Direction | Byte | Description |
|------|-----------|------|-------------|
| 1 | Interface → Typewriter | 0xF9 or 0xF8 | Mode select (terminal/typewriter) |
| 2 | Interface → Typewriter | 0xFD | SELECT command |
| 3 | Typewriter → Interface | 0x04 | EOT (End of Transmission) |
| 4 | Interface → Typewriter | 0xF4 | ? (Reset margins and new line?) |
| 5 | Interface → Typewriter | 0xB1 | Pitch setting |
| 6 | Interface → Typewriter | 0xB1 | Pitch setting (repeated) |
| 7 | Interface → Typewriter | 0x8B | ? (Only if not at column 1) |
| 8 | Interface → Typewriter | 0x00 × N | Space to restore column position |

---

## Known Command Bytes

| Byte | Name | Description |
|------|------|-------------|
| 0xFE | INIT? | Interface announces presence |
| 0xFD | SELECT | Select/configure command |
| 0xF9 | TERMINAL MODE | Arduino can read keyboard |
| 0xF8 | TYPEWRITER MODE | Arduino cannot read keyboard |
| 0xF4 | RESET MARGINS + NEWLINE | From their code comments |
| 0xB1 | PITCH 10 | 10 characters per inch |
| 0xB2 | PITCH 12 | 12 characters per inch |
| 0xB3 | PITCH 15 | 15 characters per inch |
| 0x8B | ? | Something before column restore |
| 0x00 | SPACE | Move one column right |
| 0x04 | EOT | End of Transmission |
| 0x30 | DEVICE TYPE | Typewriter identification |
| 0x7F | DEL | Direction change acknowledgment |

---

## Questions

1. **Why 0xB1 twice?** Is this intentional or a quirk?

2. **What is 0x8B?** Looking at their code:
   ```cpp
   case 0x8B:
       // we don't remember if underline is on
       sendraw(0x8B);
       break;
   ```
   They associate it with "not underlined" / underline off. Why send this before column restore?

3. **The 0x00 spacing** — So if typewriter was at column 5, interface sends 5× 0x00 to restore position?

4. **When does SELECT happen?** 
   - Only at startup?
   - When switching modes?
   - Can it be triggered anytime?