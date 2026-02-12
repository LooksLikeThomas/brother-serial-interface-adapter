0x00 = None
0x01 = None
0x02 = None
0x04 = None
0x05 = None
0x06 = None
0x07 = 0xF5 // BEL
0x08 = 0x03 // BS
0x09 = [ 0x8B,0x00,0x00,0x00,0x00 ] Moves manually to next HT
0x0A = 0x9F // LF
0x0B = IMPLEMENTED
0x0C = (0x9F for i in range (66)) // FF
0x0D = 0x9E // CR
0x0E = None
0x0F = None  // SI
0x10 = None  // DLE
0x11 = [0xF9, 0xFD, 0xF7, 0xF4, 0xB1, 0xB1] ([0x00, 0x00, 0x04, 0x00, 0x00, 0x00])  // DC1
0x12 = None  // DC2
0x13 = [0xF8]  // DC3
0x14 = None  // DC4
0x15 = None  // NAK
0x16 = None  // SYN
0x17 = None  // ETB
0x18 = None  // CAN
0x19 = None  // EM
0x1A = None  // SUB
0x1B = TODO  // ESC
0x1C = None  // FS
0x1D = None  // GS
0x1E = None  // RS
0x1F = None  // US
0x20 = 0x00  // Space
0x7F = None  // DEL

// Escape Sequenzes

// ESC + HT + n: Absolute Horizontal Tab Movement (CX, AX)
[ 0x1B, 0x09, n ] = None if pos == n-1 else [0x8B] + [0x00 if n-1 > pos else 0x03] * abs(pos - (n-1))
// Target position = (n - 1) x HMI, n = 1~126 (no NUL/DEL)
// Moves from left platen edge, ignores margins, not stored as HT stop.
//
// Response:
//   No movement needed → no response
//   Move right → 0x8B + (0x00 × steps)
//   Move left  → 0x8B + (0x03 × steps)

// ESC + LF: Reverse paper feed (CX)
[ 0x1B, 0x0A ]      = None 

// ESC + VT + n: Absolute VT movement (CX)
[ 0x1B, 0x0B, n ]   = None id pos == n-1 else ([0x9F] * pos - n) if (n>pos) else None 

// ESC + FF + n: Set page length; ESC S resets to default (CX, AX)
[ 0x1B, 0x0C, n ]   = None

// ESC + CR + P: Reset printer (CX, AX)
[ 0x1B, 0x0D, 0x50 ]= [0xF4, 0xB1, 0x8B, 0xFD, 0x7F]

// ESC + RS + n: Set VMI (Only 9,13,17,25); ESC S resets to 1/6 inch (CX) 
[ 0x1B, 0x1E, n ]   = None 

// ESC + US + n: Set HMI (Only 9,11,13); ESC S resets to PITCH default (CX, AX)
[ 0x1B, 0x1F, 9 ]   = 0xB3
[ 0x1B, 0x1F, 11 ]  = 0xB2
[ 0x1B, 0x1F, 13 ]  = 0xB1
[ 0x1B, 0x1F, 13 ]  = 0xB1 // Send again even if HMI = setHMI

// ESC + ": Auto LF "ON" (CX, AX)
[ 0x1B, 0x22 ]      = None // No different trait noticable maybe dip override

// ESC + #: Auto LF "OFF" (CX, AX)
[ 0x1B, 0x23 ]      = None // No different trait noticable maybe dip override

// ESC + &: Clear bold, shadow, double print (CX, AX)
[ 0x1B, 0x26 ]      = None

// ESC + -: Set VT at current position (CX)
[ 0x1B, 0x2D ]      = None

// ESC + /: Set auto backward print (CX)
[ 0x1B, 0x2F ]      = None

// ESC + 0: Set right margin at current position (CX, AX)
[ 0x1B, 0x30 ]      = None // Does set Right margin. Influences Auto LF when reaching margin

// ESC + 1: Set HT at current position (CX, AX)
[ 0x1B, 0x31 ]      = None // sets HT

// ESC + 2: Clear all HT, VT clear (CX, AX)
[ 0x1B, 0x32 ]      = None // Clears HT

// ESC + 8: Clear current position HT (CX, AX)
[ 0x1B, 0x38 ]      = None // Does its thing

// ESC + 9: Set left margin at current position (CX, AX)
[ 0x1B, 0x39 ]      = None // Does its thing

// ESC + C: Clear top margin, bottom margin clear (CX)
[ 0x1B, 0x43 ]      = None

// ESC + D: Feed form by reverse 1/12 inch (CX)
[ 0x1B, 0x44 ]      = None

// ESC + E: Set auto underline (CX, AX)
[ 0x1B, 0x45 ]      = [ 0x8A ]

// ESC + F: Set double-strike print mode (CX)
[ 0x1B, 0x46 ]      = None

// ESC + L: Set bottom margin at current position (CX)
[ 0x1B, 0x4C ]      = None

// ESC + O: Set bold print set (CX)
[ 0x1B, 0x4F ]      =  None

// ESC + R: Clear auto underline (CX, AX)
[ 0x1B, 0x52 ]      = [ 0x8B ]

// ESC + S: Reset to switch panel, DIP switch (CX, AX)
[ 0x1B, 0x53 ]      = [ 0x9E, 0xB1 ]

// ESC + T: Set top margin at current position (CX)
[ 0x1B, 0x54 ]      = None

// ESC + U: Feed form by 1/12 inch (CX)
[ 0x1B, 0x55 ]      = None

// ESC + W: Set shadow print (CX)
[ 0x1B, 0x57 ]      = None

// ESC + X: Clear underline, auto strike-out, shadow, and double-strike print (CX, AX)
[ 0x1B, 0x58 ]      = [ 0x8B ]

// ESC + Y: Print 20H character (CX, AX)
[ 0x1B, 0x59 ]      = [ 0x88,0x55,0x89 ]

// ESC + Z: Print 7FH character (CX, AX)
[ 0x1B, 0x5A ]      = [ 0x3D,0x00 ]

// ESC + \: Clear auto backward print (CX)
[ 0x1B, 0x5C ]      = None
