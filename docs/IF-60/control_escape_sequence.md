
# Standard Control Codes

| Symbol | Code | Function | CX | AX |
|--------|------|----------|----|----|
| BEL<br>(Bell) | 07 H | Acoustic alarm sounds about 2 sec. | O | O |
| BS<br>(Back Space) | 08 H | Moves the carriage backward one character. | O | O |
| LF<br>(Line Feed) | 0A H | Feeds the form one line after one line of data is printed from the LF buffer. The subsequent data is over-printed in the same position as the carriage does not return to the left margin.<br><br>After one line of data is printed from the buffer, the form is fed the number of lines designated by an ESC sequence minus the number of lines already fed on that page.<br>(If the number of lines fed reaches the value of top margin,) The carriage does not return to the left margin. | O | O |
| FF<br>(Form Feed) | 0C H | 1) Prints one line of data from the buffer.<br>2) Then, feeds the form one line if so set by DIP switch or "ESC". (This is effective even when CR alone is entered as CR is always followed by LF.)<br>3) Carriage return is effective even if print data is not received or entered. | O | O |
| CR<br>(Carriage Return) | 0D H |  | O | O |
| DC1<br>(Device Control 1) | 11 H | Puts IF 60 in Select state. | O | O |
| DC3<br>(Device Control 3) | 13 H | Puts IF 60 in Deselect state. | O | O |
| VT<br>(Vertical Tabulation) | 0B H | After printing data up to VT, this command feeds paper to the next VT position. Does not operate if the next VT position is not set.<br>(Does not return to left margin.) | O | O |
| HT<br>(Horizontal Tabulation) | 09 H | Carriage moves to the next HT position. Does not operate if the next HT position is not set. | O | O |
| ESC<br>(Escape) | 1B H | Extension code which, combined with the following code, makes control code. | O | O |

O → Ok

# ESC (escape) Codes

**Note 1:**
While auto backward printing is set, inputting any of the ESC sequences marked with a dagger (†) at the left of the table causes a carriage return (CR) which moves the carriage to the left margin and resets forward printing.

| Symbol | Function | CX | AX |
|--------|----------|----|----|
| ESC + HT + n | Absolute HT movement | O | O |
| † ESC + LF | Reverse paper feed | O | |
| ESC + VT + n | Absolute VT movement | O | |
| ESC + FF + n | Set page length; ESC S resets to default setting | O | O |
| ESC + CR + P | Reset printer | O | O |
| ESC + RS + n | Set VMI; ESC S resets to 1/6 inch | O | |
| ESC + US + n | Set HMI; ESC S resets to PITCH default setting | O | O |
| ESC + " | Auto LF "ON" | O | O |
| † ESC + # | Auto LF "OFF" | O | O |
| † ESC + & | Clear bold, shadow, double print | O | O |
| ESC + — | Set VT at current position | O | |
| † ESC + / | Set auto backward print | O | |
| ESC + 0 | Set right margin at current position | O | O |
| ESC + 1 | Set HT at current position | O | O |
| ESC + 2 | Clear all HT, VT clear | O | O |
| ESC + 8 | Clear current position HT | O | O |
| ESC + 9 | Set left margin at current position | O | O |
| ESC + C | Clear top margin, bottom margin clear | O | |
| † ESC + D | Feed form by reverse 1/12 inch | O | |
| ESC + E | Set auto underline | O | O |
| † ESC + F | Set double-strike print mode | O | |
| ESC + L | Set bottom margin at current position | O | |
| † ESC + O | Set bold print set | O | |
| ESC + R | Clear auto underline | O | O |
| ESC + S | Reset to switch panel, DIP switch | O | O |
| † ESC + T | Set top margin at current position | O | |
| † ESC + U | Feed form by 1/12 inch | O | |
| † ESC + W | Set shadow print | O | |
| ESC + X | Clear underline, auto strike-out, shadow, and double-strike print | O | O |
| ESC + Y | Print 20H character | O | O |
| ESC + Z | Print 7FH character | O | O |
| † ESC + \ | Clear auto backward print | O | |

O → Ok

---

# ESC Sequence Functions

## Print Format

### (1) Setting Character Pitch (HMI)
ESC + US + n sets character pitch.
ESC + S resets HMI to the pitch specified by PITCH select key.
HMI = (n - 1) x 1/120

The n specifies 13, 11 and 9.

After HMI is set, carriage moves in the amount of HMI in each print or space.

ESC + S resets HMI to the pitch specified by PITCH select key.

### (2) Setting Line Pitch (VMI)
ESC + RS + n sets line pitch.
ESC + S resets VMI to 1/6 inch.
VMI = (n - 1) x 1/48

The n specifies 9, 13, 17 and 25.

ESC + S resets VMI to the pitch specified by LINE space select key.

### (3) Setting Page Length
ESC + FF + n sets page length.
ESC + S resets page length to DIP SW.
Page length = n x VMI

The n specifies 1 - 126 excepting NUL and DEL codes and the typewriter interprets the existing VMI as one line. The page length is stored in memory as the absolute position measured with reference to the top of the page. Therefore, if you change the VMI, the number of lines per page also changes.

ESC + S resets page length to DIP SW.

### (4) Setting Left Margin
ESC + 9 sets left margin.

The code sets left margin at present position.

Absolute HT movement or BS enables carriage to move further to the left than the left margin position. (New left margin can be set)

In case the setting position is larger than right margin or the distance between right and left margins is less than 24/120 inch, the new margin cannot be set.

### (5) Setting Right Margin
ESC + 0 sets right margin.

The code sets right margin at present position.

Absolute HT movement enables carriage to move further to the right than the right margin position. (New right margin can be set)

In case the setting position is smaller than left margin or the distance between right and left margins is less than 24/120 inch, the new margin cannot be set.

---

### (6) Setting HT
ESC + 1 sets HT position.
ESC + 8 clears present position.
ESC + 2 clears whole HT positions.

HT position is set at present position and can be set up to 10 places.

To clear present HT position only, input ESC + 8; to clear whole HT positions, input ESC + 2, which also clears whole VT positions.

### (7) Setting VT
ESC + - sets VT position.
ESC + 2 clears whole VT positions.

VT position is set at present position and can be set in 10 places. Present VT position to default.

ESC + 2 clears not only whole VT positions but also all HT positions.

### (8) Setting Top Margin
ESC + T sets top margin.
ESC + C, Page Length Setting, or Remote Resetting clears top margin.

Top margin is set at present position.

Paper is automatically fed in the amount of top margin, when it reaches page top by LF.

VT, absolute VT movement, or reverse LF enables paper feed within top margin. (New top margin can be set)

Top margin can be cleared by changing page length, or by remote resetting or ESC + C, however, when skip perforation is set, top margin returns to 1-inch margin.

In case the position falls within bottom margin, the new margin cannot be set.

### (9) Setting Bottom Margin
ESC + L sets bottom margin.
ESC + C, Page Length Setting, or Remote Resetting clears bottom margin.

Bottom margin is set at present position.

Paper is automatically fed to the following page top, when it reaches bottom margin by LF, Auto LF or Half LF.

VT or absolute VT movement enables paper feed within bottom margin.

Bottom margin can be cleared by changing page length, or by remote resetting or ESC + C, however, when skip perforation is set, bottom margin returns to 1-inch margin.

In case the setting position falls within top margin, the new margin cannot be set.

### (10) Absolute HT Movement
ESC + HT + n sets absolute HT movement.
Movement range = (n - 1) x HMI

The n specifies 1 - 126 excepting NUL and DEL codes and the range can be set in present HMI/120-inch increments.

---

This function makes carriage move directly from left end of platen to set position, but is not stored as HT. (Margins are ignored.)

Does not operate when set position goes beyond right end of platen.

### (11) Absolute VT Movement
ESC + VT + n sets absolute VT movement.
Movement range = (n - 1) x VMI

The n specifies 1 - 126 excepting NUL and DEL codes and the range can be set in present VMI/48-inch increments.

This function feeds paper directly from page top to set position, but is not stored as VT.

This function move within top and bottom margins (margins are ignored). In case the setting position goes beyond page length, it does not operate.

### (12) Reverse LF
ESC + LF sets reverse LF.

Feeds paper in reverse direction in the amount of present VMI.

### (13) Half LF
ESC + U sets half LF.

Feeds paper by 1/12 inch.

### (14) Reverse half LF
ESC + D sets reverse half LF.

The paper is fed reversely by 1/12 inch.

### (15) Auto Backward Print
ESC + / sets auto backward print.
ESC + \ clears auto backward print.

When this mode is set, the typewriter executes logic seeking. However, if the ESC sequences marked with a dagger in the table on page 48 are input, carrier moves to left margin by CR code and the typewriter starts forward print.

In Printer Mode, Auto Backward Print is the default.

### (16) Auto LF
ESC + " sets auto LF.
ESC + # clears auto LF.

When CR code is input with auto LF in set, the typewriter automatically engages LF.