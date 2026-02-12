"""
Brother Serial Interface - Test Definitions

Each Test has a comment (printed as header) and a list of Steps.
Steps are executed in order. Only steps with capture=True are logged.

State assumptions:
- Typewriter starts SELECTED, carriage at home (col 1, line 1)
- Each test must leave the typewriter in a known state for the next test
- After printing tests: CR to return carriage, LF to advance line
- After movement tests: return to home position
"""

from dataclasses import dataclass, field


@dataclass
class Step:
    name: str
    bytes: list
    wait: float = 0.5
    capture: bool = False
    comment: str = ""

    def fmt_sent(self) -> str:
        """Format the sent bytes as hex string."""
        return "[ " + ", ".join(f"0x{b:02X}" for b in self.bytes) + " ]"

    def fmt_result(self, si: list = None, so: list = None) -> str:
        """Format a complete log line for a captured step."""
        sent = self.fmt_sent()

        if si is None or len(si) == 0:
            si_str = "None"
        else:
            si_str = "[ " + ", ".join(f"0x{b:02X}" for b in si) + " ]"

        if so and any(b != 0 for b in so):
            so_str = " ([ " + ", ".join(f"0x{b:02X}" for b in so) + " ])"
        else:
            so_str = ""

        comment_str = f" {self.comment}" if self.comment else ""
        return f"{sent} = {si_str}{so_str}{comment_str}"


@dataclass
class Test:
    comment: str
    steps: list = field(default_factory=list)

    def fmt_header(self) -> str:
        """Format the test comment as a log header."""
        return self.comment



TESTS_CONTROL = [

    # =========================================================================
    # Control Codes 0x00 - 0x06 (expect no output)
    # =========================================================================

    Test('// 0x00 NUL', [ Step('NUL', [0x00], wait=6, capture=True)]),

    Test('// 0x01 SOH', [Step('SOH', [0x01], wait=6, capture=True)]),

    Test('// 0x02 STX', [Step('STX', [0x02], wait=6, capture=True)]),

    Test('// 0x03 ETX', [Step('ETX', [0x03], wait=6, capture=True)]),

    Test('// 0x04 EOT', [Step('EOT', [0x04], wait=6, capture=True)]),

    Test('// 0x05 ENQ', [Step('ENQ', [0x05], wait=6, capture=True)]),

    Test('// 0x06 ACK', [Step('ACK', [0x06], wait=6, capture=True)]),

    # =========================================================================
    # Control Codes 0x07 - 0x0D (functional)
    # =========================================================================

    Test('// 0x07 BEL', [Step('BEL', [0x07], wait=2, capture=True)]),

    # State: carriage at home. Print something first so BS has somewhere to go
    Test('// 0x08 BS - Backspace', [
        # Print Test Descriptor
        Step('print',       list(b'TEST BACKSPACE'), wait=4),
        Step('CR+LF',       [0x0D, 0x0A], wait=2),
        # Print "ABC" to move carriage right
        Step('print ABC',   [0x41,0x42,0x43], wait=2),
        # Backspace one position and Capture
        Step('BS',          [0x08], wait=6, capture=True),
        # Print D over C
        Step('print D',   [0x44], wait=1),
        # Move down for next test.
        Step('2x CR+LF',     [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    # State: carriage at home. Set a tab stop at col 20 first.
    Test('// 0x09 HT - Horizontal Tab', [
        # Print Test Descriptor
        Step('print',           list(b'TEST HORIZONTAL TAB'), wait=4),
        Step('CR+LF',           [0x0D, 0x0A], wait=2),
        # ESC+HT+17: absolute move to col 17
        Step('move to col 17',  [0x1B, 0x09, 17], wait=2),
        # ESC+1: set HT at current position
        Step('set HT stop',     [0x1B, 0x31], wait=1),
        # Print Stop Indicator
        Step('print',       list(b'I  TAB'), wait=4),
        # CR back to home
        Step('CR',              [0x0D], wait=1),
        # Print Start Indicator
        Step('print',       list(b'HT1'), wait=4),
        # Tab: should jump to col 16
        Step('HT',              [0x09], wait=6, capture=True),
        # Print "HT" to verify position
        Step('print HT',        list(b'HT'), wait=2),
        # ESC+2: clear all HT/VT
        Step('clear all HT',    [0x1B, 0x32], wait=1),
        # Move down for next test.
        Step('2x CR+LF',     [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// 0x0A LF - Line Feed', [
        # Print Test Descriptor
        Step('print',             list(b'TEST LINE FEED'), wait=4),
        Step('CR+LF',            [0x0D, 0x0A], wait=2),

        # --- Test 1: Single LF ---
        Step('print',             list(b'SINGLE LF'), wait=4),
        Step('LF',                [0x0A], wait=6, capture=True, comment='// Single LF, does carriage stay at current col?'),
        Step('print',             list(b'SINGLE LF'), wait=4),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),

        # --- Test 2: Double LF ---
        Step('print',             list(b'DOUBLE LF'), wait=4),
        Step('2x LF',            [0x0A, 0x0A], wait=6, capture=True, comment='// Two consecutive LFs'),
        Step('print',             list(b'DOUBLE LF'), wait=4),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),

        # --- Test 3: CR then LF ---
        Step('move to col 15',    [0x1B, 0x09, 15], wait=2),
        Step('print',             list(b'CR THEN LF'), wait=4),
        Step('CR then LF',       [0x0D, 0x0A], wait=6, capture=True, comment='// CR first, then LF — standard newline order'),
        Step('print',             list(b'CR THEN LF'), wait=4),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),

        # --- Test 4: LF then CR ---
        Step('move to col 15',    [0x1B, 0x09, 15], wait=2),
        Step('print',             list(b'LF THEN CR'), wait=4),
        Step('LF then CR',       [0x0A, 0x0D], wait=6, capture=True, comment='// LF first, then CR — reversed order'),
        Step('print',             list(b'LF THEN CR'), wait=4),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),

        # --- Test 5: LF without CR (from mid-line) ---
        Step('move to col 17',    [0x1B, 0x09, 17], wait=2),
        Step('print',             list(b'LF NO CR'), wait=4),
        Step('LF only',          [0x0A], wait=6, capture=True, comment='// LF from col 17, no CR — does col persist?'),
        Step('print',             list(b'LF NO CR'), wait=4),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),

        # --- Test 6: Multiple CR without LF ---
        Step('move to col 17',    [0x1B, 0x09, 17], wait=2),
        Step('print',             list(b'MULTI CR'), wait=4),
        Step('3x CR',            [0x0D, 0x0D, 0x0D], wait=6, capture=True, comment='// Three CRs, no LF — should stay on same line'),
        Step('print',             list(b'MULTI CR'), wait=4),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),

        # Move down for next test.
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    # State: carriage at home. Set a VT stop a few lines down first.
    Test('// 0x0B VT - Vertical Tab', [
        # Print Test Descriptor
        Step('print',           list(b'TEST VERTICAL TAB'), wait=4),
        Step('CR+LF',           [0x0D, 0x0A], wait=2),
        # 3x LF: absolute move down 3 lines
        Step('move down 3',     [0x0A, 0x0A, 0x0A], wait=2),
        # ESC+1: set VT at current position
        Step('set VT stop',     [0x1B, 0x31], wait=1),
        # Print Stop Indicator
        Step('print',       list(b'I  TAB'), wait=4),
        # CR back to home
        Step('CR',              [0x0D], wait=1),
        # Move back up to starting line (reverse LF x3)
        Step('reverse LF',      [0x1B, 0x0A, 0x1B, 0x0A, 0x1B, 0x0A], wait=2),
        # Print Start Indicator
        Step('print',       list(b'VT1'), wait=4),
        # VT: should jump down to the VT stop line
        Step('VT',              [0x0B], wait=6, capture=True),
        # Print "VT" to verify position
        Step('print VT',        list(b'VT'), wait=2),
        # ESC+2: clear all HT/VT
        Step('clear all VT',    [0x1B, 0x32], wait=1),
        # Move down for next test.
        Step('2x CR+LF',     [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    # FF feeds through the entire page — needs long wait
    Test('// 0x0C FF - Form Feed', [Step('FF', [0x0C], wait=20, capture=True)]),

    # Print something first so CR has somewhere to return from
    Test('// 0x0D CR - Carriage Return', [
        # Print Test Descriptor
        Step('print',           list(b'TEST CARRIAGE RETURN'), wait=4),
        Step('CR+LF',           [0x0D, 0x0A], wait=2),
        # Print to move carriage
        Step('print',           list(b'I        RETURN'), wait=4),
        # CR back to home
        Step('CR',          [0x0D], wait=6, capture=True),
        # Verify position on Paper
        Step('print',           list(b'CARRIAGE'), wait=4),
        # Move down for next test.
        Step('2x CR+LF',     [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    # =========================================================================
    # Control Codes 0x0E - 0x1F
    # =========================================================================

    Test('// 0x0E SO - Shift Out', [Step('SO', [0x0E], wait=2, capture=True)]),

    Test('// 0x0F SI - Shift In', [Step('SI', [0x0F], wait=2, capture=True)]),

    Test('// 0x10 DLE', [Step('DLE', [0x10], wait=2, capture=True)]),

    # DC1 is SELECT — deselect first so we can see the select response
    Test('// 0x11 DC1 - SELECT', [
        Step('deselect',  [0x13], wait=2),                   # DC3: deselect first
        Step('DC1',       [0x11], wait=4, capture=True),     # DC1: select
        # State: typewriter is now selected again
    ]),

    Test('// 0x12 DC2', [Step('DC2', [0x12], wait=2, capture=True)]),

    # DC3 is DESELECT
    Test('// 0x13 DC3 - DESELECT', [
        Step('DC3',     [0x13], wait=4, capture=True),       # DC3: deselect
        Step('reselect', [0x11], wait=2),                     # DC1: reselect for next tests
    ]),

    Test('// 0x14 DC4', [Step('DC4', [0x14], wait=2, capture=True)]),

    Test('// 0x15 NAK', [Step('NAK', [0x15], wait=2, capture=True)]),

    Test('// 0x16 SYN', [Step('SYN', [0x16], wait=2, capture=True)]),

    Test('// 0x17 ETB', [Step('ETB', [0x17], wait=2, capture=True)]),

    Test('// 0x18 CAN', [Step('CAN', [0x18], wait=2, capture=True)]),

    Test('// 0x19 EM', [Step('EM', [0x19], wait=2, capture=True)]),

    Test('// 0x1A SUB', [Step('SUB', [0x1A], wait=2, capture=True)]),

    # ESC alone: next byte will be interpreted as ESC sequence.
    # Send ESC+S (reset to DIP) to consume the ESC cleanly.
    Test('// 0x1B ESC (alone)', [
        Step('ESC',         [0x1B], wait=6, capture=True),
        Step('consume ESC', [0x53], wait=1),                  # 'S' → ESC+S resets to DIP defaults
    ]),

    Test('// 0x1C FS', [Step('FS', [0x1C], wait=2, capture=True)]),

    Test('// 0x1D GS', [Step('GS', [0x1D], wait=2, capture=True)]),

    Test('// 0x1E RS', [Step('RS', [0x1E], wait=2, capture=True)]),

    Test('// 0x1F US', [Step('US', [0x1F], wait=2, capture=True)]),
]

def _build_printable_test() -> Test:
    """Build a single test that prints a classical ASCII table grid.

    Output on paper:
        0 1 2 3 4 5 6 7 8 9 A B C D E F
      2   ! " # $ % & ' ( ) * + , - . /
      3 0 1 2 3 4 5 6 7 8 9 : ; < = > ?
      4 @ A B C D E F G H I J K L M N O
      5 P Q R S T U V W X Y Z [ \\ ] ^ _
      6 ` a b c d e f g h i j k l m n o
      7 p q r s t u v w x y z { | } ~ DEL

    Each char uses absolute HT positioning so dead keys don't shift the grid.
    A space is printed after every char to separate them.
    """

    # Grid layout constants
    COL_START = 3       # First data column (after row label)
    COL_SPACING = 2     # Columns between each character
    ROW_LABEL_COL = 1   # Column for row label (2-7)

    steps = [
        # Print Test Descriptor
        Step('print',  list(b'TEST PRINTABLE CHARS'), wait=4),
        Step('CR+LF',  [0x0D, 0x0A], wait=2),
    ]

    # --- Header row: 0 1 2 3 4 5 6 7 8 9 A B C D E F ---
    header_labels = '0123456789ABCDEF'
    for col_idx, label in enumerate(header_labels):
        col = COL_START + col_idx * COL_SPACING
        steps.append(Step(f'move to col {col}', [0x1B, 0x09, col], wait=0))
        steps.append(Step(f'print "{label}"', [ord(label)], wait=0))

    steps.append(Step('CR+LF', [0x0D, 0x0A], wait=4))

    # --- Data rows: 0x20-0x7F, 16 chars per row ---
    for row in range(2, 8):
        # Print row label (2-7)
        steps.append(Step(f'move to col {ROW_LABEL_COL}', [0x1B, 0x09, ROW_LABEL_COL], wait=0))
        steps.append(Step(f'print row "{row}"', [ord(str(row))], wait=0))

        # Print 16 characters in this row
        for col_idx in range(16):
            code = row * 16 + col_idx
            col = COL_START + col_idx * COL_SPACING

            # Absolute HT to grid position
            steps.append(Step(f'move to col {col}', [0x1B, 0x09, col], wait=0.2))

            if code == 0x7F:
                label = '0x7F DEL'
            else:
                label = f'0x{code:02X} "{chr(code)}"'

            # Print the character (captured)
            steps.append(Step(label, [code], wait=0.2, capture=True))
            # Space after to separate from next and clear dead keys
            steps.append(Step('space', [0x20], wait=0))

        steps.append(Step('CR+LF', [0x0D, 0x0A], wait=2))

    # Move down for next test group
    steps.append(Step('2x CR+LF', [0x0D, 0x0A, 0x0D, 0x0A], wait=1))
    return Test('// Printable Characters 0x20 - 0x7F', steps)


TESTS_PRINTABLE = [
    _build_printable_test(),
]

TESTS_ESCAPE = [

    # =========================================================================
    # ESC Sequences
    # State: selected, carriage at home after printable char tests
    # =========================================================================

    # --- Movement ---

    Test('// ESC + HT + n: Absolute HT Movement', [
        # Print Test Descriptor
        Step('print',      list(b'TEST ABSOLUTE HT MOVEMENT'), wait=4),
        Step('CR+LF',          [0x0D, 0x0A], wait=2),
        # Known state: home position
        Step('reset printer',        [0x1B, 0x0D, 0x50], wait=2),
        # Verify position on paper
        Step('print "HT1"',          [0x48, 0x54, 0x31], wait=2),
        # Move to H1 again
        Step('abs HT to col 1',      [0x1B, 0x09, 1], wait=4),
        # Move to H17 and Capture
        Step('abs HT to col 17',     [0x1B, 0x09, 17], wait=6, capture=True, comment='// Move right HT1 to HT17'),
        # Verify position on paper
        Step('print "HT17"',         [0x48, 0x54, 0x31, 0x37], wait=2),
        # Move to H17 again
        Step('abs HT to col 17',     [0x1B, 0x09, 17], wait=6),
        # Move to H17 and Capture
        Step('abs HT to col 17',     [0x1B, 0x09, 17], wait=6, capture=True, comment='// Already at target HT17'),
        # Move to H9 and Capture
        Step('abs HT to col 9',      [0x1B, 0x09, 9], wait=6, capture=True, comment='// Move left HT17 to HT9'),
        # Verify position on paper
        Step('print "HT9"',          [0x48, 0x54, 0x39], wait=2),
        # Move Back fot next Test
        Step('2x CR+LF',                [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// ESC + VT + n: Absolute VT Movement', [
        # Print Test Descriptor
        Step('print',      list(b'TEST ABSOLUTE VT MOVEMENT'), wait=4),
        Step('CR+LF',          [0x0D, 0x0A], wait=2),
        # Known state: line 1
        Step('reset printer',        [0x1B, 0x0D, 0x50], wait=2),
        # Verify position on paper
        Step('print "VT1"',          [0x56, 0x54, 0x31], wait=2),
        # Move to VT3 and Capture
        Step('abs VT to line 3',     [0x1B, 0x0B, 3], wait=6, capture=True, comment='// Move down VT1 to VT3'),
        # Verify position on paper
        Step('print "VT3"',          [0x56, 0x54, 0x33], wait=2),
        # Move to VT3 and Capture
        Step('abs VT to line 3',    [0x1B, 0x0B, 3], wait=6, capture=True, comment='// Already at target'),
        # Move to VT2 and Capture
        Step('abs VT to line 2',     [0x1B, 0x0B, 2], wait=6, capture=True, comment='// Move up VT3 to VT2'),
        # Verify position on paper
        Step('print "VT2"',          [0x56, 0x54, 0x32], wait=2),
        # Move Back fot next Test
        Step('2x CR+LF',                [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    # --- Paper Feed ---

    Test('// ESC + LF: Reverse paper feed (one VMI unit up)', [
        # Print Test Descriptor
        Step('print',      list(b'TEST REVERSE LF'), wait=4),
        Step('CR+LF',          [0x0D, 0x0A], wait=2),
        # Go down one line so we can reverse
        Step('LF first',         [0x0A], wait=1),
        # Verify position on paper
        Step('print LF1',        [0x4C, 0x46, 0x31], wait=2),
        # LF Reverse
        Step('reverse LF',       [0x1B, 0x0A], wait=6, capture=True),
        # Verify position on paper
        Step('print LF0',        [0x4C, 0x46, 0x30], wait=2),
        # State: back where we started vertically
        Step('2x CR+LF',                [0x0D, 0x0A, 0x0D, 0x0A], wait=2),
    ]),

    Test('// ESC + U: Half LF (1/12 inch down)', [
        # Print Test Descriptor
        Step('print',      list(b'TEST SUBSCRIPT'), wait=4),
        Step('CR+LF',          [0x0D, 0x0A], wait=2),
        # Print baseline text (Normal)
        Step('print NRM',      list(b'NRM'), wait=2),
        # Perform Half LF (1/12" Down)
        Step('half LF',        [0x1B, 0x55], wait=6, capture=True),
        # Print subscript text (should be lower)
        Step('print SUB',      list(b'SUB'), wait=2),
        # Undo: Reverse Half LF (Up) to return to baseline
        Step('undo: reverse',  [0x1B, 0x44], wait=2),
        # Reset line position
        Step('2x CR+LF',                [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// ESC + D: Reverse half LF (1/12 inch up)', [
        # Print Test Descriptor
        Step('print',      list(b'TEST SUPERSCRIPT'), wait=4),
        Step('CR+LF',          [0x0D, 0x0A], wait=2),
        # Print baseline text (Normal)
        Step('print NRM',      list(b'NRM'), wait=2),
        # Perform Reverse Half LF (1/12" Up)
        Step('reverse half LF',[0x1B, 0x44], wait=6, capture=True),
        # Print superscript text (should be higher)
        Step('print SUP',      list(b'SUP'), wait=2),
        # Undo: Half LF (Down) to return to baseline
        Step('undo: down',     [0x1B, 0x55], wait=2),
        # Reset line position
        Step('2x CR+LF',                [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    # --- Page Setup ---

    # --- Page Length & Form Feed Interaction ---
    # Manual: Page length = n × VMI, stored as absolute position from top of page.
    # Changing VMI also changes the number of lines per page.
    Test('// ESC + FF + n: Set page length and test FF behavior', [
        # Print Test Descriptor
        Step('print',           list(b'TEST PAGE LENGTH'), wait=4),
        Step('CR+LF',           [0x0D, 0x0A], wait=2),

        # --- Test 1: Set short page (8 lines), FF should advance to next page ---
        # Print start marker
        Step('print',           list(b'LEN8 START'), wait=4),
        Step('CR+LF',           [0x0D, 0x0A], wait=2),
        # Reset printer to ensure we start at line 1, top of page
        Step('reset printer',   [0x1B, 0x0D, 0x50], wait=6),
        # Set page length 8
        Step('set page len 8',  [0x1B, 0x0C, 8], wait=6, capture=True,  comment='// Set Page length 8'),
        # FF: should feed to top of next 8-line page
        Step('FF LEN8',         [0x0C], wait=20, capture=True, comment='// FF with Page length 8'),
        # Print to verify we landed at top of new page
        Step('print',           list(b'LEN8 FF'), wait=4),
        Step('CR+LF',           [0x0D, 0x0A], wait=1),

        # --- Test 2: Reset and set page length to 16 lines, FF should advance to next page ---
        # Print start marker
        Step('print',           list(b'LEN16 START'), wait=4),
        Step('CR+LF',           [0x0D, 0x0A], wait=2),
        # Reset printer to ensure we start at line 1, top of page
        Step('reset printer',   [0x1B, 0x0D, 0x50], wait=6),
        # Set page length 16
        Step('set page len 16',   [0x1B, 0x0C, 16], wait=6, capture=True, comment='// Set Page length 16'),
        # FF: should feed further now (16-line page)
        Step('FF LEN16',        [0x0C], wait=20, capture=True, comment='// FF with Page length 16'),
        Step('print',             list(b'LEN16 FF'), wait=4),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),

        # --- Cleanup: reset page length to DIP default ---
        Step('reset to DIP',     [0x1B, 0x53], wait=1),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// ESC + CR + P: Reset printer', [
        # Print Test Descriptor
        Step('print',               list(b'TEST RESET PRINTER'), wait=4),
        Step('CR+LF',               [0x0D, 0x0A], wait=2),
        # --- Test 1: Reset from home position (col 1) ---
        Step('reset from home',     [0x1B, 0x0D, 0x50], wait=6, capture=True, comment='// Reset at col 1'),
        Step('CR+LF',               [0x0D, 0x0A], wait=1),
        # --- Test 2: Reset from mid-line position ---
        Step('move to col 9',       [0x1B, 0x09, 9], wait=2),
        Step('reset from col 9',    [0x1B, 0x0D, 0x50], wait=6, capture=True, comment='// Reset at col 9, does carriage return home?'),
        # Move down for next test.
        Step('2x CR+LF',            [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    # --- Pitch / Spacing ---

    Test('// ESC + US + n: Set HMI (Only 9,11,13); ESC S resets to PITCH default (CX, AX)', [
        # Print Test Descriptor
        Step('print',               list(b'TEST SET HMI'), wait=4),
        Step('LF+CR',               [0x0A, 0x0D], wait=2),
        # --- Test 1: Set HMI to 9 (15 characters per inch, tightest spacing) ---
        Step('set HMI 9 (15cpi)',   [0x1B, 0x1F, 9],  wait=6, capture=True, comment='// HMI=9, 15cpi — narrowest pitch'),
        Step('print PITCH15',       [0x50, 0x49, 0x54, 0x43, 0x48, 0x31, 0x35], wait=6),
        Step('LF+CR',               [0x0A, 0x0D], wait=2),
        # --- Test 2: Set HMI to 11 (12 characters per inch, medium spacing) ---
        Step('set HMI 11 (12cpi)',  [0x1B, 0x1F, 11], wait=6, capture=True, comment='// HMI=11, 12cpi — medium pitch'),
        Step('print PITCH12',       [0x50, 0x49, 0x54, 0x43, 0x48, 0x31, 0x32], wait=6),
        Step('LF+CR',               [0x0A, 0x0D], wait=2),
        # --- Test 3: Set HMI to 13 (10 characters per inch, widest spacing) ---
        Step('set HMI 13 (10cpi)',  [0x1B, 0x1F, 13], wait=6, capture=True, comment='// HMI=13, 10cpi — widest pitch'),
        Step('print PITCH10',       [0x50, 0x49, 0x54, 0x43, 0x48, 0x31, 0x30], wait=6),
        Step('LF+CR',               [0x0A, 0x0D], wait=2),
        # --- Test 4: Set same HMI again to check idempotent behavior ---
        Step('set HMI 13 again',    [0x1B, 0x1F, 13], wait=6, capture=True, comment='// Send again when HMI already 13 — is response identical?'),
        # --- Cleanup: reset to DIP switch defaults ---
        Step('reset to DIP',        [0x1B, 0x53], wait=1),
        Step('2x CR+LF',            [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// ESC + RS + n: Set VMI (Only 9,13,17,25); ESC S resets to 1/6 inch', [
        # Print Test Descriptor
        Step('print',             list(b'TEST SET VMI'), wait=4),
        Step('LF+CR',            [0x0A, 0x0D], wait=2),
        Step('print',            list(b'VMI START'), wait=4),
        Step('LF+CR',            [0x0A, 0x0D], wait=2),
        # --- Test 1: Set VMI to 9 (tightest line spacing) ---
        Step('set VMI 9',        [0x1B, 0x1E, 9],  wait=6, capture=True, comment='// VMI=9 — tightest line spacing'),
        Step('print',            list(b'VMI9'), wait=4),
        Step('LF+CR',            [0x0A, 0x0D], wait=1),
        # --- Test 2: Set VMI to 13 (default 1/6 inch spacing) ---
        Step('set VMI 13',       [0x1B, 0x1E, 13], wait=6, capture=True, comment='// VMI=13 — default 1/6 inch'),
        Step('print',            list(b'VMI13'), wait=4),
        Step('LF+CR',            [0x0A, 0x0D], wait=1),
        # --- Test 3: Set VMI to 17 (wider line spacing) ---
        Step('set VMI 17',       [0x1B, 0x1E, 17], wait=6, capture=True, comment='// VMI=17 — wider spacing'),
        Step('print',            list(b'VMI17'), wait=4),
        Step('LF+CR',            [0x0A, 0x0D], wait=1),
        # --- Test 4: Set VMI to 25 (widest line spacing) ---
        Step('set VMI 25',       [0x1B, 0x1E, 25], wait=6, capture=True, comment='// VMI=25 — widest spacing'),
        Step('print',            list(b'VMI25'), wait=4),
        Step('LF+CR',            [0x0A, 0x0D], wait=1),
        # --- Cleanup: reset to DIP switch defaults ---
        Step('reset to DIP',     [0x1B, 0x53], wait=1),
        Step('2x CR+LF',            [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    # --- Margins ---

    Test('// ESC + 9: Set left margin at current position', [
        # Print Test Descriptor
        Step('print',             list(b'TEST LEFT MARGIN'), wait=4),
        Step('CR+LF',            [0x0D, 0x0A], wait=2),
        # Move to col 17 and set left margin
        Step('move to col 17',    [0x1B, 0x09, 17], wait=2),
        Step('set left margin',   [0x1B, 0x39], wait=2, capture=True, comment='// Set left margin at col 17'),
        # Print margin indicator at the margin position
        Step('print',             list(b'I    MARGIN'), wait=2),
        # CR: should carriage return to col 16 (margin), not col 1
        Step('CR',                [0x0D], wait=1, capture=True, comment='// Return to home at col 17'),
        # Print start indicator at home to see where CR lands
        Step('print',             list(b'LEFT'), wait=1),
        # --- Cleanup: reset to DIP defaults to clear margin ---
        Step('clear margins',     [0x1B, 0x42], wait=1),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// ESC + 0: Set right margin at current position', [
        # Print Test Descriptor
        Step('print',             list(b'TEST RIGHT MARGIN'), wait=4),
        Step('CR+LF',            [0x0D, 0x0A], wait=2),
        # Move to col 17 and set right margin
        Step('move to col 21',    [0x1B, 0x09, 21], wait=2),
        # Print 12 Letters
        Step('print',             list(b'RIGHT      I'), wait=4),
        Step('set right margin',  [0x1B, 0x30], wait=2, capture=True, comment='// Set right margin at col 33'),
        # CR back to home
        Step('CR',                [0x0D], wait=1, capture=True, comment='// Return to home at col 1'),
        # Move to Col 27
        Step('move to col 27',    [0x1B, 0x09, 27], wait=1),
        Step('print',             list(b'MARGIN'), wait=2),
        # Test: try to move past right margin
        Step('try over RM',       list(b'    '), wait=2, capture=True, comment='// Try over writing over Right Margin'),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # --- Cleanup: reset to DIP defaults to clear margin ---
        Step('clear margins',     [0x1B, 0x42], wait=1),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// ESC + T: Set top margin at current position', [
        # Print Test Descriptor
        Step('print',             list(b'TEST TOP MARGIN'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=2),
        # Reset to known state at top of page
        Step('reset printer',     [0x1B, 0x0D, 0x50], wait=2),
        # Move down 8 lines and set top margin
        Step('abs VT to line 8',  [0x1B, 0x0B, 8], wait=2),
        Step('set top margin',    [0x1B, 0x54], wait=2, capture=True, comment='// Set top margin at line 8'),
        Step('print',             list(b'TOP MARGIN'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # Test: LF past top of page should auto-feed by top margin amount
        # Move to line 1 via absolute VT (manual says this is allowed within top margin)
        Step('abs VT to line 1',  [0x1B, 0x0B, 1], wait=2, capture=True, comment='// Move into top margin area — manual says VT/reverse LF allowed'),
        Step('print',             list(b'LINE1?'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # --- Cleanup ---
        Step('clear margins',     [0x1B, 0x43], wait=1),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// ESC + L: Set bottom margin at current position', [
        # Print Test Descriptor
        Step('print',             list(b'TEST BOTTOM MARGIN'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # Reset to known state at top of page
        Step('reset printer',     [0x1B, 0x0D, 0x50], wait=1),
        # Move down 16 lines and set bottom margin
        Step('abs VT to line 16', [0x1B, 0x0B, 16], wait=2),
        Step('set bottom margin', [0x1B, 0x4C], wait=2, capture=True, comment='// Set bottom margin at line 16'),
        Step('print',             list(b'BOTTOM MARGIN'), wait=2),
        # CR back to start of line, then LF toward bottom margin
        Step('CR',                [0x0D], wait=1),
        # Test: LF at bottom margin should auto-feed to next page top
        Step('LF at margin',      [0x0A], wait=6, capture=True, comment='// LF at bottom margin — should feed to next page top'),
        Step('print',             list(b'AFTER LF'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # --- Cleanup ---
        Step('clear margins',     [0x1B, 0x43], wait=1),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// ESC + C: Clear top and bottom margins', [
        # Print Test Descriptor
        Step('print',             list(b'TEST CLEAR MARGINS'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # Set both margins so we have something to clear
        Step('reset printer',     [0x1B, 0x0D, 0x50], wait=1),
        Step('abs VT to line 8',  [0x1B, 0x0B, 8], wait=2),
        Step('set top margin',    [0x1B, 0x54], wait=1),
        Step('abs VT to line 16', [0x1B, 0x0B, 16], wait=2),
        Step('set bottom margin', [0x1B, 0x4C], wait=1),
        # Clear both margins
        Step('clear margins',     [0x1B, 0x43], wait=4, capture=True, comment='// Clear both top and bottom margins'),
        # --- Cleanup ---
        Step('reset to DIP',      [0x1B, 0x53], wait=1),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    # --- Tab Stops ---

    Test('// ESC + 1: Set HT at current position', [
        # Print Test Descriptor
        Step('print',             list(b'TEST SET HT STOP'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # Move to col 24 and set HT stop
        Step('move to col 24',    [0x1B, 0x09, 24], wait=2),
        Step('set HT stop',       [0x1B, 0x31], wait=2, capture=True, comment='// Set HT stop at col 24'),
        # Print stop indicator
        Step('print',             list(b'  TAB'), wait=1),
        # CR back to home
        Step('CR',                [0x0D], wait=2),
        # Print start indicator
        Step('print',             list(b'HT1'), wait=2),
        # Tab: should jump to col 24
        Step('HT',                [0x09], wait=4, capture=True, comment='// HT should jump to col 24'),
        # Verify position on paper
        Step('print',             list(b'HT'), wait=1),
        # --- Cleanup ---
        Step('clear all HT/VT',   [0x1B, 0x32], wait=1),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// ESC + 8: Clear HT at current position', [
        # Print Test Descriptor
        Step('print',             list(b'TEST CLEAR HT STOP'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # Set two HT stops: col 16 and col 32
        Step('move to col 16',    [0x1B, 0x09, 16], wait=2),
        Step('set HT stop',       [0x1B, 0x31], wait=0),
        Step('move to col 32',    [0x1B, 0x09, 32], wait=2),
        Step('set HT stop',       [0x1B, 0x31], wait=0),
        # Clear HT at col 16 only
        Step('move to col 16',    [0x1B, 0x09, 16], wait=2),
        Step('clear HT at pos',   [0x1B, 0x38], wait=2, capture=True, comment='// Clear HT stop at col 16, col 32 should remain'),
        # CR back to home
        Step('CR',                [0x0D], wait=1),
        # Print start indicator
        Step('print',             list(b'HT1'), wait=2),
        # Tab: should skip col 16 and jump to col 32
        Step('HT',                [0x09], wait=2, capture=True, comment='// HT should skip cleared col 16, jump to col 32'),
        Step('print',             list(b'HT'), wait=2),
        # --- Cleanup ---
        Step('clear all HT/VT',   [0x1B, 0x32], wait=1),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// ESC + 2: Clear all HT and VT', [
        # Print Test Descriptor
        Step('print',             list(b'TEST CLEAR ALL STOPS'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # Set HT stops at col 16 and col 32
        Step('move to col 16',    [0x1B, 0x09, 16], wait=0),
        Step('set HT stop',       [0x1B, 0x31], wait=2),
        Step('move to col 32',    [0x1B, 0x09, 32], wait=0),
        Step('set HT stop',       [0x1B, 0x31], wait=2),
        # Clear all stops
        Step('clear all HT/VT',   [0x1B, 0x32], wait=2, capture=True, comment='// Clear all HT and VT stops'),
        # CR back to home
        Step('CR',                [0x0D], wait=1),
        # Print start indicator
        Step('print',             list(b'HT1 '), wait=1),
        # Tab: no stops set, should do nothing or default behavior
        Step('HT',                [0x09], wait=2, capture=True, comment='// HT with no stops — what happens?'),
        Step('print',             list(b'HT?'), wait=1),
        # --- Cleanup ---
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// ESC + -: Set VT at current position', [
        # Print Test Descriptor
        Step('print',             list(b'TEST SET VT STOP'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # Move down 8 lines and set VT stop
        Step('abs VT to line 8',  [0x1B, 0x0B, 8], wait=2),
        Step('set VT stop',       [0x1B, 0x2D], wait=2, capture=True, comment='// Set VT stop at line 8'),
        # Print stop indicator
        Step('print',             list(b'  VSTOP'), wait=2),
        # Move back to top
        Step('reset printer',     [0x1B, 0x0D, 0x50], wait=1),
        # Print start indicator
        Step('print',             list(b'VT1'), wait=2),
        # VT: should jump to line 8
        Step('VT',                [0x0B], wait=2, capture=True, comment='// VT should jump to line 8'),
        Step('print',             list(b'VT'), wait=1),
        # --- Cleanup ---
        Step('clear all HT/VT',   [0x1B, 0x32], wait=1),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    # --- Print Modes ---
    Test('// ESC + E: Set auto underline', [
        # Print Test Descriptor
        Step('print',             list(b'TEST UNDERLINE'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # Print normal text for comparison
        Step('print',             list(b'NORMAL '), wait=2),
        # Enable underline
        Step('enable underline',  [0x1B, 0x45], wait=1, capture=True, comment='// Enable auto underline'),
        # Print underlined text to verify on paper
        Step('print',             list(b'UNDERLINED'), wait=2),
        # Disable underline before next test
        Step('clear underline',   [0x1B, 0x52], wait=1),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// ESC + R: Clear auto underline', [
        # Print Test Descriptor
        Step('print',             list(b'TEST CLEAR UNDERLINE'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # Enable underline first
        Step('enable underline',  [0x1B, 0x45], wait=1),
        Step('print',             list(b'UNDERLINE '), wait=4),
        # Clear underline
        Step('clear underline',   [0x1B, 0x52], wait=2, capture=True, comment='// Clear auto underline'),
        # Print after clearing to verify it stopped
        Step('print',             list(b'NORMAL'), wait=1),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// ESC + O: Set bold print', [
        # Print Test Descriptor
        Step('print',             list(b'TEST BOLD'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # Print normal text for comparison
        Step('print',             list(b'NORMAL '), wait=2),
        # Enable bold
        Step('enable bold',       [0x1B, 0x4F], wait=2, capture=True, comment='// Enable bold print'),
        # Print bold text to verify on paper
        Step('print',             list(b'BOLD'), wait=1),
        # Disable bold
        Step('clear bold',        [0x1B, 0x26], wait=1),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// ESC + W: Set shadow print', [
        # Print Test Descriptor
        Step('print',             list(b'TEST SHADOW'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # Print normal text for comparison
        Step('print',             list(b'NORMAL '), wait=1),
        # Enable shadow
        Step('enable shadow',     [0x1B, 0x57], wait=2, capture=True, comment='// Enable shadow print'),
        # Print shadow text to verify on paper
        Step('print',             list(b'SHADOW'), wait=1),
        # Disable shadow
        Step('clear shadow',      [0x1B, 0x26], wait=1),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// ESC + F: Set double-strike print mode', [
        # Print Test Descriptor
        Step('print',             list(b'TEST DOUBLE STRIKE'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # Print normal text for comparison
        Step('print',             list(b'NORMAL '), wait=2),
        # Enable double-strike
        Step('enable dbl strike', [0x1B, 0x46], wait=2, capture=True, comment='// Enable double-strike print'),
        # Print double-strike text to verify on paper
        Step('print',             list(b'DBLSTRIKE'), wait=1),
        # Disable double-strike
        Step('clear dbl strike',  [0x1B, 0x26], wait=1),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// ESC + &: Clear bold, shadow, double-strike', [
        # Print Test Descriptor
        Step('print',             list(b'TEST CLEAR BOLD/SHADOW/DS'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # Enable all three modes
        Step('enable bold',       [0x1B, 0x4F], wait=0),
        Step('enable shadow',     [0x1B, 0x57], wait=0),
        Step('enable dbl strike', [0x1B, 0x46], wait=0),
        Step('print',             list(b'ALL ON '), wait=2),
        # Clear bold, shadow, double-strike
        Step('clear all',         [0x1B, 0x26], wait=2, capture=True, comment='// Clear bold + shadow + double-strike'),
        Step('print',             list(b'ALL OFF'), wait=1),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # Test clearing when nothing is set
        Step('clear again',       [0x1B, 0x26], wait=2, capture=True, comment='// Clear when none set — is response identical?'),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// ESC + X: Clear underline, strike-out, shadow, double-strike', [
        # Print Test Descriptor
        Step('print',             list(b'TEST CLEAR ALL MODES'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # Enable underline, shadow, double-strike
        Step('enable underline',  [0x1B, 0x45], wait=0),
        Step('enable shadow',     [0x1B, 0x57], wait=0),
        Step('enable dbl strike', [0x1B, 0x46], wait=0),
        Step('print',             list(b'ALL ON '), wait=2),
        # Clear all modes
        Step('clear all',         [0x1B, 0x58], wait=2, capture=True, comment='// Clear underline + shadow + double-strike'),
        Step('print',             list(b'ALL OFF'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # Test clearing when nothing is set
        Step('clear again',       [0x1B, 0x58], wait=2, capture=True, comment='// Clear when none set — is response identical?'),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    # --- Backward Print ---

    Test('// ESC + /: Set auto backward print', [
        # Print Test Descriptor
        Step('print',             list(b'TEST BACKWARD PRINT'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # Print forward reference text first
        Step('print',             list(b'FORWARD '), wait=2),
        # Move to col 40 so there is space to print backward
        Step('move to col 40',    [0x1B, 0x09, 40], wait=2),
        # Enable backward print
        Step('enable backward',   [0x1B, 0x2F], wait=4, capture=True, comment='// Enable auto backward print — logic seeking active'),
        # Print backward to verify right-to-left
        Step('print',             list(b'BACKWARD'), wait=2),
        # Test: CR should move to left margin and start forward print
        Step('CR',                [0x0D], wait=4, capture=True, comment='// CR in backward mode — should go to left margin and switch to forward'),
        Step('print',             list(b'        AFTER CR'), wait=1),
        # --- Cleanup ---
        Step('clear backward',    [0x1B, 0x5C], wait=1),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// ESC + \\: Clear auto backward print', [
        # Print Test Descriptor
        Step('print',             list(b'TEST CLEAR BACKWARD'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # Enable backward, move out so we can see the effect
        Step('move to col 40',    [0x1B, 0x09, 40], wait=2),
        Step('enable backward',   [0x1B, 0x2F], wait=1),
        Step('print',             list(b'BACKWARD '), wait=4),
        # Clear backward print
        Step('clear backward',    [0x1B, 0x5C], wait=6, capture=True, comment='// Clear auto backward print — should resume forward'),
        Step('print',             list(b'FORWARD'), wait=4),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    # --- Auto LF ---

    Test('// ESC + ": Auto LF ON', [
        # Print Test Descriptor
        Step('print',             list(b'TEST AUTO LF ON'), wait=4),
        Step('CR+LF',            [0x0D, 0x0A], wait=2),
        # Enable auto LF — CR should now also advance one line
        Step('auto LF on',       [0x1B, 0x22], wait=2, capture=True, comment='// Enable auto LF — CR now includes LF'),
        # Print A then CR — should move to next line automatically
        Step('print',             list(b'LINE A'), wait=4),
        Step('CR',                [0x0D], wait=2, capture=True, comment='// CR with auto LF — should advance to next line'),
        # Print B then CR — should be on yet another line
        Step('print',             list(b'LINE B'), wait=4),
        Step('CR+LF',             [0x0D], wait=2, capture=True, comment='// CR+LF with auto LF: What happens to the LF?'),
        # Print C to verify we are on a third line
        Step('print',             list(b'LINE C/D?'), wait=2),
        # --- Cleanup ---
        Step('reset to DIP',      [0x1B, 0x53], wait=1),
        Step('2x CR+LF',          [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// ESC + #: Auto LF OFF', [
        # Print Test Descriptor
        Step('print',             list(b'TEST AUTO LF OFF'), wait=4),
        Step('CR+LF',            [0x0D, 0x0A], wait=2),
        # Enable auto LF first
        Step('auto LF on',       [0x1B, 0x22], wait=1),
        Step('print',             list(b'AUTO ON'), wait=4),
        Step('CR',                [0x0D], wait=1),
        Step('print',             list(b'NEXT LINE'), wait=4),
        # Disable auto LF
        Step('auto LF off',      [0x1B, 0x23], wait=6, capture=True, comment='// Disable auto LF — CR should no longer advance line'),
        # CR should now stay on same line — print overwrites
        Step('print',             list(b'SAME LINE'), wait=4),
        Step('CR',                [0x0D], wait=6, capture=True, comment='// CR without auto LF — should stay on same line'),
        Step('print',             list(b'OVERWRITE'), wait=4),
        # --- Cleanup ---
        Step('reset to DIP',      [0x1B, 0x53], wait=1),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    # --- Special Characters ---
    Test('// ESC + Y: Print 0x20 (Space) character', [
        # Print Test Descriptor
        Step('print',             list(b'TEST PRINT 0x20 CHAR'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # Print I markers around the space character to see it on paper
        Step('print',             list(b'I'), wait=1),
        Step('print space char',  [0x1B, 0x59], wait=2, capture=True, comment='// ESC+Y prints the physical glyph for 0x20'),
        Step('print',             list(b'I'), wait=1),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    Test('// ESC + Z: Print 0x7F (DEL) character', [
        # Print Test Descriptor
        Step('print',             list(b'TEST PRINT 0x7F CHAR'), wait=2),
        Step('CR+LF',            [0x0D, 0x0A], wait=1),
        # Print I markers around the DEL character to see it on paper
        Step('print',             list(b'I'), wait=1),
        Step('print DEL char',    [0x1B, 0x5A], wait=2, capture=True, comment='// ESC+Z prints the physical glyph for 0x7F'),
        Step('print',             list(b'I'), wait=1),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),

    # --- Reset ---
    Test('// ESC + S: Reset to DIP switch defaults', [
        # Print Test Descriptor
        Step('print',             list(b'TEST RESET TO DIP'), wait=4),
        Step('CR+LF',            [0x0D, 0x0A], wait=2),
        # Change several settings away from defaults
        Step('set HMI 9',         [0x1B, 0x1F, 9], wait=1),
        Step('enable underline',  [0x1B, 0x45], wait=1),
        Step('enable bold',       [0x1B, 0x4F], wait=1),
        # Print styled text to show current state
        Step('print',             list(b'MODIFIED '), wait=4),
        # Reset to DIP defaults — should clear all changes
        Step('reset to DIP',      [0x1B, 0x53], wait=6, capture=True, comment='// Reset all settings to DIP switch defaults'),
        # Print after reset to verify default styling
        Step('print',             list(b'DEFAULT'), wait=4),
        Step('2x CR+LF',         [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),
]