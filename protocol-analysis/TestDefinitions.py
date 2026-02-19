"""
Brother Serial Interface - Test Definitions
Each Test has a comment (printed as header) and a list of Steps.
Steps are executed in order. Only steps with capture=True are logged.
State assumptions:
- Typewriter starts SELECTED, carriage at home (col 1, line 1)
- Each test must leave the typewriter in a known state for the next test
- After printing tests: CR to return carriage, LF to advance line
- After movement tests: return to home position
Test Groups:
1. CONTROL   - Control codes 0x00-0x06, 0x0E-0x1F (non-functional / simple)
2. PRINTABLE - ASCII table 0x20-0x7F
3. MOVEMENT  - Absolute HT/VT, Reverse LF, Half LF (sub/superscript)
4. SPACING   - HMI (pitch) and VMI (line spacing)
5. MODES     - Underline, bold, shadow, double-strike, backward print, auto LF
6. MARGINS   - Left/right margins (horizontal, no paper feed)
7. TABS      - HT/VT stop set, clear individual, clear all
8. SPECIAL   - Print space/DEL glyphs, reset to DIP
9. PAPERFEED - Page length, FF, top/bottom margins, reset printer
               (grouped last to avoid mid-run paper feeding)
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
		if so and any((b != 0 and b != 0xFF) for b in so):
			so_str = " ([ " + ", ".join(f"0x{b:02X}" for b in so) + " ])"
		else:
			so_str = ""
		comment_str = f" {self.comment}" if self.comment else ""
		return f"{sent} = {si_str}{so_str}{comment_str}"

@dataclass
class Test:
	comment: str
	steps: list = field(default_factory=list)
	compatible_series: list = field(default_factory=lambda: ["A", "C"])

	def fmt_header(self) -> str:
		"""Format the test comment as a log header."""
		return self.comment

# =============================================================================
# 1. CONTROL CODES
#    - 0x00-0x06: expect no output
#    - 0x07 BEL, 0x08 BS, 0x0A LF, 0x0D CR: functional
#    - 0x0E-0x1F: misc control (SO, SI, DLE, DC1-DC4, etc.)
#    NOTE: HT (0x09), VT (0x0B), FF (0x0C) moved to TABS / PAPERFEED groups
# =============================================================================
TESTS_CONTROL = [
	# --- 0x00-0x06: expect no output ---
	Test('// 0x00 NUL', [Step('NUL', [0x00], wait=2, capture=True)]),
	Test('// 0x01 SOH', [Step('SOH', [0x01], wait=2, capture=True)]),
	Test('// 0x02 STX', [Step('STX', [0x02], wait=2, capture=True)]),
	Test('// 0x03 ETX', [Step('ETX', [0x03], wait=2, capture=True)]),
	Test('// 0x04 EOT', [Step('EOT', [0x04], wait=2, capture=True)]),
	Test('// 0x05 ENQ', [Step('ENQ', [0x05], wait=2, capture=True)]),
	Test('// 0x06 ACK', [Step('ACK', [0x06], wait=2, capture=True)]),
	# --- 0x07 BEL ---
	Test('// 0x07 BEL', [Step('BEL', [0x07], wait=2, capture=True)]),
	# --- 0x08 BS - Backspace ---
	Test('// 0x08 BS - Backspace', [
		Step('print',	    list(b'TEST BACKSPACE'), wait=4),
		Step('CR+LF',	    [0x0D, 0x0A], wait=1),
		Step('print ABC',	[0x41, 0x42, 0x43], wait=2),
		Step('BS',	        [0x08], wait=1, capture=True),
		Step('print D',	    [0x44], wait=1),
		Step('2x CR+LF',	[0x0D, 0x0A, 0x0D, 0x0A], wait=1),
	]),
	# --- 0x0A LF - Line Feed (all LF/CR interaction tests) ---
	Test('// 0x0A LF - Line Feed', [
		Step('print',	    list(b'TEST LINE FEED'), wait=2),
		Step('CR+LF',	    [0x0D, 0x0A], wait=2),
		# Single LF
		Step('print',	    list(b'SINGLE LF'), wait=2),
		Step('LF',	        [0x0A], wait=2, capture=True, comment='// Single LF, does carriage stay at current col?'),
		Step('print',	    list(b'SINGLE LF'), wait=2),
		Step('CR+LF',	    [0x0D, 0x0A], wait=1),
		# Double LF
		Step('print',	    list(b'DOUBLE LF'), wait=2),
		Step('2x LF',	    [0x0A, 0x0A], wait=2, capture=True, comment='// Two consecutive LFs'),
		Step('print',	    list(b'DOUBLE LF'), wait=2),
		Step('CR+LF',	    [0x0D, 0x0A], wait=1),
		# CR then LF
		Step('move to col 15',	[0x1B, 0x09, 15], wait=2),
		Step('print',	        list(b'CR THEN LF'), wait=2),
		Step('CR then LF',	    [0x0D, 0x0A], wait=2, capture=True, comment='// CR first, then LF — standard newline order'),
		Step('print',	        list(b'CR THEN LF'), wait=2),
		Step('CR+LF',	        [0x0D, 0x0A], wait=1),
		# LF then CR
		Step('move to col 15',	[0x1B, 0x09, 15], wait=2),
		Step('print',	        list(b'LF THEN CR'), wait=2),
		Step('LF then CR',	    [0x0A, 0x0D], wait=2, capture=True, comment='// LF first, then CR — reversed order'),
		Step('print',	        list(b'LF THEN CR'), wait=2),
		Step('CR+LF',	        [0x0D, 0x0A], wait=1),
		# LF without CR from mid-line
		Step('move to col 17',	[0x1B, 0x09, 17], wait=2),
		Step('print',	        list(b'LF NO CR'), wait=2),
		Step('LF only',	        [0x0A], wait=2, capture=True, comment='// LF from col 17, no CR — does col persist?'),
		Step('print',	        list(b'LF NO CR'), wait=2),
		Step('CR+LF',	        [0x0D, 0x0A], wait=1),
		# Multiple CR without LF
		Step('move to col 17',	[0x1B, 0x09, 17], wait=2),
		Step('print',	        list(b'MULTI CR'), wait=2),
		Step('3x CR',	        [0x0D, 0x0D, 0x0D], wait=2, capture=True, comment='// Three CRs, no LF — should stay on same line'),
		Step('print',	        list(b'MULTI CR'), wait=2),
		Step('CR+LF',	        [0x0D, 0x0A], wait=1),
		Step('2x CR+LF',	    [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
	]),
	# --- 0x0D CR - Carriage Return ---
	Test('// 0x0D CR - Carriage Return', [
		Step('print',	    list(b'TEST CARRIAGE RETURN'), wait=2),
		Step('CR+LF',	    [0x0D, 0x0A], wait=2),
		Step('print',	    list(b'I        RETURN'), wait=2),
		Step('CR',	        [0x0D], wait=2, capture=True),
		Step('print',	    list(b'CARRIAGE'), wait=2),
		Step('2x CR+LF',	[0x0D, 0x0A, 0x0D, 0x0A], wait=1),
	]),
	# --- 0x0E-0x1F: misc control codes ---
	Test('// 0x0E SO - Shift Out', [Step('SO', [0x0E], wait=2, capture=True)]),
	Test('// 0x0F SI - Shift In', [Step('SI', [0x0F], wait=2, capture=True)]),
	Test('// 0x10 DLE', [Step('DLE', [0x10], wait=2, capture=True)]),
	# DC1 is SELECT — deselect first so we can see the select response
	Test('// 0x11 DC1 - SELECT', [
		Step('deselect',	[0x13], wait=2),
		Step('DC1',	        [0x11], wait=4, capture=True),
	]),
	Test('// 0x12 DC2', [Step('DC2', [0x12], wait=2, capture=True)]),
	# DC3 is DESELECT
	Test('// 0x13 DC3 - DESELECT', [
		Step('DC3',	        [0x13], wait=4, capture=True),
		Step('reselect',	[0x11], wait=2),
	]),
	Test('// 0x14 DC4', [Step('DC4', [0x14], wait=2, capture=True)]),
	Test('// 0x15 NAK', [Step('NAK', [0x15], wait=2, capture=True)]),
	Test('// 0x16 SYN', [Step('SYN', [0x16], wait=2, capture=True)]),
	Test('// 0x17 ETB', [Step('ETB', [0x17], wait=2, capture=True)]),
	Test('// 0x18 CAN', [Step('CAN', [0x18], wait=2, capture=True)]),
	Test('// 0x19 EM',  [Step('EM', [0x19], wait=2, capture=True)]),
	Test('// 0x1A SUB', [Step('SUB', [0x1A], wait=2, capture=True)]),
	# ESC alone: consume with ESC+S (reset to DIP)
	Test('// 0x1B ESC (alone)', [
		Step('ESC',	[0x1B], wait=6, capture=True),
		Step('consume ESC',	[0x53], wait=1),
	]),
	Test('// 0x1C FS', [Step('FS', [0x1C], wait=2, capture=True)]),
	Test('// 0x1D GS', [Step('GS', [0x1D], wait=2, capture=True)]),
	Test('// 0x1E RS', [Step('RS', [0x1E], wait=2, capture=True)]),
	Test('// 0x1F US', [Step('US', [0x1F], wait=2, capture=True)]),
]

# =============================================================================
# 2. PRINTABLE CHARACTERS 0x20-0x7F
# =============================================================================
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
	COL_START = 3
	COL_SPACING = 2
	ROW_LABEL_COL = 1

	steps = [
		Step('print',	list(b'TEST PRINTABLE CHARS'), wait=4),
		Step('CR+LF',	[0x0D, 0x0A], wait=2),
	]

	# Header row: 0 1 2 3 4 5 6 7 8 9 A B C D E F
	header_labels = '0123456789ABCDEF'
	for col_idx, label in enumerate(header_labels):
		col = COL_START + col_idx * COL_SPACING
		steps.append(Step(f'move to col {col}', [0x1B, 0x09, col], wait=0.2))
		steps.append(Step(f'print "{label}"', [ord(label)], wait=0.2))
	steps.append(Step('CR+LF', [0x0D, 0x0A], wait=4))

	# Data rows: 0x20-0x7F, 16 chars per row
	for row in range(2, 8):
		steps.append(Step(f'move to col {ROW_LABEL_COL}', [0x1B, 0x09, ROW_LABEL_COL], wait=0))
		steps.append(Step(f'print row "{row}"', [ord(str(row))], wait=0))
		for col_idx in range(16):
			code = row * 16 + col_idx
			col = COL_START + col_idx * COL_SPACING
			steps.append(Step(f'move to col {col}', [0x1B, 0x09, col], wait=0.2))
			if code == 0x7F:
				label = '0x7F DEL'
			else:
				label = f'0x{code:02X} "{chr(code)}"'
			steps.append(Step(label, [code], wait=0.2, capture=True))
			steps.append(Step('space', [0x20], wait=0))
		steps.append(Step('CR+LF', [0x0D, 0x0A], wait=2))

	steps.append(Step('2x CR+LF', [0x0D, 0x0A, 0x0D, 0x0A], wait=1))

	return Test('// Printable Characters 0x20 - 0x7F', steps)

TESTS_PRINTABLE = [
	_build_printable_test(),
]

# =============================================================================
# 3. MOVEMENT
#    - ESC + HT + n: Absolute horizontal movement
#    - ESC + VT + n: Absolute vertical movement
#    - ESC + LF: Reverse paper feed (one VMI unit up)
#    - ESC + U: Half LF (subscript, 1/12 inch down)
#    - ESC + D: Reverse half LF (superscript, 1/12 inch up)
# =============================================================================
TESTS_MOVEMENT = [
	Test('// ESC + HT + n: Absolute HT Movement', [
		Step('print',	        list(b'TEST ABSOLUTE HT MOVEMENT'), wait=4),
		Step('CR+LF',	        [0x0D, 0x0A], wait=2),
		Step('reset printer',	[0x1B, 0x0D, 0x50], wait=2),
		Step('print "HT1"',	    [0x48, 0x54, 0x31], wait=2),
		Step('abs HT to col 1',	[0x1B, 0x09, 1], wait=4),
		Step('abs HT to col 17',[0x1B, 0x09, 17], wait=6, capture=True, comment='// Move right HT1 to HT17'),
		Step('print "HT17"',	[0x48, 0x54, 0x31, 0x37], wait=2),
		Step('abs HT to col 17',[0x1B, 0x09, 17], wait=6),
		Step('abs HT to col 17',[0x1B, 0x09, 17], wait=6, capture=True, comment='// Already at target HT17'),
		Step('abs HT to col 9',	[0x1B, 0x09, 9], wait=6, capture=True, comment='// Move left HT17 to HT9'),
		Step('print "HT9"',	    [0x48, 0x54, 0x39], wait=2),
		Step('2x CR+LF',	    [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
	]),
	Test('// ESC + VT + n: Absolute VT Movement', [
		Step('print',	list(b'TEST ABSOLUTE VT MOVEMENT'), wait=4),
		Step('CR+LF',	[0x0D, 0x0A], wait=2),
		Step('reset printer',	[0x1B, 0x0D, 0x50], wait=2),
		Step('print "VT1"',	[0x56, 0x54, 0x31], wait=2),
		Step('abs VT to line 3',	[0x1B, 0x0B, 3], wait=6, capture=True, comment='// Move down VT1 to VT3'),
		Step('print "VT3"',	[0x56, 0x54, 0x33], wait=2),
		Step('abs VT to line 3',	[0x1B, 0x0B, 3], wait=6, capture=True, comment='// Already at target'),
		Step('abs VT to line 2',	[0x1B, 0x0B, 2], wait=6, capture=True, comment='// Move up VT3 to VT2'),
		Step('print "VT2"',	[0x56, 0x54, 0x32], wait=2),
		Step('2x CR+LF',	[0x0D, 0x0A, 0x0D, 0x0A], wait=1),
	]),
	Test('// ESC + LF / ESC + U / ESC + D: Paper micro-movement', [
		# --- Reverse LF ---
		Step('print',	list(b'TEST REVERSE LF + SUB/SUPERSCRIPT'), wait=4),
		Step('CR+LF',	[0x0D, 0x0A], wait=2),
		Step('LF first',	[0x0A], wait=1),
		Step('print LF1',	[0x4C, 0x46, 0x31], wait=2),
		Step('reverse LF',	[0x1B, 0x0A], wait=6, capture=True, comment='// ESC+LF Reverse LF — one VMI unit up'),
		Step('print LF0',	[0x4C, 0x46, 0x30], wait=2),
		Step('CR+LF',	[0x0D, 0x0A], wait=2),
		# --- Half LF (subscript) ---
		Step('print NRM',	list(b'NRM'), wait=2),
		Step('half LF',	[0x1B, 0x55], wait=6, capture=True, comment='// ESC+U half LF down (subscript)'),
		Step('print SUB',	list(b'SUB'), wait=2),
		Step('undo: reverse',	[0x1B, 0x44], wait=2),
		Step('CR+LF',	[0x0D, 0x0A], wait=1),
		# --- Reverse half LF (superscript) ---
		Step('print NRM',	list(b'NRM'), wait=2),
		Step('reverse half LF',	[0x1B, 0x44], wait=6, capture=True, comment='// ESC+D reverse half LF up (superscript)'),
		Step('print SUP',	list(b'SUP'), wait=2),
		Step('undo: down',	[0x1B, 0x55], wait=2),
		Step('2x CR+LF',	[0x0D, 0x0A, 0x0D, 0x0A], wait=1),
	], compatible_series=["C"]),
]

# =============================================================================
# 4. SPACING
#    - ESC + US + n: Set HMI (horizontal motion index / pitch)
#    - ESC + RS + n: Set VMI (vertical motion index / line spacing)
# =============================================================================
TESTS_SPACING = [
	Test('// ESC + US + n / ESC + RS + n: HMI and VMI', [
		Step('print',	            list(b'TEST HMI AND VMI'), wait=2),
		Step('LF+CR',	            [0x0A, 0x0D], wait=1),
		# --- HMI: 15cpi (tightest) ---
		Step('set HMI 9 (15cpi)',	[0x1B, 0x1F, 9], wait=2, capture=True, comment='// HMI=9, 15cpi — narrowest pitch'),
		Step('print PITCH15',	    [0x50, 0x49, 0x54, 0x43, 0x48, 0x31, 0x35], wait=2),
		Step('LF+CR',	            [0x0A, 0x0D], wait=1),
		# --- HMI: 12cpi (medium) ---
		Step('set HMI 11 (12cpi)',	[0x1B, 0x1F, 11], wait=2, capture=True, comment='// HMI=11, 12cpi — medium pitch'),
		Step('print PITCH12',	    [0x50, 0x49, 0x54, 0x43, 0x48, 0x31, 0x32], wait=2),
		Step('LF+CR',	            [0x0A, 0x0D], wait=1),
		# --- HMI: 10cpi (widest) ---
		Step('set HMI 13 (10cpi)',	[0x1B, 0x1F, 13], wait=2, capture=True, comment='// HMI=13, 10cpi — widest pitch'),
		Step('print PITCH10',	    [0x50, 0x49, 0x54, 0x43, 0x48, 0x31, 0x30], wait=2),
		Step('LF+CR',	            [0x0A, 0x0D], wait=1),
		# --- HMI: idempotent check ---
		Step('set HMI 13 again',	[0x1B, 0x1F, 13], wait=2, capture=True, comment='// Send again when HMI already 13 — is response identical?'),
		Step('LF+CR',	            [0x0A, 0x0D], wait=1),
		# --- VMI: tightest ---
		Step('reset to DIP',	    [0x1B, 0x53], wait=1),
		Step('print',	            list(b'VMI START'), wait=1),
		Step('LF+CR',	            [0x0A, 0x0D], wait=1),
		Step('set VMI 9',	        [0x1B, 0x1E, 9], wait=2, capture=True, comment='// VMI=9 — tightest line spacing'),
		Step('print',	            list(b'VMI9'), wait=1),
		Step('LF+CR',	            [0x0A, 0x0D], wait=1),
		# --- VMI: default 1/6 inch ---
		Step('set VMI 13',	        [0x1B, 0x1E, 13], wait=2, capture=True, comment='// VMI=13 — default 1/6 inch'),
		Step('print',	            list(b'VMI13'), wait=1),
		Step('LF+CR',	            [0x0A, 0x0D], wait=1),
		# --- VMI: wider ---
		Step('set VMI 17',	        [0x1B, 0x1E, 17], wait=2, capture=True, comment='// VMI=17 — wider spacing'),
		Step('print',	            list(b'VMI17'), wait=1),
		Step('LF+CR',	            [0x0A, 0x0D], wait=1),
		# --- VMI: widest ---
		Step('set VMI 25',	        [0x1B, 0x1E, 25], wait=2, capture=True, comment='// VMI=25 — widest spacing'),
		Step('print',	            list(b'VMI25'), wait=1),
		Step('LF+CR',	            [0x0A, 0x0D], wait=1),
		# --- Cleanup ---
		Step('reset to DIP',	    [0x1B, 0x53], wait=1),
		Step('2x CR+LF',	        [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
	], compatible_series=["C"]),
]

# =============================================================================
# 5. PRINT MODES
#    - ESC + E / ESC + R: Underline on/off
#    - ESC + O: Bold, ESC + W: Shadow, ESC + F: Double-strike
#    - ESC + &: Clear bold/shadow/double-strike
#    - ESC + X: Clear underline/shadow/double-strike
#    - ESC + / / ESC + \: Backward print on/off
#    - ESC + " / ESC + #: Auto LF on/off
# =============================================================================
TESTS_MODES = [
	Test('// Underline: ESC + E (on) / ESC + R (off)', [
		Step('print',	            list(b'TEST UNDERLINE ON OFF'), wait=2),
		Step('CR+LF',	            [0x0D, 0x0A], wait=1),
		# Enable underline
		Step('print',	            list(b'NORMAL '), wait=2),
		Step('enable underline',	[0x1B, 0x45], wait=1, capture=True, comment='// Enable auto underline'),
		Step('print',	            list(b'UNDERLINED '), wait=2),
		# Disable underline
		Step('clear underline',	    [0x1B, 0x52], wait=2, capture=True, comment='// Clear auto underline'),
		Step('print',	            list(b'NORMAL'), wait=1),
		Step('2x CR+LF',	        [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
	]),
	Test('// Bold / Shadow / Double-Strike / Clear: ESC + O, W, F, &, X', [
		Step('print',	            list(b'TEST BOLD SHADOW DBLSTRIKE'), wait=2),
		Step('CR+LF',	            [0x0D, 0x0A], wait=1),
		# --- Bold ---
		Step('print',	            list(b'NORMAL '), wait=2),
		Step('enable bold',	        [0x1B, 0x4F], wait=2, capture=True, comment='// Enable bold print'),
		Step('print',	            list(b'BOLD '), wait=1),
		Step('clear bold',	        [0x1B, 0x26], wait=1),
		Step('CR+LF',	            [0x0D, 0x0A], wait=1),
		# --- Shadow ---
		Step('print',	            list(b'NORMAL '), wait=1),
		Step('enable shadow',	    [0x1B, 0x57], wait=2, capture=True, comment='// Enable shadow print'),
		Step('print',	            list(b'SHADOW '), wait=1),
		Step('clear shadow',	    [0x1B, 0x26], wait=1),
		Step('CR+LF',	            [0x0D, 0x0A], wait=1),
		# --- Double-strike ---
		Step('print',	            list(b'NORMAL '), wait=2),
		Step('enable dbl strike',	[0x1B, 0x46], wait=2, capture=True, comment='// Enable double-strike print'),
		Step('print',	            list(b'DBLSTRIKE '), wait=1),
		Step('clear dbl strike',	[0x1B, 0x26], wait=1),
		Step('CR+LF',	            [0x0D, 0x0A], wait=1),
		# --- ESC + &: Clear bold + shadow + double-strike ---
		Step('enable bold',	        [0x1B, 0x4F], wait=0),
		Step('enable shadow',	    [0x1B, 0x57], wait=0),
		Step('enable dbl strike',	[0x1B, 0x46], wait=0),
		Step('print',	            list(b'ALL ON '), wait=2),
		Step('clear B/S/DS',	    [0x1B, 0x26], wait=2, capture=True, comment='// ESC+& clear bold + shadow + double-strike'),
		Step('print',	            list(b'ALL OFF '), wait=1),
		Step('clear B/S/DS again',	[0x1B, 0x26], wait=2, capture=True, comment='// ESC+& when none set — is response identical?'),
		Step('CR+LF',	            [0x0D, 0x0A], wait=1),
		# --- ESC + X: Clear underline + shadow + double-strike ---
		Step('enable underline',	[0x1B, 0x45], wait=0),
		Step('enable shadow',	    [0x1B, 0x57], wait=0),
		Step('enable dbl strike',	[0x1B, 0x46], wait=0),
		Step('print',	            list(b'ALL ON '), wait=2),
		Step('clear U/S/DS',	    [0x1B, 0x58], wait=2, capture=True, comment='// ESC+X clear underline + shadow + double-strike'),
		Step('print',	            list(b'ALL OFF '), wait=2),
		Step('clear U/S/DS again',	[0x1B, 0x58], wait=2, capture=True, comment='// ESC+X when none set — is response identical?'),
		Step('2x CR+LF',	        [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
	]),
	Test('// Backward Print: ESC + / (on) / ESC + \\\\ (off)', [
		Step('print',	            list(b'TEST BACKWARD PRINT ON OFF'), wait=2),
		Step('CR+LF',	            [0x0D, 0x0A], wait=1),
		# Enable backward print
		Step('print',	            list(b'FORWARD '), wait=2),
		Step('move to col 40',	    [0x1B, 0x09, 40], wait=2),
		Step('enable backward',	    [0x1B, 0x2F], wait=4, capture=True, comment='// Enable auto backward print'),
		Step('print',	            list(b'BACKWARD'), wait=2),
		# CR in backward mode
		Step('CR',	                [0x0D], wait=4, capture=True, comment='// CR in backward mode — should go to left margin and switch to forward'),
		Step('print',	            list(b'        AFTER CR'), wait=1),
		Step('CR+LF',	            [0x0D, 0x0A], wait=1),
		# Clear backward, verify forward resumes
		Step('move to col 40',	    [0x1B, 0x09, 40], wait=2),
		Step('enable backward',	    [0x1B, 0x2F], wait=1),
		Step('print',	            list(b'BACKWARD '), wait=4),
		Step('clear backward',	    [0x1B, 0x5C], wait=6, capture=True, comment='// Clear auto backward print — should resume forward'),
		Step('print',	            list(b'FORWARD'), wait=4),
		Step('2x CR+LF',	        [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
	], compatible_series=["C"]),
	Test('// Auto LF: ESC + " (on) / ESC + # (off)', [
		Step('print',	list(b'TEST AUTO LF ON/OFF'), wait=4),
		Step('CR+LF',	[0x0D, 0x0A], wait=2),
		# --- Auto LF ON ---
		Step('auto LF on',	[0x1B, 0x22], wait=2, capture=True, comment='// Enable auto LF — CR now includes LF'),
		Step('print',	    list(b'LINE A'), wait=4),
		Step('CR',	        [0x0D], wait=2, capture=True, comment='// CR with auto LF — should advance to next line'),
		Step('print',	    list(b'LINE B'), wait=4),
		Step('CR+LF',	    [0x0D], wait=2, capture=True, comment='// CR+LF with auto LF: What happens to the LF?'),
		Step('print',	    list(b'LINE C/D?'), wait=2),
		Step('CR+LF',	    [0x0D, 0x0A], wait=1),
		# --- Auto LF OFF ---
		Step('auto LF off',	[0x1B, 0x23], wait=6, capture=True, comment='// Disable auto LF — CR should no longer advance line'),
		Step('print',	    list(b'SAME LINE'), wait=4),
		Step('CR',	        [0x0D], wait=6, capture=True, comment='// CR without auto LF — should stay on same line'),
		Step('print',	    list(b'OVERWRITE'), wait=4),
		# --- Cleanup ---
		Step('reset to DIP',[0x1B, 0x53], wait=1),
		Step('2x CR+LF',	[0x0D, 0x0A, 0x0D, 0x0A], wait=1),
	]),
]

TESTS_NEWLINE_QUIRKS = [
    Test('// Newline Quirks: Look-ahead buffer and swallowed CR testing', [
        Step('print',          list(b'TEST NEWLINE QUIRKS'), wait=2),
        Step('CR+LF',          [0x0D, 0x0A], wait=2),

        # === BASELINE: Standalone CR from mid-line ===
        Step('move to col 15', [0x1B, 0x09, 15], wait=2),
        Step('print',          list(b'CR ALONE (WAIT)'), wait=2),
        Step('CR alone',       [0x0D], wait=10.0, capture=True, comment='// Standalone CR from mid-line with 10s wait — baseline: does IF60 hold CR pending or emit immediately?'),
        Step('print',          list(b'AFTER CR'), wait=2),
        Step('CR+LF',          [0x0D, 0x0A], wait=1),

        # === LF+CR from col 1 (original test, replicated) ===
        Step('print',          list(b'LF CR COL1'), wait=2),
        Step('CR',             [0x0D], wait=1),  # ensure at col 1
        Step('LF then CR c1',  [0x0A, 0x0D], wait=10.0, capture=True, comment='// LF+CR from col 1 — is CR swallowed because already at margin?'),
        Step('print',          list(b'AFTER'), wait=2),
        Step('CR+LF',          [0x0D, 0x0A], wait=1),

        # === LF+CR from mid-line (the key disambiguation) ===
        Step('move to col 15', [0x1B, 0x09, 15], wait=2),
        Step('print',          list(b'LF CR MID'), wait=2),
        Step('LF then CR mid', [0x0A, 0x0D], wait=10.0, capture=True, comment='// LF+CR from mid-line — if CR appears: position-based suppression. If swallowed: lookahead mechanism.'),
        Step('print',          list(b'AFTER'), wait=2),
        Step('CR+LF',          [0x0D, 0x0A], wait=1),

        # === LF ... wait ... CR (timing separation) ===
        Step('move to col 15', [0x1B, 0x09, 15], wait=2),
        Step('print',          list(b'LF WAIT CR'), wait=2),
        Step('LF only',        [0x0A], wait=5.0, capture=True, comment='// LF alone — establish baseline, wait for any pending output'),
        Step('CR after wait',  [0x0D], wait=5.0, capture=True, comment='// CR sent 5s after LF — if CR appears: lookahead has timeout. If swallowed: not time-based.'),
        Step('print',          list(b'AFTER'), wait=2),
        Step('CR+LF',          [0x0D, 0x0A], wait=1),

        # === LF+CR+printable (flush theory) ===
        Step('move to col 15', [0x1B, 0x09, 15], wait=2),
        Step('print',          list(b'LF CR SPACE'), wait=2),
        Step('LF CR Space',    [0x0A, 0x0D, 0x20], wait=2, capture=True, comment='// LF+CR+Space — does printable char flush the swallowed CR?'),
        Step('print',          list(b'AFTER SPACE'), wait=2),
        Step('CR+LF',          [0x0D, 0x0A], wait=1),

        # === Double CR (clean pair) ===
        Step('move to col 15', [0x1B, 0x09, 15], wait=2),
        Step('print',          list(b'DOUBLE CR'), wait=2),
        Step('2x CR',          [0x0D, 0x0D], wait=5.0, capture=True, comment='// 2x CR from mid-line — both emitted, or second swallowed?'),
        Step('print',          list(b'AFTER'), wait=2),
        Step('CR+LF',          [0x0D, 0x0A], wait=1),

        # === Triple CR + printable (your existing test) ===
        Step('move to col 15', [0x1B, 0x09, 15], wait=2),
        Step('print',          list(b'TRIPLE CR FLUSH'), wait=2),
        Step('3x CR + Char',   [0x0D, 0x0D, 0x0D, 0x58], wait=3, capture=True, comment='// 3x CR + "X" — does printable char prevent swallowing?'),
        Step('CR+LF',          [0x0D, 0x0A], wait=1),

        # === CR+LF coalescing verification ===
        Step('move to col 15', [0x1B, 0x09, 15], wait=2),
        Step('print',          list(b'CR LF MERGE'), wait=2),
        Step('CR then LF',     [0x0D, 0x0A], wait=2, capture=True, comment='// CR+LF from mid-line — expect [0x02, 0xB1] (coalesced)'),
        Step('print',          list(b'AFTER'), wait=2),
        Step('CR+LF',          [0x0D, 0x0A], wait=1),

        # --- Cleanup ---
        Step('2x CR+LF',       [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
    ]),
]

# =============================================================================
# 6. HORIZONTAL MARGINS
#    - ESC + 9: Set left margin
#    - ESC + 0: Set right margin
#    - ESC + B: Clear left and right margins
#    No significant paper feed — safe to run mid-page.
# =============================================================================
TESTS_MARGINS_H = [
	Test('// Left / Right Margin: ESC + 9, ESC + 0, ESC + B', [
		Step('print',	        list(b'TEST LEFT RIGHT MARGINS'), wait=4),
		Step('CR+LF',	        [0x0D, 0x0A], wait=2),
		# --- Left margin at col 17 ---
		Step('move to col 17',	[0x1B, 0x09, 17], wait=2),
		Step('set left margin',	[0x1B, 0x39], wait=2, capture=True, comment='// Set left margin at col 17'),
		Step('print',	        list(b'I    MARGIN'), wait=2),
		Step('CR',	            [0x0D], wait=1, capture=True, comment='// CR — should return to col 17 (left margin)'),
		Step('print',	        list(b'LEFT'), wait=1),
		Step('CR+LF',	        [0x0D, 0x0A], wait=1),
		# --- Right margin at col 33 ---
		Step('reset printer',	    [0x1B, 0x0D, 0x50], wait=1),
		Step('move to col 21',	[0x1B, 0x09, 21], wait=2),
		Step('print',	        list(b'RIGHT      I'), wait=4),
		Step('set right margin',[0x1B, 0x30], wait=2, capture=True, comment='// Set right margin at col 33'),
		Step('CR',	            [0x0D], wait=1, capture=True, comment='// CR — should return to col 1'),
		Step('move to col 27',	[0x1B, 0x09, 27], wait=1),
		Step('print',	        list(b'MARGIN'), wait=2),
		Step('try over RM',	    list(b'    '), wait=2, capture=True, comment='// Try writing past right margin'),
		Step('CR+LF',	        [0x0D, 0x0A], wait=1),
		# --- Cleanup ---
		Step('reset printer',	    [0x1B, 0x0D, 0x50], wait=1),
		Step('2x CR+LF',	    [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
	]),
]

# =============================================================================
# 7. TAB STOPS
#    - 0x09 HT: Horizontal tab (moved from control codes)
#    - 0x0B VT: Vertical tab (moved from control codes)
#    - ESC + 1: Set HT/VT at current position
#    - ESC + 8: Clear HT at current position
#    - ESC + -: Set VT at current position
#    - ESC + 2: Clear all HT and VT
# =============================================================================
TESTS_TABS = [
	Test('// HT Stops: 0x09 HT, ESC + 1 (set), ESC + 8 (clear one), ESC + 2 (clear all)', [
		Step('print',	        list(b'TEST HT STOPS'), wait=2),
		Step('CR+LF',	        [0x0D, 0x0A], wait=1),
		# --- Set HT at col 17, test 0x09 HT ---
		Step('move to col 17',	[0x1B, 0x09, 17], wait=2),
		Step('set HT stop',	    [0x1B, 0x31], wait=2, capture=True, comment='// ESC+1 set HT stop at col 17'),
		Step('print',	        list(b'I  TAB'), wait=4),
		Step('CR',	            [0x0D], wait=1),
		Step('print',	        list(b'HT1'), wait=4),
		Step('HT',	            [0x09], wait=6, capture=True, comment='// 0x09 HT — should jump to col 17'),
		Step('print',	        list(b'HT'), wait=2),
		Step('CR+LF',	        [0x0D, 0x0A], wait=1),
		# --- Set second HT at col 32 ---
		Step('clear all HT/VT',	[0x1B, 0x32], wait=1),
		Step('move to col 16',	[0x1B, 0x09, 16], wait=2),
		Step('set HT stop',	    [0x1B, 0x31], wait=0),
		Step('move to col 32',	[0x1B, 0x09, 32], wait=2),
		Step('set HT stop',	    [0x1B, 0x31], wait=0),
		# --- Clear HT at col 16 only ---
		Step('move to col 16',	[0x1B, 0x09, 16], wait=2),
		Step('clear HT at pos',	[0x1B, 0x38], wait=2, capture=True, comment='// ESC+8 clear HT at col 16, col 32 remains'),
		Step('CR',	            [0x0D], wait=1),
		Step('print',	        list(b'HT1'), wait=2),
		Step('HT',	            [0x09], wait=2, capture=True, comment='// HT should skip cleared col 16, jump to col 32'),
		Step('print',	        list(b'HT'), wait=2),
		Step('CR+LF',	        [0x0D, 0x0A], wait=1),
		# --- Clear all HT/VT ---
		Step('clear all HT/VT',	[0x1B, 0x32], wait=2, capture=True, comment='// ESC+2 clear all HT and VT stops'),
		Step('CR',	            [0x0D], wait=1),
		Step('print',	        list(b'HT1 '), wait=1),
		Step('HT',	            [0x09], wait=2, capture=True, comment='// HT with no stops — what happens?'),
		Step('print',	        list(b'HT?'), wait=1),
		Step('2x CR+LF',	    [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
	]),
	Test('// VT Stops: 0x0B VT, ESC + - (set)', [
		Step('print',	            list(b'TEST VT STOPS'), wait=2),
		Step('CR+LF',	            [0x0D, 0x0A], wait=1),
		# --- Set VT stop at line 3 down from here ---
		Step('move down 3',	        [0x0A, 0x0A, 0x0A], wait=2),
		Step('set VT stop (ESC+1)',	[0x1B, 0x31], wait=1, capture=True, comment='// ESC+1 set VT stop 3 lines down'),
		Step('print',	            list(b'I  TAB'), wait=4),
		Step('CR',	                [0x0D], wait=1),
		Step('reverse LF x3',	    [0x1B, 0x0A, 0x1B, 0x0A, 0x1B, 0x0A], wait=2),
		Step('print',	            list(b'VT1'), wait=4),
		Step('VT',	                [0x0B], wait=6, capture=True, comment='// 0x0B VT — should jump to VT stop'),
		Step('print',	            list(b'VT'), wait=2),
		Step('clear all HT/VT',	    [0x1B, 0x32], wait=1),
		Step('CR+LF',	            [0x0D, 0x0A], wait=1),
		# --- Set VT via ESC + - at line 8 ---
		Step('abs VT to line 8',	[0x1B, 0x0B, 8], wait=2),
		Step('set VT stop (ESC+-)',	[0x1B, 0x2D], wait=2, capture=True, comment='// ESC+- set VT stop at line 8'),
		Step('print',	            list(b'  VSTOP'), wait=2),
		Step('reset printer',	    [0x1B, 0x0D, 0x50], wait=1),
		Step('print',	            list(b'VT1'), wait=2),
		Step('VT',	                [0x0B], wait=2, capture=True, comment='// VT should jump to line 8'),
		Step('print',	            list(b'VT'), wait=1),
		# --- Cleanup ---
		Step('clear all HT/VT',	    [0x1B, 0x32], wait=1),
		Step('2x CR+LF',	        [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
	], compatible_series=["C"]),
]

# =============================================================================
# 8. SPECIAL
#    - ESC + Y: Print 0x20 (space) glyph
#    - ESC + Z: Print 0x7F (DEL) glyph
#    - ESC + S: Reset to DIP switch defaults
# =============================================================================
TESTS_SPECIAL = [
	Test('// ESC + Y / ESC + Z: Print space and DEL glyphs', [
		Step('print',	        list(b'TEST PRINT SPACE/DEL GLYPHS'), wait=2),
		Step('CR+LF',	        [0x0D, 0x0A], wait=1),
		# Space glyph
		Step('print',	        list(b'I'), wait=1),
		Step('print space char',[0x1B, 0x59], wait=2, capture=True, comment='// ESC+Y prints the physical glyph for 0x20'),
		Step('print',	        list(b'I '), wait=1),
		# DEL glyph
		Step('print',	        list(b'I'), wait=1),
		Step('print DEL char',	[0x1B, 0x5A], wait=2, capture=True, comment='// ESC+Z prints the physical glyph for 0x7F'),
		Step('print',	        list(b'I'), wait=1),
		Step('2x CR+LF',	    [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
	]),
	Test('// ESC + S: Reset to DIP switch defaults', [
		Step('print',	        list(b'TEST RESET TO DIP'), wait=4),
		Step('CR+LF',	        [0x0D, 0x0A], wait=2),
		# Change several settings away from defaults
		Step('set HMI 9',	    [0x1B, 0x1F, 9], wait=1),
		Step('enable underline',[0x1B, 0x45], wait=1),
		Step('enable bold',	    [0x1B, 0x4F], wait=1),
		Step('print',	        list(b'MODIFIED '), wait=4),
		# Reset
		Step('reset to DIP',	[0x1B, 0x53], wait=6, capture=True, comment='// Reset all settings to DIP switch defaults'),
		Step('print',	        list(b'DEFAULT'), wait=4),
		Step('2x CR+LF',	    [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
	]),
	Test('// ESC + CR + P: Reset printer', [
		Step('print',	        list(b'TEST RESET PRINTER'), wait=4),
		Step('CR+LF',	        [0x0D, 0x0A], wait=2),
		# Reset from home
		Step('reset from home',	[0x1B, 0x0D, 0x50], wait=6, capture=True, comment='// Reset at col 1'),
		Step('CR+LF',	        [0x0D, 0x0A], wait=1),
		# Reset from mid-line
		Step('move to col 9',	[0x1B, 0x09, 9], wait=2),
		Step('reset from col 9',[0x1B, 0x0D, 0x50], wait=6, capture=True, comment='// Reset at col 9, does carriage return home?'),
		Step('2x CR+LF',	    [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
	]),
]

# =============================================================================
# 9. PAPER FEED (grouped last — these move paper significantly)
#    - 0x0C FF: Form feed
#    - ESC + FF + n: Set page length
#    - ESC + T: Set top margin
#    - ESC + L: Set bottom margin
#    - ESC + C: Clear top and bottom margins
#    - ESC + CR + P: Reset printer
# =============================================================================
TESTS_PAPERFEED = [
	Test('// Top / Bottom Margins: ESC + T, ESC + L, ESC + C', [
		Step('print',	            list(b'TEST TOP/BOTTOM MARGINS'), wait=2),
		Step('CR+LF',	            [0x0D, 0x0A], wait=2),
		# --- Top margin at line 8 ---
		Step('reset printer',	    [0x1B, 0x0D, 0x50], wait=2),
		Step('abs VT to line 8',    [0x1B, 0x0B, 8], wait=2),
		Step('set top margin',	    [0x1B, 0x54], wait=2, capture=True, comment='// Set top margin at line 8'),
		Step('print',	            list(b'TOP MARGIN'), wait=2),
		Step('CR+LF',	            [0x0D, 0x0A], wait=1),
		# Try moving into top margin area
		Step('abs VT to line 1',    [0x1B, 0x0B, 1], wait=2, capture=True, comment='// Move into top margin area — manual says VT/reverse LF allowed'),
		Step('print',	            list(b'LINE1?'), wait=2),
		Step('CR+LF',	            [0x0D, 0x0A], wait=1),
		# --- Bottom margin at line 16 ---
		Step('clear margins',	    [0x1B, 0x43], wait=1),
		Step('reset printer',	    [0x1B, 0x0D, 0x50], wait=1),
		Step('abs VT to line 16',	[0x1B, 0x0B, 16], wait=2),
		Step('set bottom margin',	[0x1B, 0x4C], wait=2, capture=True, comment='// Set bottom margin at line 16'),
		Step('print',	            list(b'BOTTOM MARGIN'), wait=2),
		Step('CR',	                [0x0D], wait=1),
		Step('LF at margin',	    [0x0A], wait=6, capture=True, comment='// LF at bottom margin — should feed to next page top'),
		Step('print',	            list(b'AFTER LF'), wait=2),
		Step('CR+LF',	            [0x0D, 0x0A], wait=1),
		# --- Clear both margins ---
		Step('reset printer',	    [0x1B, 0x0D, 0x50], wait=1),
		Step('abs VT to line 8',	[0x1B, 0x0B, 8], wait=2),
		Step('set top margin',	    [0x1B, 0x54], wait=1),
		Step('abs VT to line 16',	[0x1B, 0x0B, 16], wait=2),
		Step('set bottom margin',	[0x1B, 0x4C], wait=1),
		Step('clear margins',	    [0x1B, 0x43], wait=4, capture=True, comment='// ESC+C clear both top and bottom margins'),
		# --- Cleanup ---
		Step('reset to DIP',	    [0x1B, 0x53], wait=1),
		Step('2x CR+LF',	        [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
	], compatible_series=["C"]),
	Test('// Page Length + FF: ESC + FF + n, 0x0C FF', [
		Step('print',	        list(b'TEST PAGE LENGTH AND FF'), wait=4),
		Step('CR+LF',	        [0x0D, 0x0A], wait=2),
		# --- FF alone first (default page length) ---
		Step('print',	        list(b'FF DEFAULT'), wait=4),
		Step('CR+LF',	        [0x0D, 0x0A], wait=1),
		Step('FF default',	    [0x0C], wait=20, capture=True, comment='// FF with default page length'),
		Step('print',	        list(b'AFTER FF'), wait=4),
		Step('CR+LF',	        [0x0D, 0x0A], wait=1),
		# --- Page length 8 + FF ---
		Step('print',	        list(b'LEN8 START'), wait=4),
		Step('CR+LF',	        [0x0D, 0x0A], wait=2),
		Step('reset printer',	[0x1B, 0x0D, 0x50], wait=6),
		Step('set page len 8',	[0x1B, 0x0C, 8], wait=6, capture=True, comment='// Set page length 8'),
		Step('FF LEN8',	        [0x0C], wait=20, capture=True, comment='// FF with page length 8'),
		Step('print',	        list(b'LEN8 FF'), wait=4),
		Step('CR+LF',	        [0x0D, 0x0A], wait=1),
		# --- Page length 16 + FF ---
		Step('print',	        list(b'LEN16 START'), wait=4),
		Step('CR+LF',	        [0x0D, 0x0A], wait=2),
		Step('reset printer',	[0x1B, 0x0D, 0x50], wait=6),
		Step('set page len 16',	[0x1B, 0x0C, 16], wait=6, capture=True, comment='// Set page length 16'),
		Step('FF LEN16',	    [0x0C], wait=20, capture=True, comment='// FF with page length 16'),
		Step('print',	        list(b'LEN16 FF'), wait=4),
		Step('CR+LF',	        [0x0D, 0x0A], wait=1),
		# --- Cleanup ---
		Step('reset to DIP',	[0x1B, 0x53], wait=1),
		Step('2x CR+LF',	    [0x0D, 0x0A, 0x0D, 0x0A], wait=1),
	]),
]