"""
Serial Protocol Decoder Module
"""

from dataclasses import dataclass
from typing import Union
import numpy as np


# =============================================================================
# ASCII Lookup Table
# =============================================================================

ASCII_TABLE = {
    # Control Codes (0-31)
    0: 'NUL', 1: 'SOH', 2: 'STX', 3: 'ETX', 4: 'EOT', 5: 'ENQ', 6: 'ACK', 7: 'BEL',
    8: 'BS',  9: 'HT',  10: 'LF', 11: 'VT', 12: 'FF',  13: 'CR', 14: 'SO', 15: 'SI',
    16: 'DLE', 17: 'DC1', 18: 'DC2', 19: 'DC3', 20: 'DC4', 21: 'NAK', 22: 'SYN', 23: 'ETB',
    24: 'CAN', 25: 'EM',  26: 'SUB', 27: 'ESC', 28: 'FS',  29: 'GS',  30: 'RS',  31: 'US',
    # Printable Characters (32-126)
    32: 'Space', 33: '!',   34: '"',   35: '#',   36: '$',   37: '%',   38: '&',   39: "'",
    40: '(',     41: ')',   42: '*',   43: '+',   44: ',',   45: '-',   46: '.',   47: '/',
    48: '0',     49: '1',   50: '2',   51: '3',   52: '4',   53: '5',   54: '6',   55: '7',
    56: '8',     57: '9',   58: ':',   59: ';',   60: '<',   61: '=',   62: '>',   63: '?',
    64: '@',     65: 'A',   66: 'B',   67: 'C',   68: 'D',   69: 'E',   70: 'F',   71: 'G',
    72: 'H',     73: 'I',   74: 'J',   75: 'K',   76: 'L',   77: 'M',   78: 'N',   79: 'O',
    80: 'P',     81: 'Q',   82: 'R',   83: 'S',   84: 'T',   85: 'U',   86: 'V',   87: 'W',
    88: 'X',     89: 'Y',   90: 'Z',   91: '[',   92: '\\',  93: ']',   94: '^',   95: '_',
    96: '`',     97: 'a',   98: 'b',   99: 'c',   100: 'd',  101: 'e',  102: 'f',  103: 'g',
    104: 'h',    105: 'i',  106: 'j',  107: 'k',  108: 'l',  109: 'm',  110: 'n',  111: 'o',
    112: 'p',    113: 'q',  114: 'r',  115: 's',  116: 't',  117: 'u',  118: 'v',  119: 'w',
    120: 'x',    121: 'y',  122: 'z',  123: '{',  124: '|',  125: '}',  126: '~',
    # Delete (127)
    127: 'DEL'
}


def byte_to_ascii_label(value: int) -> str:
    """Convert byte value to human-readable ASCII label."""
    if isinstance(value, bytes):
        value = value[0]
    return ASCII_TABLE.get(value, f'{value:02X}h')

# =============================================================================
# Annotation
# =============================================================================

@dataclass
class Annotation:
    """
    A display annotation for plotting.
    """
    channel: int                       # Channel this annotation belongs to
    start: float                       # Start time (seconds)
    end: float                         # End time (seconds)
    text: str                          # The label to display
    row: str = "default"               # Row/category for grouping


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RawFrame:
    """A single frame of raw sampled bits."""
    channel: int                       # Data channel this frame belongs to
    bits: list[int]                    # Raw sampled bit values (0 or 1)
    edge_indices: list[int]            # Sample indices where each bit was captured
    timestamps: list[float]            # Time (seconds) of each bit

    @property
    def start_time(self) -> float:
        return self.timestamps[0] if self.timestamps else 0.0

    @property
    def end_time(self) -> float:
        return self.timestamps[-1] if self.timestamps else 0.0

    def __len__(self) -> int:
        return len(self.bits)

    def __str__(self) -> str:
        return "0b" + "".join(str(bit) for bit in self.bits)

    def to_annotations(self, row: str = "bits") -> list[Annotation]:
        """Convert frame to bit-level annotations."""
        annotations = []
        for i, (bit, ts) in enumerate(zip(self.bits, self.timestamps)):
            if i + 1 < len(self.timestamps):
                end = self.timestamps[i + 1]
            elif i > 0:
                duration = ts - self.timestamps[i - 1]
                end = ts + duration
            else:
                end = ts

            annotations.append(Annotation(
                channel=self.channel,
                start=ts,
                end=end,
                text=str(bit),  # Simple string
                row=row
            ))
        return annotations


@dataclass
class DecodedByte:
    """A decoded byte with reference to its raw frame."""
    value: int                         # Decoded byte value (0-127)
    raw_value: int                     # Raw byte value (before inversion)
    ascii_label: str                   # Human-readable ASCII label
    raw_frame: RawFrame                # Underlying raw frame data

    @property
    def channel(self) -> int:
        return self.raw_frame.channel

    @property
    def start_time(self) -> float:
        return self.raw_frame.start_time

    @property
    def end_time(self) -> float:
        return self.raw_frame.end_time

    @property
    def hex_str(self) -> str:
        return f"0x{self.value:02X}"

    @property
    def raw_hex_str(self) -> str:
        return f"0x{self.raw_value:02X}"

    def __repr__(self) -> str:
        return f"DecodedByte({self.hex_str} '{self.ascii_label}')"

    def to_annotation(self, row: str = "bytes") -> Annotation:
        """Convert to a single byte-level annotation."""
        # Format: Raw Hex -> Decoded Value "ASCII"
        # Example: 0x55 -> 0x55 "U"
        label = f"{self.raw_hex_str} -> {self.hex_str} \"{self.ascii_label}\""

        return Annotation(
            channel=self.channel,
            start=self.start_time,
            end=self.end_time,
            text=label,  # Simple string
            row=row
        )


# =============================================================================
# Result Wrapper Classes
# =============================================================================

@dataclass
class RawDecodeResult:
    """Result from RawDecoder.decode() for a single channel."""
    channel: int
    frames: list[RawFrame]
    sample_rate: float

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def __getitem__(self, index):
        return self.frames[index]

    def __str__(self) -> str:
        return ", ".join(str(frame) for frame in self.frames)

    def to_annotations(self, row: str = "bits") -> list[Annotation]:
        """Get all bit-level annotations for all frames."""
        annotations = []
        for frame in self.frames:
            annotations.extend(frame.to_annotations(row=row))
        return annotations


@dataclass
class BrotherDecodeResult:
    """Result from BrotherSerialDecoder.decode()."""
    channel: int
    bytes: list[DecodedByte]
    raw: RawDecodeResult

    def __len__(self) -> int:
        return len(self.bytes)

    def __iter__(self):
        return iter(self.bytes)

    def __getitem__(self, index):
        return self.bytes[index]

    def __str__(self) -> str:
        return self.as_ascii()

    def to_annotations(self, include_bits: bool = False,
                       bits_row: str = "bits",
                       bytes_row: str = "bytes") -> list[Annotation]:
        annotations = []
        if include_bits:
            annotations.extend(self.raw.to_annotations(row=bits_row))
        for byte in self.bytes:
            annotations.append(byte.to_annotation(row=bytes_row))
        return annotations

    def as_bytes(self) -> bytes:
        return bytes(b.value for b in self.bytes)

    def as_hex_string(self) -> str:
        return " ".join(f"{b.value:02X}" for b in self.bytes)

    def as_ascii(self) -> str:
        return ",".join( byte_to_ascii_label(b.value) for b in self.bytes)



# =============================================================================
# RawDecoder
# =============================================================================

class RawDecoder:
    """
    Generic decoder for clock-synchronized serial protocols.
    """

    # Default channel assignments for Brother protocol
    DEFAULT_CLOCK_CH = 2
    DEFAULT_READY_CH = 4

    def __init__(
        self,
        channel_data: dict,
        time_data: np.ndarray,
        clock_ch: int = DEFAULT_CLOCK_CH,
        ready_ch: int = DEFAULT_READY_CH,
        clock_edge: str = "rising",
        debounce_us: float = 2.0,
        bits_per_frame: int = 8
    ):
        self.channel_data = channel_data
        self.time_data = time_data
        self.clock_ch = clock_ch
        self.ready_ch = ready_ch
        self.clock_edge = clock_edge.lower()
        self.debounce_us = debounce_us
        self.bits_per_frame = bits_per_frame

        # Pre-calculate timing
        sample_interval = time_data[1] - time_data[0]
        self.sample_rate = 1 / sample_interval
        self.debounce_samples = max(1, int(debounce_us * 1e-6 * self.sample_rate))

        # Pre-calculate valid clock edges
        self._valid_edges = self._find_valid_edges()

    def _find_valid_edges(self) -> list[int]:
        """Find all valid clock edges (with debounce and ready gating)."""
        clock = self.channel_data[self.clock_ch]
        ready = self.channel_data[self.ready_ch]

        if self.clock_edge == "rising":
            raw_edges = np.where((clock[:-1] == 0) & (clock[1:] == 1))[0]
            expected_level = 1
        else:
            raw_edges = np.where((clock[:-1] == 1) & (clock[1:] == 0))[0]
            expected_level = 0

        valid_edges = []
        for edge in raw_edges:
            # Check bounds for debounce
            if edge + 1 + self.debounce_samples >= len(clock):
                continue
            # Check stable level
            if not np.all(clock[edge + 1 : edge + 1 + self.debounce_samples] == expected_level):
                continue
            # Check READY (Active Low)
            if ready[edge] != 0:
                continue
            valid_edges.append(edge)

        return valid_edges

    def decode(
        self,
        data_ch: Union[int, list[int]] = 0
    ) -> Union[RawDecodeResult, tuple[RawDecodeResult, ...]]:
        if isinstance(data_ch, int):
            return self._decode_channel(data_ch)
        else:
            return tuple(self._decode_channel(ch) for ch in data_ch)

    def _decode_channel(self, data_ch: int) -> RawDecodeResult:
        data = self.channel_data[data_ch]
        raw_bits = [int(data[e]) for e in self._valid_edges]
        raw_timestamps = [float(self.time_data[e]) for e in self._valid_edges]

        frames = []
        num_complete = len(raw_bits) // self.bits_per_frame

        for i in range(num_complete):
            start = i * self.bits_per_frame
            end = start + self.bits_per_frame

            frame = RawFrame(
                channel=data_ch,
                bits=raw_bits[start:end],
                edge_indices=self._valid_edges[start:end],
                timestamps=raw_timestamps[start:end]
            )
            frames.append(frame)

        return RawDecodeResult(
            channel=data_ch,
            frames=frames,
            sample_rate=self.sample_rate
        )


# =============================================================================
# BrotherSerialDecoder
# =============================================================================

class BrotherSerialDecoder:
    """
    Decoder for the Brother serial interface protocol.
    """

    BITS_PER_FRAME = 8
    DATA_BITS = 7

    def __init__(self, raw_result: RawDecodeResult):
        self.raw_result = raw_result

    def decode(self) -> BrotherDecodeResult:
        decoded_bytes = []
        for frame in self.raw_result.frames:
            decoded_bytes.append(self._decode_frame(frame))

        return BrotherDecodeResult(
            channel=self.raw_result.channel,
            bytes=decoded_bytes,
            raw=self.raw_result
        )

    def _decode_frame(self, frame: RawFrame) -> DecodedByte:
        if len(frame.bits) != self.BITS_PER_FRAME:
            raise ValueError(f"Expected {self.BITS_PER_FRAME} bits, got {len(frame.bits)}")

        start_bit = frame.bits[0]
        raw_data_bits = frame.bits[1:]

        # Conditional inversion
        if start_bit == 1:
            data_bits = [1 - b for b in raw_data_bits]
        else:
            data_bits = raw_data_bits

        # Decode MSB-first
        value = 0
        for i, bit in enumerate(data_bits):
            value |= bit << (self.DATA_BITS - 1 - i)

        # Calc raw value (what was actually on the wire, including start bit)
        raw_value = 0
        for i, bit in enumerate(frame.bits):
            raw_value |= bit << (self.BITS_PER_FRAME - 1 - i)

        return DecodedByte(
            value=value,
            raw_value=raw_value,
            ascii_label=byte_to_ascii_label(value),
            raw_frame=frame
        )