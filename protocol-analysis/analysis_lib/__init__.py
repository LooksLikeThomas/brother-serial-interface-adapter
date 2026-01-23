"""
Brother Serial Interface Analysis Library

A toolkit for decoding and visualizing the Brother Serial Interface protocol.

Modules:
    decoder: Protocol decoding (RawDecoder, BrotherSerialDecoder)
    plotting: Signal visualization (DigitalPlot, plot_digital)
    keysight_mso: Keysight oscilloscope control
"""

from .decoder import (
    RawDecoder,
    BrotherSerialDecoder,
    RawDecodeResult,
    BrotherDecodeResult,
    RawFrame,
    DecodedByte,
    Annotation,
)

from .plotting import (
    DigitalPlot,
    plot_digital,
    Style,
    DEFAULT_STYLE,
    DEFAULT_LABELS,
)

from .keysight_mso import KeysightInfiniiVisionMSO

__all__ = [
    # Decoders
    'RawDecoder',
    'BrotherSerialDecoder',
    'RawDecodeResult',
    'BrotherDecodeResult',
    'RawFrame',
    'DecodedByte',
    'Annotation',
    # Plotting
    'DigitalPlot',
    'plot_digital',
    'Style',
    'DEFAULT_STYLE',
    'DEFAULT_LABELS',
    # Hardware
    'KeysightInfiniiVisionMSO',
]

__version__ = '0.1.0'