"""
Signal Capture Module
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pickle
import re
import json
import numpy as np
from vcd import VCDWriter

# Import the shared constant
from .constants import DEFAULT_LABELS

@dataclass
class Capture:
    """
    Attributes:
        time_data: Array of timestamps (seconds) for each sample
        packed_data: Raw 8-bit packed channel data (uint8)
        channel_data: Dictionary mapping channel index to binary arrays
        typewriter: Typewriter model (e.g., "EM-31", "AX20")
        interface: Interface model (e.g., "IF-60")
        name: Descriptive name for the capture
        info: Additional information/notes
        keyboard_setting: DIP switch setting for keyboard mode (0, 1, or 2)
        interface_dip_switches: 16-bit DIP switch configuration
        timestamp: When capture was taken (auto-set to now if not provided)
    """
    # Raw scope data
    time_data: np.ndarray
    packed_data: np.ndarray
    channel_data: Dict[int, np.ndarray]

    # Device metadata
    typewriter: str
    interface: str

    # Capture identification
    name: str
    info: str = ""

    # Configuration
    keyboard_setting: int = 0
    interface_dip_switches: int = 0b00111100000000

    channel_labels: Dict[int, str] = field(default_factory=dict)

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def sample_rate(self) -> float:
        """Calculate sample rate from time data."""
        if len(self.time_data) < 2:
            return 0.0
        return 1.0 / (self.time_data[1] - self.time_data[0])

    @property
    def duration(self) -> float:
        """Total capture duration in seconds."""
        if len(self.time_data) == 0:
            return 0.0
        return self.time_data[-1] - self.time_data[0]

    @property
    def num_samples(self) -> int:
        return len(self.time_data)

    def __repr__(self) -> str:
        return (
            f"Capture('{self.typewriter}' + '{self.interface}', "
            f"'{self.name}', {self.num_samples:,} samples, "
            f"{self.duration*1000:.2f}ms @ {self.sample_rate/1e6:.1f}MHz)"
        )

    def get_info(self) -> str:
        ch_info = []
        for ch in sorted(self.channel_data.keys()):
            name = self.channel_labels.get(ch, f"D{ch}")
            ch_info.append(f"{ch}:{name}")

        return "\n".join([
            f"Capture: {self.name}",
            f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Devices: {self.typewriter} / {self.interface}",
            f"Data: {self.sample_rate/1e6:.2f} MHz, Channels: {', '.join(ch_info)}",
            f"Notes: {self.info}"
        ])

    @classmethod
    def from_scope_data(
        cls,
        time_data: np.ndarray,
        packed_data: np.ndarray,
        channel_data: Dict[int, np.ndarray],
        typewriter: str = "AX20",
        interface: str = "IF60",
        name: str = "Unnamed",
        info: str = "",
        keyboard_setting: int = 0,
        interface_dip_switches: int = 0,
        labels: Optional[List[str]] = None
    ) -> "Capture":

        # Apply default labels if none provided
        if labels is None:
            # Create map from default constant
            label_map = {i: label for i, label in enumerate(DEFAULT_LABELS)}
        else:
            # Create map from provided list
            label_map = {i: label for i, label in enumerate(labels)}

        return cls(
            time_data=time_data,
            packed_data=packed_data,
            channel_data=channel_data,
            typewriter=typewriter,
            interface=interface,
            name=name,
            info=info,
            keyboard_setting=keyboard_setting,
            interface_dip_switches=interface_dip_switches,
            channel_labels=label_map,
            timestamp=datetime.now()
        )

    def save(self, directory: Path = None) -> Path:
        if directory is None:
            directory = Path(__file__).parent.parent / "signal_captures"

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        safe_name = self._sanitize_filename(f"{self.typewriter}_{self.interface}_{self.name}")
        pkl_path = directory / f"{safe_name}.pkl"
        vcd_path = directory / f"{safe_name}.vcd"

        if pkl_path.exists():
            ts = self.timestamp.strftime("%Y%m%d_%H%M%S")
            pkl_path = directory / f"{safe_name}_{ts}.pkl"
            vcd_path = directory / f"{safe_name}_{ts}.vcd"

        with open(pkl_path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._save_vcd(vcd_path)
        return pkl_path

    def _save_vcd(self, filepath: Path) -> None:
        metadata = {
            "typewriter": self.typewriter,
            "interface": self.interface,
            "name": self.name,
            "info": self.info,
            "labels": self.channel_labels
        }

        with open(filepath, 'w') as f:
            with VCDWriter(f, timescale='1 ns', date=str(self.timestamp),
                           comment=json.dumps(metadata)) as writer:

                # Register Metadata
                meta_vars = {k: writer.register_var("Metadata", k, 'string')
                             for k in ["typewriter", "interface", "name", "info"]}

                # Register Data Signals
                mod_name = f"{self._sanitize_filename(self.typewriter)}_{self._sanitize_filename(self.interface)}"
                signals = {}
                sorted_channels = sorted(self.channel_data.keys())

                for ch in sorted_channels:
                    # Retrieve label from dict or default to D{ch}
                    sig_name = self.channel_labels.get(ch, f"D{ch}")
                    signals[ch] = writer.register_var(mod_name, sig_name, 'wire', size=1)

                # Write initial metadata
                for k, var in meta_vars.items():
                    writer.change(var, 0, str(metadata.get(k, "")))

                # Write data
                if len(self.time_data) > 0:
                    # Normalize time
                    time_ticks = ((self.time_data - self.time_data[0]) * 1e9).astype(np.int64)

                    for i, t in enumerate(time_ticks):
                        for ch in sorted_channels:
                            val = int(self.channel_data[ch][i])
                            writer.change(signals[ch], t, val)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        s = re.sub(r'[^\w\-]', '_', name)
        return re.sub(r'_+', '_', s).strip('_') or "unnamed"

    @classmethod
    def load(cls, filepath: Path) -> "Capture":
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            return pickle.load(f)