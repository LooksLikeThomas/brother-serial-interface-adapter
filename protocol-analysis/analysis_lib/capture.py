"""
Signal Capture Module

Structured capture dataclass for organizing oscilloscope signal captures with metadata.

Usage:
    from analysis_lib import Capture

    # Capture from scope
    time_data, packed, channel_data = scope.read_current_pod_data(pod=1, mode='RAW')

    # Create Capture with metadata
    capture = Capture.from_scope_data(
        time_data, packed, channel_data,
        typewriter="EM-31",
        interface="IF-60",
        name="Letter_U_Test",
        keyboard_setting=2
    )

    # Save
    filepath = capture.save()

    # Load later
    capture = Capture.load(filepath)
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict
import pickle
import re
import numpy as np


@dataclass
class Capture:
    """
    A structured capture from the oscilloscope containing raw signal data and metadata.

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
    interface_dip_switches: int = 0

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
        """Number of samples in capture."""
        return len(self.time_data)

    def __repr__(self) -> str:
        """String representation of capture."""
        return (
            f"Capture('{self.typewriter}' + '{self.interface}', "
            f"'{self.name}', {self.num_samples:,} samples, "
            f"{self.duration*1000:.2f}ms @ {self.sample_rate/1e6:.1f}MHz)"
        )

    def get_info(self) -> str:
        """Get detailed information about capture."""
        info_lines = [
            f"Capture: {self.name}",
            f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Devices:",
            f"  Typewriter: {self.typewriter}",
            f"  Interface: {self.interface}",
            "",
            "Configuration:",
            f"  Keyboard Setting: {self.keyboard_setting}",
            f"  DIP Switches: 0b{self.interface_dip_switches:016b} (0x{self.interface_dip_switches:04X})",
            "",
            "Capture Data:",
            f"  Samples: {self.num_samples:,}",
            f"  Duration: {self.duration*1000:.3f} ms",
            f"  Sample Rate: {self.sample_rate/1e6:.2f} MHz",
            f"  Channels: {sorted(self.channel_data.keys())}",
        ]
        if self.info:
            info_lines.extend(["", f"Notes: {self.info}"])

        return "\n".join(info_lines)

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
    ) -> "Capture":
        """
        Factory method to create Capture from scope.read_current_pod_data() output.

        Args:
            time_data: Array of timestamps (seconds)
            packed_data: Raw 8-bit packed channel data
            channel_data: Dictionary mapping channel index to binary arrays
            typewriter: Typewriter model (default: "AX20")
            interface: Interface model (default: "IF60")
            name: Capture name (default: "Unnamed")
            info: Additional information (default: "")
            keyboard_setting: Keyboard DIP switch setting 0/1/2 (default: 0)
            interface_dip_switches: 16-bit DIP switch config (default: 0)

        Returns:
            Capture object

        Usage:
            time_data, packed, channel_data = scope.read_current_pod_data(pod=1, mode='RAW')
            capture = Capture.from_scope_data(
                time_data, packed, channel_data,
                typewriter="EM-31",
                interface="IF-60",
                name="Letter_U_Test",
                keyboard_setting=2,
                interface_dip_switches=0b0000000000000000
            )
        """
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
            timestamp=datetime.now()
        )

    def save(self, directory: Path = None) -> Path:
        """
        Save capture to pickle file.

        Args:
            directory: Target directory (defaults to signal-captures/)

        Returns:
            Path to saved file

        File naming: {Typewriter}_{Interface}_{Name}.pkl
        If duplicate exists: {Typewriter}_{Interface}_{Name}_{YYYYMMDD_HHMMSS}.pkl
        """
        if directory is None:
            directory = Path(__file__).parent.parent / "signal-captures"

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Sanitize components for filesystem
        typewriter_safe = self._sanitize_filename(self.typewriter)
        interface_safe = self._sanitize_filename(self.interface)
        name_safe = self._sanitize_filename(self.name)

        # Build filename
        base_filename = f"{typewriter_safe}_{interface_safe}_{name_safe}"

        # Check for collision
        filepath = directory / f"{base_filename}.pkl"
        if filepath.exists():
            # Add timestamp suffix
            timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
            filepath = directory / f"{base_filename}_{timestamp_str}.pkl"

        # Save with pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        return filepath

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Convert string to filesystem-safe filename component."""
        # Replace spaces and special chars with underscores
        sanitized = re.sub(r'[^\w\-]', '_', name)
        # Collapse multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized or "unnamed"

    @classmethod
    def load(cls, filepath: Path) -> "Capture":
        """
        Load capture from pickle file.

        Args:
            filepath: Path to .pkl file

        Returns:
            Loaded Capture object

        Usage:
            capture = Capture.load('signal-captures/EM-31_IF-60_Letter_U_Test.pkl')
        """
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            capture = pickle.load(f)

        return capture
