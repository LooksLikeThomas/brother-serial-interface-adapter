"""
Keysight/Agilent InfiniiVision 7000A Series MSO Controller

Supports MSO7104A, MSO7054A, MSO7034A, MSO7014A, etc.

Usage:
    from keysight_mso import KeysightInfiniiVisionMSO

    with KeysightInfiniiVisionMSO(address) as scope:
        scope.setup_digital_channels(channels=range(8), threshold=2.5)
        # For Deep Memory (RAW) capture:
        time_d, packed, chans, info = scope.acquire_pod_high_resolution(pod=1)

        # For Segmented Memory capture:
        scope.setup_segmented_capture(num_segments=3)
        scope.arm_trigger()
        # ... send data that triggers scope ...
        segments = scope.wait_and_read_all_segments(pod=1, timeout=5.0)
"""

import pyvisa
import numpy as np
import time


class ScpiCommandError(Exception):
    """Exception raised when an SCPI command fails"""

    def __init__(self, command, error_response):
        self.command = command
        self.error_response = error_response
        # Parse error code and message
        parts = error_response.split(",", 1)
        try:
            self.error_code = int(parts[0])
            self.error_message = parts[1].strip('"') if len(parts) > 1 else "Unknown error"
        except ValueError:
            self.error_code = -1
            self.error_message = error_response

        super().__init__(
            f"SCPI command '{command}' failed: [{self.error_code}] {self.error_message}"
        )


class ScpiErrorCheckedResource:
    """Wrapper for pyvisa Resource that automatically checks for SCPI errors after each write"""

    def __init__(self, resource: pyvisa.resources.Resource, check_errors=True):
        self._resource = resource
        self.check_errors_enabled = check_errors

    def __getattr__(self, name):
        """Pass through all attributes to the underlying resource"""
        return getattr(self._resource, name)

    def write(self, command):
        """Write command and check for errors"""
        result = self._resource.write(command)
        if self.check_errors_enabled:
            self._check_error(command)
        return result

    def write_no_check(self, command):
        """Write command WITHOUT checking for errors (Crucial for binary transfers)"""
        return self._resource.write(command)

    def query(self, command):
        """Query command - no error check needed as query itself would fail"""
        return self._resource.query(command)

    def read_raw(self):
        """Read raw bytes from the instrument"""
        return self._resource.read_raw()

    def _check_error(self, command):
        """Check error queue and raise exception if error occurred"""
        try:
            error = self._resource.query(":SYSTem:ERRor?")
            code = error.split(",")[0]
            if code != "+0" and code != "0":
                raise ScpiCommandError(command, error.strip())
        except ScpiCommandError:
            # Re-raise our own error
            raise
        except Exception:
            # If the error check itself fails (e.g., timeout), don't mask the original operation
            pass

    def get_all_errors(self):
        """Drain and return all errors from the error queue"""
        errors = []
        while True:
            try:
                error = self._resource.query(":SYSTem:ERRor?")
                code = error.split(",")[0]
                if code == "+0" or code == "0":
                    break
                errors.append(error.strip())
            except Exception:
                break
        return errors

    def clear_errors(self):
        """Clear all errors from the queue"""
        self.get_all_errors()


class KeysightInfiniiVisionMSO:
    """Agilent/Keysight InfiniiVision 7000A Series MSO Controller"""

    def __init__(self, resource_string, check_errors=True):
        self.rm = pyvisa.ResourceManager()

        # 1. Open the Real Resource
        raw_resource = self.rm.open_resource(resource_string)

        # 2. Configure Chunk Sizes
        raw_resource.chunk_size = 10 * 1024 * 1024  # 10 MB
        raw_resource.timeout = 30000                # 30 seconds

        # 4. Wrap it
        self.scope = ScpiErrorCheckedResource(raw_resource, check_errors=check_errors)

        # Clear any SCPI error queue entries
        self.scope.clear_errors()

        # Verify connection
        idn = self.scope.query("*IDN?")
        print(f"Connected to: {idn.strip()}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        if hasattr(self, 'scope'):
            self.scope._resource.close()
        if hasattr(self, 'rm'):
            self.rm.close()
        print("Connection closed")

    # =========================================================================
    # Basic Control
    # =========================================================================
    def write(self, command):
        """Pass the write command down to the internal scope object."""
        return self.scope.write(command)

    def query(self, command):
        """Pass the query command down to the internal scope object."""
        return self.scope.query(command)

    def read(self):
        """Pass the read command down to the internal scope object."""
        return self.scope.read()

    def reset(self):
        self.scope.write("*RST")
        self.scope.query("*OPC?")

    def run(self):
        self.scope.write(":RUN")

    def stop(self):
        self.scope.write(":STOP")

    def acquire_single(self):
        self.scope.write(":SINGle")
        self.scope.query("*OPC?")

    def digitize(self, sources=None):
        if sources:
            source_str = ",".join(sources)
            self.scope.write(f":DIGitize {source_str}")
        else:
            self.scope.write(":DIGitize")

    def autoscale(self):
        self.scope.write(":AUToscale")

    # =========================================================================
    # Digital Channel Setup
    # =========================================================================

    def setup_digital_channels(self, channels=range(8), threshold=1.4, time_scale=1e-3):
        # Ensure we're in normal mode (not segmented) for standard capture
        self.scope.write(":ACQuire:MODE RTIMe")

        self.scope.write(":DISPlay:LABel ON")

        # Determine pods
        pod1_channels = [ch for ch in channels if ch < 8]
        pod2_channels = [ch for ch in channels if ch >= 8]

        if pod1_channels:
            self.scope.write(":POD1:DISPlay ON")
            self.scope.write(f":POD1:THReshold {threshold}")
        else:
            self.scope.write(":POD1:DISPlay OFF")

        if pod2_channels:
            self.scope.write(":POD2:DISPlay ON")
            self.scope.write(f":POD2:THReshold {threshold}")
        else:
            self.scope.write(":POD2:DISPlay OFF")

        for ch in range(16):
            if ch in channels:
                self.scope.write(f":DIGital{ch}:DISPlay ON")
            else:
                self.scope.write(f":DIGital{ch}:DISPlay OFF")

        self.scope.write(f":TIMebase:SCALe {time_scale}")
        self.scope.write(":TIMebase:POSition 0")
        print(f"Digital channels {list(channels)} enabled, threshold={threshold}V")

    def set_digital_label(self, channel, label, display=True):
        self.scope.write(f':DIGital{channel}:LABel "{label[:10]}"')
        state = "ON" if display else "OFF"
        self.scope.write(f":DISPlay:LABel {state}")

    # =========================================================================
    # Trigger Setup
    # =========================================================================

    def setup_digital_trigger(self, channel=0, slope="POSitive"):
        self.scope.write(":TRIGger:MODE EDGE")
        self.scope.write(f":TRIGger:EDGE:SOURce DIGital{channel}")
        self.scope.write(f":TRIGger:EDGE:SLOPe {slope}")
        self.scope.write(":TRIGger:SWEep NORMal")
        print(f"Edge trigger on D{channel}, slope={slope}")

    def arm_trigger(self):
        self.scope.write(":SINGle")

    def force_trigger(self):
        """Send software trigger"""
        self.scope.write("*TRG")

    def set_trigger_delay(self, delay):
        """
        Set trigger delay (horizontal position).

        Args:
            delay: Delay in seconds (time from trigger to reference point)
        """
        self.scope.write(f":TIMebase:POSition {delay}")

    # =========================================================================
    # Data Acquisition - Standard
    # =========================================================================

    def read_current_pod_data(self, pod=1, points=None, mode="NORMal"):
        """Read currently displayed data."""
        self.scope.write(":STOP")  # Stop required for consistent read
        self.scope.write(":WAVeform:FORMat BYTE")
        self.scope.write(f":WAVeform:SOURce POD{pod}")
        self.scope.write(f":WAVeform:POINts:MODE {mode}")

        if mode in ["RAW", "MAXimum"]:
            # FW 6.x fix: Explicitly request MAXimum points to fill buffer
            self.scope.write(":WAVeform:POINts MAXimum")
        elif points:
            self.scope.write(f":WAVeform:POINts {points}")

        preamble = self.scope.query(":WAVeform:PREamble?").split(",")
        x_increment = float(preamble[4])
        x_origin = float(preamble[5])

        # BYPASS ERROR CHECK for data download
        self.scope.write_no_check(":WAVeform:DATA?")
        raw_data = self.scope.read_raw()

        header_len = 2 + int(chr(raw_data[1]))
        packed_data = np.frombuffer(raw_data[header_len:-1], dtype=np.uint8)

        # Unpack channels
        channel_data = {}
        base_ch = 0 if pod == 1 else 8
        for bit in range(8):
            channel_data[base_ch + bit] = (packed_data >> bit) & 1

        time_data = np.arange(len(packed_data)) * x_increment + x_origin

        return time_data, packed_data, channel_data

    def read_pod_data_high_resolution(self, pod=1, points=None):
        """
        Read pod data with maximum resolution using RAW mode.
        """
        # 1. Scope must be STOPPED
        self.scope.write(":STOP")

        # 2. Timebase must be MAIN
        self.scope.write(":TIMebase:MODE MAIN")

        # 3. Configure Waveform Source
        self.scope.write(":WAVeform:FORMat BYTE")
        self.scope.write(":WAVeform:UNSigned ON")
        self.scope.write(f":WAVeform:SOURce POD{pod}")
        self.scope.write(":WAVeform:POINts:MODE RAW")

        # 4. FW 6.x FIX: Force MAXimum points
        if points is None:
            self.scope.write(":WAVeform:POINts MAXimum")
        else:
            self.scope.write(f":WAVeform:POINts {points}")

        # 5. Get Preamble
        preamble = self.scope.query(":WAVeform:PREamble?").split(",")
        x_increment = float(preamble[4])
        x_origin = float(preamble[5])

        # 6. Fetch Data (BYPASS ERROR CHECKING)
        # Using write_no_check avoids the "Invalid literal for int()" crash
        self.scope.write_no_check(":WAVeform:DATA?")
        raw_data = self.scope.read_raw()

        # 7. Parse IEEE 488.2 Block
        header_len = 2 + int(chr(raw_data[1]))
        packed_data = np.frombuffer(raw_data[header_len:-1], dtype=np.uint8)

        # 8. Unpack
        channel_data = {}
        base_ch = 0 if pod == 1 else 8
        for bit in range(8):
            channel_data[base_ch + bit] = (packed_data >> bit) & 1

        time_data = np.arange(len(packed_data)) * x_increment + x_origin

        info = {
            "points": len(packed_data),
            "x_increment": x_increment,
            "x_origin": x_origin,
            "duration": len(packed_data) * x_increment
        }

        print(f"Deep capture: {len(packed_data):,} points retrieved.")

        return time_data, packed_data, channel_data, info

    def acquire_pod_high_resolution(self, pod=1, points=None, timeout_ms=30000):
        """
        Full Sequence: Setup -> Arm -> Wait -> Fetch Deep Memory
        """
        # Setup for deep acquisition
        self.scope.write(":DISPlay:VECTors OFF")  # Important for RAW
        self.scope.write(":ACQuire:MODE RTIMe")
        self.scope.write(":TIMebase:MODE MAIN")
        self.scope.write(":ACQuire:TYPE NORMal")

        # Set timeout
        old_timeout = self.scope.timeout
        self.scope.timeout = timeout_ms

        try:
            # Arm and Wait
            self.scope.write(":SINGle")

            # We assume external code triggers the scope,
            # or we wait for the trigger event
            self.scope.query("*OPC?")

            return self.read_pod_data_high_resolution(pod=pod, points=points)
        finally:
            self.scope.timeout = old_timeout

    # =========================================================================
    # Data Acquisition - Software Segmented (Long Capture + Split)
    # =========================================================================

    def segment_long_capture(self, time_data, packed_data, channel_data,
                              idle_threshold=1e-3, channels=None):
        """
        Segment a long capture into multiple segments based on activity.

        Finds idle periods (no edges on any monitored channel) longer than
        idle_threshold and splits the capture at those points.

        Args:
            time_data: Time array from capture
            packed_data: Packed data array from capture
            channel_data: Dict of channel arrays from capture
            idle_threshold: Minimum idle time (seconds) to split on (default 1ms)
            channels: List of channels to monitor for activity (default: all in channel_data)

        Returns:
            list: List of (time_data, packed_data, channel_data, timetag) tuples
                  timetag is relative to start of original capture
        """
        if channels is None:
            channels = list(channel_data.keys())

        if len(time_data) == 0:
            return []

        # Calculate sample interval
        if len(time_data) < 2:
            return [(time_data, packed_data, channel_data, 0.0)]

        sample_interval = time_data[1] - time_data[0]
        idle_samples = int(idle_threshold / sample_interval)

        # Find edges on any monitored channel
        # Edge = any change in any channel
        combined_activity = np.zeros(len(time_data), dtype=bool)
        for ch in channels:
            if ch in channel_data:
                ch_data = channel_data[ch]
                # Mark positions where there's a change
                edges = np.diff(ch_data.astype(np.int8)) != 0
                combined_activity[1:] |= edges

        # Find activity indices (where any edge occurs)
        activity_indices = np.where(combined_activity)[0]

        if len(activity_indices) == 0:
            # No activity at all
            print("Warning: No activity detected in capture")
            return []

        # Find gaps between activity that exceed idle_threshold
        segments = []
        segment_start = 0

        for i in range(1, len(activity_indices)):
            gap = activity_indices[i] - activity_indices[i-1]

            if gap > idle_samples:
                # End current segment at last activity + small margin
                segment_end = activity_indices[i-1] + min(idle_samples // 2, 100)
                segment_end = min(segment_end, len(time_data))

                # Extract segment
                seg_time, seg_packed, seg_channels = self._extract_segment(
                    time_data, packed_data, channel_data,
                    segment_start, segment_end
                )

                # Timetag relative to original capture start
                timetag = time_data[segment_start] - time_data[0]

                segments.append((seg_time, seg_packed, seg_channels, timetag))

                # Start new segment at next activity - small margin
                segment_start = activity_indices[i] - min(idle_samples // 2, 100)
                segment_start = max(segment_start, 0)

        # Don't forget the last segment
        segment_end = activity_indices[-1] + min(idle_samples // 2, 100)
        segment_end = min(segment_end, len(time_data))

        seg_time, seg_packed, seg_channels = self._extract_segment(
            time_data, packed_data, channel_data,
            segment_start, segment_end
        )
        timetag = time_data[segment_start] - time_data[0]
        segments.append((seg_time, seg_packed, seg_channels, timetag))

        print(f"Segmented capture into {len(segments)} segments")
        return segments

    def _extract_segment(self, time_data, packed_data, channel_data, start_idx, end_idx):
        """Extract a segment from the capture arrays."""
        seg_time = time_data[start_idx:end_idx].copy()
        seg_packed = packed_data[start_idx:end_idx].copy()
        seg_channels = {}
        for ch, data in channel_data.items():
            seg_channels[ch] = data[start_idx:end_idx].copy()
        return seg_time, seg_packed, seg_channels

    def capture_and_segment(self, pod=1, idle_threshold=1e-3, channels=None,
                            timeout_ms=30000):
        """
        Convenience method: Capture with high resolution and automatically segment.

        Args:
            pod: Pod number (1 or 2)
            idle_threshold: Minimum idle time (seconds) to split on (default 1ms)
            channels: List of channels to monitor for activity (default: all)
            timeout_ms: Capture timeout in milliseconds

        Returns:
            list: List of (time_data, packed_data, channel_data, timetag) tuples
        """
        # Capture
        time_data, packed_data, channel_data, info = self.acquire_pod_high_resolution(
            pod=pod, timeout_ms=timeout_ms
        )

        print(f"Captured {info['points']:,} points, duration {info['duration']*1000:.2f}ms")

        # Segment
        return self.segment_long_capture(
            time_data, packed_data, channel_data,
            idle_threshold=idle_threshold,
            channels=channels
        )

    def setup_segmented_capture(self, num_segments=10):
        """
        Setup segmented memory acquisition.

        Segmented memory captures multiple trigger events, each into a separate
        segment. Ideal for capturing bursty signals like protocol handshakes
        with dead time between them.

        Args:
            num_segments: Number of segments to capture (2-2000, depends on memory)

        Note:
            - Points per segment is determined by timebase setting
            - Use setup_digital_channels(time_scale=...) to set segment window size
            - Each segment captures 10 divisions worth of data
        """
        # Enable segmented mode
        self.scope.write(":ACQuire:MODE SEGMented")
        self.scope.write(f":ACQuire:SEGMented:COUNt {num_segments}")

        print(f"Segmented capture configured: {num_segments} segments")

    def get_segmented_config_count(self):
        """
        Get the configured (requested) number of segments.

        Returns:
            int: Number of segments configured via setup_segmented_capture()
        """
        return int(self.scope.query(":ACQuire:SEGMented:COUNt?"))

    def get_acquired_segment_count(self):
        """
        Get the number of segments actually captured so far.

        Returns:
            int: Number of segments currently in acquisition memory
        """
        return int(self.scope.query(":WAVeform:SEGMented:COUNt?"))

    def get_segment_timetag(self, index):
        """
        Get the time tag for a specific segment.

        Args:
            index: Segment index (1-based)

        Returns:
            float: Time in seconds relative to first segment trigger
        """
        self.scope.write(f":ACQuire:SEGMented:INDex {index}")
        return float(self.scope.query(":WAVeform:SEGMented:TTAG?"))

    def read_segment_pod_data(self, segment_index, pod=1):
        """
        Read pod data from a specific segment.

        Args:
            segment_index: Segment index (1-based)
            pod: Pod number (1 or 2)

        Returns:
            tuple: (time_data, packed_data, channel_data, timetag)
        """
        # Select the segment
        self.scope.write(f":ACQuire:SEGMented:INDex {segment_index}")

        # Get timetag for this segment
        timetag = float(self.scope.query(":WAVeform:SEGMented:TTAG?"))

        # Configure waveform read
        self.scope.write(":WAVeform:FORMat BYTE")
        self.scope.write(f":WAVeform:SOURce POD{pod}")

        # Get preamble
        preamble = self.scope.query(":WAVeform:PREamble?").split(",")
        x_increment = float(preamble[4])
        x_origin = float(preamble[5])

        # Read data (bypass error check for binary transfer)
        self.scope.write_no_check(":WAVeform:DATA?")
        raw_data = self.scope.read_raw()

        # Parse IEEE 488.2 block header
        header_len = 2 + int(chr(raw_data[1]))
        packed_data = np.frombuffer(raw_data[header_len:-1], dtype=np.uint8)

        # Unpack into individual channels
        channel_data = {}
        base_ch = 0 if pod == 1 else 8
        for bit in range(8):
            channel_data[base_ch + bit] = (packed_data >> bit) & 1

        # Generate time array
        time_data = np.arange(len(packed_data)) * x_increment + x_origin

        return time_data, packed_data, channel_data, timetag

    def wait_and_read_all_segments(self, pod=1, timeout=10.0, poll_interval=0.1):
        """
        Wait for segmented capture to complete and read all segments.

        This method polls until either:
        - All configured segments are captured, OR
        - Timeout is reached

        Then reads whatever segments were captured.

        Args:
            pod: Pod number (1 or 2)
            timeout: Maximum time to wait in seconds
            poll_interval: How often to check segment count in seconds

        Returns:
            list: List of (time_data, packed_data, channel_data, timetag) tuples
                  One tuple per captured segment.
        """
        target_count = self.get_segmented_config_count()
        start_time = time.time()

        # Poll until all segments captured or timeout
        while True:
            acquired = self.get_acquired_segment_count()

            if acquired >= target_count:
                print(f"All {acquired} segments captured")
                break

            elapsed = time.time() - start_time
            if elapsed >= timeout:
                print(f"Timeout reached. Captured {acquired}/{target_count} segments")
                break

            time.sleep(poll_interval)

        # Stop acquisition
        self.scope.write(":STOP")

        # Get final count
        num_segments = self.get_acquired_segment_count()

        if num_segments == 0:
            print("Warning: No segments captured!")
            # Reset to normal mode before returning
            self.scope.write(":ACQuire:MODE RTIMe")
            return []

        # Read all segments
        segments = []
        for i in range(1, num_segments + 1):  # 1-based indexing
            segment_data = self.read_segment_pod_data(i, pod)
            segments.append(segment_data)

        print(f"Read {len(segments)} segments from memory")

        # Reset to normal mode for subsequent standard captures
        self.scope.write(":ACQuire:MODE RTIMe")

        return segments