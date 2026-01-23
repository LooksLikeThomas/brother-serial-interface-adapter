"""
Keysight/Agilent InfiniiVision 7000A Series MSO Controller

Supports MSO7104A, MSO7054A, MSO7034A, MSO7014A, etc.

Usage:
    from keysight_mso import KeysightInfiniiVisionMSO

    with KeysightInfiniiVisionMSO(address) as scope:
        scope.setup_digital_channels(channels=range(8), threshold=2.5)
        # For Deep Memory (RAW) capture:
        time_d, packed, chans, info = scope.acquire_pod_high_resolution(pod=1)
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
        except Exception:
            # If the error check itself fails, we don't want to mask the original operation
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
    # Data Acquisition
    # =========================================================================

    def read_current_pod_data(self, pod=1, points=None, mode="NORMal"):
        """Read currently displayed data."""
        self.scope.write(":STOP") # Stop required for consistent read
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
        self.scope.write(":DISPlay:VECTors OFF") # Important for RAW
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