"""
Digital Signal Plotting Module

Visualizes digital signals with optional protocol annotations.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable
import copy


@runtime_checkable
class AnnotationLike(Protocol):
    """Protocol for annotation objects."""
    channel: int
    start: float
    end: float
    row: str
    text: str


@dataclass
class Style:
    """Visual styling configuration."""
    signal_width: float = 2.0
    baseline_width: float = 1.0
    baseline_color: str = "0.4"
    baseline_dash: tuple = (0, (4, 4))
    clock_marker_width: float = 0.8
    clock_marker_alpha: float = 0.7
    font_label: float = 12
    font_annotation: float = 8.0
    font_axis: float = 12
    font_title: float = 14
    signal_color: str = "black"
    background: str = "white"
    dpi: int = 300

    # Absolute offset for byte labels (in X-axis units)
    byte_label_offset: float = 0.1


DEFAULT_STYLE = Style()
DEFAULT_LABELS = ["SI", "SO", "SCK", "KBACK", "READY", "KBRQ"]

plt.rcParams["font.sans-serif"] = ["TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"


class DigitalPlot:
    """
    Digital signal plotter with annotation support.
    """

    TIME_SCALES = {
        "s": (1.0, "Time (s)"),
        "ms": (1e3, "Time (ms)"),
        "us": (1e6, "Time (µs)"),
    }

    def __init__(
        self,
        time: np.ndarray,
        channels: dict[int, np.ndarray],
        labels: list[str] = None,
        style: Style = None,
    ):
        self.time = time
        self.channels = channels
        self.labels = labels or DEFAULT_LABELS
        self.style = style or DEFAULT_STYLE
        self.annotations: list[AnnotationLike] = []

    def add_annotations(self, annotations: list[AnnotationLike]) -> "DigitalPlot":
        self.annotations.extend(annotations)
        return self

    def render(
        self,
        channel_order: list[int] = None,
        time_unit: str = "ms",
        title: str = "Digital Capture",
        figsize: tuple[float, float] = (12, 6),
        clock_markers: tuple[int, int] = (2, 0),
        show: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        if channel_order is None:
            channel_order = list(range(min(6, len(self.channels))))

        n_channels = len(channel_order)

        # Group annotations by channel
        ann_by_ch = {ch: [] for ch in channel_order}
        for ann in self.annotations:
            if ann.channel in ann_by_ch:
                ann_by_ch[ann.channel].append(ann)

        # Scale time
        scale, xlabel = self.TIME_SCALES.get(time_unit, (1.0, "Time (s)"))
        t = self.time * scale
        t_start_val = t[0]
        t_end_val = t[-1]

        # Padding
        t_span_total = t_end_val - t_start_val
        t_view_max = t_end_val + (t_span_total * 0.05)

        fig, ax = plt.subplots(figsize=figsize, dpi=self.style.dpi, layout="constrained")

        signal_height = 0.65

        # --- CALCULATE DYNAMIC Y POSITIONS ---
        y_positions = {}
        current_y = 0.0

        for ch in channel_order:
            # Add 0.4 padding BELOW channels 0 (SI) and 1 (SO)
            if ch in [0, 1]:
                current_y += 0.4

            y_positions[ch] = current_y

            # Advance for next channel
            current_y += 1.0

        # Draw channels
        for ch in channel_order:
            y_base = y_positions[ch]
            self._draw_channel(ax, t, ch, 0, y_base, signal_height,
                               t_start_val, t_end_val, ann_by_ch[ch], scale)

        # Draw clock markers
        if clock_markers and clock_markers[0] in self.channels:
            self._draw_clock_markers(ax, t, clock_markers, channel_order, y_positions, signal_height)

        # Configure axes limits
        max_y = current_y - (1.0 - signal_height)
        ax.set_ylim(-0.5, max_y + 0.5)
        ax.set_xlim(t_start_val, t_view_max)

        # Limit axis line
        ax.spines["bottom"].set_bounds(t_start_val, t_end_val)

        # Configure Ticks (Standard)
        if time_unit == "ms":
            tick_interval = 0.1
        elif time_unit == "us":
            tick_interval = 100.0
        elif time_unit == "s":
            tick_interval = 0.0001
        else:
            tick_interval = 0.1

        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))

        ax.tick_params(axis='x', direction='out', length=5, width=1,
                      colors=self.style.signal_color, labelsize=self.style.font_axis)

        # Y-axis labels
        yticks = [y_positions[ch] + (signal_height/2) for ch in channel_order]
        ylabels = [self.labels[ch] if ch < len(self.labels) else f"D{ch}"
                   for ch in channel_order]

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=self.style.font_label, color=self.style.signal_color)
        ax.set_xlabel(xlabel, fontsize=self.style.font_axis, color=self.style.signal_color)
        ax.set_title(title, fontsize=self.style.font_title, color=self.style.signal_color)

        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.spines["bottom"].set_color(self.style.signal_color)
        ax.tick_params(axis='y', length=0)
        ax.set_facecolor(self.style.background)
        fig.patch.set_facecolor(self.style.background)

        if show:
            plt.show()

        return fig, ax

    def _draw_channel(
        self,
        ax: plt.Axes,
        time: np.ndarray,
        ch: int,
        ch_index: int,
        y_base: float,
        signal_height: float,
        t_start: float,
        t_end: float,
        annotations: list,
        time_scale: float,
    ):
        s = self.style
        y_top = y_base + signal_height

        ax.hlines(
            y_base, t_start, t_end,
            colors=s.baseline_color,
            linewidths=s.baseline_width,
            linestyles=s.baseline_dash,
        )

        self._draw_signal(ax, time, self.channels[ch], y_base, y_top)

        for ann in annotations:
            self._draw_annotation(ax, ann, y_base, time_scale)

    def _draw_signal(
        self,
        ax: plt.Axes,
        time: np.ndarray,
        signal: np.ndarray,
        y_low: float,
        y_high: float,
    ):
        if len(signal) == 0:
            return

        transitions = np.where(np.diff(signal) != 0)[0] + 1
        x = [time[0]]
        y = [y_high if signal[0] else y_low]

        for idx in transitions:
            t = time[idx]
            prev_level = y_high if signal[idx - 1] else y_low
            curr_level = y_high if signal[idx] else y_low
            x.extend([t, t])
            y.extend([prev_level, curr_level])

        x.append(time[-1])
        y.append(y[-1])

        ax.plot(x, y, lw=self.style.signal_width, color=self.style.signal_color, zorder=2)

    def _draw_clock_markers(
        self,
        ax: plt.Axes,
        time: np.ndarray,
        markers: tuple[int, int],
        channel_order: list[int],
        y_positions: dict,
        signal_height: float,
    ):
        clock_ch, _ = markers

        if clock_ch not in channel_order:
            return

        clock = self.channels[clock_ch]
        edges = np.where((clock[:-1] == 0) & (clock[1:] == 1))[0] + 1

        if len(edges) == 0:
            return

        try:
            y_bottom = y_positions[channel_order[0]]
            if len(channel_order) >= 3:
                top_ch = channel_order[2]
                y_top = y_positions[top_ch] + signal_height
            else:
                y_top = y_positions[channel_order[-1]] + signal_height

            ax.vlines(
                time[edges],
                y_bottom, y_top,
                colors="gray",
                linestyles="dotted",
                linewidths=self.style.clock_marker_width,
                alpha=self.style.clock_marker_alpha,
                zorder=0,
            )
        except (IndexError, KeyError):
            pass

    def _draw_annotation(
        self,
        ax: plt.Axes,
        ann: AnnotationLike,
        y_base: float,
        time_scale: float,
    ):
        s = self.style
        t_start = ann.start * time_scale
        t_end = ann.end * time_scale

        if ann.row == "bits":
            y_pos = y_base - 0.05
            ax.text(
                t_start,
                y_pos,
                ann.text,
                fontsize=s.font_annotation,
                ha="center",
                va="top",
                color="black"
            )
        else:
            t_center = t_start + (t_end - t_start) * 0.5
            y_pos = y_base - 0.35

            ax.text(
                t_center, y_pos,
                ann.text,
                fontsize=s.font_annotation + 2,
                ha="center",
                va="top",
                color="black",
                bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="black", lw=1)
            )


def plot_digital(
    time_data: np.ndarray,
    channel_data: dict,
    channels: list[int] = None,
    labels: list[str] = None,
    annotations: list = None,
    time_unit: str = "ms",
    figsize: tuple[float, float] = (12, 6),
    title: str = "Digital Capture",
    show: bool = True,
    debounce_us: float = 2.0
) -> tuple[plt.Figure, plt.Axes]:
    if debounce_us > 0:
        # Calculate minimum samples required for a valid pulse
        sample_rate = 1.0 / (time_data[1] - time_data[0])
        min_samples = int(debounce_us * 1e-6 * sample_rate)

        filtered_data = {}
        for ch, signal in channel_data.items():
            sig = signal.copy() # Work on a copy

            # Find all signal transitions (edges)
            edges = np.where(np.diff(sig) != 0)[0] + 1

            # Iterate through edges and suppress short pulses
            for i in range(len(edges) - 1):
                width = edges[i+1] - edges[i]
                if width < min_samples:
                    # Glitch detected: Overwrite this segment with the previous stable value
                    # (effectively flattening the pulse)
                    sig[edges[i]:edges[i+1]] = sig[edges[i]-1]

            filtered_data[ch] = sig

        # Use the filtered data for plotting
        channel_data = filtered_data

    plot = DigitalPlot(time_data, channel_data, labels)
    if annotations:
        plot.add_annotations(annotations)
    return plot.render(
        channel_order=channels,
        time_unit=time_unit,
        title=title,
        figsize=figsize,
        show=show,
    )


def plot_digital_normalized(
    time_data: np.ndarray,
    channel_data: dict,
    channels: list[int] = None,
    normalization_source_channels: list[int] = None,
    labels: list[str] = None,
    annotations: list = None,
    figsize: tuple[float, float] = (12, 6),
    title: str = "Digital Capture (Normalized Edges)",
    show: bool = True,
    debounce_us: float = 2.0,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots digital signals with a normalized X-axis.
    Applies signal stability check (debouncing).
    """
    if channels is None:
        channels = list(range(min(6, len(channel_data))))

    if normalization_source_channels is None:
        normalization_source_channels = [2, 3, 4, 5]

    valid_norm_channels = [ch for ch in normalization_source_channels if ch in channel_data]

    # Calculate Debounce Samples
    sample_interval = time_data[1] - time_data[0]
    sample_rate = 1 / sample_interval
    debounce_samples = max(1, int(debounce_us * 1e-6 * sample_rate))

    # 1. Identify Valid Transitions
    transition_indices = set()
    transition_indices.add(0)
    transition_indices.add(len(time_data) - 1)

    for ch in valid_norm_channels:
        data = channel_data[ch]
        raw_transitions = np.where(np.diff(data) != 0)[0] + 1

        for t in raw_transitions:
            if t + debounce_samples >= len(data):
                continue
            new_level = data[t]
            if np.all(data[t : t + debounce_samples] == new_level):
                transition_indices.add(t)

    warp_indices = np.sort(list(transition_indices))
    warp_times = time_data[warp_indices]

    # 2. Create warped data
    normalized_time = np.arange(len(warp_indices), dtype=float)
    normalized_channels = {}
    for ch, data in channel_data.items():
        normalized_channels[ch] = data[warp_indices]

    # 3. Warp Annotations (REVERTED TO INTERP)
    warped_annotations = []
    if annotations:
        for ann in annotations:
            # Linear interp maps absolute timestamps to new normalized axis
            new_start = np.interp(ann.start, warp_times, normalized_time)
            new_end = np.interp(ann.end, warp_times, normalized_time)

            new_ann = copy.copy(ann)
            new_ann.start = new_start
            new_ann.end = new_end
            warped_annotations.append(new_ann)

    # 4. Custom style
    norm_style = replace(DEFAULT_STYLE, byte_label_offset=4.0)

    # 5. Render
    plot = DigitalPlot(
        normalized_time,
        normalized_channels,
        labels=labels,
        style=norm_style
    )
    if warped_annotations:
        plot.add_annotations(warped_annotations)

    fig, ax = plot.render(
        channel_order=channels,
        time_unit="s",
        title=title,
        figsize=figsize,
        show=False
    )

    # 6. Custom Axis Ticks
    ax.xaxis.set_major_locator(FixedLocator(normalized_time))

    def format_time_label(x, pos):
        idx = int(round(x))
        if 0 <= idx < len(warp_times):
            t_us = warp_times[idx] * 1e6
            s = f"{t_us:.1f}"
            if s.endswith(".0"):
                return s[:-2]
            return s
        return ""

    ax.xaxis.set_major_formatter(FuncFormatter(format_time_label))

    tick_font_size = DEFAULT_STYLE.font_axis * 0.75
    ax.tick_params(axis='x', which='major', labelsize=tick_font_size)

    # Minor ticks: Every 10 microseconds
    t_min = time_data[0]
    t_max = time_data[-1]
    step_10us = 10e-6

    grid_times = np.arange(t_min, t_max + step_10us, step_10us)
    grid_x_coords = np.interp(grid_times, warp_times, normalized_time)

    ax.xaxis.set_minor_locator(FixedLocator(grid_x_coords))
    ax.tick_params(axis='x', which='minor', length=3, color='gray', direction='out')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    ax.set_xlabel("Time of Edge (µs)", fontsize=DEFAULT_STYLE.font_axis)

    if show:
        plt.show()

    return fig, ax