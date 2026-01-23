"""
Digital Signal Plotting Module

Visualizes digital signals with optional protocol annotations.
"""

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class AnnotationLike(Protocol):
    """Protocol for annotation objects."""
    channel: int
    start: float
    end: float
    row: str
    @property
    def label(self) -> str: ...


@dataclass
class Style:
    """Visual styling configuration."""
    # Line styles
    signal_width: float = 2.0
    baseline_width: float = 1.0
    baseline_color: str = "0.4"
    baseline_dash: tuple = (4, 1, 0.2, 1)  # In data units, will be scaled
    clock_marker_width: float = 0.8
    clock_marker_alpha: float = 0.7

    # Font sizes
    font_label: float = 12
    font_annotation: float = 7.0
    font_axis: float = 12
    font_title: float = 14

    # Colors
    signal_color: str = "black"
    background: str = "white"

    # DPI for export
    dpi: int = 300


DEFAULT_STYLE = Style()
DEFAULT_LABELS = ["SI", "SO", "SCK", "KBACK", "READY", "KBRQ"]

plt.rcParams["font.sans-serif"] = ["TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"


class DigitalPlot:
    """
    Digital signal plotter with annotation support.

    Usage:
        plot = DigitalPlot(time_data, channel_data)
        plot.add_annotations(annotations)
        fig, ax = plot.render()
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
        """Add annotations to the plot. Returns self for chaining."""
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
        """
        Render the digital signal plot.
        """
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
        t_min, t_max = t[0], t[-1]

        # Create figure with constrained_layout
        fig, ax = plt.subplots(figsize=figsize, dpi=self.style.dpi, layout="constrained")

        # Each channel gets 1 unit of height, signal goes from y to y+0.8
        signal_height = 0.8

        # Draw each channel (bottom to top in display, so reverse the order for y positions)
        for display_idx, ch in enumerate(channel_order):
            y_base = (n_channels - 1 - display_idx)  # Channel 0 at top
            self._draw_channel(ax, t, ch, display_idx, y_base, signal_height,
                               t_min, t_max, ann_by_ch[ch], scale)

        # Draw clock markers
        if clock_markers and clock_markers[0] in self.channels:
            self._draw_clock_markers(ax, t, clock_markers, channel_order, n_channels, signal_height)

        # Configure axes
        ax.set_xlim(t_min, t_max)
        ax.set_ylim(-0.5, n_channels - 0.5 + signal_height)

        # Y-axis: show channel labels
        yticks = [(n_channels - 1 - i) + signal_height / 2 for i in range(n_channels)]
        ylabels = [self.labels[ch] if ch < len(self.labels) else f"D{ch}"
                   for ch in channel_order]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=self.style.font_label, color=self.style.signal_color)

        # X-axis
        ax.set_xlabel(xlabel, fontsize=self.style.font_axis, color=self.style.signal_color)
        ax.tick_params(axis='x', labelsize=self.style.font_axis, labelcolor=self.style.signal_color, color=self.style.signal_color)

        # Title
        ax.set_title(title, fontsize=self.style.font_title, color=self.style.signal_color)

        # Style spines
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.spines["bottom"].set_color(self.style.signal_color)
        ax.tick_params(axis='y', length=0)  # Hide y tick marks

        # Colors
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
        t_min: float,
        t_max: float,
        annotations: list,
        time_scale: float,
    ):
        """Draw a single channel with its signal and annotations."""
        s = self.style
        y_top = y_base + signal_height

        # Baseline (dashed reference line)
        # Offset odd rows by half the first dash
        dash_offset = (s.baseline_dash[0] / 2) if (ch_index % 2 == 1) else 0
        ax.hlines(
            y_base, t_min, t_max,
            colors=s.baseline_color,
            linewidths=s.baseline_width,
            linestyles=(dash_offset, s.baseline_dash),
        )

        # Digital signal
        self._draw_signal(ax, time, self.channels[ch], y_base, y_top)

        # Annotations
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
        """Draw digital signal waveform."""
        if len(signal) == 0:
            return

        # Find transitions
        transitions = np.where(np.diff(signal) != 0)[0] + 1

        # Build path
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
        n_channels: int,
        signal_height: float,
    ):
        """Draw vertical lines at clock rising edges."""
        clock_ch, target_ch = markers
        if clock_ch not in self.channels:
            return
        if clock_ch not in channel_order or target_ch not in channel_order:
            return

        clock = self.channels[clock_ch]
        edges = np.where(np.diff(clock) == 1)[0]

        if len(edges) == 0:
            return

        # Find y positions
        clock_display_idx = channel_order.index(clock_ch)
        target_display_idx = channel_order.index(target_ch)

        y_clock = (n_channels - 1 - clock_display_idx) + signal_height
        y_target = (n_channels - 1 - target_display_idx)

        y_bottom = min(y_clock, y_target)
        y_top = max(y_clock, y_target)

        ax.vlines(
            time[edges],
            y_bottom, y_top,
            colors="gray",
            linestyles="dotted",
            linewidths=self.style.clock_marker_width,
            alpha=self.style.clock_marker_alpha,
            zorder=0,
        )

    def _draw_annotation(
        self,
        ax: plt.Axes,
        ann: AnnotationLike,
        y_base: float,
        time_scale: float,
    ):
        """Draw a single annotation."""
        s = self.style
        y_pos = y_base - 0.15  # Below the baseline
        t_start = ann.start * time_scale
        t_end = ann.end * time_scale

        if ann.row == "bits":
            t_center = t_start + (t_end - t_start) * 0.35
            ax.text(
                t_center, y_pos,
                ann.label,
                fontsize=s.font_annotation,
                ha="center",
                va="top",
            )
        else:
            ax.text(
                t_end, y_pos,
                ann.label,  # Removed prefix
                fontsize=s.font_annotation,
                ha="left",
                va="top",
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
) -> tuple[plt.Figure, plt.Axes]:
    """
    Convenience function for quick plotting.
    """
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