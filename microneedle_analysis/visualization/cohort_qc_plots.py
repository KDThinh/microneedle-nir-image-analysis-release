"""Figures for cohort QC: spaghetti + median, RMS bar chart, before/after exclusion."""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerBase

from microneedle_analysis.analysis.smoothing import (
    lowess_smooth,
    rolling_mean_smooth,
    savgol_smooth,
)

logger = logging.getLogger(__name__)

AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 11
TITLE_FONTSIZE = 12


class HandlerOverlayBandLine(HandlerBase):
    """Render uncertainty band and center line overlaid in legend."""

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        band_proto, line_proto = orig_handle
        band = plt.Rectangle(
            (xdescent, ydescent),
            width,
            height,
            facecolor=band_proto.get_facecolor(),
            edgecolor="none",
            alpha=band_proto.get_alpha(),
            transform=trans,
        )
        y_mid = ydescent + 0.5 * height
        line = Line2D(
            [xdescent, xdescent + width],
            [y_mid, y_mid],
            color=line_proto.get_color(),
            linewidth=line_proto.get_linewidth(),
            transform=trans,
        )
        return [band, line]


_DEFAULT_TREND: Dict[str, Any] = {
    "enabled": True,
    "lowess_frac": 0.1,
    "savgol_window": 51,
    "savgol_polyorder": 3,
    "rolling_window": 50,
}


def _plot_trend_overlays(
    ax,
    frames: np.ndarray,
    y: np.ndarray,
    trend_smoothing: Optional[Dict[str, Any]],
) -> Tuple[List[Any], List[str]]:
    """Overlay LOWESS, Savitzky-Golay, and rolling mean on the primary cohort curve."""
    handles: list = []
    labels: list[str] = []
    if trend_smoothing is None:
        cfg = _DEFAULT_TREND
    else:
        cfg = {**_DEFAULT_TREND, **trend_smoothing}
    if not cfg.get("enabled", True):
        return handles, labels
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return handles, labels
    try:
        y_lo = lowess_smooth(y, frac=float(cfg["lowess_frac"]))
        (ln,) = ax.plot(
            frames,
            y_lo,
            color="C2",
            linestyle="-",
            linewidth=1.6,
            label="trend LOWESS",
        )
        handles.append(ln)
        labels.append("trend LOWESS")
    except Exception as e:
        logger.warning("cohort_qc before/after: LOWESS failed: %s", e)
    try:
        y_sg = savgol_smooth(
            y,
            window_length=int(cfg["savgol_window"]),
            polyorder=int(cfg["savgol_polyorder"]),
        )
        (ln,) = ax.plot(
            frames,
            y_sg,
            color="C3",
            linestyle="--",
            linewidth=1.6,
            label="trend Savitzky-Golay",
        )
        handles.append(ln)
        labels.append("trend Savitzky-Golay")
    except Exception as e:
        logger.warning("cohort_qc before/after: Savitzky-Golay failed: %s", e)
    try:
        y_rm = rolling_mean_smooth(y, window=int(cfg["rolling_window"]))
        (ln,) = ax.plot(
            frames,
            y_rm,
            color="C4",
            linestyle=":",
            linewidth=1.6,
            label="trend rolling mean",
        )
        handles.append(ln)
        labels.append("trend rolling mean")
    except Exception as e:
        logger.warning("cohort_qc before/after: rolling mean failed: %s", e)
    return handles, labels


def save_cohort_qc_spaghetti_median(
    frames: np.ndarray,
    X: np.ndarray,
    spot_ids: np.ndarray,
    value_label: str,
    base_path: str,
    format: str = "svg",
) -> None:
    """Per-tip traces + cohort median (median across tips at each frame)."""
    if X.size == 0:
        logger.warning("cohort_qc_plots: empty matrix; skip spaghetti plot")
        return

    cohort_median = np.median(X, axis=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, sid in enumerate(spot_ids):
        ax.plot(frames, X[i], alpha=0.75, linewidth=1.0, label=f"Tip {sid}")
    ax.plot(frames, cohort_median, color="black", linewidth=2.0, label="median")
    ax.set_xlabel("frame", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(value_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("background corrected per-tip traces", fontsize=TITLE_FONTSIZE)
    ax.legend(loc="best", fontsize=LEGEND_FONTSIZE, ncol=2)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, base_path, format)


def save_cohort_qc_rms_bar(
    spot_ids: np.ndarray,
    rms_distance: np.ndarray,
    threshold: float,
    base_path: str,
    format: str = "svg",
    flagged: Optional[np.ndarray] = None,
) -> None:
    """Bar chart of RMS distance per tip with threshold line."""
    if rms_distance.size == 0:
        logger.warning("cohort_qc_plots: no RMS data; skip bar chart")
        return

    n = len(spot_ids)
    flagged = flagged if flagged is not None else np.zeros(n, dtype=bool)
    colors = ["C3" if f else "C0" for f in flagged]

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(range(n), rms_distance, color=colors)
    if np.isfinite(threshold):
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=1.2, label="threshold")
        ax.legend()
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(s) for s in spot_ids])
    ax.set_xlabel("spot_id", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("RMS distance to LOO median ref", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Leave-One-Out cross validation", fontsize=TITLE_FONTSIZE)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)
    fig.tight_layout()
    _save(fig, base_path, format)


def save_cohort_qc_before_after(
    frames: np.ndarray,
    X: np.ndarray,
    flagged: np.ndarray,
    value_label: str,
    base_path: str,
    format: str = "svg",
    trend_smoothing: Optional[Dict[str, Any]] = None,
) -> None:
    """Per-frame median and mean: all tips vs after excluding flagged tips.

    Trend overlays (LOWESS, Savitzky-Golay, rolling mean) use the primary curve:
    after-exclusion median/mean when available, otherwise all tips.
    """
    if X.size == 0:
        return

    n_before = X.shape[0]
    keep = ~np.asarray(flagged, dtype=bool)
    n_after = int(keep.sum())

    median_all = np.median(X, axis=0)
    q25_all = np.percentile(X, 25, axis=0)
    q75_all = np.percentile(X, 75, axis=0)
    mean_all = np.mean(X, axis=0)
    std_all = np.std(X, axis=0)

    if n_after == 0:
        median_kept = None
        q25_kept = None
        q75_kept = None
        mean_kept = None
        std_kept = None
    else:
        median_kept = np.median(X[keep], axis=0)
        q25_kept = np.percentile(X[keep], 25, axis=0)
        q75_kept = np.percentile(X[keep], 75, axis=0)
        mean_kept = np.mean(X[keep], axis=0)
        std_kept = np.std(X[keep], axis=0)

    y_trend_median = median_kept if median_kept is not None else median_all
    y_trend_mean = mean_kept if mean_kept is not None else mean_all

    fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)

    def _legend_combo_handle(color: str):
        """Legend glyph showing both uncertainty band and center line."""
        return (
            Patch(facecolor=color, edgecolor="none", alpha=0.2),
            Line2D([0], [0], color=color, linewidth=2.0),
        )

    ax = axes[0]
    ax.fill_between(
        frames,
        q25_all,
        q75_all,
        color="C0",
        alpha=0.2,
        edgecolor="none",
        label=f"IQR, all tips (n={n_before})",
    )
    ax.plot(frames, median_all, color="C0", alpha=0.9)
    if median_kept is not None and n_after < n_before:
        ax.fill_between(
            frames,
            q25_kept,
            q75_kept,
            color="C1",
            alpha=0.2,
            edgecolor="none",
            label=f"IQR, after excluding flagged (n={n_after})",
        )
        ax.plot(
            frames,
            median_kept,
            color="C1",
            label=None,
            alpha=0.9,
        )
    trend_h_med, trend_l_med = _plot_trend_overlays(
        ax, frames, y_trend_median, trend_smoothing
    )
    ax.set_ylabel(value_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Per-frame median across tips", fontsize=TITLE_FONTSIZE)
    median_handles: List[Any] = [_legend_combo_handle("C0")]
    median_labels = [f"median + IQR, all tips (n={n_before})"]
    if median_kept is not None and n_after < n_before:
        median_handles.append(_legend_combo_handle("C1"))
        median_labels.append(f"median + IQR, after excluding flagged (n={n_after})")
    median_handles.extend(trend_h_med)
    median_labels.extend(trend_l_med)
    ax.legend(
        median_handles,
        median_labels,
        handler_map={tuple: HandlerOverlayBandLine()},
        loc="best",
        fontsize=LEGEND_FONTSIZE,
    )
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)

    ax = axes[1]
    ax.fill_between(
        frames,
        mean_all - std_all,
        mean_all + std_all,
        color="C0",
        alpha=0.2,
        edgecolor="none",
        label=f"SD, all tips (n={n_before})",
    )
    ax.plot(frames, mean_all, color="C0", alpha=0.9)
    if mean_kept is not None and n_after < n_before:
        ax.fill_between(
            frames,
            mean_kept - std_kept,
            mean_kept + std_kept,
            color="C1",
            alpha=0.2,
            edgecolor="none",
            label=f"SD, after excluding flagged (n={n_after})",
        )
        ax.plot(
            frames,
            mean_kept,
            color="C1",
            label=None,
            alpha=0.9,
        )
    trend_h_mean, trend_l_mean = _plot_trend_overlays(
        ax, frames, y_trend_mean, trend_smoothing
    )
    ax.set_xlabel("frame", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(value_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Per-frame mean across tips", fontsize=TITLE_FONTSIZE)
    mean_handles: List[Any] = [_legend_combo_handle("C0")]
    mean_labels = [f"mean + SD, all tips (n={n_before})"]
    if mean_kept is not None and n_after < n_before:
        mean_handles.append(_legend_combo_handle("C1"))
        mean_labels.append(f"mean + SD, after excluding flagged (n={n_after})")
    mean_handles.extend(trend_h_mean)
    mean_labels.extend(trend_l_mean)
    ax.legend(
        mean_handles,
        mean_labels,
        handler_map={tuple: HandlerOverlayBandLine()},
        loc="best",
        fontsize=LEGEND_FONTSIZE,
    )
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)

    fig.tight_layout()
    _save(fig, base_path, format)


def run_all_cohort_qc_plots(
    qc: dict,
    value_label: str,
    output_dir: str,
    format: str = "svg",
    trend_smoothing: Optional[Dict[str, Any]] = None,
) -> None:
    """Save all three figures into output_dir with fixed basenames."""
    os.makedirs(output_dir, exist_ok=True)
    X = qc.get("X")
    spot_ids = qc.get("spot_ids")
    frames = qc.get("frames")
    rms = qc.get("rms_distance")
    flagged = qc.get("flagged")
    threshold = qc.get("threshold", np.nan)

    if X is None or spot_ids is None or frames is None:
        return

    base1 = os.path.join(output_dir, "cohort_qc_spaghetti_median")
    save_cohort_qc_spaghetti_median(frames, X, spot_ids, value_label, base1, format=format)

    if rms is not None and rms.size:
        base2 = os.path.join(output_dir, "cohort_qc_rms_bar")
        save_cohort_qc_rms_bar(spot_ids, rms, float(threshold), base2, format=format, flagged=flagged)

    if flagged is not None and flagged.size == X.shape[0]:
        base3 = os.path.join(output_dir, "cohort_qc_before_after")
        save_cohort_qc_before_after(
            frames,
            X,
            flagged,
            value_label,
            base3,
            format=format,
            trend_smoothing=trend_smoothing,
        )


def _save(fig: Any, base_path: str, format: str) -> None:
    fig.savefig(f"{base_path}.svg", format=format, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{base_path}.png", format="png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
