"""Cohort QC: leave-one-out median reference, RMS distance, robust outlier flags."""

from typing import Any, Dict, List, Optional, Tuple

import logging
import numpy as np

from microneedle_analysis.analysis.smoothing import exponential_smoothing

logger = logging.getLogger(__name__)


def build_smoothed_matrix(
    normalized_data: Dict[int, Any],
    smoothing_alpha: float,
    column_key: str = "normalized_intensity",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack per-tip smoothed traces into X (n_tips, n_frames).

    Returns
    -------
    X : ndarray
        Smoothed values per tip per frame.
    spot_ids : ndarray
        Row order matches X.
    frames : ndarray
        Common frame index per column.
    """
    if not normalized_data:
        return np.zeros((0, 0)), np.array([]), np.array([])

    spot_ids_sorted = sorted(normalized_data.keys())
    first_id = spot_ids_sorted[0]
    df0 = normalized_data[first_id]
    if "frame" not in df0.columns or column_key not in df0.columns:
        logger.warning("cohort_qc: missing frame or %s in normalized_data", column_key)
        return np.zeros((0, 0)), np.array([]), np.array([])

    frames_common = df0["frame"].values
    rows: List[np.ndarray] = []
    used_ids: List[int] = []

    for spot_id in spot_ids_sorted:
        df = normalized_data[spot_id]
        if "frame" not in df.columns or column_key not in df.columns:
            continue
        if not np.array_equal(df["frame"].values, frames_common):
            logger.warning("cohort_qc: skipping spot %s (frame vector mismatch)", spot_id)
            continue
        y_raw = df[column_key].astype(float).values
        if y_raw.size == 0:
            continue
        y_smooth = exponential_smoothing(y_raw, alpha=smoothing_alpha)
        rows.append(y_smooth.astype(float))
        used_ids.append(spot_id)

    if not rows:
        return np.zeros((0, 0)), np.array([]), np.array([])

    X = np.vstack(rows)
    return X, np.asarray(used_ids, dtype=int), np.asarray(frames_common, dtype=float)


def leave_one_out_median_reference(X: np.ndarray) -> np.ndarray:
    """X: (n, T). ref[i,t] = median(X[j,t], j != i)."""
    n, T = X.shape
    ref = np.empty_like(X)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        ref[i, :] = np.median(X[mask, :], axis=0)
    return ref


def rms_deviation_to_reference(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """RMS over time for each row."""
    return np.sqrt(np.mean((X - ref) ** 2, axis=1))


def median_mad_threshold(d: np.ndarray, mad_lambda: float) -> Tuple[float, float, float]:
    """threshold = median(d) + mad_lambda * 1.4826 * MAD(d)."""
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    scale = 1.4826 * mad
    threshold = med + mad_lambda * scale
    return med, mad, threshold


def compute_cohort_qc(
    normalized_data: Dict[int, Any],
    smoothing_alpha: float,
    mad_lambda: float = 3.0,
    column_key: str = "normalized_intensity",
) -> Dict[str, Any]:
    """
    Full cohort QC: smoothed matrix, LOO median ref, RMS, flags.

    If n_tips < 2, flagged are all False (no threshold).
    """
    X, spot_ids, frames = build_smoothed_matrix(
        normalized_data, smoothing_alpha=smoothing_alpha, column_key=column_key
    )
    n = X.shape[0]
    out: Dict[str, Any] = {
        "X": X,
        "spot_ids": spot_ids,
        "frames": frames,
        "rms_distance": np.array([]),
        "flagged": np.array([], dtype=bool),
        "threshold": np.nan,
        "median_d": np.nan,
        "mad_d": np.nan,
        "mad_lambda": mad_lambda,
        "ref": None,
    }

    if n == 0:
        return out

    ref = leave_one_out_median_reference(X)
    out["ref"] = ref
    d = rms_deviation_to_reference(X, ref)
    out["rms_distance"] = d

    if n < 2:
        out["flagged"] = np.zeros(n, dtype=bool)
        logger.warning(
            "cohort_qc: n_tips=%d < 2; skipping MAD-based flags (no outliers).",
            n,
        )
        med_d, mad_d, threshold = median_mad_threshold(d, mad_lambda)
        out["median_d"] = med_d
        out["mad_d"] = mad_d
        out["threshold"] = threshold
        return out

    med_d, mad_d, threshold = median_mad_threshold(d, mad_lambda)
    out["median_d"] = med_d
    out["mad_d"] = mad_d
    out["threshold"] = threshold
    out["flagged"] = d > threshold
    return out
