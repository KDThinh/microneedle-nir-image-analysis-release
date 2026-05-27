"""Exponential smoothing and minimum finding."""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


def baseline_correction(normalized_intensity, n_frames_baseline=50):
    """
    Apply baseline correction using linear fitting.
    
    Parameters:
    -----------
    normalized_intensity : pd.Series or np.ndarray
        Normalized intensity values
    n_frames_baseline : int
        Number of frames to use for baseline calculation at start and end
        
    Returns:
    --------
    corrected_intensity : np.ndarray
        Baseline-corrected intensity
    baseline_array : np.ndarray
        Baseline array
    """
    # Support both pandas Series and NumPy arrays
    if hasattr(normalized_intensity, "iloc"):
        # pandas Series / DataFrame column
        first_point = normalized_intensity.iloc[:n_frames_baseline].mean()
        last_point = normalized_intensity.iloc[-n_frames_baseline:].mean()
        values = normalized_intensity.values
    else:
        # Assume 1D NumPy array-like
        arr = np.asarray(normalized_intensity, dtype=float)
        first_point = arr[:n_frames_baseline].mean()
        last_point = arr[-n_frames_baseline:].mean()
        values = arr
    
    num_frames = len(values)
    slope = (last_point - first_point) / (num_frames - 1)
    intercept = first_point
    
    baseline_array = np.array([slope * i + intercept for i in range(num_frames)])
    corrected_intensity = values / baseline_array
    
    return corrected_intensity, baseline_array


def exponential_smoothing(intensity_data, alpha=0.01):
    """
    Apply exponential smoothing to intensity data.
    
    Parameters:
    -----------
    intensity_data : np.ndarray or pd.Series
        Intensity values to smooth
    alpha : float
        Smoothing factor (0 < alpha <= 1)
        
    Returns:
    --------
    smoothed_intensity : np.ndarray
        Smoothed intensity values
    """
    # Ensure 1D NumPy array for statsmodels
    data_array = np.asarray(intensity_data, dtype=float)
    model = SimpleExpSmoothing(data_array)
    fit = model.fit(smoothing_level=alpha, optimized=False)
    # fittedvalues is a pandas Series; convert to NumPy array
    return np.asarray(fit.fittedvalues, dtype=float)


def lowess_smooth(y, frac=0.1):
    """
    Locally weighted regression (LOWESS) trend for 1D series.

    Parameters
    ----------
    y : array-like
        Values along the time axis.
    frac : float
        Fraction of points used in each local regression (larger = smoother).

    Returns
    -------
    np.ndarray
        Smoothed values, same length as y.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return y.copy()
    x = np.arange(n, dtype=float)
    out = lowess(y, x, frac=float(frac), return_sorted=False)
    out = np.asarray(out, dtype=float)
    # statsmodels: 1D = smoothed y in input order; older 2D = columns (x, yfitted)
    if out.ndim == 1:
        return out
    return np.asarray(out[:, 1], dtype=float)


def savgol_smooth(y, window_length=51, polyorder=3):
    """
    Savitzky-Golay filter (preserves peaks better than moving average).

    Window length is reduced automatically when the series is shorter than requested.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return y.copy()
    po = int(polyorder)
    po = max(1, min(po, n - 2))
    wl = min(int(window_length), n)
    if wl % 2 == 0:
        wl -= 1
    wl = max(wl, 3)
    if wl <= po:
        wl = po + 1 + (1 - (po + 1) % 2)
    if wl > n:
        wl = n if n % 2 == 1 else n - 1
    if wl <= po or wl < 3:
        return y.copy()
    return np.asarray(savgol_filter(y, wl, po), dtype=float)


def rolling_mean_smooth(y, window=50):
    """Centered rolling mean; edges use partial windows (min_periods=1)."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y.copy()
    w = max(1, int(window))
    return pd.Series(y).rolling(window=w, center=True, min_periods=1).mean().to_numpy(dtype=float)


def find_minima(df, time_in_minutes, min_time_range=(210, 700), exposure_per_frame=20):
    """
    Find minimum intensity within specified time range.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'smoothed_intensity' and 'frame' columns
    time_in_minutes : np.ndarray
        Time array in minutes
    min_time_range : tuple
        (min_time, max_time) range in minutes for finding minimum
    exposure_per_frame : float
        Exposure time per frame in seconds
        
    Returns:
    --------
    min_time : float
        Time of minimum intensity in minutes
    min_intensity : float
        Minimum intensity value
    min_frame : int
        Frame number of minimum
    """
    filtered_df = df[
        (time_in_minutes >= min_time_range[0]) &
        (time_in_minutes <= min_time_range[1])
    ]
    
    if len(filtered_df) == 0:
        raise ValueError(f"No data points found in time range {min_time_range}")
    
    min_idx = filtered_df['smoothed_intensity'].idxmin()
    min_intensity = filtered_df.loc[min_idx, 'smoothed_intensity']
    min_frame = filtered_df.loc[min_idx, 'frame']
    min_time = min_frame * exposure_per_frame / 60
    
    return min_time, min_intensity, min_frame

