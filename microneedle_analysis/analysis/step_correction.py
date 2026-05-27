"""
Algorithms for correcting sudden step-change artifacts in intensity traces.
"""

import numpy as np
import pandas as pd
from typing import Optional


def robust_std(x: np.ndarray) -> float:
    """
    Calculate Robust Standard Deviation using MAD (Median Absolute Deviation).
    This prevents artifacts themselves from inflating the noise estimate.
    
    Parameters:
    -----------
    x : np.ndarray
        Input array
        
    Returns:
    --------
    float
        Robust standard deviation estimate
    """
    mad = np.median(np.abs(x - np.median(x)))
    return 1.4826 * mad


def auto_correct_multistep(
    df: pd.DataFrame, 
    window: int = 20, 
    sigma_threshold: float = 3.5, 
    max_iter: int = 5, 
    verbose: bool = False
) -> pd.DataFrame:
    """
    Iteratively detects and corrects multiple baseline shifts for EACH spot.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'spot_id', 'frame', and 'mean_intensity'.
    window : int
        Window size for local baseline calculation (frames).
    sigma_threshold : float
        Sensitivity. Minimum size of jump (in robust std devs) to correct.
    max_iter : int
        Maximum number of correction passes per spot.
    verbose : bool
        If True, prints details of each correction.
        
    Returns:
    --------
    pd.DataFrame
        Original dataframe with added 'mean_intensity_corrected' and 'corrections_count' columns.
    """
    corrected_dfs = []
    
    if verbose:
        print(f"{'Spot ID':<8} | {'Iter':<4} | {'Frame':<6} | {'Jump Mag':<10} | {'Status'}")
        print("-" * 55)

    for spot_id, group in df.groupby('spot_id'):
        # Ensure we work on a sorted copy
        group = group.sort_values('frame').copy()
        
        # Working copy of the signal
        y = group['mean_intensity'].values.astype(float)
        x = group['frame'].values
        
        corrections_count = 0
        correction_frames = []  # Track frames where corrections were applied
        
        for i in range(max_iter):
            # 1. Calculate derivatives (frame-to-frame change)
            diffs = np.diff(y)
            if len(diffs) == 0: break
                
            # 2. Robust Noise Estimation
            noise_est = robust_std(diffs)
            # Fallback if signal is perfectly flat
            if noise_est == 0: noise_est = np.std(diffs) 
            if noise_est == 0: break 
            
            # 3. Find largest remaining jump
            max_jump_idx = np.argmax(np.abs(diffs))
            max_jump_val = diffs[max_jump_idx]
            detected_frame = x[max_jump_idx + 1]
            
            # 4. Check Significance
            if np.abs(max_jump_val) > (sigma_threshold * noise_est):
                
                # Calculate local delta using window
                idx = max_jump_idx + 1
                start_pre = max(0, idx - window)
                end_post = min(len(y), idx + window)
                
                pre_chunk = y[start_pre : idx]
                post_chunk = y[idx : end_post]
                
                if len(pre_chunk) > 0 and len(post_chunk) > 0:
                    mu_pre = np.mean(pre_chunk)
                    mu_post = np.mean(post_chunk)
                    delta = mu_post - mu_pre
                    
                    # Apply Correction: Shift everything AFTER this point down by delta
                    y[idx:] = y[idx:] - delta
                    
                    # Record the frame where correction was applied
                    correction_frames.append(int(detected_frame))
                    
                    if verbose:
                        print(f"{spot_id:<8} | {i+1:<4} | {detected_frame:<6} | {delta:<10.2f} | Corrected")
                    corrections_count += 1
                else:
                    break # Cannot correct at edge
            else:
                break # No more significant jumps found
                
        group['mean_intensity_corrected'] = y
        group['corrections_count'] = corrections_count
        # Mark frames where corrections occurred (1 = corrected, 0 = not corrected)
        group['correction_applied'] = group['frame'].isin(correction_frames).astype(int)
        corrected_dfs.append(group)
    
    return pd.concat(corrected_dfs)