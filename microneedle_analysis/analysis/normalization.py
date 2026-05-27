"""Intensity normalization and background correction."""

import numpy as np
import pandas as pd


def normalize_intensities(tracked_spot_data, starting_frame=100, ending_frame=3000):
    """
    Normalize intensities for each tracked spot.
    
    Parameters:
    -----------
    tracked_spot_data : dict
        Dictionary with tracking data
    starting_frame : int
        Starting frame for normalization baseline
    ending_frame : int
        Ending frame for normalization baseline
        
    Returns:
    --------
    normalized_data : dict
        Dictionary with normalized intensity DataFrames for each spot
    """
    normalized_data = {}
    
    for idx, spot_data in tracked_spot_data.items():
        df = pd.DataFrame(spot_data, columns=['frame', 'x', 'y', 'mean_intensity'])
        first_frames_avg = df['mean_intensity'][starting_frame:ending_frame].mean()
        df['normalized_intensity'] = df['mean_intensity'] / first_frames_avg
        normalized_data[idx] = df
    
    return normalized_data


def normalize_background(average_background_intensities, starting_frame=100, ending_frame=3000):
    """
    Normalize background intensities.
    
    Parameters:
    -----------
    average_background_intensities : list
        List of average background intensities per frame
    starting_frame : int
        Starting frame for normalization baseline
    ending_frame : int
        Ending frame for normalization baseline
        
    Returns:
    --------
    normalized_background_intensities : np.ndarray
        Normalized background intensities
    """
    first_frames_bg_avg = np.mean(
        average_background_intensities[starting_frame:ending_frame]
    )
    normalized_background_intensities = (
        np.array(average_background_intensities) / first_frames_bg_avg
    )
    return normalized_background_intensities


def apply_background_correction(normalized_data, normalized_background_intensities):
    """
    Apply background correction to normalized intensities.
    
    Parameters:
    -----------
    normalized_data : dict
        Dictionary with normalized intensity DataFrames
    normalized_background_intensities : np.ndarray
        Normalized background intensities
        
    Returns:
    --------
    corrected_data : dict
        Dictionary with background-corrected DataFrames
    average_data : pd.DataFrame
        Average normalized intensity with std per frame
    """
    all_normalized_intensities = []
    
    for idx, df in normalized_data.items():
        df['normalized_intensity_bg_corrected'] = (
            df['normalized_intensity'] / normalized_background_intensities[:len(df)]
        )
        all_normalized_intensities.append(df[['frame', 'normalized_intensity_bg_corrected']])
    
    all_data = pd.concat(all_normalized_intensities)
    average_data = all_data.groupby('frame')['normalized_intensity_bg_corrected'].agg(['mean', 'std'])
    
    return normalized_data, average_data

