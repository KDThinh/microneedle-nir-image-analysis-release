"""Intensity curve visualization with smoothing and minima annotation."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 11
TITLE_FONTSIZE = 12


def plot_smoothing_results(processing_results=None, tracked_spot_data=None,
                          exposure_per_frame=20, alpha=0.01,
                          min_time_range=(210, 700), n_frames_baseline=50,
                          save_path=None, shift_frame=None, shift_end_frame=None,
                          artifact_correction_frames=None):
    """
    Plot exponential smoothing results for each spot with minimum annotation.
    
    Parameters:
    -----------
    processing_results : dict, optional
        Dictionary containing pre-calculated processing results for each spot.
        Each entry should have keys: 'detrended', 'smoothed', 't_min', 'min_intensity', 'time_in_minutes'
        If provided, tracked_spot_data and other parameters are ignored.
    tracked_spot_data : dict, optional
        Dictionary with tracking data (used only if processing_results is None, for backward compatibility)
    exposure_per_frame : float
        Exposure time per frame in seconds
    alpha : float
        Smoothing factor (used only if processing_results is None)
    min_time_range : tuple
        (min_time, max_time) range in minutes for finding minimum (used only if processing_results is None)
    n_frames_baseline : int
        Number of frames for baseline calculation (used only if processing_results is None)
    save_path : str, optional
        Path to save the figure
    shift_frame : int, optional
        Frame where illumination shift starts
    shift_end_frame : int, optional
        Frame where illumination shift ends
        
    Returns:
    --------
    spot_ids : list
        List of spot IDs
    min_times : list
        List of minimum times for each spot
    """
    # Use processing_results if provided, otherwise fall back to calculating (backward compatibility)
    if processing_results is None:
        if tracked_spot_data is None:
            raise ValueError("Either processing_results or tracked_spot_data must be provided")
        
        # Backward compatibility: calculate processing results
        from microneedle_analysis.analysis.smoothing import (
            baseline_correction, exponential_smoothing, find_minima
        )
        
        processing_results = {}
        for spot_id, spot_data in tracked_spot_data.items():
            df = pd.DataFrame(spot_data, columns=['frame', 'x', 'y', 'mean_intensity'])
            time_in_minutes = df['frame'] * exposure_per_frame / 60
            
            # Normalize intensity
            first_50_frames_avg = df['mean_intensity'][:n_frames_baseline].mean()
            df['normalized_intensity'] = df['mean_intensity'] / first_50_frames_avg
            
            # Baseline correction
            corrected_intensity, _ = baseline_correction(
                df['normalized_intensity'], n_frames_baseline
            )
            
            # Exponential smoothing
            smoothed_intensity = exponential_smoothing(corrected_intensity, alpha)
            
            # Find minimum
            df['smoothed_intensity'] = smoothed_intensity
            try:
                min_time, min_intensity, min_frame = find_minima(
                    df, time_in_minutes, min_time_range, exposure_per_frame
                )
            except ValueError as e:
                print(f"Warning for spot {spot_id}: {e}")
                min_time = np.nan
                min_intensity = np.nan
                min_frame = 0
            
            processing_results[spot_id] = {
                'detrended': corrected_intensity,
                'smoothed': smoothed_intensity,
                't_min': min_time,
                'min_intensity': min_intensity,
                'min_frame': min_frame,
                'time_in_minutes': time_in_minutes
            }
    
    num_spots = len(processing_results)
    fig, axes = plt.subplots(num_spots, 1, figsize=(4.5, 1.6 * num_spots), sharex=True)
    if num_spots == 1:
        axes = [axes]
    
    min_times = []
    spot_ids = []
    
    for idx, (spot_id, proc) in enumerate(processing_results.items()):
        time_in_minutes = proc['time_in_minutes']
        # Use baseline_corrected if available (new workflow), otherwise fall back to detrended (backward compatibility)
        baseline_corrected = proc.get('baseline_corrected', proc.get('detrended'))
        smoothed = proc['smoothed']
        min_time = proc['t_min']
        min_intensity = proc['min_intensity']
        
        min_times.append(min_time)
        spot_ids.append(spot_id)
        
        # Plot
        ax = axes[idx]
        ax.plot(time_in_minutes, baseline_corrected,
               label='Baseline Corrected', color='blue', linewidth=0.5)
        ax.plot(time_in_minutes, smoothed,
               label='Smoothed', color='red')
        
        if not np.isnan(min_time):
            ax.annotate('', xy=(min_time + 10, min_intensity),
                       xytext=(min_time, min_intensity - 0.02),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
        
        # Shade annotation (illumination change) if provided
        if shift_frame is not None:
            t_shade = shift_frame * exposure_per_frame / 60.0
            ax.axvline(t_shade, color='orange', linestyle='--', linewidth=1.5, label='Shade ON' if idx == 0 else None)
        
        # Artifact correction markers if provided
        if artifact_correction_frames and spot_id in artifact_correction_frames:
            correction_frames = artifact_correction_frames[spot_id]
            if len(correction_frames) > 0:
                # Convert correction frames to time in minutes
                correction_times = np.array(correction_frames) * exposure_per_frame / 60.0
                # Get corresponding intensity values at correction times
                correction_intensities = []
                for cf in correction_frames:
                    # Find closest time point
                    time_idx = np.argmin(np.abs(time_in_minutes - (cf * exposure_per_frame / 60.0)))
                    if time_idx < len(baseline_corrected):
                        correction_intensities.append(baseline_corrected[time_idx])
                    else:
                        correction_intensities.append(baseline_corrected[-1] if len(baseline_corrected) > 0 else 0)
                
                # Plot markers for artifact corrections
                ax.scatter(correction_times, correction_intensities, 
                          color='red', marker='x', s=100, zorder=5, linewidths=2,
                          label='Artifact Correction' if idx == 0 else None)

        ax.set_ylabel('Normalized Mean Intensity', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(f'Microneedle Tip {spot_id}', fontsize=TITLE_FONTSIZE)
        ax.grid(True)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)
    
    plt.xlabel('Time (minutes)', fontsize=AXIS_LABEL_FONTSIZE)
    plt.tight_layout()
    
    if save_path:
        # Save SVG
        plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0.02)
        # Also save PNG alongside
        base, _ = os.path.splitext(save_path)
        png_path = f"{base}.png"
        plt.savefig(png_path, format='png', bbox_inches='tight', pad_inches=0.02)
        plt.show()
    else:
        plt.show()
    
    for spot_id, min_time in zip(spot_ids, min_times):
        if not np.isnan(min_time):
            print(f"Microneedle Tip ID: {spot_id}, Time of Minimum Intensity (time_min): {min_time} minutes")
    
    return spot_ids, min_times

