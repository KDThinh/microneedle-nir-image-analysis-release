"""Separated tracking visualization plots."""

from typing import Optional, Tuple, Set, Iterable

import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 11
TITLE_FONTSIZE = 12


def _apply_axis_text_style(ax):
    """Keep plot text readable and consistent across exported figures."""
    ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)


def plot_tracking_results(first_frame, tracked_spot_data, average_background_intensities,
                         normalized_data=None, normalized_background=None, average_data=None,
                         starting_frame=100, ending_frame=3000, exposure_per_frame=10,
                         save_path=None, format='svg',
                         shift_frame=None, shift_end_frame=None,
                         artifact_correction_frames=None,
                         treatment_frame=None):
    """
    Create separate visualization figures for tracking results (4 separate plots).
    
    Parameters:
    -----------
    first_frame : np.ndarray
        First frame for trajectory overlay
    tracked_spot_data : dict
        Dictionary with tracking data
    average_background_intensities : list
        List of average background intensities per frame
    normalized_data : dict, optional
        Dictionary with normalized intensity DataFrames for each spot (from normalize_intensities)
    normalized_background : np.ndarray, optional
        Normalized background intensities (from normalize_background)
    average_data : pd.DataFrame, optional
        Average normalized intensity with std per frame (from apply_background_correction)
    starting_frame : int
        Starting frame for normalization (used if normalized_data is None)
    ending_frame : int
        Ending frame for normalization (used if normalized_data is None)
    exposure_per_frame : float
        Exposure time per frame in seconds
    save_path : str, optional
        Base path to save the figures (will append suffixes for each plot)
    format : str
        File format ('svg', 'png', etc.)
    shift_frame : int, optional
        Frame where illumination shift starts
    shift_end_frame : int, optional
        Frame where illumination shift ends
    treatment_frame : int, optional
        Frame index (relative to trimmed stack) where treatment is applied

    Returns:
    --------
    average_data : pd.DataFrame
        Average normalized intensity with std per frame
    time_in_minutes : np.ndarray
        Time array in minutes
    """
    
    frame_height, frame_width = first_frame.shape
    
    # Use provided normalized data or calculate if not provided (backward compatibility)
    if normalized_data is None:
        from microneedle_analysis.analysis.normalization import (
            normalize_intensities, normalize_background, apply_background_correction
        )
        normalized_data = normalize_intensities(tracked_spot_data, starting_frame, ending_frame)
        normalized_background = normalize_background(
            average_background_intensities, starting_frame, ending_frame
        )
        _, average_data = apply_background_correction(normalized_data, normalized_background)
    
    if normalized_background is None:
        from microneedle_analysis.analysis.normalization import normalize_background
        normalized_background = normalize_background(
            average_background_intensities, starting_frame, ending_frame
        )
    
    if average_data is None:
        from microneedle_analysis.analysis.normalization import apply_background_correction
        _, average_data = apply_background_correction(normalized_data, normalized_background)
    
    # Determine base path for saving
    if save_path:
        base, _ = os.path.splitext(save_path)
        output_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
        os.makedirs(output_dir, exist_ok=True)
    else:
        base = None
    
    # ===== Panel 1: Trajectory Plot =====
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.imshow(first_frame, cmap='gray')
    
    # Track which spots have visible trajectories for legend
    visible_spots = []
    
    for idx, spot_data in tracked_spot_data.items():
        df = pd.DataFrame(spot_data, columns=['frame', 'x', 'y', 'mean_intensity'])
        
        # Check if spot has enough points and movement to be visible
        if len(df) == 0:
            continue  # Skip empty spots
        elif len(df) == 1:
            # Single point: plot as marker only
            ax1.plot(df['x'].iloc[0], df['y'].iloc[0], 
                    marker='o', markersize=6, label=f'Tip {idx}')
            visible_spots.append(idx)
        else:
            # Multiple points: check if there's actual movement
            x_range = df['x'].max() - df['x'].min()
            y_range = df['y'].max() - df['y'].min()
            
            if x_range > 0.1 or y_range > 0.1:  # Has movement (threshold: 0.1 pixels)
                # Plot trajectory line
                ax1.plot(df['x'], df['y'], label=f'Tip {idx}', linestyle='-', linewidth=1.5)
                visible_spots.append(idx)
            else:
                # No movement: plot as single point
                ax1.plot(df['x'].iloc[0], df['y'].iloc[0], 
                        marker='o', markersize=6, label=f'Tip {idx}')
                visible_spots.append(idx)
    
    ax1.set_xlabel('X Position (pixels)', fontsize=AXIS_LABEL_FONTSIZE)
    ax1.set_ylabel('Y Position (pixels)', fontsize=AXIS_LABEL_FONTSIZE)
    
    # Only show legend if there are visible spots (inside image, top-right)
    if visible_spots:
        ax1.legend(
            loc='upper right',
            frameon=True,
            facecolor='white',
            framealpha=0.8,
            edgecolor='black',
            fontsize=LEGEND_FONTSIZE,
        )
    _apply_axis_text_style(ax1)
    
    ax1.grid(True)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlim(0, frame_width)
    ax1.set_ylim(frame_height, 0)
    fig1.tight_layout()
    
    if base:
        traj_path_svg = f"{base}_trajectory.svg"
        traj_path_png = f"{base}_trajectory.png"
        fig1.savefig(traj_path_svg, format=format, bbox_inches='tight', pad_inches=0.02)
        fig1.savefig(traj_path_png, format='png', bbox_inches='tight', pad_inches=0.02)
        plt.close(fig1)
    else:
        plt.show()
        plt.close(fig1)
    
    # Number of tips and sorted IDs for subsequent plots
    num_tips = len(tracked_spot_data)
    sorted_tip_ids = sorted(tracked_spot_data.keys())

    # Plot width = 2/3 of original (8 * 2/3 ≈ 5.33) for raw_intensity and stacked plots
    plot_width = 5.2
    # ===== Tracked tips visualization: Raw intensity of each tip and background =====
    if num_tips > 0 and base:
        num_subplots_raw = num_tips + 1
        fig_raw, axes_raw = plt.subplots(num_subplots_raw, 1, figsize=(plot_width, 1.6 * num_subplots_raw), sharex=True)
        if num_subplots_raw == 1:
            axes_raw = [axes_raw]
        cmap = plt.get_cmap('tab10')
        for plot_idx, idx in enumerate(sorted_tip_ids):
            spot_data = tracked_spot_data[idx]
            df = pd.DataFrame(spot_data, columns=['frame', 'x', 'y', 'mean_intensity'])
            time_in_minutes = df['frame'] * exposure_per_frame / 60
            y_vals = df['mean_intensity'].values.astype(float)
            ax = axes_raw[plot_idx]
            color = cmap(plot_idx % 10)
            ax.plot(time_in_minutes, y_vals, label=f'Tip {idx}', linewidth=1.5, color=color)
            if shift_frame is not None:
                t_shade = shift_frame * exposure_per_frame / 60.0
                ax.axvline(t_shade, color='orange', linestyle='--', linewidth=1.5,
                           label='Shade ON' if plot_idx == 0 else None)
            if treatment_frame is not None:
                t_treat = treatment_frame * exposure_per_frame / 60.0
                ax.axvline(t_treat, color='red', linestyle='--', linewidth=1.5)
            if y_vals.size > 0:
                y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
                if y_min == y_max:
                    delta = max(abs(y_min) * 0.1, 1e-3)
                    y_min -= delta
                    y_max += delta
                pad = 0.2 * (y_max - y_min)
                ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_ylabel('Raw Intensity', fontsize=AXIS_LABEL_FONTSIZE)
            ax.grid(True)
            ax.legend(loc='upper right', fontsize=LEGEND_FONTSIZE)
            _apply_axis_text_style(ax)
        bg_ax = axes_raw[-1]
        if average_background_intensities and len(average_background_intensities) > 0:
            bg_time = np.arange(len(average_background_intensities)) * exposure_per_frame / 60
            bg_vals = np.asarray(average_background_intensities, dtype=float)
            bg_ax.plot(bg_time, bg_vals, label='Background', color='gray', linewidth=1.5, linestyle='--')
            if shift_frame is not None:
                t_shade = shift_frame * exposure_per_frame / 60.0
                bg_ax.axvline(t_shade, color='orange', linestyle='--', linewidth=1.5, label='Shade ON')
            if treatment_frame is not None:
                t_treat = treatment_frame * exposure_per_frame / 60.0
                bg_ax.axvline(t_treat, color='red', linestyle='--', linewidth=1.5)
            if bg_vals.size > 0:
                bg_min, bg_max = float(np.min(bg_vals)), float(np.max(bg_vals))
                if bg_min == bg_max:
                    delta = max(abs(bg_min) * 0.1, 1e-3)
                    bg_min -= delta
                    bg_max += delta
                pad = 0.2 * (bg_max - bg_min)
                bg_ax.set_ylim(bg_min - pad, bg_max + pad)
            bg_ax.set_ylabel('Raw Intensity', fontsize=AXIS_LABEL_FONTSIZE)
            bg_ax.grid(True)
            bg_ax.legend(loc='upper right', fontsize=LEGEND_FONTSIZE)
            _apply_axis_text_style(bg_ax)
        axes_raw[-1].set_xlabel('Time (minutes)', fontsize=AXIS_LABEL_FONTSIZE)
        _apply_axis_text_style(axes_raw[-1])
        fig_raw.tight_layout()
        fig_raw.savefig(f"{base}_tracked_spots_raw_intensity.svg", format=format, bbox_inches='tight', pad_inches=0.02)
        fig_raw.savefig(f"{base}_tracked_spots_raw_intensity.png", format='png', bbox_inches='tight', pad_inches=0.02)
        plt.close(fig_raw)

    # Panel 3 (Normalized BG-corrected before vs after baseline) removed per user request.

    # Panels 4a and 4 (average normalized intensity with error bars) removed per user request.
    # ===== Panel 5a: Overlaid per-spot normalized intensity (raw and smoothed) =====
    # Format matches tracked_spots_raw_intensity: one subplot per spot + average, width 2/3 of original
    if normalized_data is not None and len(normalized_data) > 0:
        from microneedle_analysis.analysis.smoothing import exponential_smoothing

        sorted_ids_overlay = sorted(normalized_data.keys())
        cmap = plt.get_cmap('tab10')

        num_tips_overlay = len(sorted_ids_overlay)
        n_rows = num_tips_overlay + 1
        fig5a, axes5a = plt.subplots(
            n_rows, 1, figsize=(plot_width, 1.6 * n_rows), sharex=True
        )
        if n_rows == 1:
            axes5a = [axes5a]
        else:
            axes5a = list(axes5a)

        for plot_idx, spot_id in enumerate(sorted_ids_overlay):
            ax5a = axes5a[plot_idx]
            df = normalized_data[spot_id]
            t_spot = df['frame'].values * exposure_per_frame / 60.0

            y_raw = df['normalized_intensity'].astype(float).values

            color = cmap(plot_idx % 10)
            ax5a.plot(t_spot, y_raw, color=color, alpha=0.4, linewidth=1.0, label=f'Tip {spot_id}')

            y_all_overlay = []
            if y_raw.size > 0:
                y_smooth = exponential_smoothing(y_raw, alpha=0.05)
                ax5a.plot(t_spot, y_smooth, color=color, alpha=1.0, linewidth=2.5, linestyle='--', label=None)
                y_all_overlay.append(y_smooth.astype(float))

            if shift_frame is not None:
                t_shade = shift_frame * exposure_per_frame / 60.0
                ax5a.axvline(t_shade, color='orange', linestyle='--', linewidth=1.5,
                             label='Shade ON' if plot_idx == 0 else None)
            if treatment_frame is not None:
                t_treat = treatment_frame * exposure_per_frame / 60.0
                ax5a.axvline(t_treat, color='red', linestyle='--', linewidth=1.5)

            if len(y_all_overlay) > 0:
                y_concat = np.concatenate(y_all_overlay)
                y_min, y_max = float(np.min(y_concat)), float(np.max(y_concat))
                if y_min == y_max:
                    delta = max(abs(y_min) * 0.1, 1e-3)
                    y_min -= delta
                    y_max += delta
                pad = 0.2 * (y_max - y_min)
                ax5a.set_ylim(y_min - pad, y_max + pad)

            ax5a.set_ylabel('Norm. Intensity', fontsize=AXIS_LABEL_FONTSIZE)
            ax5a.grid(True)
            ax5a.legend(loc='upper right', fontsize=LEGEND_FONTSIZE)
            _apply_axis_text_style(ax5a)

        first_id = sorted_ids_overlay[0]
        df_first = normalized_data[first_id]
        frames_common = df_first['frame'].values
        t_common = frames_common * exposure_per_frame / 60.0
        smoothed_traces = []
        for spot_id in sorted_ids_overlay:
            df = normalized_data[spot_id]
            frames = df['frame'].values
            if not np.array_equal(frames, frames_common):
                continue
            y_raw = df['normalized_intensity'].astype(float).values
            if y_raw.size == 0:
                continue
            y_smooth = exponential_smoothing(y_raw, alpha=0.05)
            smoothed_traces.append(y_smooth.astype(float))

        ax_avg = axes5a[-1]
        if len(smoothed_traces) > 0:
            smoothed_arr = np.vstack(smoothed_traces)
            mean_smooth = smoothed_arr.mean(axis=0)
            std_smooth = smoothed_arr.std(axis=0)
            ax_avg.fill_between(t_common, mean_smooth - std_smooth, mean_smooth + std_smooth,
                                alpha=0.25, color='blue', edgecolor='none')
            ax_avg.plot(t_common, mean_smooth, color='blue', linewidth=1.8, label='Average')
            if shift_frame is not None:
                t_shade = shift_frame * exposure_per_frame / 60.0
                ax_avg.axvline(t_shade, color='orange', linestyle='--', linewidth=1.5, label='Shade ON')
            if treatment_frame is not None:
                t_treat = treatment_frame * exposure_per_frame / 60.0
                ax_avg.axvline(t_treat, color='red', linestyle='--', linewidth=1.5)
            y_lower, y_upper = mean_smooth - std_smooth, mean_smooth + std_smooth
            y_min, y_max = float(np.min(y_lower)), float(np.max(y_upper))
            if y_min == y_max:
                delta = max(abs(y_min) * 0.1, 1e-3)
                y_min -= delta
                y_max += delta
            pad = 0.2 * (y_max - y_min)
            ax_avg.set_ylim(y_min - pad, y_max + pad)
        ax_avg.set_ylabel('Norm. Intensity', fontsize=AXIS_LABEL_FONTSIZE)
        ax_avg.grid(True)
        ax_avg.legend(loc='upper right', fontsize=LEGEND_FONTSIZE)
        _apply_axis_text_style(ax_avg)

        axes5a[-1].set_xlabel('Time (minutes)', fontsize=AXIS_LABEL_FONTSIZE)
        _apply_axis_text_style(axes5a[-1])
        fig5a.tight_layout()

        if base:
            overlay_svg = f"{base}_stacked_spots_raw_and_smoothed.svg"
            overlay_png = f"{base}_stacked_spots_raw_and_smoothed.png"
            fig5a.savefig(
                overlay_svg,
                format=format,
                bbox_inches='tight',
                pad_inches=0.02,
            )
            fig5a.savefig(
                overlay_png,
                format='png',
                bbox_inches='tight',
                pad_inches=0.02,
            )
            plt.close(fig5a)
        else:
            plt.show()
            plt.close(fig5a)

    return average_data, time_in_minutes


def plot_ratiometric_iaa_ref_panel(
    iaa_result: dict,
    ref_result: dict,
    save_path: str,
    format: str = 'svg',
    iaa_label: str = 'IAA',
    ref_label: str = 'Ref',
    ylim: Optional[Tuple[float, float]] = None,
    smoothing_alpha: float = 0.05,
    exclude_spot_ids_iaa: Optional[Iterable[int]] = None,
    exclude_spot_ids_ref: Optional[Iterable[int]] = None,
):
    """
    Create an overlaid IAA/Ref response figure (no ratio panel).

    Parameters
    ----------
    ylim : (ymin, ymax), optional
        If set, applied to the y-axis of all four subplots.
    smoothing_alpha : float
        Exponential smoothing alpha for per-tip traces (match pipeline ``smoothing.alpha`` when set from config).
    exclude_spot_ids_iaa, exclude_spot_ids_ref : optional sets of spot_id to omit from that channel's mean/std.
    """
    t_iaa = iaa_result['time_in_minutes']
    column_key = 'normalized_intensity'
    avg_key = 'average_data'

    ex_iaa: Set[int] = set(exclude_spot_ids_iaa) if exclude_spot_ids_iaa else set()
    ex_ref: Set[int] = set(exclude_spot_ids_ref) if exclude_spot_ids_ref else set()

    base, _ = os.path.splitext(save_path)
    output_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
    os.makedirs(output_dir, exist_ok=True)

    def _get_avg_data(result: dict, key: str):
        """Get data by key, fallback to average_data if None or empty. Avoids DataFrame truthiness."""
        val = result.get(key)
        if val is None:
            return result['average_data']
        if hasattr(val, 'empty') and val.empty:
            return result['average_data']
        return val

    from microneedle_analysis.analysis.smoothing import exponential_smoothing

    def _compute_smoothed_mean_std_from_normalized_data(
        result: dict,
        col: str,
        alpha: float,
        exclude: Set[int],
    ):
        """
        Compute per-frame mean/std after applying exponential_smoothing to each spot trace.
        """
        normalized_data = result.get('normalized_data')
        if not normalized_data:
            return None

        spot_ids = sorted(normalized_data.keys())
        if not spot_ids:
            return None

        first_id = spot_ids[0]
        df_first = normalized_data[first_id]
        if 'frame' not in df_first or col not in df_first:
            return None

        frames_common = df_first['frame'].values
        traces = []
        for spot_id in spot_ids:
            if spot_id in exclude:
                continue
            df = normalized_data[spot_id]
            if 'frame' not in df or col not in df:
                continue
            frames = df['frame'].values
            if not np.array_equal(frames, frames_common):
                continue
            y_raw = df[col].astype(float).values
            if y_raw.size == 0:
                continue
            y_smooth = exponential_smoothing(y_raw, alpha=alpha)
            traces.append(y_smooth.astype(float))

        if not traces:
            return None

        arr = np.vstack(traces)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        return frames_common, mean, std

    try:
        computed_iaa = _compute_smoothed_mean_std_from_normalized_data(
            iaa_result, column_key, smoothing_alpha, ex_iaa
        )
        computed_ref = _compute_smoothed_mean_std_from_normalized_data(
            ref_result, column_key, smoothing_alpha, ex_ref
        )

        if (computed_iaa is None or computed_ref is None) and (ex_iaa or ex_ref):
            logger.warning(
                "Ratiometric: after spot exclusion, no per-tip traces remain for IAA and/or Ref; "
                "skipping panel for %s",
                base,
            )
            return

        if computed_iaa is not None and computed_ref is not None:
            frames_iaa, mean_iaa_full, std_iaa_full = computed_iaa
            frames_ref, mean_ref_full, std_ref_full = computed_ref

            frames_iaa_set = set(map(int, frames_iaa))
            frames_ref_set = set(map(int, frames_ref))
            frames_common = sorted(frames_iaa_set & frames_ref_set)
            if not frames_common:
                logger.warning(
                    "Ratiometric: no overlapping frames (IAA: %d, Ref: %d)",
                    len(frames_iaa_set), len(frames_ref_set),
                )
                return

            frames_iaa_arr = np.asarray(frames_iaa, dtype=int)
            frames_ref_arr = np.asarray(frames_ref, dtype=int)
            pos_map_iaa = {f: i for i, f in enumerate(frames_iaa_arr)}
            pos_map_ref = {f: i for i, f in enumerate(frames_ref_arr)}
            pos_iaa = [pos_map_iaa[f] for f in frames_common]
            pos_ref = [pos_map_ref[f] for f in frames_common]

            mean_iaa = mean_iaa_full[pos_iaa].astype(float)
            std_iaa = std_iaa_full[pos_iaa].astype(float)
            mean_ref = mean_ref_full[pos_ref].astype(float)
            std_ref = std_ref_full[pos_ref].astype(float)

            t = t_iaa[pos_iaa]
        else:
            avg_iaa = _get_avg_data(iaa_result, avg_key)
            avg_ref = _get_avg_data(ref_result, avg_key)

            if avg_iaa is None or avg_ref is None or len(avg_iaa) == 0 or len(avg_ref) == 0:
                logger.debug("Skipping ratiometric panel: no average data")
                return

            frames_iaa = set(avg_iaa.index)
            frames_ref = set(avg_ref.index)
            frames_common = sorted(frames_iaa & frames_ref)
            if not frames_common:
                logger.warning(
                    "Ratiometric: no overlapping frames (IAA: %d, Ref: %d)",
                    len(frames_iaa),
                    len(frames_ref),
                )
                return

            mean_iaa = avg_iaa.loc[frames_common, 'mean'].values.astype(float)
            std_iaa = avg_iaa.loc[frames_common, 'std'].values.astype(float)
            mean_ref = avg_ref.loc[frames_common, 'mean'].values.astype(float)
            std_ref = avg_ref.loc[frames_common, 'std'].values.astype(float)

            positions = [np.where(avg_iaa.index == f)[0][0] for f in frames_common]
            t = t_iaa[positions]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.fill_between(t, mean_iaa - std_iaa, mean_iaa + std_iaa, alpha=0.25, color='C0', edgecolor='none')
        ax.plot(t, mean_iaa, color='C0', linewidth=1.6, label=iaa_label)
        ax.fill_between(t, mean_ref - std_ref, mean_ref + std_ref, alpha=0.25, color='C1', edgecolor='none')
        ax.plot(t, mean_ref, color='C1', linewidth=1.6, label=ref_label)
        ax.set_ylabel('Smoothed Norm. Intensity', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xlabel('Time (minutes)', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(f'{iaa_label} and {ref_label} overlaid', fontsize=TITLE_FONTSIZE)
        ax.grid(True)
        ax.legend(loc='best', fontsize=LEGEND_FONTSIZE)
        _apply_axis_text_style(ax)

        fig.tight_layout()

        if ylim is not None:
            ax.set_ylim(ylim)

        fig.savefig(f"{base}.svg", format=format, bbox_inches='tight', pad_inches=0.02)
        fig.savefig(f"{base}.png", format='png', bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)

        rat_df = pd.DataFrame({
            'Time (minutes)': t,
            f'{iaa_label} (mean)': mean_iaa,
            f'{iaa_label} (std)': std_iaa,
            f'{ref_label} (mean)': mean_ref,
            f'{ref_label} (std)': std_ref,
        })
        rat_df.to_csv(f"{base}.csv", index=False)
        logger.info("Ratiometric panel saved: %s", base)
    except Exception as e:
        logger.warning("Ratiometric panel failed: %s", e, exc_info=True)

