"""High-level pipeline for complete microneedle analysis workflow."""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Set

import numpy as np
import pandas as pd

from microneedle_analysis.core.loader import load_tiff, subtract_background, correct_illumination_shift
from microneedle_analysis.core.detection import detect_tips
from microneedle_analysis.core.tracking import track_spots
from microneedle_analysis.analysis.step_correction import auto_correct_multistep
from microneedle_analysis.analysis.normalization import (
    normalize_intensities, normalize_background, apply_background_correction
)
from microneedle_analysis.analysis.smoothing import exponential_smoothing, find_minima
from microneedle_analysis.analysis.cohort_qc import compute_cohort_qc
from microneedle_analysis.visualization.tracking_plot import (
    plot_tracking_results,
    plot_ratiometric_iaa_ref_panel,
)
from microneedle_analysis.visualization.cohort_qc_plots import run_all_cohort_qc_plots
from microneedle_analysis.visualization.intensity_curves import plot_smoothing_results
from microneedle_analysis.visualization.tracking_video import export_tracking_video
from microneedle_analysis.io.exporter import (
    export_tracking_data,
    export_average_intensity,
    export_tiff_stack,
)
from microneedle_analysis.io.config import load_config, find_profile_by_path

logger = logging.getLogger(__name__)


def _flagged_spot_ids_from_cohort_qc(qc: Optional[Dict[str, Any]]) -> Set[int]:
    """spot_id values marked flagged in cohort_qc result dict."""
    if not qc:
        return set()
    sids = qc.get("spot_ids")
    fl = qc.get("flagged")
    if sids is None or fl is None or len(sids) == 0:
        return set()
    return {int(sids[i]) for i in range(len(sids)) if bool(fl[i])}


def _parse_ratiometric_iaa_ref_ylim(raw: Any) -> Optional[Tuple[float, float]]:
    """Parse config `visualization.ratiometric_iaa_ref_ylim` into a (ymin, ymax) tuple or None."""
    if raw is None:
        return None
    try:
        if len(raw) != 2:
            logger.warning(
                "Invalid visualization.ratiometric_iaa_ref_ylim (expected length 2): %s",
                raw,
            )
            return None
        return (float(raw[0]), float(raw[1]))
    except (TypeError, ValueError) as e:
        logger.warning(
            "Invalid visualization.ratiometric_iaa_ref_ylim (expected two numbers): %s (%s)",
            raw,
            e,
        )
        return None


def _apply_profile_to_config(config: Dict[str, Any], profile_data: Dict[str, Any]) -> None:
    """Merge profile_data into config.

    Supported overrides (when present in profile YAML):
    - Root: `exposure_per_frame`
    - `detection` dict (and legacy top-level `threshold_rel`)
    - `tracking` dict keys via dedicated top-level aliases (spots_to_track, max_frame, skip_initial_frames)
    - `normalization` dict keys via dedicated top-level aliases (normalization_starting_frame, normalization_ending_frame)
    - Root `treatment_frame` (mapped to `visualization.treatment_frame`)
    - `visualization` dict (merged into config['visualization'], e.g. `ratiometric_iaa_ref_ylim`)
    - `smoothing` dict (merged into config['smoothing'])
    """
    if not profile_data:
        return

    # `exposure_per_frame` is used throughout the pipeline to build the time axis.
    # We treat profile-provided exposure_per_frame as authoritative and propagate it
    # to the root `config['exposure_per_frame']` for backward compatibility with the
    # rest of the pipeline code.
    exposure_override = profile_data.get("exposure_per_frame", None)
    if exposure_override is None and "visualization" in profile_data and isinstance(profile_data["visualization"], dict):
        exposure_override = profile_data["visualization"].get("exposure_per_frame", None)
    if exposure_override is None and "smoothing" in profile_data and isinstance(profile_data["smoothing"], dict):
        exposure_override = profile_data["smoothing"].get("exposure_per_frame", None)
    if exposure_override is not None:
        config["exposure_per_frame"] = exposure_override

    if "threshold_rel" in profile_data:
        config.setdefault("detection", {})["threshold_rel"] = profile_data["threshold_rel"]
    if "detection" in profile_data:
        for k, v in profile_data["detection"].items():
            config.setdefault("detection", {})[k] = v
    if "spots_to_track" in profile_data:
        config.setdefault("tracking", {})["spots_to_track"] = profile_data["spots_to_track"]
    if "max_frame" in profile_data:
        config.setdefault("tracking", {})["max_frame"] = profile_data["max_frame"]
    if "skip_initial_frames" in profile_data:
        config.setdefault("tracking", {})["skip_initial_frames"] = profile_data["skip_initial_frames"]
    if "normalization_starting_frame" in profile_data:
        config.setdefault("normalization", {})["starting_frame"] = profile_data["normalization_starting_frame"]
    if "normalization_ending_frame" in profile_data:
        config.setdefault("normalization", {})["ending_frame"] = profile_data["normalization_ending_frame"]
    if "treatment_frame" in profile_data:
        config.setdefault("visualization", {})["treatment_frame"] = profile_data["treatment_frame"]

    # Merge visualization/smoothing dicts for true per-profile overrides.
    if "visualization" in profile_data and isinstance(profile_data["visualization"], dict):
        for k, v in profile_data["visualization"].items():
            config.setdefault("visualization", {})[k] = v
    if "smoothing" in profile_data and isinstance(profile_data["smoothing"], dict):
        for k, v in profile_data["smoothing"].items():
            config.setdefault("smoothing", {})[k] = v
    if "cohort_qc" in profile_data and isinstance(profile_data["cohort_qc"], dict):
        for k, v in profile_data["cohort_qc"].items():
            config.setdefault("cohort_qc", {})[k] = v


def add_timestamp_to_output_dir(output_dir: str) -> str:
    """
    Add a timestamp suffix to the output directory name.
    
    Parameters:
    -----------
    output_dir : str
        Original output directory path (can be relative or absolute)
        
    Returns:
    --------
    output_dir_with_timestamp : str
        Output directory path with timestamp appended
        Examples:
        - './Exp_1_Results' -> './Exp_1_Results_20250127_143022'
        - 'Exp_1_Results' -> 'Exp_1_Results_20250127_143022'
        - 'C:/path/to/Exp_1_Results' -> 'C:/path/to/Exp_1_Results_20250127_143022'
    """
    if not output_dir:
        return output_dir
    
    # Generate timestamp in format: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Check if path starts with './' to preserve relative path format
    is_relative_dot = output_dir.startswith('./')
    
    # Normalize the path to handle separators
    normalized = os.path.normpath(output_dir)
    
    # Split into directory and base name
    dir_path = os.path.dirname(normalized)
    base_name = os.path.basename(normalized)
    
    # Add timestamp to base name
    base_name_with_timestamp = f"{base_name}_{timestamp}"
    
    # Reconstruct path
    if dir_path and dir_path != '.':
        # Has a directory component
        result = os.path.join(dir_path, base_name_with_timestamp)
    else:
        # Just the base name
        result = base_name_with_timestamp
    
    # Restore './' prefix if it was there originally
    if is_relative_dot and not result.startswith('./'):
        result = f"./{result}"
    
    return result


class MicroneedlePipeline:
    """High-level pipeline for microneedle analysis."""
    
    def __init__(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        config_path: Optional[str] = None,
        radius: Optional[int] = None,
        correct_shift: bool = False,
        shift_frame: Optional[int] = None,
        shift_end_frame: Optional[int] = None,
        export_video: bool = False,
        skip_output_timestamp: bool = False,
    ):
        """
        Initialize pipeline.
        
        Parameters:
        -----------
        file_path : str
            Path to input TIFF file
        output_dir : str, optional
            Output directory. If None, uses same directory as input file.
            If provided, a timestamp (YYYYMMDD_HHMMSS) is automatically appended to the directory name.
        config_path : str, optional
            Path to YAML config file. If None, uses default config.
        radius : int, optional
            Rolling ball radius for background subtraction. If None, background
            subtraction is skipped (default).
        correct_shift : bool
            Whether to apply illumination shift (baseline) correction.
        shift_frame : int, optional
            Frame where illumination shift starts (auto-detect if None).
        shift_end_frame : int, optional
            Frame where illumination shift ends (for multi-frame shifts).
        export_video : bool
            If True, export an MP4 video showing tracked tips.
        """
        import matplotlib
        _bk = matplotlib.get_backend().lower()
        if _bk not in ("agg", "svg", "pdf", "ps", "cairo", "template"):
            try:
                matplotlib.use("Agg")
            except RuntimeError:
                pass

        self.file_path = file_path
        # Set output directory and add timestamp (unless skip_output_timestamp)
        if output_dir:
            # If output_dir is a relative path (starts with './' or is just a folder name),
            # resolve it relative to the input file's directory instead of current working directory
            normalized_output = os.path.normpath(output_dir)
            is_relative = (
                output_dir.startswith('./') or 
                output_dir.startswith('.\\') or
                (not os.path.isabs(normalized_output) and not os.path.splitdrive(normalized_output)[0])
            )
            
            if is_relative:
                # Remove './' or '.\\' prefix if present
                if output_dir.startswith('./'):
                    clean_output = output_dir[2:]
                elif output_dir.startswith('.\\'):
                    clean_output = output_dir[2:]
                else:
                    clean_output = output_dir
                # Resolve relative to input file's directory
                input_dir = os.path.dirname(file_path)
                resolved_output = os.path.join(input_dir, clean_output)
                self.output_dir = resolved_output if skip_output_timestamp else add_timestamp_to_output_dir(resolved_output)
            else:
                # Absolute path - use as is
                self.output_dir = output_dir if skip_output_timestamp else add_timestamp_to_output_dir(output_dir)
        else:
            self.output_dir = os.path.dirname(file_path)
        self.config = load_config(config_path)

        # Apply profile matching file path so spots_to_track etc. are set even without CLI --profile
        if config_path and os.path.exists(config_path) and file_path:
            _profile_name, _profile_data = find_profile_by_path(config_path, file_path)
            if _profile_data:
                _apply_profile_to_config(self.config, _profile_data)
                logger.info("Applied profile '%s' (matched by file path) to config.", _profile_name)
        
        # Setup logging
        # Ensure output directory exists for log file
        os.makedirs(self.output_dir, exist_ok=True)
        log_file = os.path.join(self.output_dir, 'analysis.log')
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, mode='w', encoding='utf-8')
            ],
            force=True  # Override any existing configuration
        )
        
        # Preprocessing options
        self.radius = radius
        self.correct_shift = correct_shift
        self.shift_frame = shift_frame
        self.shift_end_frame = shift_end_frame
        self.export_video = export_video
        
        # Data storage
        self.tiff_stack = None
        self.first_frame = None
        self.skip_initial_frames = 0
        self.coordinates = None
        self.initial_coordinates = None
        self.tracked_spot_data = {}
        self.average_background_intensities = []
        self.average_data = None
        self.time_in_minutes = None
        
    def run(self, exclude_spots: Optional[list] = None):
        """
        Run complete analysis pipeline.
        
        Parameters:
        -----------
        exclude_spots : list, optional
            List of spot IDs to exclude from analysis
            
        Returns:
        --------
        results : dict
            Dictionary containing all analysis results
        """
        logger.info("=" * 60)
        logger.info("Starting Microneedle Analysis Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Load TIFF
        logger.info("\n[1/12] Loading TIFF file...")
        self.tiff_stack, self.first_frame = load_tiff(self.file_path)

        # Skip initial frames and optionally limit length (tracking config)
        track_config = self.config['tracking']
        skip_initial_frames = track_config.get('skip_initial_frames') or 0
        self.skip_initial_frames = skip_initial_frames
        if skip_initial_frames > 0:
            n_before = self.tiff_stack.shape[0]
            self.tiff_stack = self.tiff_stack[skip_initial_frames:]
            self.first_frame = self.tiff_stack[0]
            logger.info(f"Skipping first {skip_initial_frames} frames; analyzing {len(self.tiff_stack)} frames (original total {n_before}).")

        max_frame = track_config.get('max_frame')
        if max_frame is not None:
            if max_frame <= 0:
                logger.warning(f"tracking.max_frame={max_frame} is not positive. Ignoring limit.")
            else:
                n_total = self.tiff_stack.shape[0]
                limit = min(max_frame, n_total)
                if limit < n_total:
                    logger.info(f"Limiting analysis to first {limit} frames (of {n_total} total).")
                    self.tiff_stack = self.tiff_stack[:limit]
                    self.first_frame = self.tiff_stack[0]

        # Step 2: Optional illumination shift correction (baseline correction)
        if self.correct_shift:
            logger.info("\n[2/12] Correcting illumination shift (baseline correction)...")
            # Convert to indices relative to trimmed stack (after skip_initial_frames)
            shift_frame_eff = (self.shift_frame - self.skip_initial_frames) if self.shift_frame is not None else None
            shift_end_frame_eff = (self.shift_end_frame - self.skip_initial_frames) if self.shift_end_frame is not None else None
            if shift_frame_eff is not None and shift_frame_eff < 0:
                shift_frame_eff = 0
            if shift_end_frame_eff is not None and shift_end_frame_eff < 0:
                shift_end_frame_eff = 0
            self.tiff_stack, detected_frame, shift_amount = correct_illumination_shift(
                self.tiff_stack,
                shift_frame=shift_frame_eff,
                shift_end_frame=shift_end_frame_eff,
                auto_detect=(self.shift_frame is None),
                visualize=False,
            )
            if detected_frame is not None:
                if self.shift_end_frame is not None:
                    logger.info(f"  Shift applied from frame {detected_frame} to {self.shift_end_frame}")
                else:
                    logger.info(f"  Shift detected/applied at frame: {detected_frame}")
                logger.info(f"  Shift amount (average): {shift_amount:.2f}")
        else:
            logger.info("\n[2/12] Skipping illumination shift correction (not requested)")
        
        # Step 3: Optional background subtraction
        bg_config = self.config['background_subtraction']
        if self.radius is not None:
            logger.info("\n[3/12] Performing background subtraction...")
            self.tiff_stack, _ = subtract_background(
                self.tiff_stack,
                radius=self.radius,
                visualize=bg_config['visualize'],
            )
            self.first_frame = self.tiff_stack[0]
        else:
            logger.info("\n[3/12] Skipping background subtraction (no radius specified)")
            self.first_frame = self.tiff_stack[0]

        # Step 4: Export processed TIFF stack used for analysis
        # Use suffix consistent with preprocess: background-corrected vs generic processed
        logger.info("\n[4/12] Exporting processed TIFF stack used for analysis...")
        suffix = "_background_corrected" if self.radius is not None else "_processed"
        processed_tiff_path = export_tiff_stack(
            self.tiff_stack,
            self.file_path,
            output_dir=self.output_dir,
            suffix=suffix,
        )
        logger.info(f"Processed TIFF exported to: {processed_tiff_path}")
        
        # Step 5: Detect tips
        logger.info("\n[5/12] Detecting microneedle tips...")
        det_config = self.config['detection']
        self.coordinates = detect_tips(
            self.first_frame,
            min_distance=det_config['min_distance'],
            threshold_rel=det_config['threshold_rel'],
            visualize=det_config['visualize']
        )
        self.initial_coordinates = self.coordinates.copy()
        
        # Step 6: Track spots
        logger.info("\n[6/12] Tracking spots across frames...")
        track_config = self.config['tracking']
        spots_to_track = track_config.get('spots_to_track')
        if spots_to_track is not None:
            logger.info("spots_to_track from config: %s (tracking %d of %d detected spots)", spots_to_track, len(spots_to_track), len(self.coordinates))
        else:
            logger.info("spots_to_track: all (tracking all %d detected spots)", len(self.coordinates))
        # Convert ROI from list to tuple if provided
        bg_roi = track_config.get('background_roi')
        if bg_roi is not None:
            bg_roi = tuple(bg_roi)  # Convert list to tuple
        
        self.tracked_spot_data, self.average_background_intensities = track_spots(
            self.tiff_stack,
            self.first_frame,
            self.coordinates,
            spots_to_track=spots_to_track,
            search_range=track_config['search_range'],
            diameter=track_config['diameter'],
            min_distance=track_config['min_distance'],
            threshold_rel=track_config['threshold_rel'],
            background_roi=bg_roi
        )
        
        # Optional: export tracking video (before exclusions so all tracked data is shown)
        if self.export_video:
            tiff_file_name = os.path.splitext(os.path.basename(self.file_path))[0]
            video_save_path = os.path.join(
                self.output_dir,
                f"tracking_video_{tiff_file_name}.mp4",
            )
            logger.info(f"\n[Video] Exporting tracking video to: {video_save_path}")
            export_tracking_video(
                self.tiff_stack,
                self.tracked_spot_data,
                output_path=video_save_path,
                fps=10,
            )

        # Step 7: Exclude spots if needed
        if exclude_spots:
            logger.info(f"\nExcluding spots: {exclude_spots}")
            self.initial_coordinates = [
                coord for i, coord in enumerate(self.initial_coordinates)
                if i not in exclude_spots
            ]
            self.tracked_spot_data = {
                idx: data for idx, data in self.tracked_spot_data.items()
                if idx not in exclude_spots
            }
        
        # Step 8: Artifact Correction
        corr_config = self.config.get('artifact_correction', {})
        df_corrected = None  # Store corrected DataFrame for later use in CSV export
        self.artifact_correction_frames = {}  # Store frames where corrections occurred per spot
        if corr_config.get('enabled', False):
            logger.info("\n[7/12] Applying artifact correction...")
            
            # Convert tracked_spot_data to DataFrame format
            correction_data = []
            for spot_id, spot_data in self.tracked_spot_data.items():
                for row in spot_data:
                    frame_idx, x, y, intensity = row
                    correction_data.append({
                        'spot_id': spot_id,
                        'frame': int(frame_idx),
                        'mean_intensity': float(intensity)
                    })
            df = pd.DataFrame(correction_data)
            
            # Apply correction
            df_corrected = auto_correct_multistep(
                df,
                window=corr_config.get('window', 20),
                sigma_threshold=corr_config.get('sigma_threshold', 3.5),
                max_iter=corr_config.get('max_iter', 5),
                verbose=False
            )
            
            # Extract correction frames for each spot
            for spot_id in self.tracked_spot_data.keys():
                spot_df = df_corrected[df_corrected['spot_id'] == spot_id].sort_values('frame')
                # Get frames where correction was applied (correction_applied == 1)
                correction_frames = spot_df[spot_df['correction_applied'] == 1]['frame'].tolist()
                self.artifact_correction_frames[spot_id] = correction_frames
            
            # Update tracked_spot_data with corrected intensities
            for spot_id in self.tracked_spot_data.keys():
                spot_df = df_corrected[df_corrected['spot_id'] == spot_id].sort_values('frame')
                # Reconstruct the data structure: [frame, x, y, mean_intensity_corrected]
                original_data = self.tracked_spot_data[spot_id]
                corrected_data = []
                for i, row in enumerate(original_data):
                    frame_idx, x, y, _ = row
                    # Find corresponding corrected intensity
                    corrected_row = spot_df[spot_df['frame'] == int(frame_idx)]
                    if len(corrected_row) > 0:
                        corrected_intensity = corrected_row['mean_intensity_corrected'].iloc[0]
                        corrected_data.append([frame_idx, x, y, corrected_intensity])
                    else:
                        corrected_data.append(row)  # Keep original if not found
                self.tracked_spot_data[spot_id] = corrected_data
            
            # Count unique corrections per spot (corrections_count is duplicated per frame)
            total_corrections = df_corrected.groupby('spot_id')['corrections_count'].max().sum()
            logger.info(f"Artifact correction completed. Total corrections applied: {total_corrections}")
        else:
            logger.info("\n[7/12] Skipping artifact correction (disabled in config)")
            self.artifact_correction_frames = {}
        
        # Step 9: Background-corrected intensity and normalization
        # All sensor intensity used for analysis is background-corrected (tip/background ratio)
        # before normalization, baseline correction, averaging, or export.
        logger.info("\n[8/12] Computing background-corrected intensities and normalizing ratios...")
        norm_config = self.config['normalization']
        # Convert to indices relative to trimmed stack (after skip_initial_frames)
        n_frames_bg = len(self.average_background_intensities)
        start_frame = max(0, norm_config['starting_frame'] - self.skip_initial_frames)
        end_frame = norm_config['ending_frame'] - self.skip_initial_frames
        if end_frame <= start_frame:
            end_frame = n_frames_bg
        end_frame = min(end_frame, n_frames_bg)
        window_frames = max(0, end_frame - start_frame)
        logger.info(
            "[8/12] Normalization window frames=%s | start_frame=%s end_frame=%s | "
            "raw_norm_start=%s raw_norm_end=%s | skip_initial_frames=%s | n_frames_bg=%s",
            window_frames,
            start_frame,
            end_frame,
            norm_config.get('starting_frame'),
            norm_config.get('ending_frame'),
            self.skip_initial_frames,
            n_frames_bg,
        )

        # Convert background intensities to numpy array for vectorized operations
        bg_array = np.array(self.average_background_intensities, dtype=float)

        final_corrected_spots: Dict[int, pd.DataFrame] = {}
        all_normalized_rows = []

        for spot_id, spot_data in self.tracked_spot_data.items():
            df = pd.DataFrame(spot_data, columns=['frame', 'x', 'y', 'mean_intensity'])
            frames = df['frame'].astype(int).values
            intensities = df['mean_intensity'].astype(float).values

            # Ensure frames are within background array bounds
            valid_mask = (frames >= 0) & (frames < len(bg_array))
            if not np.all(valid_mask):
                logger.warning(
                    f"Spot {spot_id}: some frames are outside background range; "
                    "clamping to available background frames."
                )
                frames = np.clip(frames, 0, len(bg_array) - 1)

            bg_values = bg_array[frames]

            # Calculate raw background-corrected intensity (tip / background)
            ratio = np.zeros_like(intensities, dtype=float)
            positive_bg = bg_values > 0
            ratio[positive_bg] = intensities[positive_bg] / bg_values[positive_bg]
            # If background is zero or negative, leave ratio at 0 (or could be np.nan)

            df['background_intensity'] = bg_values
            df['background_corrected_intensity'] = ratio

            # Normalize background-corrected intensity using frames [start_frame, end_frame)
            norm_window_mask = (
                (frames >= start_frame) &
                (frames < end_frame) &
                positive_bg
            )
            # Log baseline details once to help validate normalization window overrides.
            if spot_id == next(iter(self.tracked_spot_data.keys()), None):
                logger.info(
                    "[8/12] Spot %s normalization frames used=%s (positive_bg=%s)",
                    spot_id,
                    int(np.sum(norm_window_mask)),
                    int(np.sum(positive_bg)),
                )
            if np.any(norm_window_mask):
                baseline = np.mean(ratio[norm_window_mask])
            else:
                # Fallback: use all positive-bg frames; if still none, default to 1.0
                if np.any(positive_bg):
                    baseline = np.mean(ratio[positive_bg])
                else:
                    baseline = 1.0
                    logger.warning(
                        f"Spot {spot_id}: no valid frames for normalization; "
                        "using baseline=1.0."
                    )

            if baseline == 0 or not np.isfinite(baseline):
                baseline = 1.0
                logger.warning(
                    f"Spot {spot_id}: baseline for normalization is zero or NaN; "
                    "forcing baseline=1.0."
                )

            normalized_ratio = ratio / baseline
            df['normalized_intensity'] = normalized_ratio

            final_corrected_spots[spot_id] = df
            all_normalized_rows.append(df[['frame', 'normalized_intensity']])

        # Background ratio is 1 by definition (background / background)
        normalized_background = np.ones(len(bg_array), dtype=float)

        # Compute average normalized intensity with std per frame across spots
        if all_normalized_rows:
            all_data = pd.concat(all_normalized_rows)
            self.average_data = all_data.groupby('frame')['normalized_intensity'].agg(['mean', 'std'])
        else:
            self.average_data = pd.DataFrame({'mean': [], 'std': []})

        # Calculate time array
        viz_config = self.config['visualization']
        exposure_per_frame = self.config.get(
            'exposure_per_frame',
            viz_config.get('exposure_per_frame', 30),
        )
        self.time_in_minutes = np.arange(len(self.average_data)) * exposure_per_frame / 60
        
        logger.info("Normalization completed.")
        
        # Step 10: Smoothing & Analysis
        logger.info("\n[9/12] Performing smoothing and analysis...")
        smooth_config = self.config['smoothing']
        
        processing_results = {}
        spot_ids = []
        min_times = []
        
        for spot_id, spot_data in self.tracked_spot_data.items():
            df_norm = final_corrected_spots[spot_id]
            normalized_for_smoothing = df_norm['normalized_intensity'].values

            smoothed = exponential_smoothing(
                normalized_for_smoothing,
                alpha=smooth_config['alpha']
            )
            
            # Find minimum
            df_smooth = pd.DataFrame({
                'frame': df_norm['frame'].values,
                'smoothed_intensity': smoothed
            })
            time_in_minutes_spot = df_norm['frame'].values * exposure_per_frame / 60
            
            try:
                min_time, min_intensity, min_frame = find_minima(
                    df_smooth,
                    time_in_minutes_spot,
                    min_time_range=tuple(smooth_config['min_time_range']),
                    exposure_per_frame=exposure_per_frame
                )
            except ValueError as e:
                logger.warning(f"Warning for spot {spot_id}: {e}")
                min_time = np.nan
                min_intensity = np.nan
                min_frame = 0
            
            processing_results[spot_id] = {
                'baseline_corrected': normalized_for_smoothing,  # normalized ratio (input to smoothing)
                'smoothed': smoothed,
                't_min': min_time,
                'min_intensity': min_intensity,
                'min_frame': min_frame,
                'time_in_minutes': time_in_minutes_spot
            }
            
            spot_ids.append(spot_id)
            min_times.append(min_time)
        
        logger.info("Smoothing and analysis completed.")

        cohort_qc_result: Optional[Dict[str, Any]] = None
        cohort_cfg = self.config.get("cohort_qc", {})
        if cohort_cfg.get("enabled", True):
            mad_lambda = float(cohort_cfg.get("mad_lambda", 3.0))
            cohort_qc_result = compute_cohort_qc(
                final_corrected_spots,
                smoothing_alpha=smooth_config["alpha"],
                mad_lambda=mad_lambda,
                column_key="normalized_intensity",
            )
            try:
                cq_df = pd.DataFrame(
                    {
                        "spot_id": cohort_qc_result.get("spot_ids", []),
                        "rms_distance": cohort_qc_result.get("rms_distance", []),
                        "flagged_outlier": cohort_qc_result.get("flagged", []),
                    }
                )
                if not cq_df.empty:
                    cq_df["threshold"] = cohort_qc_result.get("threshold", np.nan)
                else:
                    logger.info("Cohort QC: no tips in matrix; cohort_qc.csv is empty.")
                cohort_csv = os.path.join(self.output_dir, "cohort_qc.csv")
                cq_df.to_csv(cohort_csv, index=False)
                logger.info("Cohort QC table saved to %s", cohort_csv)
            except Exception as e:
                logger.warning("Could not save cohort_qc.csv: %s", e)
            try:
                trend_smoothing_cfg = smooth_config.get("trend_smoothing")
                run_all_cohort_qc_plots(
                    cohort_qc_result,
                    value_label="Smoothed normalized intensity",
                    output_dir=self.output_dir,
                    format=viz_config.get("format", "svg"),
                    trend_smoothing=trend_smoothing_cfg,
                )
                logger.info("Cohort QC figures saved under %s", self.output_dir)
            except Exception as e:
                logger.warning("Cohort QC plots failed: %s", e, exc_info=True)
        
        # Step 11: Correlation (removed from workflow)
        # Step 11: Export Data
        logger.info("\n[11/12] Exporting comprehensive data...")
        
        # Create comprehensive CSV with all processing steps
        comprehensive_data = []
        for spot_id in spot_ids:
            spot_data = self.tracked_spot_data[spot_id]
            df_tracked = pd.DataFrame(spot_data, columns=['frame', 'x', 'y', 'mean_intensity'])
            
            # Get raw and corrected intensities
            if corr_config.get('enabled', False) and df_corrected is not None:
                # Get raw and corrected intensities from df_corrected
                spot_df_corr = df_corrected[df_corrected['spot_id'] == spot_id].sort_values('frame')
                # Merge with tracked data to get x, y coordinates
                df_combined = df_tracked[['frame', 'x', 'y']].merge(
                    spot_df_corr[['frame', 'mean_intensity', 'mean_intensity_corrected']],
                    on='frame',
                    how='left'
                )
                raw_intensity = df_combined['mean_intensity'].values  # Raw from before correction
                corrected_intensity = df_combined['mean_intensity_corrected'].values  # Corrected
            else:
                # No artifact correction - raw and corrected are the same
                raw_intensity = df_tracked['mean_intensity'].values
                corrected_intensity = df_tracked['mean_intensity'].values
                df_combined = df_tracked
            
            # Get normalized/background-corrected data for this spot
            df_norm = final_corrected_spots[spot_id]
            
            # Get processing results
            proc = processing_results[spot_id]
            
            # Combine all data
            df_comprehensive = pd.DataFrame({
                'spot_id': spot_id,
                'frame': df_combined['frame'].values,
                'x': df_combined['x'].values,
                'y': df_combined['y'].values,
                'raw_intensity': raw_intensity,
                'corrected_intensity': corrected_intensity,
                # Background and background-corrected (tip/background) intensity
                'background_intensity': df_norm['background_intensity'].values,
                'background_corrected_intensity': df_norm['background_corrected_intensity'].values,
                # Normalized ratio (background-corrected intensity normalized over the chosen window)
                'normalized_intensity': df_norm['normalized_intensity'].values,
                # Background ratio is 1 by construction after normalization step
                'normalized_background': normalized_background[:len(df_norm)],
                # Smoothing results
                'smoothed': proc['smoothed'],
                'time_minutes': proc['time_in_minutes']
            })
            
            comprehensive_data.append(df_comprehensive)
        
        comprehensive_df = pd.concat(comprehensive_data, ignore_index=True)
        comprehensive_csv_path = os.path.join(self.output_dir, 'comprehensive_analysis_data.csv')
        comprehensive_df.to_csv(comprehensive_csv_path, index=False)
        logger.info(f"Comprehensive data exported to: {comprehensive_csv_path}")
        
        logger.info("Data export completed.")
        
        # Step 13: Visualization
        logger.info("\n[12/12] Creating visualizations...")
        
        # Tracking visualization (filenames independent of original TIFF name)
        viz_save_path = os.path.join(
            self.output_dir,
            'tracked_spots_visualization'
        )
        
        # Shift frame indices for plots are relative to trimmed stack
        shift_frame_viz = (self.shift_frame - self.skip_initial_frames) if self.shift_frame is not None else None
        shift_end_frame_viz = (self.shift_end_frame - self.skip_initial_frames) if self.shift_end_frame is not None else None
        if shift_frame_viz is not None and shift_frame_viz < 0:
            shift_frame_viz = None
        if shift_end_frame_viz is not None and shift_end_frame_viz < 0:
            shift_end_frame_viz = None
        # Treatment frame index for plots (relative to trimmed stack)
        treatment_frame_cfg = self.config.get("visualization", {}).get("treatment_frame")
        treatment_frame_viz = None
        if treatment_frame_cfg is not None:
            treatment_frame_viz = treatment_frame_cfg - self.skip_initial_frames
            if treatment_frame_viz < 0:
                treatment_frame_viz = None
        plot_tracking_results(
            self.first_frame,
            self.tracked_spot_data,
            self.average_background_intensities,
            normalized_data=final_corrected_spots,
            normalized_background=normalized_background,
            average_data=self.average_data,
            starting_frame=start_frame,
            ending_frame=end_frame,
            exposure_per_frame=exposure_per_frame,
            save_path=viz_save_path,
            format=viz_config['format'],
            shift_frame=shift_frame_viz,
            shift_end_frame=shift_end_frame_viz,
            artifact_correction_frames=self.artifact_correction_frames,
            treatment_frame=treatment_frame_viz,
        )
        
        # Export tracking data (raw tracking export)
        export_tracking_data(self.tracked_spot_data, self.file_path, self.output_dir)
        export_average_intensity(
            self.average_data, self.time_in_minutes, self.file_path, self.output_dir
        )
        
        logger.info("Visualization completed.")
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("=" * 60)
        
        return {
            'tracked_spot_data': self.tracked_spot_data,
            'normalized_data': final_corrected_spots,
            'normalized_background': normalized_background,
            'average_data': self.average_data,
            'time_in_minutes': self.time_in_minutes,
            'processing_results': processing_results,
            'spot_ids': spot_ids,
            'min_times': min_times,
            'comprehensive_df': comprehensive_df,
            'cohort_qc': cohort_qc_result,
        }


def run_pipeline(
    file_path: str,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    exclude_spots: Optional[list] = None,
    radius: Optional[int] = None,
    correct_shift: bool = False,
    shift_frame: Optional[int] = None,
    shift_end_frame: Optional[int] = None,
    spots_to_track_override: Optional[list] = None,
    export_video: bool = False,
    max_frame_override: Optional[int] = None,
    skip_initial_frames_override: Optional[int] = None,
    normalization_starting_frame_override: Optional[int] = None,
    normalization_ending_frame_override: Optional[int] = None,
    profile_data: Optional[Dict[str, Any]] = None,
    skip_output_timestamp: bool = False,
):
    """
    Convenience function to run the complete pipeline.
    
    Parameters:
    -----------
    file_path : str
        Path to input TIFF file
    output_dir : str, optional
        Output directory
    config_path : str, optional
        Path to YAML config file
    exclude_spots : list, optional
        List of spot IDs to exclude
    radius : int, optional
        Rolling ball radius for background subtraction. If None, background
        subtraction is skipped (default).
    correct_shift : bool
        Whether to apply illumination shift (baseline) correction.
    shift_frame : int, optional
        Frame where illumination shift starts (auto-detect if None).
    shift_end_frame : int, optional
        Frame where illumination shift ends (for multi-frame shifts).
    spots_to_track_override : list, optional
        If provided, overrides tracking.spots_to_track from config.
    export_video : bool
        If True, export an MP4 video showing tracked tips.
    max_frame_override : int, optional
        If provided, overrides tracking.max_frame from config for this run.
    skip_initial_frames_override : int, optional
        If provided, overrides tracking.skip_initial_frames from config for this run.
    normalization_starting_frame_override : int, optional
        If provided, overrides normalization.starting_frame from config for this run.
    normalization_ending_frame_override : int, optional
        If provided, overrides normalization.ending_frame from config for this run.
    profile_data : dict, optional
        If provided (e.g. from --profile or auto-detected profile), profile keys are merged
        into config so that spots_to_track, max_frame, etc. are applied.
    Returns:
    --------
    results : dict
        Analysis results
    """
    pipeline = MicroneedlePipeline(
        file_path,
        output_dir,
        config_path,
        radius=radius,
        correct_shift=correct_shift,
        shift_frame=shift_frame,
        shift_end_frame=shift_end_frame,
        export_video=export_video,
        skip_output_timestamp=skip_output_timestamp,
    )
    # Merge profile_data into config so profile settings (e.g. spots_to_track) are applied
    _apply_profile_to_config(pipeline.config, profile_data or {})
    # Explicit overrides (CLI or passed args) take precedence
    if spots_to_track_override is not None:
        pipeline.config["tracking"]["spots_to_track"] = spots_to_track_override
    # If max_frame_override is provided, override tracking.max_frame
    if max_frame_override is not None:
        pipeline.config["tracking"]["max_frame"] = max_frame_override
    # If skip_initial_frames override is provided, override tracking.skip_initial_frames
    if skip_initial_frames_override is not None:
        pipeline.config["tracking"]["skip_initial_frames"] = skip_initial_frames_override
    # If normalization overrides are provided, override normalization parameters
    if normalization_starting_frame_override is not None:
        pipeline.config["normalization"]["starting_frame"] = normalization_starting_frame_override
    if normalization_ending_frame_override is not None:
        pipeline.config["normalization"]["ending_frame"] = normalization_ending_frame_override
    return pipeline.run(exclude_spots=exclude_spots)


def discover_iaa_ref_files(
    folder_path: str,
    keywords: List[str],
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Discover TIFF files in folder_path and subfolders, partitioned by keyword.
    
    Walks folder_path to find leaf directories containing .tif files. For each
    such directory, partitions files by keyword (case-insensitive substring match).
    
    Parameters:
    -----------
    folder_path : str
        Root folder to scan (e.g., path to Fig-3)
    keywords : list of str
        Keywords to match in filenames (e.g., ['IAA', 'Ref'])
        
    Returns:
    --------
    mapping : dict
        { experiment_folder_abs_path: [(file_path, keyword), ...], ... }
        Each key is the absolute path of an experiment folder (leaf dir with .tif).
        Each value is a list of (file_path, keyword) for files in that folder.
    """
    folder_path = os.path.abspath(folder_path)
    if not os.path.isdir(folder_path):
        return {}

    def _is_output_dir(dirname: str) -> bool:
        """Skip output directories from previous runs."""
        d = dirname.lower()
        return (
            d.startswith("output_timestamped")
            or d == "iaa_output"
            or d == "ref_output"
        )

    keywords_lower = [k.lower() for k in keywords]
    result: Dict[str, List[Tuple[str, str]]] = {}

    for root, dirs, files in os.walk(folder_path):
        # Do not descend into output directories from previous runs
        dirs[:] = [d for d in dirs if not _is_output_dir(d)]
        tif_files = [f for f in files if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]
        if not tif_files:
            continue
        
        # Consider this directory an "experiment folder" if it contains .tif files
        exp_folder = os.path.abspath(root)
        matched: List[Tuple[str, str]] = []
        
        for tif_file in tif_files:
            file_path = os.path.join(root, tif_file)
            fname_lower = tif_file.lower()
            for i, kw in enumerate(keywords_lower):
                if kw in fname_lower:
                    matched.append((file_path, keywords[i]))
                    break
        
        if matched:
            result[exp_folder] = matched
    
    return result


def run_iaa_ref_folder_pipeline(
    folder_path: str,
    keywords: List[str],
    output_subdirs: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    profile_data: Optional[Dict[str, Any]] = None,
    export_video: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run pipeline on IAA and Ref TIFF files discovered in folder_path.
    
    For each experiment folder (leaf dir with .tif files), creates
    output_timestamped/IAA_output and output_timestamped/Ref_output, then runs
    the pipeline on each matched file.
    
    Parameters:
    -----------
    folder_path : str
        Root folder to scan
    keywords : list of str
        Keywords to match (e.g., ['IAA', 'Ref'])
    output_subdirs : list of str, optional
        Output subfolder names per keyword (e.g., ['IAA_output', 'Ref_output']).
        If None, uses [f'{kw}_output' for kw in keywords].
    config_path : str, optional
        Path to YAML config
    profile_data : dict, optional
        Profile data (correct_shift, spots_to_track, etc.) from folder_keywords profile
    export_video : bool
        If True, export tracking video per file
        
    Returns:
    --------
    results : dict
        { experiment_folder: [result_dict1, result_dict2, ...], ... }
    """
    profile_data = profile_data or {}
    # Strip output_dir from profile to prevent pipeline from using it (we use our computed path)
    profile_data_no_output = {k: v for k, v in profile_data.items() if k != "output_dir"}

    if output_subdirs is None:
        output_subdirs = [f"{kw}_output" for kw in keywords]

    # Ensure keyword -> output_subdir mapping
    keyword_to_subdir = {}
    for i, kw in enumerate(keywords):
        keyword_to_subdir[kw] = output_subdirs[i] if i < len(output_subdirs) else f"{kw}_output"

    discovered = discover_iaa_ref_files(folder_path, keywords)
    if not discovered:
        logger.warning("No TIFF files matching keywords %s found in %s", keywords, folder_path)
        return {}

    # Single timestamp for entire run (shared across all experiment folders)
    folder_path_abs = os.path.abspath(folder_path)
    base_output_root = add_timestamp_to_output_dir(os.path.join(folder_path_abs, "output_timestamped"))

    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for exp_folder, file_list in discovered.items():
        # Use single output_timestamped; place each exp_folder's outputs under it
        # When folder_path is Exp-1: base_output = base_output_root (IAA_output, Ref_output directly inside)
        # When folder_path is Fig-3 with Exp-1, Exp-2: base_output = base_output_root/Fig3-Bokchoi/Exp-1 etc.
        if os.path.normpath(exp_folder) == os.path.normpath(folder_path_abs):
            base_output = base_output_root
        else:
            exp_rel = os.path.relpath(exp_folder, folder_path_abs)
            base_output = os.path.join(base_output_root, exp_rel)

        results_by_keyword: Dict[str, Dict[str, Any]] = {}

        for file_path, keyword in file_list:
            subdir = keyword_to_subdir.get(keyword, f"{keyword}_output")
            output_dir = os.path.join(base_output, subdir)
            os.makedirs(output_dir, exist_ok=True)
            
            if exp_folder not in all_results:
                all_results[exp_folder] = []
            
            # Per-keyword spots_to_track: spots_to_track_IAA, spots_to_track_Ref, or spots_to_track
            spots_key = f"spots_to_track_{keyword}"
            spots_override = profile_data_no_output.get(spots_key) or profile_data_no_output.get("spots_to_track")
            
            logger.info("Processing %s: %s -> %s", keyword, file_path, output_dir)
            
            try:
                result = run_pipeline(
                    file_path=file_path,
                    output_dir=output_dir,
                    config_path=config_path,
                    radius=profile_data_no_output.get("radius"),
                    correct_shift=profile_data_no_output.get("correct_shift", False),
                    shift_frame=profile_data_no_output.get("shift_frame"),
                    shift_end_frame=profile_data_no_output.get("shift_end_frame"),
                    spots_to_track_override=spots_override,
                    max_frame_override=profile_data_no_output.get("max_frame"),
                    skip_initial_frames_override=profile_data_no_output.get("skip_initial_frames"),
                    normalization_starting_frame_override=profile_data_no_output.get("normalization_starting_frame"),
                    normalization_ending_frame_override=profile_data_no_output.get("normalization_ending_frame"),
                    profile_data=profile_data_no_output,
                    export_video=export_video,
                    skip_output_timestamp=True,
                )
                all_results[exp_folder].append(result)
                results_by_keyword[keyword] = result
            except Exception as e:
                logger.exception("Failed to process %s: %s", file_path, e)
                raise

        # Ratiometric panel when both IAA and Ref are present
        if 'IAA' in results_by_keyword and 'Ref' in results_by_keyword:
            # Merge profile into config (same as run_pipeline). load_config() alone does not
            # include profile blocks — e.g. visualization.ratiometric_iaa_ref_ylim on the profile.
            config = load_config(config_path)
            _apply_profile_to_config(config, profile_data_no_output)
            viz_format = config.get('visualization', {}).get('format', 'svg')
            ylim_raw = config.get('visualization', {}).get('ratiometric_iaa_ref_ylim')
            ratiometric_ylim = _parse_ratiometric_iaa_ref_ylim(ylim_raw)
            smoothing_alpha = float(config.get('smoothing', {}).get('alpha', 0.05))
            ratiometric_path = os.path.join(base_output, 'ratiometric_IAA_Ref')
            try:
                plot_ratiometric_iaa_ref_panel(
                    iaa_result=results_by_keyword['IAA'],
                    ref_result=results_by_keyword['Ref'],
                    save_path=ratiometric_path,
                    format=viz_format,
                    iaa_label='IAA',
                    ref_label='Ref',
                    ylim=ratiometric_ylim,
                    smoothing_alpha=smoothing_alpha,
                )
                logger.info("Ratiometric panel saved under %s", ratiometric_path)
            except Exception as e:
                logger.warning("Ratiometric panel failed: %s", e)

            iaa_qc = results_by_keyword['IAA'].get('cohort_qc')
            ref_qc = results_by_keyword['Ref'].get('cohort_qc')
            flagged_iaa = _flagged_spot_ids_from_cohort_qc(iaa_qc)
            flagged_ref = _flagged_spot_ids_from_cohort_qc(ref_qc)
            if flagged_iaa or flagged_ref:
                exclusion_path = os.path.join(base_output, 'ratiometric_IAA_Ref_flagged_tips_exclusion')
                try:
                    plot_ratiometric_iaa_ref_panel(
                        iaa_result=results_by_keyword['IAA'],
                        ref_result=results_by_keyword['Ref'],
                        save_path=exclusion_path,
                        format=viz_format,
                        iaa_label='IAA',
                        ref_label='Ref',
                        ylim=ratiometric_ylim,
                        smoothing_alpha=smoothing_alpha,
                        exclude_spot_ids_iaa=flagged_iaa,
                        exclude_spot_ids_ref=flagged_ref,
                    )
                    logger.info(
                        "Ratiometric panel (excluding flagged tips) saved under %s",
                        exclusion_path,
                    )
                except Exception as e:
                    logger.warning("Ratiometric flagged-exclusion panel failed: %s", e)
            else:
                logger.debug(
                    "Skipping ratiometric_IAA_Ref_flagged_tips_exclusion (no cohort QC flags on IAA or Ref)."
                )

    return all_results

