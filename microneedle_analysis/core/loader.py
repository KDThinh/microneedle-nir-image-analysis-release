"""TIFF file loading and background subtraction."""

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.restoration import rolling_ball


def load_tiff(file_path):
    """
    Load TIFF stack from file.
    
    Parameters:
    -----------
    file_path : str
        Path to the TIFF file
        
    Returns:
    --------
    tiff_stack : np.ndarray
        TIFF stack array with shape (frames, height, width)
    first_frame : np.ndarray
        First frame of the stack
        
    Raises:
    -------
    FileNotFoundError
        If the file is not found
    Exception
        If an error occurs during loading
    """
    try:
        tiff_stack = tiff.imread(file_path)
        first_frame = tiff_stack[0]
        print(f"Loaded successfully. TIFF stack shape (frame, y, x): {tiff_stack.shape}")
        return tiff_stack, first_frame
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    except Exception as e:
        raise Exception(f"An error occurred loading TIFF: {e}")


def subtract_background(tiff_stack, radius=25, visualize=False):
    """
    Perform rolling ball background subtraction.
    
    Parameters:
    -----------
    tiff_stack : np.ndarray
        Input TIFF stack
    radius : int
        Radius of the rolling ball in pixels
    visualize : bool
        Whether to display visualization of background subtraction
        
    Returns:
    --------
    subtracted_stack : np.ndarray
        Background-subtracted TIFF stack
    background_stack : np.ndarray
        Background stack (for visualization)
    """
    background_stack = np.zeros_like(tiff_stack, dtype=np.float64)
    subtracted_stack = np.zeros_like(tiff_stack, dtype=np.float64)
    
    for i in range(tiff_stack.shape[0]):
        background_stack[i] = rolling_ball(tiff_stack[i], radius=radius)
        subtracted_stack[i] = tiff_stack[i] - background_stack[i]
    
    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        im1 = axes[0].imshow(tiff_stack[0], cmap='gray')
        axes[0].set_title('Original First Frame')
        plt.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
        
        im2 = axes[1].imshow(subtracted_stack[0], cmap='gray')
        axes[1].set_title('Background-Subtracted First Frame')
        plt.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
        
        im3 = axes[2].imshow(background_stack[0], cmap='gray')
        axes[2].set_title('Difference Image (First Frame)')
        plt.colorbar(im3, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
    
    return subtracted_stack, background_stack


def correct_illumination_shift(tiff_stack, shift_frame=None, shift_end_frame=None,
                                pre_shift_window=10, post_shift_window=10, 
                                auto_detect=True, detection_threshold=0.1, visualize=False):
    """
    Correct for sudden illumination increase affecting all pixel intensities.
    
    This function corrects for baseline shifts caused by light environment changes
    (e.g., switching from low light to higher light conditions). It can handle both
    instantaneous shifts (single frame) and gradual shifts (multiple frames).
    
    Parameters:
    -----------
    tiff_stack : np.ndarray
        Input TIFF stack with shape (frames, height, width)
    shift_frame : int, optional
        Frame number where light environment change starts.
        If None and auto_detect=True, will attempt to detect automatically.
        If None and auto_detect=False, raises ValueError.
    shift_end_frame : int, optional
        Frame number where light environment change ends (for multi-frame shifts).
        If None, shift is treated as instantaneous at shift_frame.
        If provided, shift is treated as gradual from shift_frame to shift_end_frame.
    pre_shift_window : int
        Number of frames before shift_frame to use for baseline calculation
    post_shift_window : int
        Number of frames after shift_frame (or shift_end_frame if provided) to use for shift calculation
    auto_detect : bool
        If True, automatically detect the shift frame by analyzing frame-to-frame
        intensity changes. If False, shift_frame must be provided.
    detection_threshold : float
        Relative threshold for detecting shift (fractional increase in average intensity).
        Used only when auto_detect=True.
    visualize : bool
        Whether to display visualization of the correction
        
    Returns:
    --------
    corrected_stack : np.ndarray
        Illumination-shift-corrected TIFF stack
    shift_frame : int
        Frame number where shift starts
    shift_amount : float
        Calculated shift amount that was subtracted
    """
    if tiff_stack.ndim != 3:
        raise ValueError(f"Expected 3D array (frames, height, width), got {tiff_stack.ndim}D")
    
    n_frames, height, width = tiff_stack.shape
    
    # Auto-detect shift frame if requested
    if auto_detect and shift_frame is None:
        print("Auto-detecting illumination shift frame...")
        shift_frame = _detect_illumination_shift(tiff_stack, threshold=detection_threshold)
        if shift_frame is None:
            print("Warning: Could not detect illumination shift. Returning original stack.")
            return tiff_stack.copy(), None, 0.0
        print(f"Detected illumination shift at frame {shift_frame}")
    elif shift_frame is None:
        raise ValueError("shift_frame must be provided when auto_detect=False")
    
    # Validate shift_frame
    if shift_frame < 0 or shift_frame >= n_frames:
        raise ValueError(f"shift_frame {shift_frame} is out of range [0, {n_frames-1}]")
    
    # Handle multi-frame shift range
    if shift_end_frame is not None:
        if shift_end_frame <= shift_frame:
            raise ValueError(f"shift_end_frame ({shift_end_frame}) must be greater than shift_frame ({shift_frame})")
        if shift_end_frame >= n_frames:
            raise ValueError(f"shift_end_frame {shift_end_frame} is out of range [0, {n_frames-1}]")
        
        # For multi-frame shifts: calculate baseline before shift starts and after shift ends
        # Calculate pixel-wise baselines (mean across frames, but per pixel)
        pre_start = max(0, shift_frame - pre_shift_window)
        pre_end = shift_frame
        pre_shift_frames = tiff_stack[pre_start:pre_end]
        pre_shift_baseline = np.mean(pre_shift_frames, axis=0)  # Shape: (height, width) - per pixel
        
        # Post-shift baseline: average intensity after the transition completes
        post_start = shift_end_frame + 1
        post_end = min(n_frames, post_start + post_shift_window)
        if post_end <= post_start:
            post_end = min(n_frames, shift_end_frame + post_shift_window)
            post_start = shift_end_frame + 1
        post_shift_frames = tiff_stack[post_start:post_end]
        post_shift_baseline = np.mean(post_shift_frames, axis=0)  # Shape: (height, width) - per pixel
        
        # Calculate pixel-wise shift amount
        # Positive shift_amount_map  -> illumination increased after shift (we subtract to bring it down)
        # Negative shift_amount_map -> illumination decreased after shift (we subtract a negative, i.e. add,
        #                               to bring it up to the pre-shift baseline)
        shift_amount_map = post_shift_baseline - pre_shift_baseline  # Shape: (height, width)
        shift_amount_mean = np.mean(shift_amount_map)  # For reporting
        
        print(f"Multi-frame shift detected: frames {shift_frame} to {shift_end_frame}")
        print(f"Pre-shift baseline intensity (before frame {shift_frame}): {np.mean(pre_shift_baseline):.2f} (pixel-wise)")
        print(f"Post-shift baseline intensity (after frame {shift_end_frame}): {np.mean(post_shift_baseline):.2f} (pixel-wise)")
        print(f"Calculated shift amount (average): {shift_amount_mean:.2f} (pixel-wise correction)")
        if shift_amount_mean < 0:
            print("Detected negative shift (illumination decreased). "
                  "Lifting post-shift frames up to match pre-shift baseline.")
        else:
            print("Detected positive shift (illumination increased). "
                  "Lowering post-shift frames down to match pre-shift baseline.")
        print(f"Applying pixel-wise correction with interpolation during transition frames...")
        
        # Apply pixel-wise correction with interpolation during transition
        corrected_stack = tiff_stack.copy().astype(np.float64)
        
        # For transition frames, interpolate corrected values smoothly
        # Strategy: Interpolate the corrected values between the frame before transition
        # and the target corrected value (pre_shift_baseline level)
        transition_span = shift_end_frame - shift_frame
        if transition_span > 0:
            # Get the frame just before transition as reference
            reference_frame = max(0, shift_frame - 1)
            reference_value = corrected_stack[reference_frame].astype(np.float64)
            
            # Target corrected value after full correction (should be pre_shift_baseline level)
            # After full correction: post_shift_baseline - shift_amount_map = pre_shift_baseline
            target_corrected = pre_shift_baseline
            
            for i in range(shift_frame, shift_end_frame + 1):
                # Linear interpolation progress: 0 at shift_frame, 1 at shift_end_frame
                progress = (i - shift_frame) / transition_span
                
                # Interpolate corrected value between reference (before transition) and target
                # This ensures smooth transition without spikes
                corrected_stack[i] = reference_value * (1 - progress) + target_corrected * progress
        else:
            # Single frame transition: apply full correction
            corrected_stack[shift_frame] -= shift_amount_map
        
        # Apply full correction to frames after transition completes
        corrected_stack[shift_end_frame + 1:] -= shift_amount_map[np.newaxis, :, :]
        
    else:
        # Single-frame shift
        # Calculate pixel-wise baseline intensities (mean across frames, but per pixel)
        # Pre-shift baseline: average intensity before the shift
        pre_start = max(0, shift_frame - pre_shift_window)
        pre_end = shift_frame
        pre_shift_frames = tiff_stack[pre_start:pre_end]
        pre_shift_baseline = np.mean(pre_shift_frames, axis=0)  # Shape: (height, width) - per pixel
        
        # Post-shift baseline: average intensity immediately after the shift
        post_start = shift_frame
        post_end = min(n_frames, shift_frame + post_shift_window)
        post_shift_frames = tiff_stack[post_start:post_end]
        post_shift_baseline = np.mean(post_shift_frames, axis=0)  # Shape: (height, width) - per pixel
        
        # Calculate pixel-wise shift amount
        # Positive shift_amount_map  -> illumination increased after shift (we subtract to bring it down)
        # Negative shift_amount_map -> illumination decreased after shift (we subtract a negative, i.e. add,
        #                               to bring it up to the pre-shift baseline)
        shift_amount_map = post_shift_baseline - pre_shift_baseline  # Shape: (height, width)
        shift_amount_mean = np.mean(shift_amount_map)  # For reporting
        
        print(f"Pre-shift baseline intensity: {np.mean(pre_shift_baseline):.2f} (pixel-wise)")
        print(f"Post-shift baseline intensity: {np.mean(post_shift_baseline):.2f} (pixel-wise)")
        print(f"Calculated shift amount (average): {shift_amount_mean:.2f} (pixel-wise correction)")
        if shift_amount_mean < 0:
            print("Detected negative shift (illumination decreased). "
                  "Lifting post-shift frames up to match pre-shift baseline.")
        else:
            print("Detected positive shift (illumination increased). "
                  "Lowering post-shift frames down to match pre-shift baseline.")
        print(f"Applying pixel-wise correction to frames {shift_frame} onwards...")
        
        # Apply pixel-wise correction: subtract shift map from all frames after shift_frame
        # Note: if shift_amount_map is negative, subtracting it will *increase* intensities appropriately.
        corrected_stack = tiff_stack.copy().astype(np.float64)
        # Broadcasting: shift_amount_map shape (H, W) is subtracted from each frame (H, W)
        corrected_stack[shift_frame:] -= shift_amount_map[np.newaxis, :, :]
    
    # Ensure non-negative values (clip to 0)
    corrected_stack = np.clip(corrected_stack, 0, None)
    
    # Convert back to original dtype if possible, otherwise keep as float64
    if np.issubdtype(tiff_stack.dtype, np.integer):
        # For integer types, we might lose precision, so keep as float64
        # User can cast manually if needed
        pass
    else:
        corrected_stack = corrected_stack.astype(tiff_stack.dtype)
    
    # Visualization
    if visualize:
        # For visualization, use mean values for display (pre_shift_baseline and post_shift_baseline are now arrays)
        pre_baseline_mean = np.mean(pre_shift_baseline)
        post_baseline_mean = np.mean(post_shift_baseline)
        _visualize_illumination_correction(tiff_stack, corrected_stack, shift_frame, 
                                          shift_amount_mean, pre_baseline_mean, 
                                          post_baseline_mean, shift_end_frame=shift_end_frame)
    
    # Return mean shift amount for compatibility (actual correction uses pixel-wise map)
    return corrected_stack, shift_frame, shift_amount_mean


def _detect_illumination_shift(tiff_stack, threshold=0.1, window_size=10):
    """
    Automatically detect frame where illumination shift occurred.
    
    Parameters:
    -----------
    tiff_stack : np.ndarray
        TIFF stack with shape (frames, height, width)
    threshold : float
        Relative threshold for detecting shift (fractional increase)
    window_size : int
        Number of frames to average for smoothing frame-to-frame changes
        
    Returns:
    --------
    shift_frame : int or None
        Frame number where shift was detected, or None if not detected
    """
    n_frames = tiff_stack.shape[0]
    
    # Calculate average intensity per frame
    frame_averages = np.array([np.mean(frame) for frame in tiff_stack])
    
    # Smooth the frame averages to reduce noise
    if window_size > 1:
        smoothed = np.convolve(frame_averages, np.ones(window_size)/window_size, mode='same')
    else:
        smoothed = frame_averages
    
    # Calculate frame-to-frame changes
    frame_changes = np.diff(smoothed)
    
    # Normalize changes by pre-change baseline
    baseline = np.mean(smoothed[:n_frames//4])  # Use first quarter as baseline
    normalized_changes = frame_changes / baseline if baseline > 0 else frame_changes
    
    # Find frames where change exceeds threshold
    significant_changes = np.where(normalized_changes > threshold)[0]
    
    if len(significant_changes) > 0:
        # Return the first significant change
        shift_frame = int(significant_changes[0])
        return shift_frame
    
    return None


def _visualize_illumination_correction(original_stack, corrected_stack, shift_frame,
                                       shift_amount, pre_baseline, post_baseline,
                                       shift_end_frame=None):
    """
    Visualize illumination shift correction.
    
    Parameters:
    -----------
    original_stack : np.ndarray
        Original TIFF stack
    corrected_stack : np.ndarray
        Corrected TIFF stack
    shift_frame : int
        Frame where shift starts
    shift_amount : float
        Shift amount that was corrected
    pre_baseline : float
        Pre-shift baseline intensity
    post_baseline : float
        Post-shift baseline intensity
    shift_end_frame : int, optional
        Frame where shift ends (for multi-frame shifts)
    """
    n_frames = original_stack.shape[0]
    
    # Calculate average intensity per frame for both stacks
    original_avg = np.array([np.mean(frame) for frame in original_stack])
    corrected_avg = np.array([np.mean(frame) for frame in corrected_stack])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Determine shift range label
    if shift_end_frame is not None:
        shift_label = f'Shift frames {shift_frame}-{shift_end_frame}'
        correction_start_frame = shift_end_frame + 1
    else:
        shift_label = f'Shift at frame {shift_frame}'
        correction_start_frame = shift_frame
    
    # Plot 1: Average intensity vs frame (original)
    axes[0, 0].plot(original_avg, 'b-', label='Original', linewidth=1.5)
    axes[0, 0].axvline(x=shift_frame, color='r', linestyle='--', 
                      label=shift_label)
    if shift_end_frame is not None:
        axes[0, 0].axvline(x=shift_end_frame, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].axvspan(shift_frame, shift_end_frame, alpha=0.2, color='red', 
                          label='Shift transition period')
    axes[0, 0].axhline(y=pre_baseline, color='g', linestyle=':', 
                      label=f'Pre-shift baseline ({pre_baseline:.2f})')
    axes[0, 0].axhline(y=post_baseline, color='orange', linestyle=':', 
                      label=f'Post-shift baseline ({post_baseline:.2f})')
    axes[0, 0].set_xlabel('Frame Number')
    axes[0, 0].set_ylabel('Average Intensity')
    axes[0, 0].set_title('Original Stack: Average Intensity vs Frame')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Average intensity vs frame (corrected)
    axes[0, 1].plot(corrected_avg, 'g-', label='Corrected', linewidth=1.5)
    axes[0, 1].axvline(x=shift_frame, color='r', linestyle='--', 
                     label=shift_label)
    if shift_end_frame is not None:
        axes[0, 1].axvline(x=shift_end_frame, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].axvspan(shift_frame, shift_end_frame, alpha=0.2, color='red')
    axes[0, 1].axhline(y=pre_baseline, color='g', linestyle=':', 
                      label=f'Target baseline ({pre_baseline:.2f})')
    axes[0, 1].set_xlabel('Frame Number')
    axes[0, 1].set_ylabel('Average Intensity')
    axes[0, 1].set_title(f'Corrected Stack: Average Intensity vs Frame\n(Shift: {shift_amount:.2f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Frame comparison (before shift)
    if shift_frame > 0:
        frame_before = shift_frame - 1
        axes[1, 0].imshow(original_stack[frame_before], cmap='gray')
        axes[1, 0].set_title(f'Original Frame {frame_before} (Before Shift)')
        axes[1, 0].axis('off')
    
    # Plot 4: Frame comparison (after shift)
    if correction_start_frame < n_frames:
        frame_after = min(correction_start_frame + 10, n_frames - 1)
        im = axes[1, 1].imshow(corrected_stack[frame_after], cmap='gray')
        axes[1, 1].set_title(f'Corrected Frame {frame_after} (After Shift)')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
