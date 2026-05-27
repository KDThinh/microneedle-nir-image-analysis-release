"""
Microneedle tip tracking using optimized Greedy Local Search (ROI-based).
Updated for performance and drift tracking stability.
"""

import numpy as np
from skimage.feature import peak_local_max


def track_spots(tiff_stack, first_frame, coordinates, spots_to_track=None,
                search_range=10, diameter=5, min_distance=3, 
                threshold_rel=0.5, background_roi=None):
    """
    Track selected spots (microneedle tips) across all frames using 
    ROI-based nearest-neighbor tracking.
    
    Parameters:
    -----------
    tiff_stack : np.ndarray
        TIFF stack array (Frames, Height, Width)
    first_frame : np.ndarray
        First frame for reference (usually tiff_stack[0])
    coordinates : list
        Initial coordinates of spots to track [(row, col), ...]
    spots_to_track : list or None
        List of spot indices to track. If None, tracks all spots in 'coordinates'.
    search_range : int
        Radius (in pixels) to search around the previous location. 
        Should be larger than the max drift per frame.
    diameter : int
        Diameter of the spot for intensity calculation.
    min_distance : int
        Minimum distance separating peaks (for peak_local_max).
    threshold_rel : float
        Relative threshold for peak detection.
    background_roi : tuple or None
        ROI for background calculation as (y_min, y_max, x_min, x_max).
        If None, calculates background by excluding tracked spots from entire frame.
        If provided, calculates average intensity within this rectangular region,
        **automatically excluding tracked spots that fall within the ROI**.
        This allows the ROI to overlap with needle tip locations while still
        calculating accurate background intensity.
        
    Returns:
    --------
    tracked_spot_data : dict
        Keys: spot IDs. Values: List of [frame, x, y, mean_intensity].
        (Note: Output is x, y for plotting; Input is row, col).
    average_background_intensities : list
        List of average background intensities per frame.
    """
    
    # Filter spots if specific indices are requested
    if spots_to_track is not None:
        selected_spots = [coordinates[i] for i in spots_to_track]
    else:
        selected_spots = [list(c) for c in coordinates]  # Deep copy to list of lists
    
    print(f"Tracking {len(selected_spots)} spots over {len(tiff_stack)} frames...")
    
    # Initialize data structure
    tracked_spot_data = {idx: [] for idx in range(len(selected_spots))}
    average_background_intensities = []
    
    # Pre-calculate offsets for intensity circle mask (optimization)
    # We create a small mask grid just once, not every frame
    r_mask = int(diameter) + 2
    y_grid, x_grid = np.ogrid[-r_mask:r_mask+1, -r_mask:r_mask+1]
    intensity_mask = x_grid**2 + y_grid**2 <= (diameter / 2)**2

    for frame_no, frame in enumerate(tiff_stack):
        
        # --- 1. SPOT TRACKING ---
        for idx, current_pos in enumerate(selected_spots):
            y_prev, x_prev = current_pos  # y=row, x=col
            
            if frame_no == 0:
                # First frame: just record the initial position and intensity
                y_new, x_new = y_prev, x_prev
            else:
                # Subsequent frames: Search within ROI
                r = int(search_range)
                
                # Define ROI bounds (clamping to image edges)
                y_min = max(0, int(y_prev - r))
                y_max = min(frame.shape[0], int(y_prev + r + 1))
                x_min = max(0, int(x_prev - r))
                x_max = min(frame.shape[1], int(x_prev + r + 1))
                
                roi = frame[y_min:y_max, x_min:x_max]
                
                # Detect peaks in ROI
                local_peaks = peak_local_max(
                    roi,
                    min_distance=min_distance,
                    threshold_rel=threshold_rel
                )
                
                if len(local_peaks) > 0:
                    # Convert local ROI coords back to global image coords
                    # local_peaks is (row, col)
                    global_peaks = local_peaks + np.array([y_min, x_min])
                    
                    # Find the peak closest to the previous position
                    dists = np.linalg.norm(global_peaks - np.array([y_prev, x_prev]), axis=1)
                    min_idx = np.argmin(dists)
                    
                    if dists[min_idx] <= search_range:
                        y_new, x_new = global_peaks[min_idx]
                        selected_spots[idx] = [y_new, x_new]  # Update for next frame
                    else:
                        # Nearest peak is too far (jumped?), keep old pos
                        y_new, x_new = y_prev, x_prev
                else:
                    # No peaks found in ROI, keep old pos
                    y_new, x_new = y_prev, x_prev

            # --- Calculate Intensity for this spot ---
            # Extract small slice for intensity calculation
            yi, xi = int(y_new), int(x_new)
            
            # bounds for intensity slice
            y_i_min = max(0, yi - r_mask)
            y_i_max = min(frame.shape[0], yi + r_mask + 1)
            x_i_min = max(0, xi - r_mask)
            x_i_max = min(frame.shape[1], xi + r_mask + 1)
            
            # This handles edge cases where the mask might be larger than the image corner
            # We slice the mask to match the actual image slice if near edge
            mask_slice_y_start = r_mask - (yi - y_i_min)
            mask_slice_y_end = mask_slice_y_start + (y_i_max - y_i_min)
            mask_slice_x_start = r_mask - (xi - x_i_min)
            mask_slice_x_end = mask_slice_x_start + (x_i_max - x_i_min)
            
            current_mask = intensity_mask[mask_slice_y_start:mask_slice_y_end, 
                                        mask_slice_x_start:mask_slice_x_end]
            
            spot_slice = frame[y_i_min:y_i_max, x_i_min:x_i_max]
            
            if spot_slice.shape == current_mask.shape and spot_slice.size > 0:
                mean_intensity = np.mean(spot_slice[current_mask])
            else:
                mean_intensity = 0
            
            # Store data: [Frame, X(col), Y(row), Intensity]
            tracked_spot_data[idx].append([frame_no, x_new, y_new, mean_intensity])

        # --- 2. BACKGROUND CALCULATION ---
        if background_roi is not None:
            # Use user-defined ROI for background calculation
            # IMPORTANT: Tracked spots are automatically excluded from ROI even if ROI overlaps with them
            y_min_roi, y_max_roi, x_min_roi, x_max_roi = background_roi
            
            # Clamp ROI to image bounds
            y_min_roi = max(0, int(y_min_roi))
            y_max_roi = min(frame.shape[0], int(y_max_roi))
            x_min_roi = max(0, int(x_min_roi))
            x_max_roi = min(frame.shape[1], int(x_max_roi))
            
            # Extract ROI region
            bg_region = frame[y_min_roi:y_max_roi, x_min_roi:x_max_roi]
            
            # Create mask for ROI: start with all pixels = True (included)
            bg_mask_roi = np.ones(bg_region.shape, dtype=bool)
            rr_roi, cc_roi = np.ogrid[:bg_region.shape[0], :bg_region.shape[1]]
            
            # Exclude tracked spots from ROI background calculation
            # This ensures accurate background even if ROI overlaps with needle tip locations
            for idx in tracked_spot_data:
                # Get the position just recorded for this frame
                _, s_x, s_y, _ = tracked_spot_data[idx][-1]
                
                # Check if this tracked spot falls within the ROI bounds
                if (y_min_roi <= s_y < y_max_roi and x_min_roi <= s_x < x_max_roi):
                    # Convert spot position to ROI-local coordinates
                    s_y_local = s_y - y_min_roi
                    s_x_local = s_x - x_min_roi
                    
                    # Mask out a circular region around the spot (exclude from background)
                    r_bg = int(diameter/2) + 2
                    y_local_min = max(0, int(s_y_local - r_bg))
                    y_local_max = min(bg_region.shape[0], int(s_y_local + r_bg + 1))
                    x_local_min = max(0, int(s_x_local - r_bg))
                    x_local_max = min(bg_region.shape[1], int(s_x_local + r_bg + 1))
                    
                    # Set mask to False for pixels within circle around spot
                    bg_mask_roi[y_local_min:y_local_max, x_local_min:x_local_max] &= \
                        ((rr_roi[y_local_min:y_local_max, :] - s_y_local)**2 + 
                         (cc_roi[:, x_local_min:x_local_max] - s_x_local)**2 > (diameter / 2)**2)
            
            # Calculate average background intensity from ROI, excluding tracked spots
            average_background_intensity = np.mean(bg_region[bg_mask_roi])
        else:
            # Original method: Create a boolean mask of the whole frame, excluding spots
            bg_mask = np.ones(frame.shape, dtype=bool)
            
            rr, cc = np.ogrid[:frame.shape[0], :frame.shape[1]]
            
            for idx in tracked_spot_data:
                # Get the position just recorded for this frame
                _, s_x, s_y, _ = tracked_spot_data[idx][-1]
                
                # Mask out a circle around the spot
                # Optimization: Only calculate distance for a bounding box around the spot
                r_bg = int(diameter/2) + 2
                y_min = max(0, int(s_y - r_bg))
                y_max = min(frame.shape[0], int(s_y + r_bg + 1))
                x_min = max(0, int(s_x - r_bg))
                x_max = min(frame.shape[1], int(s_x + r_bg + 1))
                
                bg_mask[y_min:y_max, x_min:x_max] &= \
                    ((rr[y_min:y_max, :] - s_y)**2 + (cc[:, x_min:x_max] - s_x)**2 > (diameter / 2)**2)

            average_background_intensity = np.mean(frame[bg_mask])
        
        average_background_intensities.append(average_background_intensity)
        
        # Update progress on single line
        if frame_no % 10 == 0 or frame_no == len(tiff_stack) - 1:
            progress = (frame_no + 1) / len(tiff_stack) * 100
            print(f"\rProcessed frame [{frame_no + 1}/{len(tiff_stack)}] ({progress:.1f}%)", end='', flush=True)
    
    # Print newline after completion
    print()  # Move to next line after progress completes

    return tracked_spot_data, average_background_intensities
