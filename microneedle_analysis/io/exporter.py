"""CSV and TIFF export functions."""

import os
import pandas as pd
import numpy as np
import tifffile as tiff


def export_tracking_data(tracked_spot_data, file_path, output_dir=None):
    """
    Export tracking data to CSV.
    
    Parameters:
    -----------
    tracked_spot_data : dict
        Dictionary with tracking data
    file_path : str
        Path to input TIFF file (for naming output)
    output_dir : str, optional
        Output directory. If None, uses same directory as input file.
        
    Returns:
    --------
    long_format_df : pd.DataFrame
        Exported DataFrame
    save_path : str
        Path where file was saved
    """
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    
    all_spot_data = []
    for spot_id, spot_data in tracked_spot_data.items():
        df = pd.DataFrame(spot_data, columns=['frame', 'x', 'y', 'mean_intensity'])
        df['spot_id'] = spot_id
        all_spot_data.append(df)
    
    long_format_df = pd.concat(all_spot_data)
    
    tiff_file_name = os.path.splitext(os.path.basename(file_path))[0]
    save_path = os.path.join(
        output_dir,
        f'tracked_spots_visualization_{tiff_file_name}.csv'
    )
    
    long_format_df.to_csv(save_path, index=False)
    print(f"Tracking data exported to: {save_path}")
    return long_format_df, save_path


def export_average_intensity(average_data, time_in_minutes, file_path, output_dir=None):
    """
    Export average normalized intensity data to CSV.
    
    Parameters:
    -----------
    average_data : pd.DataFrame
        Average intensity data with 'mean' and 'std' columns
    time_in_minutes : np.ndarray
        Time array in minutes
    file_path : str
        Path to input TIFF file (for naming output)
    output_dir : str, optional
        Output directory. If None, uses same directory as input file.
        
    Returns:
    --------
    export_df : pd.DataFrame
        Exported DataFrame
    save_path : str
        Path where file was saved
    """
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    
    export_df = pd.DataFrame({
        'Time (minutes)': time_in_minutes,
        'Average Normalized Intensity (BG Corrected)': average_data['mean'],
        'Standard Deviation (BG Corrected)': average_data['std']
    })
    
    tiff_file_name = os.path.splitext(os.path.basename(file_path))[0]
    save_path = os.path.join(
        output_dir,
        f'average_normalized_intensity_{tiff_file_name}.csv'
    )
    
    export_df.to_csv(save_path, index=False)
    print(f"Average normalized intensity data exported to: {save_path}")
    return export_df, save_path


def export_final_results(final_df, output_dir, filename='Auxin_movement_microneedle_tip_tracking.csv'):
    """
    Export final results (distance vs time_min) to CSV.
    
    Parameters:
    -----------
    final_df : pd.DataFrame
        DataFrame with spot_id, relative_distance, and time_min
    output_dir : str
        Output directory
    filename : str
        Output filename
        
    Returns:
    --------
    save_path : str
        Path where file was saved
    """
    save_path = os.path.join(output_dir, filename)
    final_df.to_csv(save_path, index=False)
    print(f"Final results exported to: {save_path}")
    return save_path


def export_tiff_stack(tiff_stack, file_path, output_dir=None, suffix='_corrected'):
    """
    Export TIFF stack to file.
    
    Parameters:
    -----------
    tiff_stack : np.ndarray
        TIFF stack array with shape (frames, height, width)
    file_path : str
        Path to input TIFF file (for naming output)
    output_dir : str, optional
        Output directory. If None, uses same directory as input file.
    suffix : str
        Suffix to add to output filename (e.g., '_corrected', '_background_subtracted')
        
    Returns:
    --------
    save_path : str
        Path where file was saved
    """
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    
    tiff_file_name = os.path.splitext(os.path.basename(file_path))[0]
    save_path = os.path.join(output_dir, f'{tiff_file_name}{suffix}.tif')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save TIFF stack
    # Use imwrite for newer tifffile versions, fallback to imsave for older versions
    if hasattr(tiff, 'imwrite'):
        tiff.imwrite(save_path, tiff_stack)
    else:
        tiff.imsave(save_path, tiff_stack)
    print(f"TIFF stack exported to: {save_path}")
    print(f"  Stack shape: {tiff_stack.shape} (frames, height, width)")
    print(f"  Data type: {tiff_stack.dtype}")
    
    return save_path

