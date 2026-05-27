"""Microneedle tip detection using peak_local_max."""

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max


def detect_tips(first_frame, min_distance=4, threshold_rel=0.5, visualize=True):
    """
    Detect microneedle tips using peak_local_max.
    
    Parameters:
    -----------
    first_frame : np.ndarray
        First frame of the TIFF stack
    min_distance : int
        Minimum pixels separating detected peaks
    threshold_rel : float
        Relative intensity threshold
    visualize : bool
        Whether to display detected tips
        
    Returns:
    --------
    coordinates : list
        List of (y, x) coordinate tuples for detected tips
    """
    coordinates = peak_local_max(
        first_frame,
        min_distance=min_distance,
        threshold_rel=threshold_rel
    )
    coordinates = sorted(coordinates, key=lambda item: item[0])
    
    if visualize:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(first_frame, cmap='gray')
        
        for i, coord in enumerate(coordinates):
            circle = plt.Circle((coord[1], coord[0]), radius=2, color='red',
                               fill=False, linewidth=1.5)
            ax.add_patch(circle)
            ax.text(coord[1] + 5, coord[0], f"ID_{i}", color='yellow',
                   fontsize=8, verticalalignment='center', horizontalalignment='center')
        
        img = ax.imshow(first_frame, cmap='gray')
        plt.colorbar(img, ax=ax, orientation='vertical')
        plt.axis('off')
        plt.title('First Image in TIFF Stack with Bright Spots Labeled')
        plt.tight_layout()
        plt.show()
    
    print("Spot ID and Coordinates:")
    for idx, spot_coords in enumerate(coordinates):
        print(f"Spot ID: {idx}, Coordinates: {spot_coords}")
    
    return coordinates

