"""ROI selection utilities for background calculation."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


def select_background_roi(first_frame, title="Select Background ROI"):
    """
    Interactive ROI selection for background calculation.
    
    Parameters:
    -----------
    first_frame : np.ndarray
        First frame of the TIFF stack to display
    title : str
        Title for the plot window
        
    Returns:
    --------
    roi : tuple or None
        ROI as (y_min, y_max, x_min, x_max) if selected, None if cancelled
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(first_frame, cmap='gray')
    ax.set_title(f"{title}\nDrag to select rectangular ROI, press Enter to confirm")
    
    roi_coords = [None]
    
    def onselect(eclick, erelease):
        """Callback for rectangle selection."""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        # Ensure min < max
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        
        roi_coords[0] = (y_min, y_max, x_min, x_max)
    
    def on_enter(event):
        """Handle Enter key press."""
        if event.key == 'enter':
            plt.close(fig)
    
    selector = RectangleSelector(ax, onselect, useblit=True,
                                button=[1], minspanx=5, minspany=5,
                                spancoords='pixels', interactive=True)
    
    fig.canvas.mpl_connect('key_press_event', on_enter)
    plt.tight_layout()
    plt.show(block=True)
    
    if roi_coords[0] is not None:
        return roi_coords[0]
    return None


def visualize_background_roi(first_frame, roi, title="Background ROI"):
    """
    Visualize the selected background ROI on the first frame.
    
    Parameters:
    -----------
    first_frame : np.ndarray
        First frame of the TIFF stack
    roi : tuple
        ROI as (y_min, y_max, x_min, x_max)
    title : str
        Title for the plot
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    y_min, y_max, x_min, x_max = roi
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(first_frame, cmap='gray')
    
    # Draw ROI rectangle
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                        fill=False, edgecolor='red', linewidth=3, linestyle='--')
    ax.add_patch(rect)
    
    ax.set_title(f"{title}\nROI: y=[{y_min}:{y_max}], x=[{x_min}:{x_max}]")
    ax.set_xlabel('X (columns)')
    ax.set_ylabel('Y (rows)')
    
    plt.tight_layout()
    return fig

