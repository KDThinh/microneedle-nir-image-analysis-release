"""Export a video showing microneedle tips being tracked frame by frame.

Dependencies:
- numpy
- matplotlib
- imageio (and imageio-ffmpeg for mp4)
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def export_tracking_video(
    tiff_stack,
    tracked_spot_data,
    output_path,
    fps: int = 10,
    cmap: str = "gray",
    point_size: int = 40,
):
    """
    Export an MP4 video showing tracked microneedle tips over the TIFF stack.

    Parameters
    ----------
    tiff_stack : np.ndarray
        3D array with shape (n_frames, height, width)
    tracked_spot_data : dict
        Dict: spot_id -> array-like of rows [frame, x, y, mean_intensity]
        (same structure as used in tracking module)
    output_path : str
        Output video path, e.g. "tracking_video.mp4"
    fps : int
        Frames per second for the video
    cmap : str
        Colormap for the grayscale image
    point_size : int
        Size of scatter points for tracked tips
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    n_frames = tiff_stack.shape[0]

    # Precompute per-frame coordinates for each tip (for current positions)
    frame_coords = {f: {"x": [], "y": [], "ids": []} for f in range(n_frames)}
    # Also store full trajectories per spot for drawing routes
    spot_trajectories = {}

    for spot_id, spot_data in tracked_spot_data.items():
        # spot_data: array-like [frame, x, y, mean_intensity]
        arr = np.asarray(spot_data)
        if arr.size == 0:
            continue
        frames = arr[:, 0].astype(int)
        xs = arr[:, 1]
        ys = arr[:, 2]

        # Store full trajectory for this spot
        spot_trajectories[spot_id] = {
            "frames": frames,
            "xs": xs,
            "ys": ys,
        }

        # Map current positions by frame
        for f, x, y in zip(frames, xs, ys):
            if 0 <= f < n_frames:
                frame_coords[f]["x"].append(x)
                frame_coords[f]["y"].append(y)
                frame_coords[f]["ids"].append(spot_id)

    # Create video writer (force FFMPEG so 'fps' is valid and we don't hit TIFF writer)
    writer = imageio.get_writer(output_path, format="FFMPEG", fps=fps)

    # Use matplotlib to render each frame with overlaid points
    fig, ax = plt.subplots(figsize=(5, 5))

    try:
        for f in range(n_frames):
            ax.clear()
            frame_img = tiff_stack[f]

            ax.imshow(frame_img, cmap=cmap, interpolation="nearest")
            ax.set_axis_off()
            ax.set_title(f"Frame {f}")

            # Plot trajectory routes for each spot up to current frame
            for sid, traj in spot_trajectories.items():
                frames = traj["frames"]
                xs = traj["xs"]
                ys = traj["ys"]
                # Only use points up to current frame to build the path
                mask = frames <= f
                if np.any(mask):
                    ax.plot(
                        xs[mask],
                        ys[mask],
                        linestyle="-",
                        linewidth=1.0,
                        alpha=0.7,
                        label=None,
                        color="cyan",
                    )

            # Plot tracked spots at this frame (if any)
            if frame_coords[f]["x"]:
                xs_now = frame_coords[f]["x"]
                ys_now = frame_coords[f]["y"]
                ids_now = frame_coords[f]["ids"]

                ax.scatter(xs_now, ys_now, s=point_size, c="red", marker="o")

                # Optional: label each spot with its ID
                for x, y, sid in zip(xs_now, ys_now, ids_now):
                    ax.text(
                        x + 2,
                        y + 2,
                        str(sid),
                        color="yellow",
                        fontsize=6,
                        ha="left",
                        va="bottom",
                    )

            # Render figure to an RGB array (backend-agnostic)
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            # Use RGBA buffer and drop alpha channel
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape((height, width, 4))
            frame_rgb = buf[..., :3].copy()

            writer.append_data(frame_rgb)

    finally:
        writer.close()
        plt.close(fig)

    print(f"Tracking video saved to: {output_path}")


