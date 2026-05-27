"""Command-line interface for microneedle analysis."""

import typer
from pathlib import Path
from typing import Optional, List
import sys

from microneedle_analysis.pipeline import run_pipeline, run_iaa_ref_folder_pipeline

app = typer.Typer(
    name="microneedle-analysis",
    help="Microneedle Tip Tracking and Fluorescence Intensity Analysis",
    add_completion=False
)


def _resolve_cli_config_path(config_path: Optional[str], required: bool = False) -> Optional[str]:
    """Resolve --config or the default config.yaml search path."""
    from microneedle_analysis.io.config import resolve_default_config_path

    if config_path is None:
        config_path = resolve_default_config_path()
    if required and (not config_path or not Path(config_path).exists()):
        typer.echo(
            "Error: Config file not found. Pass --config config.yaml or run from a directory "
            "that contains config.yaml.",
            err=True,
        )
        raise typer.Exit(1)
    return config_path


@app.command()
def analyze(
    file_path: Optional[str] = typer.Argument(
        None,
        help="Path to input TIFF file or folder (optional if using --profile; for folder_keywords profiles, pass folder to override profile's folder_path)"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: same as input file, or profile's output_dir if set)"
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file"
    ),
    radius: Optional[int] = typer.Option(
        None,
        "--radius",
        "-r",
        help="Rolling ball radius for background subtraction (default: no background correction)",
    ),
    correct_shift: bool = typer.Option(
        False,
        "--correct-shift",
        help="Apply illumination shift (baseline) correction before analysis",
    ),
    shift_frame: Optional[int] = typer.Option(
        None,
        "--shift-frame",
        help="Frame where illumination shift starts (auto-detect if not specified)",
    ),
    shift_end_frame: Optional[int] = typer.Option(
        None,
        "--shift-end-frame",
        help="Frame where illumination shift ends (for multi-frame shifts, e.g., 262-264)",
    ),
    exclude_spots: Optional[str] = typer.Option(
        None,
        "--exclude",
        "-e",
        help="Comma-separated spot IDs to exclude (e.g., 0,2,3)"
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        "-p",
        help="Use parameters from a profile in config.yaml (can use file_path/output_dir from profile)"
    ),
    export_video: bool = typer.Option(
        False,
        "--export-video",
        help="Export an MP4 video showing tracked microneedle tips",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output"
    ),
):
    """
    Run complete microneedle analysis pipeline.
    
    Behavior depends on profile type:
    - File-based profile (or no profile): analyzes a single TIFF file
    - Folder-based profile (profile_type: folder_keywords): discovers IAA/Ref .tif
      files by keyword and processes each experiment folder
    
    Examples:
        microneedle-analysis analyze input.tif --output ./results --config config.yaml
        microneedle-analysis analyze --profile raju_shade_exp1
        microneedle-analysis analyze --profile fig3_iaa_ref  # IAA/Ref folder pipeline
    """
    from microneedle_analysis.io.config import get_profile, find_profile_by_path

    config_path = _resolve_cli_config_path(config_path)

    # Load profile if specified or auto-detect by file path
    profile_data = None
    profile_name = None

    if profile:
        if config_path and Path(config_path).exists():
            profile_data = get_profile(config_path, profile)
            if profile_data:
                profile_name = profile
                typer.echo(f"Using profile: {profile_name}")
            else:
                typer.echo(f"Error: Profile '{profile}' not found in config.", err=True)
                raise typer.Exit(1)
        else:
            typer.echo(f"Error: Config file not found. Cannot load profile '{profile}'.", err=True)
            raise typer.Exit(1)
    elif file_path and config_path and Path(config_path).exists():
        # Try to auto-detect profile: folder path -> folder_keywords, file path -> file-based
        if Path(file_path).is_dir():
            from microneedle_analysis.io.config import find_profile_by_folder_path
            profile_name, profile_data = find_profile_by_folder_path(config_path, file_path)
        else:
            profile_name, profile_data = find_profile_by_path(config_path, file_path)
        if profile_data:
            typer.echo(f"Auto-detected profile: {profile_name}")

    # Branch: folder_keywords profile -> run IAA/Ref folder pipeline
    _is_folder_profile = (
        profile_data
        and (profile_data.get("profile_type") == "folder_keywords" or "folder_path" in profile_data)
    )
    if _is_folder_profile:
        from microneedle_analysis.io.config import resolve_folder_path
        _folder_path = None
        if file_path and Path(file_path).is_dir():
            _folder_path = str(Path(file_path).resolve())
        if not _folder_path:
            _folder_path = resolve_folder_path(profile_data, config_path or "") or profile_data.get("folder_path")
        if not _folder_path:
            typer.echo(
                "Error: folder_path is required for folder_keywords profile. Set folder_path in config or pass a folder path as argument.",
                err=True,
            )
            raise typer.Exit(1)
        if not Path(_folder_path).is_dir():
            typer.echo(f"Error: Folder not found: {_folder_path}", err=True)
            raise typer.Exit(1)
        _keywords = profile_data.get("file_keywords", ["IAA", "Ref"])
        _output_subdirs = profile_data.get("output_subdirs")
        typer.echo(f"Using folder_keywords profile: {profile_name or profile}")
        typer.echo(f"Folder: {_folder_path}")
        typer.echo(f"Keywords: {_keywords}")
        try:
            _results = run_iaa_ref_folder_pipeline(
                folder_path=_folder_path,
                keywords=_keywords,
                output_subdirs=_output_subdirs,
                config_path=config_path,
                profile_data=profile_data,
                export_video=export_video,
            )
            _n_exp = len(_results)
            _n_files = sum(len(v) for v in _results.values())
            typer.echo("\n[OK] IAA/Ref analysis completed successfully!")
            typer.echo(f"  Experiment folders processed: {_n_exp}")
            typer.echo(f"  Total files processed: {_n_files}")
        except Exception as e:
            typer.echo(f"\n[ERROR] Error during IAA/Ref analysis: {e}", err=True)
            if verbose:
                import traceback
                traceback.print_exc()
            raise typer.Exit(1)
        return

    # Use file_path from profile if file_path not provided and profile has file_path
    if not file_path and profile_data:
        # Ensure profile has file_path_pattern (auto-generate from file_path if needed)
        from microneedle_analysis.io.config import _ensure_file_path_pattern
        profile_data = _ensure_file_path_pattern(profile_data)
        
        profile_file_path = profile_data.get("file_path")
        if profile_file_path:
            # First try the literal path from profile
            if Path(profile_file_path).exists():
                file_path = profile_file_path
                typer.echo(f"Using file_path from profile: {file_path}")
            else:
                # On Windows, quickly try the same path on other common drive letters
                # before falling back to the expensive recursive pattern search.
                from pathlib import PureWindowsPath
                import os

                remapped_path = None
                if os.name == "nt":
                    try:
                        p = PureWindowsPath(profile_file_path)
                        original_drive = p.drive  # e.g. "H:"
                        # Keep the order intuitive: prefer G: then H: then others,
                        # but skip the drive that's already in the profile path.
                        candidate_drives = ["G:", "H:", "F:", "E:", "D:", "C:"]
                        candidate_drives = [d for d in candidate_drives if d.upper() != original_drive.upper()]
                        for drive in candidate_drives:
                            candidate = PureWindowsPath(drive, *p.parts[1:])
                            if Path(candidate).exists():
                                remapped_path = str(candidate)
                                break
                    except Exception:
                        remapped_path = None

                if remapped_path:
                    file_path = remapped_path
                    typer.echo(
                        f"File not found at literal path: {profile_file_path}\n"
                        f"Using remapped drive path instead: {file_path}"
                    )
                else:
                    # If literal/remapped path doesn't exist, try pattern-based search
                    file_path_pattern = profile_data.get("file_path_pattern")
                    if file_path_pattern:
                        typer.echo(f"File not found at literal path: {profile_file_path}")
                        typer.echo(f"Searching using pattern: {file_path_pattern}")
                        # Search for files matching the pattern across common drive letters (Windows)
                        import glob
                        import os
                        import string

                        matching_files = []
                        # Pattern format: "ParentFolder*Filename"
                        # Search in current directory first
                        if '*' in file_path_pattern:
                            # Use glob pattern with wildcard
                            parent_part, filename_part = file_path_pattern.split('*', 1)
                            # Search for files matching the pattern
                            matching_files.extend(glob.glob(f"**/*{filename_part}", recursive=True))
                            # Filter by parent folder name
                            matching_files = [f for f in matching_files if parent_part.lower() in os.path.dirname(os.path.abspath(f)).lower()]
                        else:
                            matching_files.extend(glob.glob(f"**/*{file_path_pattern}", recursive=True))

                        # On Windows, also search common drive letters (G:, H:, etc.)
                        if os.name == 'nt':  # Windows
                            for drive_letter in ['G:', 'H:', 'F:', 'E:', 'D:', 'C:']:
                                try:
                                    if '*' in file_path_pattern:
                                        parent_part, filename_part = file_path_pattern.split('*', 1)
                                        drive_files = glob.glob(f"{drive_letter}/**/*{filename_part}", recursive=True)
                                        # Filter by parent folder name
                                        drive_files = [f for f in drive_files if parent_part.lower() in os.path.dirname(os.path.abspath(f)).lower()]
                                        matching_files.extend(drive_files)
                                    else:
                                        drive_files = glob.glob(f"{drive_letter}/**/*{file_path_pattern}", recursive=True)
                                        matching_files.extend(drive_files)
                                except:
                                    # Skip drives that don't exist or aren't accessible
                                    continue

                        if matching_files:
                            # Use the first match (should be unique enough with the pattern)
                            file_path = matching_files[0]
                            typer.echo(f"Found file matching pattern '{file_path_pattern}': {file_path}")
                        else:
                            typer.echo(
                                f"Error: No files found matching pattern '{file_path_pattern}' in current directory or common drives.",
                                err=True,
                            )
                            typer.echo("Try running from a directory that contains the file, or specify --file-path directly.", err=True)
                            raise typer.Exit(1)
                    else:
                        typer.echo(f"Error: Input file not found: {profile_file_path}", err=True)
                        raise typer.Exit(1)
        else:
            # Check if profile has file_path_pattern for flexible drive letter handling
            file_path_pattern = profile_data.get("file_path_pattern")
            if file_path_pattern:
                # Search for files matching the pattern across common drive letters (Windows)
                import glob
                import os
                import string

                matching_files = []
                # Pattern format: "ParentFolder*Filename"
                # Search in current directory first
                if '*' in file_path_pattern:
                    # Use glob pattern with wildcard
                    parent_part, filename_part = file_path_pattern.split('*', 1)
                    # Search for files matching the pattern
                    matching_files.extend(glob.glob(f"**/*{filename_part}", recursive=True))
                    # Filter by parent folder name
                    matching_files = [f for f in matching_files if parent_part.lower() in os.path.dirname(os.path.abspath(f)).lower()]
                else:
                    matching_files.extend(glob.glob(f"**/*{file_path_pattern}", recursive=True))

                # On Windows, also search common drive letters (G:, H:, etc.)
                if os.name == 'nt':  # Windows
                    for drive_letter in ['G:', 'H:', 'F:', 'E:', 'D:', 'C:']:
                        try:
                            if '*' in file_path_pattern:
                                parent_part, filename_part = file_path_pattern.split('*', 1)
                                drive_files = glob.glob(f"{drive_letter}/**/*{filename_part}", recursive=True)
                                # Filter by parent folder name
                                drive_files = [f for f in drive_files if parent_part.lower() in os.path.dirname(os.path.abspath(f)).lower()]
                                matching_files.extend(drive_files)
                            else:
                                drive_files = glob.glob(f"{drive_letter}/**/*{file_path_pattern}", recursive=True)
                                matching_files.extend(drive_files)
                        except:
                            # Skip drives that don't exist or aren't accessible
                            continue

                if matching_files:
                    # Use the first match (should be unique enough with the pattern)
                    file_path = matching_files[0]
                    typer.echo(f"Found file matching pattern '{file_path_pattern}': {file_path}")
                else:
                    typer.echo(
                        f"Error: No files found matching pattern '{file_path_pattern}' in current directory or common drives.",
                        err=True,
                    )
                    typer.echo("Try running from a directory that contains the file, or specify --file-path directly.", err=True)
                    raise typer.Exit(1)
            else:
                typer.echo(
                    "Error: No file_path provided and profile does not have file_path or file_path_pattern.",
                    err=True,
                )
                typer.echo(
                    "Either provide file_path as argument or use a profile with file_path/file_path_pattern.",
                    err=True,
                )
                raise typer.Exit(1)
    elif not file_path:
        typer.echo(
            "Error: file_path is required. Either provide it as argument or use --profile with file_path in profile.",
            err=True,
        )
        raise typer.Exit(1)

    # Validate input file
    if not Path(file_path).exists():
        typer.echo(f"Error: Input file not found: {file_path}", err=True)
        raise typer.Exit(1)

    # Parse exclude spots
    exclude_list = None
    if exclude_spots:
        try:
            exclude_list = [int(s.strip()) for s in exclude_spots.split(",") if s.strip()]
        except ValueError:
            typer.echo(f"Error: Invalid exclude spots format: {exclude_spots}", err=True)
            typer.echo(
                "Expected format: comma-separated integers (e.g., 0,2,3)", err=True
            )
            raise typer.Exit(1)

    # Validate/resolve config file if provided
    if config_path and not Path(config_path).exists():
        typer.echo(f"Warning: Config file not found: {config_path}", err=True)
        typer.echo("Using default configuration.", err=True)
        config_path = None

    # Override parameters with profile values if profile is loaded
    spots_to_track = None
    max_frame_override = None
    skip_initial_frames_override = None
    normalization_starting_frame_override = None
    normalization_ending_frame_override = None

    if profile_data:
        # Output directory
        if output_dir is None and "output_dir" in profile_data:
            output_dir = profile_data.get("output_dir")
        # Background subtraction radius
        if radius is None and "radius" in profile_data:
            radius = profile_data.get("radius")
        # Illumination shift correction
        if not correct_shift and profile_data.get("correct_shift", False):
            correct_shift = True
        if shift_frame is None and "shift_frame" in profile_data:
            shift_frame = profile_data.get("shift_frame")
        if shift_end_frame is None and "shift_end_frame" in profile_data:
            shift_end_frame = profile_data.get("shift_end_frame")
        if "spots_to_track" in profile_data:
            spots_to_track = profile_data.get("spots_to_track")
        # Per-profile max_frame to override global tracking.max_frame
        if "max_frame" in profile_data:
            max_frame_override = profile_data.get("max_frame")
        # Per-profile skip_initial_frames
        if "skip_initial_frames" in profile_data:
            skip_initial_frames_override = profile_data.get("skip_initial_frames")
        # Per-profile normalization frame overrides
        if "normalization_starting_frame" in profile_data:
            normalization_starting_frame_override = profile_data.get("normalization_starting_frame")
        if "normalization_ending_frame" in profile_data:
            normalization_ending_frame_override = profile_data.get("normalization_ending_frame")

    try:
        typer.echo(f"Starting analysis of: {file_path}")
        if output_dir:
            typer.echo(f"Output directory: {output_dir}")
        if config_path:
            typer.echo(f"Using config: {config_path}")
        if exclude_list:
            typer.echo(f"Excluding spots: {exclude_list}")
        if export_video:
            typer.echo("Tracking video export: enabled")

        results = run_pipeline(
            file_path=file_path,
            output_dir=output_dir,
            config_path=config_path,
            exclude_spots=exclude_list,
            radius=radius,
            correct_shift=correct_shift,
            shift_frame=shift_frame,
            shift_end_frame=shift_end_frame,
            export_video=export_video,
            spots_to_track_override=spots_to_track,
            max_frame_override=max_frame_override,
            skip_initial_frames_override=skip_initial_frames_override,
            normalization_starting_frame_override=normalization_starting_frame_override,
            normalization_ending_frame_override=normalization_ending_frame_override,
            profile_data=profile_data,
        )

        typer.echo("\n[OK] Analysis completed successfully!")
        typer.echo(f"  Number of spots analyzed: {len(results['spot_ids'])}")

    except Exception as e:
        typer.echo(f"\n[ERROR] Error during analysis: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1)


@app.command("analyze-iaa-ref")
def analyze_iaa_ref(
    folder_path: Optional[str] = typer.Option(
        None,
        "--folder",
        "-f",
        help="Path to root folder containing experiment subfolders with IAA and Ref .tif files",
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        "-p",
        help="Use a folder_keywords profile from config.yaml (provides folder_path and file_keywords)",
    ),
    keywords: Optional[str] = typer.Option(
        None,
        "--keywords",
        "-k",
        help="Comma-separated keywords to match .tif files (e.g., IAA,Ref). Overrides profile if set.",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file",
    ),
    export_video: bool = typer.Option(
        False,
        "--export-video",
        help="Export an MP4 video showing tracked microneedle tips for each file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
):
    """
    Run pipeline on IAA and Ref TIFF files from folder-based experiments.
    
    Discovers .tif files by keyword (IAA, Ref) in each experiment subfolder,
    processes each independently, and outputs to output_timestamped/IAA_output
    and output_timestamped/Ref_output per experiment folder.
    
    Examples:
        microneedle-analysis analyze-iaa-ref --profile fig3_iaa_ref
        microneedle-analysis analyze-iaa-ref --folder path/to/Fig-3 --keywords IAA,Ref
    """
    from microneedle_analysis.io.config import get_profile, resolve_folder_path

    config_path = _resolve_cli_config_path(config_path, required=bool(profile))

    profile_data = None
    resolved_folder = None
    keywords_list = None
    output_subdirs = None

    if profile:
        if not config_path or not Path(config_path).exists():
            typer.echo("Error: Config file not found. Cannot load profile.", err=True)
            raise typer.Exit(1)
        profile_data = get_profile(config_path, profile)
        if not profile_data:
            typer.echo(f"Error: Profile '{profile}' not found in config.", err=True)
            raise typer.Exit(1)
        if profile_data.get("profile_type") != "folder_keywords" and "folder_path" not in profile_data:
            typer.echo(
                f"Error: Profile '{profile}' is not a folder_keywords profile (missing folder_path or profile_type).",
                err=True,
            )
            raise typer.Exit(1)
        typer.echo(f"Using profile: {profile}")
        resolved_folder = resolve_folder_path(profile_data, config_path) or profile_data.get("folder_path")
        if not folder_path:
            folder_path = resolved_folder
        if keywords is None:
            keywords_list = profile_data.get("file_keywords", ["IAA", "Ref"])
        output_subdirs = profile_data.get("output_subdirs")

    if keywords:
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
    if keywords_list is None:
        keywords_list = ["IAA", "Ref"]

    if not folder_path:
        typer.echo(
            "Error: folder_path is required. Use --folder or --profile with a folder_keywords profile.",
            err=True,
        )
        raise typer.Exit(1)

    if not Path(folder_path).is_dir():
        typer.echo(f"Error: Folder not found: {folder_path}", err=True)
        raise typer.Exit(1)

    try:
        typer.echo(f"Starting IAA/Ref analysis on folder: {folder_path}")
        typer.echo(f"Keywords: {keywords_list}")
        if config_path:
            typer.echo(f"Config: {config_path}")

        results = run_iaa_ref_folder_pipeline(
            folder_path=folder_path,
            keywords=keywords_list,
            output_subdirs=output_subdirs,
            config_path=config_path,
            profile_data=profile_data,
            export_video=export_video,
        )

        n_exp = len(results)
        n_files = sum(len(v) for v in results.values())
        typer.echo("\n[OK] IAA/Ref analysis completed successfully!")
        typer.echo(f"  Experiment folders processed: {n_exp}")
        typer.echo(f"  Total files processed: {n_files}")

    except Exception as e:
        typer.echo(f"\n[ERROR] Error during IAA/Ref analysis: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def preprocess(
    file_path: Optional[str] = typer.Argument(None, help="Path to input TIFF file (optional if using --profile with file_path in profile)"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory (default: same as input file)"),
    radius: Optional[int] = typer.Option(None, "--radius", "-r", help="Rolling ball radius for background subtraction (default: no background correction)"),
    correct_shift: bool = typer.Option(False, "--correct-shift", help="Apply illumination shift correction"),
    shift_frame: Optional[int] = typer.Option(None, "--shift-frame", help="Frame where illumination shift starts (auto-detect if not specified)"),
    shift_end_frame: Optional[int] = typer.Option(None, "--shift-end-frame", help="Frame where illumination shift ends (for multi-frame shifts, e.g., 262-264)"),
    suffix: Optional[str] = typer.Option(None, "--suffix", "-s", help="Suffix for output filename (default: '_background_corrected' if radius specified, '_processed' otherwise)"),
    visualize: bool = typer.Option(False, "--visualize", help="Show visualization plots"),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Use parameters from a profile in config.yaml (can use file_path from profile if file_path not provided)"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file (default: microneedle_analysis/config.yaml)"),
):
    """
    Preprocess TIFF stack: load, optionally correct illumination shift, 
    optionally apply background subtraction, and export corrected stack.
    
    Example:
        microneedle-analysis preprocess "exp-1 2025 November 17 16_18_02-1.tif"
        microneedle-analysis preprocess input.tif --radius 25
        microneedle-analysis preprocess input.tif --correct-shift --output ./processed
        microneedle-analysis preprocess input.tif --profile raju_shade_exp1
        microneedle-analysis preprocess --profile raju_shade_exp1  # Uses file_path from profile
    """
    from microneedle_analysis.core import load_tiff, subtract_background, correct_illumination_shift
    from microneedle_analysis.io import export_tiff_stack
    from microneedle_analysis.io.config import get_profile, find_profile_by_path
    
    config_path = _resolve_cli_config_path(config_path, required=bool(profile))

    # Load profile if specified or auto-detect by file path
    profile_data = None
    profile_name = None
    
    if profile:
        if config_path and Path(config_path).exists():
            profile_data = get_profile(config_path, profile)
            if profile_data:
                profile_name = profile
                typer.echo(f"Using profile: {profile_name}")
            else:
                typer.echo(f"Error: Profile '{profile}' not found in config.", err=True)
                raise typer.Exit(1)
        else:
            typer.echo(f"Error: Config file not found. Cannot load profile '{profile}'.", err=True)
            raise typer.Exit(1)
    elif file_path and config_path and Path(config_path).exists():
        # Try to auto-detect profile by file path
        profile_name, profile_data = find_profile_by_path(config_path, file_path)
        if profile_data:
            typer.echo(f"Auto-detected profile: {profile_name}")
    
    # Use file_path from profile if file_path not provided and profile has file_path
    if not file_path and profile_data:
        # Ensure profile has file_path_pattern (auto-generate from file_path if needed)
        from microneedle_analysis.io.config import _ensure_file_path_pattern
        profile_data = _ensure_file_path_pattern(profile_data)
        
        profile_file_path = profile_data.get('file_path')
        if profile_file_path:
            # First try the literal path from profile
            if Path(profile_file_path).exists():
                file_path = profile_file_path
                typer.echo(f"Using file_path from profile: {file_path}")
            else:
                # If literal path doesn't exist, try pattern-based search
                file_path_pattern = profile_data.get("file_path_pattern")
                if file_path_pattern:
                    typer.echo(f"File not found at literal path: {profile_file_path}")
                    typer.echo(f"Searching using pattern: {file_path_pattern}")
                    # Search for files matching the pattern across common drive letters (Windows)
                    import glob
                    import os

                    matching_files = []
                    # Pattern format: "ParentFolder*Filename"
                    # Search in current directory first
                    if '*' in file_path_pattern:
                        # Use glob pattern with wildcard
                        parent_part, filename_part = file_path_pattern.split('*', 1)
                        # Search for files matching the pattern
                        matching_files.extend(glob.glob(f"**/*{filename_part}", recursive=True))
                        # Filter by parent folder name
                        matching_files = [f for f in matching_files if parent_part.lower() in os.path.dirname(os.path.abspath(f)).lower()]
                    else:
                        matching_files.extend(glob.glob(f"**/*{file_path_pattern}", recursive=True))

                    # On Windows, also search common drive letters (G:, H:, etc.)
                    if os.name == 'nt':  # Windows
                        for drive_letter in ['G:', 'H:', 'F:', 'E:', 'D:', 'C:']:
                            try:
                                if '*' in file_path_pattern:
                                    parent_part, filename_part = file_path_pattern.split('*', 1)
                                    drive_files = glob.glob(f"{drive_letter}/**/*{filename_part}", recursive=True)
                                    # Filter by parent folder name
                                    drive_files = [f for f in drive_files if parent_part.lower() in os.path.dirname(os.path.abspath(f)).lower()]
                                    matching_files.extend(drive_files)
                                else:
                                    drive_files = glob.glob(f"{drive_letter}/**/*{file_path_pattern}", recursive=True)
                                    matching_files.extend(drive_files)
                            except:
                                # Skip drives that don't exist or aren't accessible
                                continue

                    if matching_files:
                        # Use the first match (should be unique enough with the pattern)
                        file_path = matching_files[0]
                        typer.echo(f"Found file matching pattern '{file_path_pattern}': {file_path}")
                    else:
                        typer.echo(
                            f"Error: No files found matching pattern '{file_path_pattern}' in current directory or common drives.",
                            err=True,
                        )
                        typer.echo("Try running from a directory that contains the file, or specify --file-path directly.", err=True)
                        raise typer.Exit(1)
                else:
                    typer.echo(f"Error: Input file not found: {profile_file_path}", err=True)
                    raise typer.Exit(1)
        else:
            # Check if profile has file_path_pattern for flexible drive letter handling
            file_path_pattern = profile_data.get("file_path_pattern")
            if file_path_pattern:
                # Search for files matching the pattern across common drive letters (Windows)
                import glob
                import os

                matching_files = []
                # Pattern format: "ParentFolder*Filename"
                # Search in current directory first
                if '*' in file_path_pattern:
                    # Use glob pattern with wildcard
                    parent_part, filename_part = file_path_pattern.split('*', 1)
                    # Search for files matching the pattern
                    matching_files.extend(glob.glob(f"**/*{filename_part}", recursive=True))
                    # Filter by parent folder name
                    matching_files = [f for f in matching_files if parent_part.lower() in os.path.dirname(os.path.abspath(f)).lower()]
                else:
                    matching_files.extend(glob.glob(f"**/*{file_path_pattern}", recursive=True))

                # On Windows, also search common drive letters (G:, H:, etc.)
                if os.name == 'nt':  # Windows
                    for drive_letter in ['G:', 'H:', 'F:', 'E:', 'D:', 'C:']:
                        try:
                            if '*' in file_path_pattern:
                                parent_part, filename_part = file_path_pattern.split('*', 1)
                                drive_files = glob.glob(f"{drive_letter}/**/*{filename_part}", recursive=True)
                                # Filter by parent folder name
                                drive_files = [f for f in drive_files if parent_part.lower() in os.path.dirname(os.path.abspath(f)).lower()]
                                matching_files.extend(drive_files)
                            else:
                                drive_files = glob.glob(f"{drive_letter}/**/*{file_path_pattern}", recursive=True)
                                matching_files.extend(drive_files)
                        except:
                            # Skip drives that don't exist or aren't accessible
                            continue

                if matching_files:
                    # Use the first match (should be unique enough with the pattern)
                    file_path = matching_files[0]
                    typer.echo(f"Found file matching pattern '{file_path_pattern}': {file_path}")
                else:
                    typer.echo(
                        f"Error: No files found matching pattern '{file_path_pattern}' in current directory or common drives.",
                        err=True,
                    )
                    typer.echo("Try running from a directory that contains the file, or specify --file-path directly.", err=True)
                    raise typer.Exit(1)
            else:
                typer.echo("Error: No file_path provided and profile does not have file_path or file_path_pattern.", err=True)
                typer.echo("Either provide file_path as argument or use a profile with file_path/file_path_pattern.", err=True)
                raise typer.Exit(1)
    elif not file_path:
        typer.echo("Error: file_path is required. Either provide it as argument or use --profile with file_path in profile.", err=True)
        raise typer.Exit(1)
    
    # Validate input file
    if not Path(file_path).exists():
        typer.echo(f"Error: Input file not found: {file_path}", err=True)
        raise typer.Exit(1)
    
    # Override parameters with profile values if profile is loaded
    if profile_data:
        if radius is None and 'radius' in profile_data:
            radius = profile_data.get('radius')
        if not correct_shift and profile_data.get('correct_shift', False):
            correct_shift = True
        if shift_frame is None and 'shift_frame' in profile_data:
            shift_frame = profile_data.get('shift_frame')
        if shift_end_frame is None and 'shift_end_frame' in profile_data:
            shift_end_frame = profile_data.get('shift_end_frame')
        if output_dir is None and 'output_dir' in profile_data:
            output_dir = profile_data.get('output_dir')
    
    # Set default suffix based on whether background correction is applied
    if suffix is None:
        suffix = "_background_corrected" if radius is not None else "_processed"
    
    try:
        typer.echo(f"Loading TIFF file: {file_path}")
        tiff_stack, first_frame = load_tiff(file_path)
        
        # Optional: Correct illumination shift
        if correct_shift:
            typer.echo("Correcting illumination shift...")
            tiff_stack, detected_frame, shift_amount = correct_illumination_shift(
                tiff_stack,
                shift_frame=shift_frame,
                shift_end_frame=shift_end_frame,
                auto_detect=(shift_frame is None),
                visualize=visualize
            )
            if detected_frame is not None:
                if shift_end_frame is not None:
                    typer.echo(f"  Shift applied from frame {detected_frame} to {shift_end_frame}")
                else:
                    typer.echo(f"  Shift detected/applied at frame: {detected_frame}")
                typer.echo(f"  Shift amount: {shift_amount:.2f}")
        
        # Apply background subtraction only if radius is specified
        if radius is not None:
            typer.echo(f"Applying background subtraction (radius={radius})...")
            corrected_stack, background_stack = subtract_background(
                tiff_stack,
                radius=radius,
                visualize=visualize
            )
        else:
            typer.echo("Skipping background subtraction (no radius specified)")
            corrected_stack = tiff_stack
        
        # Export corrected stack
        typer.echo("Exporting corrected TIFF stack...")
        output_path = export_tiff_stack(
            corrected_stack,
            file_path,
            output_dir=output_dir,
            suffix=suffix
        )
        
        typer.echo(f"\n[OK] Preprocessing completed successfully!")
        typer.echo(f"  Output file: {output_path}")
        
    except Exception as e:
        typer.echo(f"\n[ERROR] Error during preprocessing: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def create_config(
    output_path: str = typer.Option("config.yaml", "--output", "-o", help="Output path for config file"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing config file"),
):
    """
    Create a default configuration file.
    
    Example:
        microneedle-analysis create-config --output my_config.yaml
    """
    from microneedle_analysis.io.config import get_default_config, save_config
    
    output_file = Path(output_path)
    
    if output_file.exists() and not overwrite:
        typer.echo(f"Error: File already exists: {output_path}", err=True)
        typer.echo("Use --overwrite to overwrite existing file.", err=True)
        raise typer.Exit(1)
    
    try:
        config = get_default_config()
        save_config(config, str(output_file))
        typer.echo(f"[OK] Configuration file created: {output_path}")
    except Exception as e:
        typer.echo(f"Error creating config file: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def list_profiles(
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file (default: ./config.yaml, then project root, then package)",
    ),
):
    """
    List all profiles in the config file.
    
    Example:
        microneedle-analysis list-profiles
        microneedle-analysis list-profiles --config my_config.yaml
    """
    from microneedle_analysis.io.config import list_profiles as get_profiles

    config_path = _resolve_cli_config_path(config_path, required=True)

    if not Path(config_path).exists():
        typer.echo(f"Error: Config file not found: {config_path}", err=True)
        raise typer.Exit(1)
    
    try:
        profiles = get_profiles(config_path)
        if not profiles:
            typer.echo("No profiles found in config file.")
        else:
            typer.echo(f"\nProfiles in {config_path}:")
            typer.echo("=" * 60)
            for name in profiles.keys():
                typer.echo(f"{name}")
    except Exception as e:
        typer.echo(f"Error listing profiles: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def add_profile(
    profile_name: str = typer.Argument(..., help="Name of the profile"),
    file_path: Optional[str] = typer.Option(None, "--file-path", "-f", help="Full file path for this profile (use with --file-pattern for flexible matching)"),
    file_pattern: Optional[str] = typer.Option(None, "--file-pattern", help="Filename pattern to match (e.g., 'exp-1 2025 November 17' - works across different drive letters)"),
    shift_frame: Optional[int] = typer.Option(None, "--shift-frame", help="Frame where illumination shift starts"),
    shift_end_frame: Optional[int] = typer.Option(None, "--shift-end-frame", help="Frame where illumination shift ends"),
    radius: Optional[int] = typer.Option(None, "--radius", "-r", help="Background subtraction radius"),
    correct_shift: bool = typer.Option(False, "--correct-shift", help="Apply illumination shift correction"),
    spots_to_track: Optional[str] = typer.Option(
        None,
        "--spots-to-track",
        help="Comma-separated list of spot indices to track (e.g., '0,1,2'). Omit to track all.",
    ),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory for this profile"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Description of the profile"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file (default: microneedle_analysis/config.yaml)"),
):
    """
    Add or update a profile in the config file.
    
    For files on Google Drive or network drives with variable drive letters:
    - Use --file-pattern to match by filename pattern (recommended)
    - Or use --file-path with full path (will match ignoring drive letter)
    
    Example:
        # Using full path (works across different drive letters)
        microneedle-analysis add-profile raju_shade_exp1 --file-path "G:\\My Drive\\...\\file.tif" --shift-frame 262
        
        # Using filename pattern (more flexible, recommended)
        microneedle-analysis add-profile raju_shade_exp1 --file-pattern "exp-1 2025 November 17" --shift-frame 262
    """
    from microneedle_analysis.io.config import add_or_update_profile

    config_path = _resolve_cli_config_path(config_path, required=True)

    if not file_path and not file_pattern:
        typer.echo("Error: Either --file-path or --file-pattern must be provided", err=True)
        raise typer.Exit(1)
    
    try:
        profile_data = {
            'correct_shift': correct_shift,
        }
        
        if file_path:
            profile_data['file_path'] = file_path
        if file_pattern:
            profile_data['file_path_pattern'] = file_pattern
        
        if shift_frame is not None:
            profile_data['shift_frame'] = shift_frame
        if shift_end_frame is not None:
            profile_data['shift_end_frame'] = shift_end_frame
        if radius is not None:
            profile_data['radius'] = radius
        if output_dir:
            profile_data['output_dir'] = output_dir
        if spots_to_track:
            try:
                profile_data['spots_to_track'] = [
                    int(s.strip()) for s in spots_to_track.split(",") if s.strip()
                ]
            except ValueError:
                typer.echo(
                    f"Error: Invalid --spots-to-track format: {spots_to_track}. Expected comma-separated integers.",
                    err=True,
                )
                raise typer.Exit(1)
        if description:
            profile_data['description'] = description
        
        add_or_update_profile(config_path, profile_name, profile_data)
        typer.echo(f"[OK] Profile '{profile_name}' added/updated in {config_path}")
    except Exception as e:
        typer.echo(f"Error adding profile: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def remove_profile(
    profile_name: str = typer.Argument(..., help="Name of the profile to remove"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file (default: microneedle_analysis/config.yaml)"),
):
    """
    Remove a profile from the config file.
    
    Example:
        microneedle-analysis remove-profile raju_shade_exp1
    """
    from microneedle_analysis.io.config import remove_profile as delete_profile

    config_path = _resolve_cli_config_path(config_path, required=True)

    try:
        if delete_profile(config_path, profile_name):
            typer.echo(f"[OK] Profile '{profile_name}' removed from {config_path}")
        else:
            typer.echo(f"Error: Profile '{profile_name}' not found in {config_path}", err=True)
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error removing profile: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    from microneedle_analysis import __version__
    typer.echo(f"microneedle-analysis version {__version__}")


def main():
    """Entry point for CLI."""
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("\n\nInterrupted by user.", err=True)
        sys.exit(1)
    except Exception as e:
        typer.echo(f"\nUnexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

