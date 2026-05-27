"""Configuration management."""

import os
import string
from pathlib import Path
import yaml
from typing import Dict, Any, List, Tuple, Optional


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration dictionary.
    
    Returns:
    --------
    config : dict
        Default configuration dictionary
    """
    return {
        'exposure_per_frame': 30,  # seconds per frame, global for all time axes
        'background_subtraction': {
            'radius': 25,
            'visualize': False
        },
        'detection': {
            'min_distance': 4,
            'threshold_rel': 0.5,
            'visualize': True
        },
        'tracking': {
            'spots_to_track': None,  # None = track all
            'search_range': 10,  # ROI search radius (pixels)
            'diameter': 5,  # Spot diameter for intensity calculation
            'min_distance': 3,  # Minimum distance separating peaks
            'threshold_rel': 0.5,  # Relative threshold for peak detection
            'background_roi': None,  # Background ROI as (y_min, y_max, x_min, x_max). None = exclude spots from whole frame
            'skip_initial_frames': 0,  # Number of frames to skip at start of stack (0 = use all from start)
            'max_frame': None,  # None = process full stack; integer = max frame index (exclusive) after skip
        },
        'normalization': {
            'starting_frame': 100,
            'ending_frame': 3000
        },
        'visualization': {
            'exposure_per_frame': 30,  # seconds
            'format': 'svg',
            'ratiometric_iaa_ref_ylim': None,  # [ymin, ymax] for IAA/Ref ratiometric 2x2 figure, or null for auto
        },
        'smoothing': {
            'exposure_per_frame': 30,  # seconds
            'alpha': 0.01,
            'min_time_range': [210, 700],  # minutes
            'n_frames_baseline': 50,
            'trend_smoothing': {
                'enabled': True,
                'lowess_frac': 0.1,
                'savgol_window': 51,
                'savgol_polyorder': 3,
                'rolling_window': 50,
            },
        },
        'cohort_qc': {
            'enabled': True,
            # threshold = median(d) + mad_lambda * 1.4826 * MAD(d) on RMS distances
            'mad_lambda': 3.0,
        },
    }


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file or return defaults.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to YAML config file. If None, uses default config.
        
    Returns:
    --------
    config : dict
        Configuration dictionary
    """
    if config_path is None or not os.path.exists(config_path):
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Merge with defaults to ensure all keys exist
        default_config = get_default_config()
        config = _merge_config(default_config, config)
        return config
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        print("Using default configuration.")
        return get_default_config()


def _merge_config(default: Dict, user: Dict) -> Dict:
    """Recursively merge user config into default config."""
    result = default.copy()
    for key, value in user.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    config_path : str
        Path to save config file
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_profile(config_path: str, profile_name: str) -> Dict[str, Any]:
    """
    Get a specific profile from config file.
    
    Parameters:
    -----------
    config_path : str
        Path to YAML config file
    profile_name : str
        Name of the profile to retrieve
        
    Returns:
    --------
    profile : dict or None
        Profile dictionary if found, None otherwise
    """
    config = load_config(config_path)
    profiles = config.get('profiles', {})
    return profiles.get(profile_name)


def list_profiles(config_path: str) -> Dict[str, Dict[str, Any]]:
    """
    List all profiles in config file.
    
    Parameters:
    -----------
    config_path : str
        Path to YAML config file
        
    Returns:
    --------
    profiles : dict
        Dictionary of all profiles
    """
    config = load_config(config_path)
    return config.get('profiles', {})


def generate_profile_pattern(full_path: str) -> str:
    """
    Generates a flexible file_path_pattern for YAML configuration from a full file path.
    
    The pattern is constructed as: "<Parent Folder Name>*<Filename>"
    This pattern ignores the variable drive letter (G:, H:, etc.) or 
    the starting user directories, ensuring cross-computer compatibility.
    
    Parameters:
    -----------
    full_path : str
        Full file path (e.g., "G:\\My Drive\\...\\folder\\file.tif")
        
    Returns:
    --------
    pattern : str
        Pattern string in format "ParentFolder*Filename"
    """
    # 1. Normalize and clean the path to handle different OS path separators
    normalized_path = full_path.replace('\\', os.path.sep).replace('/', os.path.sep)
    
    # 2. Get the filename
    filename = os.path.basename(normalized_path)
    
    # 3. Get the path of the directory containing the file
    directory_path = os.path.dirname(normalized_path)
    
    # 4. Get the parent folder name (the last directory component)
    parent_folder = os.path.basename(directory_path)
    
    # 5. Combine them with the wildcard in the middle
    if parent_folder and filename:
        # Example output: "Raju-SHADE exp-30sec-each frame 27-Nov-2025*exp-1 2025 November 17 16_18_02-1.tif"
        pattern = f"{parent_folder}*{filename}"
        return pattern
    else:
        # Fallback if the file is at the root of a drive or structure is too shallow
        return filename


def _ensure_file_path_pattern(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure profile has file_path_pattern. If file_path is provided but file_path_pattern is not,
    automatically generate file_path_pattern from file_path.
    
    Parameters:
    -----------
    profile : dict
        Profile dictionary
        
    Returns:
    --------
    profile : dict
        Profile dictionary with file_path_pattern ensured
    """
    profile = profile.copy()  # Don't modify original
    
    # If file_path exists but file_path_pattern doesn't, generate it
    if profile.get('file_path') and not profile.get('file_path_pattern'):
        profile['file_path_pattern'] = generate_profile_pattern(profile['file_path'])
    
    return profile


def find_profile_by_path(config_path: str, file_path: str) -> tuple:
    """
    Find a profile that matches the given file path.
    
    This function tries multiple matching strategies:
    1. Exact path match (handles different drive letters)
    2. Filename match (if file_path_pattern is used in profile, or auto-generated from file_path)
    3. Relative path match (ignoring drive letter)
    
    Parameters:
    -----------
    config_path : str
        Path to YAML config file
    file_path : str
        File path to match against profiles
        
    Returns:
    --------
    profile_name, profile : tuple
        Tuple of (profile_name, profile_dict) if match found, (None, None) otherwise
    """
    profiles = list_profiles(config_path)
    
    # Normalize paths for comparison
    file_path_normalized = os.path.normpath(os.path.abspath(file_path))
    file_path_lower = file_path_normalized.lower()
    file_name = os.path.basename(file_path_normalized)
    file_name_lower = file_name.lower()
    directory_path = os.path.dirname(file_path_normalized)
    directory_path_lower = directory_path.lower()
    
    # Strategy 1: Exact path match (handles different drive letters by comparing after drive)
    for profile_name, profile in profiles.items():
        profile = _ensure_file_path_pattern(profile)  # Auto-generate pattern if needed
        profile_file_path = profile.get('file_path')
        if profile_file_path:
            try:
                profile_file_path_normalized = os.path.normpath(os.path.abspath(profile_file_path))
                # Compare paths ignoring drive letter (Windows)
                # Extract path after drive letter
                def strip_drive(path):
                    """Remove drive letter from path for comparison."""
                    if len(path) >= 2 and path[1] == ':':
                        return path[2:].replace('\\', '/')
                    return path.replace('\\', '/')
                
                file_path_no_drive = strip_drive(file_path_lower)
                profile_path_no_drive = strip_drive(profile_file_path_normalized.lower())
                
                if file_path_no_drive == profile_path_no_drive:
                    return profile_name, profile
            except (OSError, ValueError):
                # If path doesn't exist or is invalid, try other strategies
                pass
    
    # Strategy 2: Filename match (if profile uses file_path_pattern or just filename)
    for profile_name, profile in profiles.items():
        profile = _ensure_file_path_pattern(profile)  # Auto-generate pattern if needed
        # Check if profile has file_path_pattern (for flexible matching)
        file_path_pattern = profile.get('file_path_pattern')
        if file_path_pattern:
            # Pattern matching: check if parent folder and filename match
            # Pattern format: "ParentFolder*Filename"
            if '*' in file_path_pattern:
                parent_part, filename_part = file_path_pattern.split('*', 1)
                # Check if filename matches and parent folder name is in the path
                if filename_part.lower() in file_name_lower or filename_part.lower() == file_name_lower:
                    # Check if parent folder is in the directory path
                    if parent_part.lower() in directory_path_lower:
                        return profile_name, profile
            else:
                # Simple pattern matching - check if pattern is in filename
                if file_path_pattern.lower() in file_name_lower:
                    return profile_name, profile
        
        # Also check if profile's file_path ends with same filename
        profile_file_path = profile.get('file_path')
        if profile_file_path:
            profile_file_name = os.path.basename(profile_file_path)
            if profile_file_name.lower() == file_name_lower:
                return profile_name, profile
    
    return None, None


def add_or_update_profile(config_path: str, profile_name: str, profile_data: Dict[str, Any]):
    """
    Add or update a profile in the config file.
    
    Parameters:
    -----------
    config_path : str
        Path to YAML config file
    profile_name : str
        Name of the profile
    profile_data : dict
        Profile data dictionary
    """
    config = load_config(config_path)
    
    if 'profiles' not in config:
        config['profiles'] = {}
    
    config['profiles'][profile_name] = profile_data
    save_config(config, config_path)


def remove_profile(config_path: str, profile_name: str) -> bool:
    """
    Remove a profile from the config file.
    
    Parameters:
    -----------
    config_path : str
        Path to YAML config file
    profile_name : str
        Name of the profile to remove
        
    Returns:
    --------
    success : bool
        True if profile was removed, False if not found
    """
    config = load_config(config_path)
    
    if 'profiles' not in config:
        return False
    
    if profile_name in config['profiles']:
        del config['profiles'][profile_name]
        save_config(config, config_path)
        return True
    
    return False


def get_folder_keyword_profiles(config_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Get all profiles with profile_type 'folder_keywords' or that have folder_path.
    
    Parameters:
    -----------
    config_path : str
        Path to YAML config file
        
    Returns:
    --------
    profiles : dict
        Dictionary of folder_keywords profiles
    """
    profiles = list_profiles(config_path)
    return {
        name: p for name, p in profiles.items()
        if p.get('profile_type') == 'folder_keywords' or 'folder_path' in p
    }


def _strip_drive(path: str) -> str:
    """Remove drive letter from path for comparison (Windows)."""
    path = path.replace('\\', '/')
    if len(path) >= 2 and path[1] == ':':
        return path[2:]
    return path


def find_profile_by_folder_path(config_path: str, folder_path: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Find a profile that matches the given folder path (for folder_keywords profiles).
    
    Matches when the profile's folder_path equals the given path or when the given
    path is under the profile's folder_path. Handles different drive letters on Windows.
    
    Parameters:
    -----------
    config_path : str
        Path to YAML config file
    folder_path : str
        Folder path to match against profiles
        
    Returns:
    --------
    profile_name, profile : tuple
        Tuple of (profile_name, profile_dict) if match found, (None, None) otherwise
    """
    profiles = get_folder_keyword_profiles(config_path)
    if not profiles:
        return None, None
    
    folder_path_normalized = os.path.normpath(os.path.abspath(folder_path))
    folder_path_lower = folder_path_normalized.lower()
    folder_path_no_drive = _strip_drive(folder_path_lower)
    
    for profile_name, profile in profiles.items():
        profile_folder = profile.get('folder_path')
        if not profile_folder:
            continue
        try:
            profile_folder_normalized = os.path.normpath(os.path.abspath(profile_folder))
            profile_folder_lower = profile_folder_normalized.lower()
            profile_folder_no_drive = _strip_drive(profile_folder_lower)
            
            # Exact match or given path is prefix of profile (profile is more specific)
            if folder_path_no_drive == profile_folder_no_drive:
                return profile_name, profile
            # Given path is under profile's folder_path
            if folder_path_no_drive.startswith(profile_folder_no_drive.rstrip('/') + '/'):
                return profile_name, profile
            # Profile's folder_path is under given path (e.g., user passed Fig-3, profile has Fig-3/Fig3-Bokchoi)
            if profile_folder_no_drive.startswith(folder_path_no_drive.rstrip('/') + '/'):
                return profile_name, profile
        except (OSError, ValueError):
            continue
    
    return None, None


def resolve_path_for_machine(path: Optional[str]) -> Optional[str]:
    """
    Resolve a file or directory path that may exist on another Windows drive letter.

    Tries the literal path first, then alternate drive letters (Google Drive paths
    often differ between machines). Non-Windows: only the literal path is tried.

    Returns
    -------
    Absolute path string if found, else None.
    """
    from pathlib import Path, PureWindowsPath

    if not path:
        return None
    pth = Path(path)
    try:
        if pth.exists():
            return str(pth.resolve())
    except OSError:
        return None

    if os.name != "nt":
        return None

    try:
        p = PureWindowsPath(path)
        if not p.drive:
            return None
        original_upper = p.drive.upper()
        ordered = ["G:", "H:", "F:", "E:", "D:", "C:"]
        seen: set[str] = set()
        drives_to_try: List[str] = []
        for d in ordered:
            du = d.upper()
            if du == original_upper or du in seen:
                continue
            drives_to_try.append(d)
            seen.add(du)
        for letter in string.ascii_uppercase:
            d = f"{letter}:"
            du = d.upper()
            if du == original_upper or du in seen:
                continue
            if not Path(d + os.sep).exists():
                continue
            drives_to_try.append(d)
            seen.add(du)
        for drive in drives_to_try:
            candidate = PureWindowsPath(drive, *p.parts[1:])
            if Path(candidate).exists():
                return str(Path(candidate).resolve())
    except Exception:
        pass

    return None


def resolve_folder_path(profile: Dict[str, Any], config_path: str) -> Optional[str]:
    """
    Resolve folder_path from a folder_keywords profile, with drive-letter flexibility.

    Tries the literal path first, then common drive letters on Windows if not found.

    Parameters:
    -----------
    profile : dict
        Profile with folder_path
    config_path : str
        Path to config (for context, not used for resolution)

    Returns:
    --------
    resolved_path : str or None
        Resolved absolute path if folder exists, None otherwise
    """
    folder_path = profile.get("folder_path")
    if not folder_path:
        return None
    return resolve_path_for_machine(folder_path)


def resolve_default_config_path() -> Optional[str]:
    """
    Locate the default config.yaml for CLI commands.

    Search order:
    1. ./config.yaml in the current working directory
    2. Project/repository root (parent of the microneedle_analysis package directory)

    Returns:
    --------
    path : str or None
        Absolute path to the first existing candidate, or None if not found.
    """
    import microneedle_analysis

    package_dir = Path(microneedle_analysis.__file__).resolve().parent
    candidates = [
        Path.cwd() / "config.yaml",
        package_dir.parent / "config.yaml",
    ]
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return str(resolved)
    return None