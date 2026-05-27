"""Setup script for the microneedle_analysis supplementary bundle."""

import shutil
from pathlib import Path
from setuptools import find_packages, setup

root = Path(__file__).parent
readme = root / "README.md"
long_description = readme.read_text(encoding="utf-8") if readme.exists() else ""

# Ship manuscript profiles with the package (canonical file is repo-root config.yaml).
_root_config = root / "config.yaml"
_package_config = root / "microneedle_analysis" / "config.yaml"
if _root_config.exists():
    shutil.copy2(_root_config, _package_config)

setup(
    name="microneedle-analysis",
    version="1.0.0",
    description="Microneedle tip tracking and NIR imaging analysis pipeline.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Khong Duc Thinh",
    packages=find_packages(include=["microneedle_analysis", "microneedle_analysis.*"]),
    include_package_data=True,
    package_data={"microneedle_analysis": ["config.yaml"]},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.3.0",
        "tifffile>=2021.0.0",
        "scikit-image>=0.18.0",
        "statsmodels>=0.12.0",
        "scipy>=1.7.0",
        "pyyaml>=5.4.0",
        "typer>=0.4.0",
        "imageio>=2.9.0",
        "imageio-ffmpeg>=0.4.0",
    ],
    entry_points={
        "console_scripts": [
            "microneedle-analysis=microneedle_analysis.cli:main",
            "microneedle_analysis=microneedle_analysis.cli:main",
        ],
    },
)
