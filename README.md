
# Microneedle NIR Imaging Analysis

This bundle contains a curated, sanitised snapshot of the microneedle NIR
imaging analysis pipeline, and is published as a standalone archival snapshot for
reproducibility.

This codebase provides an end-to-end image analysis workflow for time-lapse
microneedle NIR imaging experiments. In brief, it:

- loads TIFF image stacks from configured experiment profiles,
- detects microneedle tips and tracks them across frames (raw tip intensity
  plus a per-frame local background sample),
- applies background correction as a per-frame tip-to-background ratio
  (tip intensity ÷ background intensity), then pretreatment (baseline) normalization,
- applies smoothing and cohort quality control: per-tip time series
  are compared to a leave-one-out (LOO) per-frame median cohort reference;
  the root-mean-square (RMS) distance of each tip to that reference is used
  to flag atypical or problematic trajectories,
- exports tabular results and analysis plots.

The core implementation is packaged under `microneedle_analysis/` and exposed
through the CLI entry point `microneedle-analysis`.

## Required inputs

To run this pipeline, provide:

1. **Raw image data**  
   Time-series TIFF stacks (`.tif` / `.tiff`) for each experiment.

2. **Configuration file**  
   `config.yaml` stores default settings and per-experiment metadata in
   `profiles:`. 
   
   The `analyze` command selects a profile; the pipeline
loads that profile’s entry from `config.yaml` and uses it as the source of
truth for file paths, experiment layout, and numeric parameters. Global
options at the top of `config.yaml` (e.g. default detection/tracking) apply
unless overridden in the profile.

   Example (from the directory that contains `config.yaml`; replace the profile
   name with a key that exists under `profiles:` in your file):

   ```bash
   microneedle-analysis analyze --profile <profile_id> --config config.yaml
   ```

   If you do not pass `--config`, the CLI searches for `config.yaml` in this
   order: the current working directory, the repository root (parent of the
   `microneedle_analysis` package), then the copy bundled with the installed
   package. When in doubt, pass `--config config.yaml` explicitly.

   Profile IDs and field definitions are documented at the top of
   [`config.yaml`](config.yaml) (naming convention, per-profile fields, and
   figure-grouped profile blocks).

   See `microneedle-analysis --help` and `microneedle-analysis analyze --help` for more information.

3. **Data root mapping**  
   Replace every `<DATA_ROOT>` in profile `folder_path` or `file_path` entries
   with an absolute path on your machine, or with a relative path from your
   working directory.

   Example (edit `config.yaml` before running a profile):

   ```yaml
   folder_path: 'D:/deposited_data/fig3_bokchoy_exoiaa/rep01'
   ```

   On Windows, if a profile path is missing, the code may try alternate drive
   letters (`G:`, `H:`, `F:`, etc.). You can bypass this entirely by setting
   explicit paths or by passing the TIFF file or experiment folder directly on
   the command line.

4. **Deposited raw data location**  
   Download the archived TIFF stacks from Zenodo:

   - Data archive: [https://doi.org/10.5281/zenodo.20406710](https://doi.org/10.5281/zenodo.20406710)

   Unpack the archive so that folder paths in `config.yaml` match your local
   layout (e.g. `<DATA_ROOT>/fig3_bokchoy_exoiaa/rep01` contains IAA and Ref
   `.tif` files for that replicate).

## Generated outputs

For each executed profile, the pipeline produces timestamped output folders
containing analysis artefacts such as:

- tracked tip trajectories and intensity tables (CSV),
- normalized/smoothed time-series outputs,
- quality-control summaries and diagnostic plots,
- plot outputs such as tracked-tip overlays, per-spot intensity curves,
  cohort-QC diagnostics (e.g., RMS distance/flag summaries), and
  average-intensity trend plots (saved in the configured format, e.g. SVG/PNG).

For full processing steps, see `microneedle_analysis/pipeline.py` (class
`MicroneedlePipeline`).

## Contents

```
repository_root/
    README.md                  (this file)
    LICENSE                    (MIT)
    requirements.txt
    setup.py
    config.yaml                (profiles and defaults for image-analysis workflows)
    microneedle_analysis/
        __init__.py
        pipeline.py            (end-to-end analysis pipeline)
        cli.py                 (command-line interface)
        core/                  (TIFF loading, tip detection, tracking)
        analysis/              (smoothing, normalisation, step correction,
                                 cohort QC)
        visualization/         (tracking, intensity-curve, cohort-QC plots,
                                 video export)
        io/                    (config loader, exporters, replicate paths)
```

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Tested with Python 3.8–3.14 on Windows and Linux.

## Quick start

### Running on your own data (no profile)

```bash
# Single TIFF stack with built-in defaults
microneedle-analysis analyze path/to/my_image_stack.tif --output ./results

# Use repo config for detection/tracking defaults
microneedle-analysis analyze path/to/my_image_stack.tif --output ./results --config config.yaml

# Alternative CLI invocation (run the module directly)
python -m microneedle_analysis.cli analyze path/to/my_image_stack.tif --output ./results --config config.yaml

# MP4 tracking overlay (requires imageio and imageio-ffmpeg)
microneedle-analysis analyze path/to/stack.tif --output ./results --export-video
```

List available profile IDs:

```bash
microneedle-analysis list-profiles --config config.yaml
```

### Reproducing a deposited experiment

After installing the package, downloading the data archive, and replacing
`<DATA_ROOT>` in `config.yaml` with your local data path:

1. List profiles (IDs follow `fig{N}_...` — see [`config.yaml`](config.yaml)):

   ```bash
   microneedle-analysis list-profiles --config config.yaml
   ```

2. Run a profile. Most manuscript experiments use **folder-based** profiles:
   each profile points at a replicate folder containing channel TIFF stacks
   (e.g. filenames matching `IAA` and `Ref` per `file_keywords` in the
   profile). Figure 3 exo-IAA profiles are an example of this layout.

   ```bash
   microneedle-analysis analyze --profile fig3_bokchoy_exoiaa_rep01 --config config.yaml
   ```

   The pipeline discovers `.tif` files in `folder_path`, processes each
   channel, and writes timestamped outputs under subfolders such as
   `IAA_output` and `Ref_output` (see `output_subdirs` in the profile).

   Alternative invocation:

   ```bash
   python -m microneedle_analysis.cli analyze --profile fig3_bokchoy_exoiaa_rep01 --config config.yaml
   ```

   To override the data location without editing the file, pass the replicate
   folder on the command line:

   ```bash
   microneedle-analysis analyze D:/deposited_data/fig3_bokchoy_exoiaa/rep01 --profile fig3_bokchoy_exoiaa_rep01 --config config.yaml
   ```

   For other figures, choose the matching `fig4_*` or `fig5_*` profile from
   `list-profiles` and use the same `--profile` pattern.

## Citation

If you use this code, please cite:

    Project: Microneedle NIR Imaging Analysis Pipeline
    Code archive: [Zenodo DOI — to be inserted]
    Data archive: https://doi.org/10.5281/zenodo.20406710

## License

MIT (see `LICENSE`).
