# SynthRain

This repository is a standalone testbed for TelcoRain-like interpolation behavior without database dependencies. 
Synthetic CML network + synthetic wet/dry classification + synthetic rainfall field -> IDW interpolation.



## Repository layout

- `run_scenarios.py`
  - Main single-scenario runner (network generation, wet/dry simulation, IDW, plots, optional CSV).
- `run_wet_sweep.py`
  - Thin CLI wrapper for wet-target sweep.
- `run_idw_sweep.py`
  - Thin CLI wrapper for IDW parameter sweep.
- `synthrain/sweeps.py`
  - Shared sweep logic used by both wrapper scripts.
- `synthrain/run_logging.py`
  - Console tee logging to `logs/<YYYYmmdd_HHMMSS>.log`.
- `configs/config.ini`
  - Default INI config used by runners.

## Install

```bash
pip install numpy pandas scipy matplotlib pillow pycomlink pypdf tqdm
```

Notes:
- `pypdf` is used for vector PDF merge/contact sheets.
- There is a fallback to `PyPDF2` if `pypdf` is not available.

## Single scenario

Run using config defaults:

```bash
python run_scenarios.py --config configs/config.ini
```

Run with CLI overrides:

```bash
python run_scenarios.py --config configs/config.ini --wet-target 0.2 --seed 0 --idw-near 12
```

Typical outputs in `io.out` folder:
- `true_field.png`
- `links.png`
- `idw_field.png`
- `idw_field.pdf`
- `diff.png`
- `scenario.json`
- `calc_dataset_synth.csv` (only when CSV export is enabled)

## Wet sweep

```bash
python run_wet_sweep.py --base-config configs/config.ini --out-root outputs_wet_sweep
```

Optional overrides:

```bash
python run_wet_sweep.py --base-config configs/config.ini --wet-targets 0.05,0.1,0.2,0.4 --seed 0 --n-sites 75
```

## IDW sweep

```bash
python run_idw_sweep.py --base-config configs/config.ini --out-root outputs_idw_sweep
```

Example focused sweep:

```bash
python run_idw_sweep.py --base-config configs/config.ini --powers 1,2,3 --nears 4,8,12 --dists 10000,30000 --n-sites-list 50 --seeds 0 --wet-targets 0.1
```

## Logging

All run scripts support:
- `--log-dir` (default: `logs`)
- `--log-to-file` / `--no-log-to-file`

When enabled, console output is mirrored to:

```text
logs/YYYYmmdd_HHMMSS.log
```

Important behavior:
- `--help` does not create log files.
- Sweep scripts call `run_scenarios.py` with `--no-log-to-file` to avoid nested log files.

## Config behavior

- INI values are loaded from `--config` (default `configs/config.ini`).
- CLI flags override INI values.
- Inline comments are supported (for example: `idw_near = 8 ; max neighbours per pixel`).

## Config quick reference (with row comments)

```ini
[io]
out = outputs                  ; output directory for one scenario
debug = true                   ; extra console debug info
seed = 0                       ; random seed

[network]
city = true                    ; use compact city bbox preset
city_bbox = 14.2,14.8,49.9,50.2 ; bbox used when city=true
noncity_bbox = 12.0,19.0,48.5,51.2 ; bbox used when city=false
; bbox = 12.0,19.0,48.5,51.2   ; explicit bbox override (highest priority)
n_sites = 50                   ; number of CML sites
mean_degree = 4                ; average link connectivity
site_sampling = poisson        ; uniform | poisson
site_min_dist_m = 3000         ; minimum spacing between sites (poisson mode)

[interp]
interp_style = "pycomlink"     ; pycomlink | custom
grid_step_m = 1000.0           ; grid resolution in meters
idw_power = 2                  ; IDW power parameter p
idw_near = 8                   ; max neighbours used per grid point
idw_dist_m = 10000             ; neighbour radius in meters (<=0 means unlimited)
dry_as_zero = true             ; true: dry links contribute as 0, false: ignored

[rain]
n_blobs = 6                    ; number of synthetic rain cells
blob_sigma_m = 6000.0          ; rain cell spread in meters
peak_mmph = 25.0               ; max rain intensity
noise_mmph = 1.0               ; link observation noise
min_rain = 0.1                 ; plotting/threshold minimum rain

[wet]
wet_mode = random              ; threshold | random | stratified
wet_target = 0.10              ; target wet fraction
wet_strata_nx = 8              ; stratified mode x bins
wet_strata_ny = 8              ; stratified mode y bins
flip_dry_to_wet = 0.02         ; random flip probability
flip_wet_to_dry = 0.10         ; random flip probability

[plot]
make_plot_titles = false       ; if true, sweeps inject per-run plot titles
title_name = IDW from links (mm/h) ; scenario plot title

[sweep]
wet_targets = 0.05,0.10,0.20,0.35,0.50 ; used by wet sweep when CLI value is omitted
```

## CLI quick reference (with row comments)

Wet sweep:
- `--base-config configs/config.ini` : INI source
- `--out-root outputs_wet_sweep` : root folder for sweep results
- `--wet-targets 0.05,0.1,0.2` : optional override of `[sweep] wet_targets`
- `--seed 0` : optional fixed seed override
- `--n-sites 75` : optional fixed `n_sites` override

IDW sweep:
- `--base-config configs/config.ini` : INI source
- `--out-root outputs_sweep` : root folder for sweep results
- `--powers 1,2,3` : IDW power sweep
- `--nears 4,8,12` : IDW neighbour-count sweep
- `--dists 10000,30000` : IDW radius sweep (meters)
- `--n-sites-list 50` : outer sweep for network size
- `--seeds 0,1` : outer sweep for random seeds
- `--wet-targets 0.1,0.3` : outer sweep for wet fractions

## IDW behavior

- `idw_dist_m` sets neighbor radius in meters (`<= 0` means unlimited radius).
- `idw_near` caps maximum neighbors used at each grid point.
- `dry_as_zero=true` includes dry links as 0 in interpolation.
- `dry_as_zero=false` completely ignores dry links in interpolation.
