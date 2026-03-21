# telcorain_synth

Lightweight synthetic testbed for TelcoRain-like rainfall interpolation.

The repository is built for one specific job: generate configurable CML-like
networks, inject realistic wet/dry and failure behavior, run IDW, and compare
the reconstructed field against known synthetic truth without needing real CML
data or database access.

## What It Simulates

- Synthetic CML site and link networks inside a configurable bbox
- Synthetic rainfall fields built from smooth rain blobs
- Wet/dry labels with target wet fraction and misclassification
- Path-averaged link observations instead of center-only sampling
- Faulty or missing links:
  - nonresponding links
  - clustered outages
  - stuck-at-zero links
  - biased links
  - extra-noisy links
- IDW interpolation with either:
  - `pycomlink`
  - built-in custom KDTree backend

## Repository Layout

- `run_scenario.py`
  - Main single-scenario CLI
- `run_scenarios.py`
  - Backward-compatible wrapper to `run_scenario.py`
- `run_wet_sweep.py`
  - Thin CLI wrapper for wet-target sweeps
- `run_idw_sweep.py`
  - Thin CLI wrapper for IDW parameter sweeps
- `synthrain/config.py`
  - Typed config model and INI loading
- `synthrain/scenario.py`
  - Pure library entrypoint: `run_scenario(config)`
- `synthrain/outputs.py`
  - Plot, CSV, and metadata writing
- `synthrain/metrics.py`
  - Field evaluation metrics
- `synthrain/sweeps.py`
  - Direct sweep execution and summary CSV creation
- `configs/config.ini`
  - Example config

## Install

Editable install:

```bash
pip install -e .
```

With `pycomlink` backend:

```bash
pip install -e .[pycomlink]
```

For development:

```bash
pip install -e .[dev]
```

## Single Scenario

Run from config:

```bash
python run_scenario.py --config configs/config.ini
```

Override selected parameters:

```bash
python run_scenario.py --config configs/config.ini --wet-target 0.2 --seed 7 --idw-near 12
```

Use the custom backend without `pycomlink`:

```bash
python run_scenario.py --config configs/config.ini --interp-style custom
```

Example fault injection:

```bash
python run_scenario.py --config configs/config.ini --interp-style custom --outage-fraction 0.15 --clustered-outage-fraction 0.2 --stuck-zero-fraction 0.1
```

Typical outputs:

- `true_field.png`
- `links.png`
- `idw_field.png`
- `idw_field.pdf`
- `diff.png`
- `scenario.json`
- `calc_dataset_synth.csv`

`scenario.json` now includes config, metrics, and realized fault counts.

## Sweeps

Wet-target sweep:

```bash
python run_wet_sweep.py --base-config configs/config.ini --out-root outputs_wet_sweep
```

IDW sweep:

```bash
python run_idw_sweep.py --base-config configs/config.ini --out-root outputs_idw_sweep
```

Focused IDW sweep:

```bash
python run_idw_sweep.py --base-config configs/config.ini --powers 1,2,3 --nears 4,8,12 --dists 10000,30000 --n-sites-list 50 --seeds 0 --wet-targets 0.1
```

Each sweep writes metric tables using values such as:

- `rmse`
- `mae`
- `bias`
- `pearson_r`
- `valid_pixel_fraction`
- `wet_hit_rate`
- `wet_miss_rate`
- `dry_false_rain_rate`

Wet sweep is comparison-oriented:

```text
outputs_wet_sweep/
  comparison.csv
  summary.csv
  manifest.json
  scenarios/
    wet0.1/
      summary.csv
      run/
  reports/
    global/
      all_runs.csv
      wet_sweep_metrics.png
```

IDW sweep is ranking-oriented because each scenario contains many IDW parameter combinations:

```text
outputs_sweep/
  leaderboard.csv
  summary.csv
  manifest.json
  scenarios/
    <scenario_tag>/
      summary.csv
      best_run.json
      reports/
      runs/
        <run_tag>/
  reports/
    global/
      all_runs.csv
      best_per_scenario.csv
      best_per_scenario_contact_sheet.pdf
```

Notes:

- `scenarios/.../runs/...` contains the raw scenario artifacts.
- `scenarios/.../reports/` contains derived visual summaries such as contact sheets and heatmaps.
- Wet sweep writes `comparison.csv` because there is only one run per wet-target scenario, so ranking would be meaningless.
- IDW sweep writes `leaderboard.csv` because it compares multiple parameter combinations inside each scenario.
- `best_run.json` is only used for IDW sweep scenarios.
- IDW sweeps use metrics to choose representative runs for global reports instead of taking the first run arbitrarily.

## Config

Config precedence:

1. built-in defaults
2. values from `--config`
3. CLI overrides

Key sections:

```ini
[io]
out = outputs ; output directory for one scenario
debug = true ; enable extra console diagnostics
seed = 0 ; base random seed for the whole scenario

[network]
city = true ; use compact city bbox preset instead of noncity bbox
city_bbox = 14.2,14.8,49.9,50.2 ; dense urban test area
noncity_bbox = 12.0,19.0,48.5,51.2 ; larger country-scale test area
n_sites = 50 ; number of synthetic microwave sites
mean_degree = 4 ; approximate average number of links per site
site_sampling = poisson ; site placement mode: poisson or uniform
site_min_dist_m = 3000 ; minimum site spacing in poisson mode

[interp]
interp_style = "pycomlink" ; interpolation backend: pycomlink or custom
grid_step_m = 1000.0 ; interpolation grid resolution in meters
idw_power = 2 ; inverse-distance weighting power
idw_near = 8 ; maximum neighbours used per grid point
idw_dist_m = 10000 ; maximum search radius in meters, <=0 means unlimited
dry_as_zero = true ; include dry links as zero instead of ignoring them

[rain]
n_blobs = 6 ; number of synthetic rain cells
blob_sigma_m = 6000.0 ; characteristic rain-cell width
peak_mmph = 25.0 ; peak rainfall intensity
background_mmph = 0.0 ; uniform background rainfall level
min_rain = 0.1 ; plotting and wet/dry threshold floor

[wet]
wet_mode = random ; wet/dry assignment mode: threshold, random, stratified
wet_target = 0.10 ; target fraction of wet links
wet_strata_nx = 8 ; stratified mode: number of x-direction bins
wet_strata_ny = 8 ; stratified mode: number of y-direction bins
flip_dry_to_wet = 0.02 ; dry-to-wet label error probability
flip_wet_to_dry = 0.10 ; wet-to-dry label error probability

[observation]
noise_mmph = 1.0 ; baseline observation noise added to links
link_path_samples = 9 ; samples taken along each link path before averaging

[faults]
outage_fraction = 0.00 ; fraction of links removed as unavailable
clustered_outage_fraction = 0.00 ; fraction removed in one spatial outage cluster
stuck_zero_fraction = 0.00 ; fraction of links forced to always report zero
bias_fraction = 0.00 ; fraction of links with multiplicative bias
bias_low = 0.85 ; lower bound of multiplicative bias factor
bias_high = 1.15 ; upper bound of multiplicative bias factor
extra_noise_fraction = 0.00 ; fraction of links with additional noise
extra_noise_mmph = 2.0 ; extra noise sigma added to selected links
```

## Python API

You can also run scenarios directly from Python:

```python
from dataclasses import replace

from synthrain import load_scenario_config, run_scenario, write_scenario_outputs

cfg = load_scenario_config("configs/config.ini")
cfg = replace(cfg, interp=replace(cfg.interp, interp_style="custom", idw_near=12))
result = run_scenario(cfg)
write_scenario_outputs(result)
print(result.metrics)
```
