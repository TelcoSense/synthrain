# User Guide

This guide explains how to run experiments with `synthrain`: what each
experiment type is for, which parameters matter, how to set them, and what
outputs to expect.

## 1. Setup

Run commands from the repository root:

```bash
git clone <repository-url>
cd synthrain
```

Create and activate a Python environment. Any standard Python environment is
fine; the examples below use `venv`.

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the package in editable mode if needed:

```bash
python -m pip install -e .
```

If you want the `pycomlink` backend:

```bash
python -m pip install -e ".[pycomlink]"
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
```

If you prefer conda, create any environment with Python 3.10 or newer, activate
it, and run the same `python -m pip install -e .` commands inside it.

For non-interactive plotting on servers, CI systems, or remote shells, it can be
useful to set Matplotlib's backend before running commands.

Linux/macOS:

```bash
export MPLBACKEND=Agg
```

Windows PowerShell:

```powershell
$env:MPLBACKEND = "Agg"
```

Most multi-line examples in this guide use Bash-style `\` line continuation.
On Windows PowerShell, either run the command on one line or replace `\` with
PowerShell's backtick continuation character.

## 2. Experiment Types

The project supports three main workflows:

- Single scenario: one synthetic network, one rain field, one IDW setting.
- Wet sweep: compare different wet-link fractions.
- IDW sweep: compare many IDW parameter combinations across one or more scenarios.

Use a single scenario when you want to inspect one case visually. Use a wet
sweep when you want to understand how rain coverage affects performance. Use an
IDW sweep when you want to choose interpolation parameters.

## 3. Single Scenario

Basic command:

```bash
python run_scenario.py --config configs/config.ini
```

Useful command with explicit parameters:

```bash
python run_scenario.py \
  --config configs/config.ini \
  --out outputs/demo_single \
  --seed 0 \
  --n-sites 50 \
  --wet-target 0.2 \
  --idw-power 2 \
  --idw-near 8 \
  --idw-dist-m 10000 \
  --interp-style pycomlink
```

The same command in Bash style:

```bash
python run_scenario.py \
  --config configs/config.ini \
  --out outputs/demo_single \
  --seed 0 \
  --n-sites 50 \
  --wet-target 0.2 \
  --idw-power 2 \
  --idw-near 8 \
  --idw-dist-m 10000 \
  --interp-style pycomlink
```

### Important Single-Scenario Parameters

`--out`
: Output folder for this run.

`--seed`
: Random seed. The same seed should recreate the same synthetic network and rain field.

`--n-sites`
: Number of microwave sites. More sites usually create more links and better spatial coverage.

`--wet-target`
: Target fraction of wet links. Low values create sparse rain observations.

`--interp-style`
: `pycomlink` or `custom`. Use `custom` if `pycomlink` is unavailable.

`--idw-power`
: IDW distance exponent. Higher values emphasize nearby links more strongly.

`--idw-near`
: Maximum number of nearby links used per grid cell.

`--idw-dist-m`
: Maximum search radius in meters. Use `0` for unlimited distance.

`--dry-as-zero` / `--no-dry-as-zero`
: If enabled, dry links contribute zero rain. If disabled, dry links are ignored.

### Single-Scenario Outputs

Expected files in the output folder:

```text
outputs/demo_single/
  true_field.png
  links.png
  idw_field.png
  idw_field.pdf
  diff.png
  scenario.json
  calc_dataset_synth.csv
```

`true_field.png`
: Synthetic rainfall truth.

`links.png`
: Link centers, wet/dry labels, and unavailable links.

`idw_field.png`
: Reconstructed rainfall field.

`diff.png`
: Difference between reconstruction and truth.

`scenario.json`
: Config, metrics, and realized fault counts.

`calc_dataset_synth.csv`
: Synthetic CML-like dataset if CSV export is enabled.

## 4. Config File

The default config is:

```text
configs/config.ini
```

Parameter precedence is:

1. Built-in defaults.
2. Values in `configs/config.ini`.
3. CLI overrides.

For example, if `configs/config.ini` has:

```ini
[interp]
idw_power = 2
```

but you run:

```bash
python run_scenario.py --config configs/config.ini --idw-power 3
```

then the run uses `idw_power = 3`.

## 5. Wet Sweep

Wet sweeps run one scenario per wet fraction.

Basic command:

```bash
python run_wet_sweep.py --base-config configs/config.ini --out-root outputs_wet_sweep
```

Focused command:

```bash
python run_wet_sweep.py \
  --base-config configs/config.ini \
  --out-root outputs_wet_sweep \
  --wet-targets 0.05,0.1,0.2,0.35,0.5 \
  --seed 0 \
  --n-sites 50
```

### Wet Sweep Parameters

`--wet-targets`
: Comma-separated list of wet-link fractions.

`--seed`
: Seed used for all wet-target scenarios.

`--n-sites`
: Number of microwave sites.

`--max-per-page`
: Maximum number of panels in contact-sheet PDFs.

### Wet Sweep Outputs

Expected structure:

```text
outputs_wet_sweep/
  comparison.csv
  summary.csv
  manifest.json
  scenarios/
    wet0.05/
      summary.csv
      run/
    wet0.1/
      summary.csv
      run/
  reports/
    global/
      all_runs.csv
      wet_sweep_contact_sheet.pdf
      wet_sweep_contact_sheet_vector.pdf
      wet_sweep_metrics.png
      wet_sweep_metrics.pdf
```

`comparison.csv`
: Compact table comparing wet targets.

`reports/global/wet_sweep_metrics.png`
: Line plots of metrics versus `wet_target`.

## 6. IDW Sweep

IDW sweeps run many combinations of:

- `idw_power`
- `idw_near`
- `idw_dist_m`
- `n_sites`
- `wet_target`
- `seed`

Basic command:

```bash
python run_idw_sweep.py --base-config configs/config.ini --out-root outputs_idw_sweep
```

Focused command:

```bash
python run_idw_sweep.py \
  --base-config configs/config.ini \
  --out-root outputs_idw_sweep \
  --powers 1,2,3 \
  --nears 4,8,12 \
  --dists 10000,30000 \
  --n-sites-list 50 \
  --wet-targets 0.2 \
  --seeds 0,1,2
```

### IDW Parameter Meaning

`--powers`
: IDW exponents. Lower values produce smoother fields. Higher values make the
interpolation more local.

`--nears`
: Maximum number of neighboring links used for each grid cell.

`--dists`
: Maximum search radius in meters. Use `0` for unlimited radius.

`--n-sites-list`
: Network sizes to test.

`--wet-targets`
: Rain coverage levels to test.

`--seeds`
: Random seeds. Use multiple seeds to measure robustness.

## 7. IDW Presets

Presets encode common experiment grids.

```bash
python run_idw_sweep.py --base-config configs/config.ini --out-root outputs_idw_quick --preset quick
```

Available presets:

`quick`
: Small sweep for testing that the pipeline works.

`poster`
: The 12-run poster grid: powers `1,2.5,4`, nears `6,12`, distances `10000,30000`,
with a 2x6 contact sheet layout.

`robust`
: Larger sweep across more powers, neighbor counts, distances, site counts, wet
targets, and seeds.

Explicit CLI values override preset values. For example:

```bash
python run_idw_sweep.py --preset robust --seeds 0,1 --n-sites-list 50
```

uses the robust preset, but only seeds `0,1` and only `n_sites = 50`.

## 8. Ranking IDW Runs

IDW sweeps rank runs within each scenario. The default ranking metric is RMSE:

```bash
python run_idw_sweep.py --base-config configs/config.ini --out-root outputs_idw_sweep
```

To rank by balanced score:

```bash
python run_idw_sweep.py \
  --base-config configs/config.ini \
  --out-root outputs_idw_sweep \
  --ranking-metric balanced_score
```

Available ranking metrics:

`rmse`
: Lower reconstruction error is better.

`balanced_score`
: Lower is better. Combines normalized RMSE, MAE, invalid-pixel penalty, wet miss
rate, dry false rain rate, and absolute bias.

`detection_score`
: Lower is better. Focuses on wet misses, dry false rain, and invalid pixels.

`valid_pixel_fraction`
: Higher is better. Useful when coverage matters more than local error.

Recommended default for research-style sweeps:

```bash
--ranking-metric balanced_score
```

Recommended default for simple visual/poster examples:

```bash
--ranking-metric rmse
```

## 9. IDW Sweep Outputs

Expected structure:

```text
outputs_idw_sweep/
  leaderboard.csv
  summary.csv
  parameter_robustness.csv
  manifest.json
  scenarios/
    nsites50_wet0.2_seed0/
      summary.csv
      best_run.json
      reports/
        idw_sweep_contact_sheet.pdf
        idw_sweep_contact_sheet_vector.pdf
        idw_fields_merged_vector.pdf
        metric_heatmap_rmse.png
        metric_heatmap_balanced_score.png
        metric_heatmap_valid_pixel_fraction.png
      runs/
        p1_n4_d10000/
        p1_n4_d30000/
        ...
  reports/
    global/
      all_runs.csv
      best_per_scenario.csv
      parameter_robustness.csv
      best_per_scenario_contact_sheet.pdf
      ALL_SCENARIOS_idw_fields_contact_sheet_vector.pdf
```

`scenarios/.../summary.csv`
: All parameter combinations for one scenario, with ranks and metrics.

`best_run.json`
: Best run for the scenario according to `--ranking-metric`.

`leaderboard.csv`
: Best run from each scenario.

`reports/global/all_runs.csv`
: All runs across all scenarios.

`parameter_robustness.csv`
: Aggregates each IDW parameter tuple across scenarios/seeds. This is the best
file for choosing robust parameters.

## 10. Reading IDW Results

Start with:

```text
outputs_idw_sweep/parameter_robustness.csv
```

Important columns:

`rank_robust`
: Rank of each IDW parameter tuple across all scenarios.

`rmse_mean`, `rmse_std`, `rmse_worst`
: Average, variability, and worst-case RMSE.

`balanced_score_mean`
: Average balanced score. Lower is better.

`valid_pixel_fraction_mean`
: Average fraction of grid pixels with valid interpolation.

`wet_miss_rate_mean`
: Fraction of truly wet pixels missed by the reconstruction.

`dry_false_rain_rate_mean`
: Fraction of truly dry pixels falsely reconstructed as rain.

A good robust setting usually has:

- low `balanced_score_mean`
- low `rmse_mean`
- low `rmse_std`
- low `wet_miss_rate_mean`
- low `dry_false_rain_rate_mean`
- high `valid_pixel_fraction_mean`

## 11. Rerendering Reports Only

Use rerendering when simulations already exist and you only want to change
contact sheets, ranking metric, or report layout.

Example: rebuild an existing IDW sweep as a 2x6 contact sheet:

```bash
python run_idw_sweep.py \
  --base-config configs/config.ini \
  --out-root poster_outputs/03_idw_sweep_3x4 \
  --rerender-only \
  --sheet-rows 2 \
  --sheet-cols 6
```

Example: rerank existing outputs by balanced score:

```bash
python run_idw_sweep.py \
  --base-config configs/config.ini \
  --out-root outputs_idw_sweep \
  --rerender-only \
  --ranking-metric balanced_score
```

This reads existing `summary.csv` files and run artifacts. It does not rerun
rain generation or interpolation.

## 12. Poster Outputs

To reproduce poster outputs:

```bash
./scripts/generate_poster_outputs.sh
```

or on PowerShell:

```powershell
.\scripts\generate_poster_outputs.ps1
```

The IDW poster section uses:

```bash
--preset poster
```

which creates a 12-panel 2x6 IDW contact sheet.

## 13. Practical Recipes

### Fast Smoke Test

```bash
python run_idw_sweep.py --base-config configs/config.ini --out-root outputs_smoke --preset quick
```

Note: `run_idw_sweep.py` takes the interpolation backend from the base config.
To use `custom`, set `interp_style = custom` in a copied config file.

### Robust IDW Selection

```bash
python run_idw_sweep.py \
  --base-config configs/config.ini \
  --out-root outputs_idw_robust \
  --preset robust \
  --ranking-metric balanced_score
```

Then inspect:

```text
outputs_idw_robust/parameter_robustness.csv
```

### Compare Sparse and Dense Networks

```bash
python run_idw_sweep.py \
  --base-config configs/config.ini \
  --out-root outputs_idw_network_density \
  --powers 1,2,3 \
  --nears 4,8,12 \
  --dists 10000,30000 \
  --n-sites-list 25,50,100 \
  --wet-targets 0.2 \
  --seeds 0,1,2 \
  --ranking-metric balanced_score
```

### Compare Rain Coverage

```bash
python run_idw_sweep.py \
  --base-config configs/config.ini \
  --out-root outputs_idw_wet_coverage \
  --powers 1,2,3 \
  --nears 4,8,12 \
  --dists 10000,30000 \
  --n-sites-list 50 \
  --wet-targets 0.05,0.1,0.2,0.35,0.5 \
  --seeds 0,1,2 \
  --ranking-metric balanced_score
```

## 14. Common Problems

### `pycomlink` Is Not Installed

Use the custom backend by setting this in a copied config:

```ini
[interp]
interp_style = custom
```

or install the optional dependency:

```bash
python -m pip install -e ".[pycomlink]"
```

### Matplotlib Tries to Open a GUI

Linux/macOS:

```bash
export MPLBACKEND=Agg
```

Windows PowerShell:

Set:

```powershell
$env:MPLBACKEND = "Agg"
```

### A Sweep Takes Too Long

Reduce the grid:

```bash
--powers 1,2 --nears 4,8 --dists 10000 --n-sites-list 50 --seeds 0 --wet-targets 0.2
```

or use:

```bash
--preset quick
```

### Contact Sheet Has the Wrong Shape

Use:

```bash
--sheet-rows 2 --sheet-cols 6 --max-per-page 12
```

For existing outputs, combine with:

```bash
--rerender-only
```

## 15. Recommended Workflow

1. Run one single scenario and inspect the plots.
2. Run `--preset quick` to check the sweep pipeline.
3. Run a focused IDW sweep with 2-3 seeds.
4. Inspect `parameter_robustness.csv`.
5. Rerender reports with the preferred ranking metric and contact-sheet layout.
6. Run the larger `robust` preset only after the focused sweep looks sensible.
