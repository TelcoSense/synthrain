# SynthRain

Synthetic CML network + synthetic wet/dry + synthetic rainfall intensity -> IDW interpolation -> PNGs.

This is a small standalone testbed to stress-test TelcoRain-like interpolation behaviour without InfluxDB/MariaDB.

## Install
```bash
pip install numpy pandas scipy matplotlib
```

## Quick start
```bash
python run_scenarios.py --out out_demo --seed 0 --city --n-sites 250 --mean-degree 3 \
  --wet-target 0.35 --wet-mode threshold --flip-dry-to-wet 0.03 --flip-wet-to-dry 0.15 \
  --grid-step-m 1000 --idw-dist-m 30000 \
  --idw-near 12 --idw-power 2.0 --dry-as-zero
```

Outputs (in `out_demo/`):
- `true_field.png` : underlying synthetic “ground truth” rainfall field
- `links.png`      : link centers with wet/dry markers
- `idw_field.png`  : interpolated rainfall field from links
- `diff.png`       : interpolated minus true (useful for bias/artefacts)

Use `--dry-as-zero` to force dry links as 0 mm/h (strongly affects IDW output).
Omit `--dry-as-zero` to *ignore* dry links (IDW uses only wet links).

## Notes
- Bbox is `lon_min,lon_max,lat_min,lat_max` (degrees). Default is a rough Czechia bbox.
- Grid is built in meters using Web Mercator (like TelcoRain with `use_mercator=True`).
  Use `--grid-step-m` (e.g. 1000 for ~1 km) and `--idw-dist-m` (e.g. 30000 for 30 km).
  If you really want to do everything in degrees, pass `--no-mercator` (not recommended).
- Wet/dry labeling:
  - `--wet-mode threshold`: picks a threshold so ~`wet_target` links are wet (can create contiguous wet/dry regions).
  - `--wet-mode random`: wet labels are i.i.d. random.
  - `--wet-mode stratified`: random *within coarse spatial strata* (`--wet-strata-nx/ny`) to spread wet links evenly.


## Using an INI config (TelcoRain-like)

Instead of passing many CLI flags, you can use:

```bash
python run_scenarios.py --config example_config.ini
```

CLI flags override config values, e.g.:

```bash
python run_scenarios.py --config example_config.ini --wet-target 0.15 --seed 3
```

### Why both `idw_near` and `idw_dist_m`?

This matches TelcoRain’s behaviour:

- `idw_dist_m` sets a **maximum radius** (meters) for eligible neighbours.
- `idw_near` caps the **maximum number of neighbours** actually used per grid point (after filtering by radius).

In dense networks, without `idw_near`, a 30 km radius could include hundreds/thousands of links per pixel.

## Wet/dry distribution modes

- `wet_mode=threshold`: wet links are chosen by rainfall intensity quantile (can create coherent wet regions).
- `wet_mode=random`: wet links are chosen uniformly at random (spatially “salt-and-pepper”).
- `wet_mode=stratified`: random wet selection within coarse spatial bins (more spatially balanced than pure random).

