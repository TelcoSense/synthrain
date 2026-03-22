"""
Compatibility helpers around the typed config model.

New code should prefer ``synthrain.config`` directly.
"""

from __future__ import annotations

from typing import Any

from synthrain.config import load_scenario_config


def load_ini_defaults(path: str) -> dict[str, Any]:
    cfg = load_scenario_config(path)
    return {
        "out": cfg.io.out,
        "seed": cfg.io.seed,
        "debug": cfg.io.debug,
        "n_sites": cfg.network.n_sites,
        "mean_degree": cfg.network.mean_degree,
        "bbox": cfg.network.bbox,
        "city": cfg.network.city,
        "city_bbox": cfg.network.city_bbox,
        "noncity_bbox": cfg.network.noncity_bbox,
        "min_length_km": cfg.network.min_length_km,
        "max_length_km": cfg.network.max_length_km,
        "site_sampling": cfg.network.site_sampling,
        "site_min_dist_m": cfg.network.site_min_dist_m,
        "interp_style": cfg.interp.interp_style,
        "grid_step_m": cfg.interp.grid_step_m,
        "grid_nx": cfg.interp.grid_nx,
        "grid_ny": cfg.interp.grid_ny,
        "idw_power": cfg.interp.idw_power,
        "idw_near": cfg.interp.idw_near,
        "idw_dist_m": cfg.interp.idw_dist_m,
        "dry_as_zero": cfg.interp.dry_as_zero,
        "n_blobs": cfg.rain.n_blobs,
        "blob_sigma_m": cfg.rain.blob_sigma_m,
        "peak_mmph": cfg.rain.peak_mmph,
        "min_rain": cfg.rain.min_rain,
        "wet_mode": cfg.wet.wet_mode,
        "wet_target": cfg.wet.wet_target,
        "wet_min_mmph": cfg.wet.wet_min_mmph,
        "flip_dry_to_wet": cfg.wet.flip_dry_to_wet,
        "flip_wet_to_dry": cfg.wet.flip_wet_to_dry,
        "wet_strata_nx": cfg.wet.wet_strata_nx,
        "wet_strata_ny": cfg.wet.wet_strata_ny,
        "noise_mmph": cfg.observation.noise_mmph,
        "export_csv": cfg.csv.export_csv,
        "csv_steps": cfg.csv.csv_steps,
        "csv_step_min": cfg.csv.csv_step_min,
        "csv_start": cfg.csv.csv_start,
        "title_name": cfg.plot.title_name,
        "show_titles": cfg.plot.show_titles,
        "font_scale": cfg.plot.font_scale,
        "wet_targets": ",".join(str(x) for x in cfg.sweep.wet_targets),
    }
