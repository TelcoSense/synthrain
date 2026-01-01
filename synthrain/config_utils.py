"""
INI config support.

Goal: allow running scenarios without dozens of CLI flags, similarly to TelcoRain's config.ini.

- Provide --config path/to/config.ini
- Values in CLI override values from config.
"""

from __future__ import annotations

import configparser
from typing import Any, Dict, Tuple


def _parse_bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean: {v}")


def _parse_bbox(v: str) -> Tuple[float, float, float, float]:
    parts = [float(p.strip()) for p in str(v).split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'lon_min,lon_max,lat_min,lat_max'")
    return (parts[0], parts[1], parts[2], parts[3])


def load_ini_defaults(path: str) -> Dict[str, Any]:
    """
    Read INI file and return argparse defaults dict.

    Supported sections/keys:
    [io] out, debug, seed
    [network] n_sites, mean_degree, bbox, city, min_length_km, max_length_km
    [interp] interp_style, grid_step_m, grid_nx, grid_ny, idw_power, idw_near, idw_dist_m, dry_as_zero
    [rain] n_blobs, blob_sigma_m, peak_mmph, noise_mmph, min_rain
    [wet] wet_mode, wet_target, wet_min_mmph, flip_dry_to_wet, flip_wet_to_dry, wet_strata_nx, wet_strata_ny
    [csv] export_csv, csv_steps, csv_step_min, csv_start
    [sweep] wet_targets (comma-separated list, e.g. 0.05,0.1,0.2)
    [plot] title_name (of the IDW plot)
    """
    cp = configparser.ConfigParser()
    read = cp.read(path)
    if not read:
        raise FileNotFoundError(path)

    d: Dict[str, Any] = {}

    def _has(sec: str, key: str) -> bool:
        return sec in cp and key in cp[sec]

    def _get(sec: str, key: str, cast, dest: str):
        if _has(sec, key):
            d[dest] = cast(cp[sec][key])

    _get("io", "out", str, "out")
    _get("io", "seed", int, "seed")
    _get("io", "debug", _parse_bool, "debug")

    _get("network", "n_sites", int, "n_sites")
    _get("network", "mean_degree", int, "mean_degree")
    if _has("network", "bbox"):
        d["bbox"] = _parse_bbox(cp["network"]["bbox"])
    _get("network", "city", _parse_bool, "city")
    _get("network", "min_length_km", float, "min_length_km")
    _get("network", "max_length_km", float, "max_length_km")
    _get("network", "site_sampling", str, "site_sampling")
    _get("network", "site_min_dist_m", float, "site_min_dist_m")

    _get("interp", "interp_style", str, "interp_style")
    _get("interp", "grid_step_m", float, "grid_step_m")
    _get("interp", "grid_nx", int, "grid_nx")
    _get("interp", "grid_ny", int, "grid_ny")
    _get("interp", "idw_power", float, "idw_power")
    _get("interp", "idw_near", int, "idw_near")
    _get("interp", "idw_dist_m", float, "idw_dist_m")
    _get("interp", "dry_as_zero", _parse_bool, "dry_as_zero")

    _get("rain", "n_blobs", int, "n_blobs")
    _get("rain", "blob_sigma_m", float, "blob_sigma_m")
    _get("rain", "peak_mmph", float, "peak_mmph")
    _get("rain", "noise_mmph", float, "noise_mmph")
    _get("rain", "min_rain", float, "min_rain")

    _get("wet", "wet_mode", str, "wet_mode")
    _get("wet", "wet_target", float, "wet_target")
    _get("wet", "wet_min_mmph", float, "wet_min_mmph")
    _get("wet", "flip_dry_to_wet", float, "flip_dry_to_wet")
    _get("wet", "flip_wet_to_dry", float, "flip_wet_to_dry")
    _get("wet", "wet_strata_nx", int, "wet_strata_nx")
    _get("wet", "wet_strata_ny", int, "wet_strata_ny")

    _get("csv", "export_csv", _parse_bool, "export_csv")
    _get("csv", "csv_steps", int, "csv_steps")
    _get("csv", "csv_step_min", int, "csv_step_min")
    _get("csv", "csv_start", str, "csv_start")

    _get("plot", "title_name", str, "title_name")

    if _has("sweep", "wet_targets"):
        d["wet_targets"] = cp["sweep"]["wet_targets"]

    return d
