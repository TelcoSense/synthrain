"""Geospatial helpers.

This synthetic testbed mimics TelcoRain's interpolation setup:

- Synthetic links are generated in lon/lat degrees (bbox).
- Interpolation (IDW) can be done in Web Mercator meters (recommended),
  so ``grid_step_m`` and ``idw_dist_m`` behave like in TelcoRain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


EARTH_RADIUS_M = 6378137.0  # sphere radius used by Web Mercator (EPSG:3857)


def lonlat_to_mercator_m(lon_deg: np.ndarray, lat_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward Web Mercator projection (EPSG:3857), in meters."""
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    # Clamp latitude to avoid inf near poles
    lat = np.clip(lat, np.deg2rad(-85.05112878), np.deg2rad(85.05112878))
    x = EARTH_RADIUS_M * lon
    y = EARTH_RADIUS_M * np.log(np.tan(np.pi / 4.0 + lat / 2.0))
    return x, y


def mercator_m_to_lonlat(x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse Web Mercator projection (EPSG:3857)."""
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    lon = np.rad2deg(x / EARTH_RADIUS_M)
    lat = np.rad2deg(2.0 * np.arctan(np.exp(y / EARTH_RADIUS_M)) - np.pi / 2.0)
    return lon, lat


def haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in km (vectorized)."""
    lon1 = np.deg2rad(np.asarray(lon1, dtype=float))
    lat1 = np.deg2rad(np.asarray(lat1, dtype=float))
    lon2 = np.deg2rad(np.asarray(lon2, dtype=float))
    lat2 = np.deg2rad(np.asarray(lat2, dtype=float))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return (EARTH_RADIUS_M / 1000.0) * c


@dataclass
class GridSpec:
    """Regular interpolation grid spec similar to TelcoRain."""

    bbox_lonlat: Tuple[float, float, float, float]
    grid_step_m: float = 1000.0
    use_mercator: bool = True
    # Optional override; if set, these take precedence over step-based sizing
    grid_nx: int | None = None
    grid_ny: int | None = None


def make_grid(spec: GridSpec) -> dict:
    """Create an interpolation grid.

    Returns a dict with:
      - xg_m, yg_m: 2D arrays in meters (used for IDW distances)
      - lon_g, lat_g: 2D arrays in degrees (for plotting/labeling)
      - bbox_m: (xmin,xmax,ymin,ymax) in meters
    """
    lon_min, lon_max, lat_min, lat_max = spec.bbox_lonlat

    if not spec.use_mercator:
        # Treat degrees as "units" (not recommended for distance-based IDW)
        xs = np.linspace(lon_min, lon_max, spec.grid_nx or 300)
        ys = np.linspace(lat_min, lat_max, spec.grid_ny or 200)
        lon_g, lat_g = np.meshgrid(xs, ys)
        xg_m = lon_g
        yg_m = lat_g
        bbox_m = (float(lon_min), float(lon_max), float(lat_min), float(lat_max))
        return {"xg_m": xg_m, "yg_m": yg_m, "lon_g": lon_g, "lat_g": lat_g, "bbox_m": bbox_m}

    # Project bbox corners to meters
    x0, y0 = lonlat_to_mercator_m(lon_min, lat_min)
    x1, y1 = lonlat_to_mercator_m(lon_max, lat_max)
    xmin, xmax = (float(min(x0, x1)), float(max(x0, x1)))
    ymin, ymax = (float(min(y0, y1)), float(max(y0, y1)))

    if spec.grid_nx is not None and spec.grid_ny is not None:
        xs = np.linspace(xmin, xmax, int(spec.grid_nx))
        ys = np.linspace(ymin, ymax, int(spec.grid_ny))
    else:
        step = float(spec.grid_step_m)
        nx = int(np.floor((xmax - xmin) / step)) + 1
        ny = int(np.floor((ymax - ymin) / step)) + 1
        xs = xmin + step * np.arange(nx, dtype=float)
        ys = ymin + step * np.arange(ny, dtype=float)

    xg_m, yg_m = np.meshgrid(xs, ys)
    lon_g, lat_g = mercator_m_to_lonlat(xg_m, yg_m)
    bbox_m = (xmin, xmax, ymin, ymax)
    return {"xg_m": xg_m, "yg_m": yg_m, "lon_g": lon_g, "lat_g": lat_g, "bbox_m": bbox_m}
