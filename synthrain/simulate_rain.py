"""
Synthetic rainfall field + wet/dry simulation.

We create a smooth "true" rainfall field on a grid (sum of Gaussians),
sample it at link centers, and then create wet flags with controllable
target wet fraction and misclassification rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import pandas as pd


@dataclass
class RainFieldSpec:
    # Parameters for blob field on an existing grid.
    n_blobs: int = 4
    blob_sigma: float = 8000.0
    peak_mmph: float = 15.0
    background_mmph: float = 0.0
    seed: int = 0


def make_true_field_on_grid(
    xg: np.ndarray, yg: np.ndarray, spec: RainFieldSpec
) -> np.ndarray:
    """Create a smooth synthetic rainfall field (mm/h) on an existing grid."""
    rng = np.random.default_rng(spec.seed)
    x_min, x_max = float(np.nanmin(xg)), float(np.nanmax(xg))
    y_min, y_max = float(np.nanmin(yg)), float(np.nanmax(yg))

    z = np.full_like(xg, float(spec.background_mmph), dtype=float)
    for _ in range(int(spec.n_blobs)):
        cx = rng.uniform(x_min, x_max)
        cy = rng.uniform(y_min, y_max)
        amp = rng.uniform(0.3, 1.0) * float(spec.peak_mmph)
        sig = float(spec.blob_sigma) * rng.uniform(0.7, 1.3)
        z += amp * np.exp(-(((xg - cx) ** 2 + (yg - cy) ** 2) / (2 * sig**2)))
    return z


def sample_field_at_points(
    xg: np.ndarray, yg: np.ndarray, z: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Bilinear sampling of z at arbitrary (x,y).
    Assumes xg/yg are regular monotonic grids produced by meshgrid.
    """
    xs = xg[0, :]
    ys = yg[:, 0]

    xi = np.interp(x, xs, np.arange(xs.size))
    yi = np.interp(y, ys, np.arange(ys.size))

    x0 = np.floor(xi).astype(int)
    y0 = np.floor(yi).astype(int)
    x1 = np.clip(x0 + 1, 0, xs.size - 1)
    y1 = np.clip(y0 + 1, 0, ys.size - 1)
    x0 = np.clip(x0, 0, xs.size - 1)
    y0 = np.clip(y0, 0, ys.size - 1)

    fx = xi - x0
    fy = yi - y0

    z00 = z[y0, x0]
    z10 = z[y0, x1]
    z01 = z[y1, x0]
    z11 = z[y1, x1]

    z0 = z00 * (1 - fx) + z10 * fx
    z1 = z01 * (1 - fx) + z11 * fx
    return z0 * (1 - fy) + z1 * fy


WetMode = Literal["threshold", "random", "stratified"]


@dataclass
class WetDrySpec:
    wet_mode: WetMode = "random"
    wet_target: float = 0.3  # desired fraction of wet links
    flip_dry_to_wet: float = 0.0  # false positive
    flip_wet_to_dry: float = 0.0  # false negative
    wet_min_mmph: float = 0.1  # base threshold (used as fallback)
    # For wet_mode="stratified": split domain into a coarse grid and sample wet
    # links independently in each cell to avoid large contiguous wet/dry regions.
    strata_nx: int = 6
    strata_ny: int = 6
    seed: int = 0


def make_wet_flags(
    r_mmph: np.ndarray,
    spec: WetDrySpec,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create wet/dry flags from per-link "true" intensity r_mmph.
    If wet_mode="stratified", x/y (same coords as r_mmph locations) are required.
    """
    rng = np.random.default_rng(spec.seed)
    r = np.asarray(r_mmph, dtype=float)

    if spec.wet_mode == "random":
        wet = rng.random(r.size) < float(spec.wet_target)

    elif spec.wet_mode == "stratified":
        if x is None or y is None:
            raise ValueError("wet_mode='stratified' requires x and y coordinates")
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        nx = max(1, int(spec.strata_nx))
        ny = max(1, int(spec.strata_ny))

        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))

        # avoid degenerate bins
        if x_max == x_min:
            x_max = x_min + 1.0
        if y_max == y_min:
            y_max = y_min + 1.0

        # map each point to a stratum cell
        ix = np.clip(((x - x_min) / (x_max - x_min) * nx).astype(int), 0, nx - 1)
        iy = np.clip(((y - y_min) / (y_max - y_min) * ny).astype(int), 0, ny - 1)

        wet = np.zeros(r.size, dtype=bool)
        for cx in range(nx):
            for cy in range(ny):
                mask = (ix == cx) & (iy == cy)
                if not np.any(mask):
                    continue
                wet[mask] = rng.random(mask.sum()) < float(spec.wet_target)

    else:
        # "threshold": choose threshold so that fraction wet ~= wet_target
        q = 1.0 - float(spec.wet_target)
        q = np.clip(q, 0.0, 1.0)
        thr = np.quantile(r, q)
        thr = max(float(spec.wet_min_mmph), float(thr))
        wet = r >= thr

    wet = wet.astype(bool)

    # Apply misclassification
    if spec.flip_dry_to_wet > 0:
        flip = (~wet) & (rng.random(r.size) < float(spec.flip_dry_to_wet))
        wet[flip] = True
    if spec.flip_wet_to_dry > 0:
        flip = wet & (rng.random(r.size) < float(spec.flip_wet_to_dry))
        wet[flip] = False

    return wet


def build_link_observations(
    links: pd.DataFrame,
    xg: np.ndarray,
    yg: np.ndarray,
    z_true: np.ndarray,
    wet_spec: WetDrySpec,
    noise_mmph: float = 0.5,
) -> pd.DataFrame:
    """
    Create per-link observed rainfall R_obs and wet flag, plus store R_true.
    """
    rng = np.random.default_rng(wet_spec.seed + 12345)

    # Expect link centers in same coordinate system as xg/yg (typically meters)
    x = links["x_center"].to_numpy(float)
    y = links["y_center"].to_numpy(float)
    r_true = sample_field_at_points(xg, yg, z_true, x, y)

    wet = make_wet_flags(r_true, wet_spec, x=x, y=y)

    r_obs = np.maximum(
        0.0, r_true + rng.normal(0.0, float(noise_mmph), size=r_true.size)
    )
    r_obs = np.where(wet, r_obs, 0.0)

    out = links.copy()
    out["R_true"] = r_true
    out["R_obs"] = r_obs
    out["wet"] = wet
    return out
