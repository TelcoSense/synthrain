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


@dataclass
class ObservationSpec:
    noise_mmph: float = 0.5
    link_path_samples: int = 9
    outage_fraction: float = 0.0
    stuck_zero_fraction: float = 0.0
    bias_fraction: float = 0.0
    bias_low: float = 0.85
    bias_high: float = 1.15
    extra_noise_fraction: float = 0.0
    extra_noise_mmph: float = 2.0
    clustered_outage_fraction: float = 0.0
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


def sample_field_along_links(
    xg: np.ndarray,
    yg: np.ndarray,
    z: np.ndarray,
    x_a: np.ndarray,
    y_a: np.ndarray,
    x_b: np.ndarray,
    y_b: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    n = max(1, int(n_samples))
    if n == 1:
        return sample_field_at_points(
            xg, yg, z, (x_a + x_b) / 2.0, (y_a + y_b) / 2.0
        )

    t = np.linspace(0.0, 1.0, n, dtype=float)
    xs = x_a[:, None] + (x_b - x_a)[:, None] * t[None, :]
    ys = y_a[:, None] + (y_b - y_a)[:, None] * t[None, :]
    samples = sample_field_at_points(xg, yg, z, xs.ravel(), ys.ravel()).reshape(
        len(x_a), n
    )
    return np.mean(samples, axis=1)


def _select_fraction(
    rng: np.random.Generator,
    candidates: np.ndarray,
    fraction: float,
) -> np.ndarray:
    idx = np.flatnonzero(candidates)
    out = np.zeros(candidates.shape[0], dtype=bool)
    if idx.size == 0 or fraction <= 0:
        return out
    count = min(idx.size, int(round(fraction * candidates.shape[0])))
    if count <= 0:
        return out
    chosen = rng.choice(idx, size=count, replace=False)
    out[chosen] = True
    return out


def _select_clustered_fraction(
    rng: np.random.Generator,
    x: np.ndarray,
    y: np.ndarray,
    candidates: np.ndarray,
    fraction: float,
) -> np.ndarray:
    idx = np.flatnonzero(candidates)
    out = np.zeros(candidates.shape[0], dtype=bool)
    if idx.size == 0 or fraction <= 0:
        return out
    count = min(idx.size, int(round(fraction * candidates.shape[0])))
    if count <= 0:
        return out
    center_idx = int(rng.choice(idx))
    d = np.hypot(x[idx] - x[center_idx], y[idx] - y[center_idx])
    chosen = idx[np.argsort(d)[:count]]
    out[chosen] = True
    return out


def _assign_faults(
    x: np.ndarray, y: np.ndarray, spec: ObservationSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(spec.seed)
    n = x.size

    available = np.ones(n, dtype=bool)
    stuck_zero = np.zeros(n, dtype=bool)
    bias_factor = np.ones(n, dtype=float)
    noise_sigma = np.full(n, float(spec.noise_mmph), dtype=float)
    labels = np.full(n, "ok", dtype=object)

    remaining = np.ones(n, dtype=bool)
    cluster_outage = _select_clustered_fraction(
        rng, x, y, remaining, float(spec.clustered_outage_fraction)
    )
    if np.any(cluster_outage):
        available[cluster_outage] = False
        labels[cluster_outage] = "cluster_outage"
        remaining &= ~cluster_outage

    outage = _select_fraction(rng, remaining, float(spec.outage_fraction))
    if np.any(outage):
        available[outage] = False
        labels[outage] = "outage"
        remaining &= ~outage

    stuck = _select_fraction(rng, remaining, float(spec.stuck_zero_fraction))
    if np.any(stuck):
        stuck_zero[stuck] = True
        labels[stuck] = "stuck_zero"
        remaining &= ~stuck

    biased = _select_fraction(rng, remaining, float(spec.bias_fraction))
    if np.any(biased):
        lo = min(float(spec.bias_low), float(spec.bias_high))
        hi = max(float(spec.bias_low), float(spec.bias_high))
        bias_factor[biased] = rng.uniform(lo, hi, size=int(np.sum(biased)))
        labels[biased] = "biased"

    noisy = _select_fraction(rng, remaining, float(spec.extra_noise_fraction))
    if np.any(noisy):
        noise_sigma[noisy] += float(spec.extra_noise_mmph)
        labels[noisy] = "noisy"
        overlap = noisy & biased
        labels[overlap] = "biased_noisy"

    return available, stuck_zero, bias_factor, noise_sigma, labels


def build_link_observations(
    links: pd.DataFrame,
    xg: np.ndarray,
    yg: np.ndarray,
    z_true: np.ndarray,
    wet_spec: WetDrySpec,
    observation_spec: ObservationSpec,
) -> pd.DataFrame:
    """
    Create per-link observed rainfall R_obs and wet flag, plus store R_true.
    """
    rng = np.random.default_rng(observation_spec.seed + 12345)

    # Expect link centers in same coordinate system as xg/yg (typically meters)
    x = links["x_center"].to_numpy(float)
    y = links["y_center"].to_numpy(float)
    if {"x_a", "y_a", "x_b", "y_b"}.issubset(links.columns):
        r_true = sample_field_along_links(
            xg,
            yg,
            z_true,
            links["x_a"].to_numpy(float),
            links["y_a"].to_numpy(float),
            links["x_b"].to_numpy(float),
            links["y_b"].to_numpy(float),
            observation_spec.link_path_samples,
        )
    else:
        r_true = sample_field_at_points(xg, yg, z_true, x, y)

    wet = make_wet_flags(r_true, wet_spec, x=x, y=y)
    available, stuck_zero, bias_factor, noise_sigma, labels = _assign_faults(
        x, y, observation_spec
    )

    r_obs = np.maximum(
        0.0,
        r_true * bias_factor + rng.normal(0.0, noise_sigma, size=r_true.size),
    )
    r_obs = np.where(wet, r_obs, 0.0)
    r_obs = np.where(stuck_zero, 0.0, r_obs)
    r_obs = np.where(available, r_obs, np.nan)

    out = links.copy()
    out["R_true"] = r_true
    out["R_obs"] = r_obs
    out["wet"] = wet
    out["available"] = available
    out["stuck_zero"] = stuck_zero
    out["bias_factor"] = bias_factor
    out["obs_noise_mmph"] = noise_sigma
    out["fault_label"] = labels
    return out
