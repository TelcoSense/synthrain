"""Synthetic CML network generation.

Sites are generated in lon/lat degrees (bbox). Links are created via k-nearest
neighbours in Web Mercator meters (so network density behaves sensibly), then
filtered by great-circle length in km.

Outputs a link table with endpoints, centers (lon/lat), and length (km).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import lonlat_to_mercator_m, mercator_m_to_lonlat, haversine_km


@dataclass
class NetworkSpec:
    n_sites: int = 200
    bbox: Tuple[float, float, float, float] = (
        12.0,
        19.0,
        48.5,
        51.2,
    )  # (lon_min,lon_max,lat_min,lat_max)
    mean_degree: int = 3  # approx. neighbours per node
    min_length_km: float = 0.5
    max_length_km: float = 30.0
    seed: int = 0
    site_sampling: str = "uniform"  # "uniform" | "poisson"
    site_min_dist_m: float = 0  # used for poisson; 0 => auto


def _poisson_disk_rect(
    width: float, height: float, r: float, rng: np.random.Generator, k: int = 30
) -> np.ndarray:
    """
    Bridson Poisson-disk sampling in a rectangle [0,width]x[0,height] with min distance r.
    Returns array of shape (N,2).
    """
    if r <= 0:
        raise ValueError("r must be > 0")

    cell = r / np.sqrt(2.0)
    gw = int(np.ceil(width / cell))
    gh = int(np.ceil(height / cell))
    grid = -np.ones((gh, gw), dtype=int)

    def grid_coords(p):
        return int(p[1] // cell), int(p[0] // cell)  # (gy,gx)

    def fits(p, pts):
        gy, gx = grid_coords(p)
        y0 = max(gy - 2, 0)
        y1 = min(gy + 3, gh)
        x0 = max(gx - 2, 0)
        x1 = min(gx + 3, gw)
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                j = grid[yy, xx]
                if j >= 0:
                    q = pts[j]
                    if np.hypot(p[0] - q[0], p[1] - q[1]) < r:
                        return False
        return True

    # initial point
    p0 = np.array([rng.uniform(0, width), rng.uniform(0, height)], dtype=float)
    pts = [p0]
    active = [0]
    gy, gx = grid_coords(p0)
    grid[gy, gx] = 0

    while active:
        idx = active[rng.integers(0, len(active))]
        base = pts[idx]
        found = False
        for _ in range(k):
            ang = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(r, 2 * r)
            cand = base + np.array([np.cos(ang) * rad, np.sin(ang) * rad])
            if not (0 <= cand[0] <= width and 0 <= cand[1] <= height):
                continue
            if fits(cand, pts):
                pts.append(cand)
                active.append(len(pts) - 1)
                gy, gx = grid_coords(cand)
                grid[gy, gx] = len(pts) - 1
                found = True
                break
        if not found:
            active.remove(idx)

    return np.asarray(pts, dtype=float)


def generate_sites(spec: NetworkSpec) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed)
    lon_min, lon_max, lat_min, lat_max = spec.bbox

    if spec.site_sampling.lower() != "poisson":
        xs = rng.uniform(lon_min, lon_max, size=spec.n_sites)
        ys = rng.uniform(lat_min, lat_max, size=spec.n_sites)
        return pd.DataFrame(
            {"site_id": np.arange(spec.n_sites, dtype=int), "lon": xs, "lat": ys}
        )

    # poisson-disk in Web Mercator meters
    x0, y0 = lonlat_to_mercator_m(np.array([lon_min]), np.array([lat_min]))
    x1, y1 = lonlat_to_mercator_m(np.array([lon_max]), np.array([lat_max]))
    xmin, xmax = float(min(x0[0], x1[0])), float(max(x0[0], x1[0]))
    ymin, ymax = float(min(y0[0], y1[0])), float(max(y0[0], y1[0]))
    width = xmax - xmin
    height = ymax - ymin

    # auto spacing if not provided
    r = float(spec.site_min_dist_m)
    if r <= 0:
        area = width * height
        r = 0.75 * np.sqrt(area / spec.n_sites)

    # try to get at least n_sites; if not enough, relax r progressively
    r_try = r
    pts = np.empty((0, 2), float)
    for _ in range(8):
        pts = _poisson_disk_rect(width, height, r_try, rng)
        if len(pts) >= spec.n_sites:
            break
        r_try *= 0.85  # relax

    if len(pts) < spec.n_sites:
        raise RuntimeError(
            f"Poisson sampling produced only {len(pts)} points < n_sites={spec.n_sites}. "
            f"Decrease site_min_dist_m or decrease n_sites."
        )

    # do not take the first N (biased, can create nests of points) but randomly pick N points
    sel = rng.choice(len(pts), size=spec.n_sites, replace=False)
    pts = pts[sel]

    x = xmin + pts[:, 0]
    y = ymin + pts[:, 1]
    lon, lat = mercator_m_to_lonlat(x, y)

    return pd.DataFrame(
        {"site_id": np.arange(len(lon), dtype=int), "lon": lon, "lat": lat}
    )


def _unique_edges(edges):
    """Normalize (a,b) with a<b and unique."""
    s = set()
    out = []
    for a, b in edges:
        if a == b:
            continue
        aa, bb = (a, b) if a < b else (b, a)
        if (aa, bb) not in s:
            s.add((aa, bb))
            out.append((aa, bb))
    return out


def generate_links(sites: pd.DataFrame, spec: NetworkSpec) -> pd.DataFrame:
    lon = sites["lon"].to_numpy(float)
    lat = sites["lat"].to_numpy(float)
    x_m, y_m = lonlat_to_mercator_m(lon, lat)
    xy_m = np.column_stack([x_m, y_m])
    tree = cKDTree(xy_m)

    # Choose k so mean_degree ~ k (undirected will be ~k but depends on pruning)
    k = max(2, int(spec.mean_degree) + 1)  # include self => +1
    _, idxs = tree.query(xy_m, k=k, workers=-1)

    edges = []
    for i in range(xy_m.shape[0]):
        for j in idxs[i, 1:]:  # skip self
            j = int(j)
            # Filter by km length
            d_km = float(haversine_km(lon[i], lat[i], lon[j], lat[j]))
            if d_km < float(spec.min_length_km) or d_km > float(spec.max_length_km):
                continue
            edges.append((i, int(j)))

    edges = _unique_edges(edges)
    if len(edges) == 0:
        raise RuntimeError(
            "No edges created. Try increasing max_length or mean_degree."
        )

    # Build link table
    a = np.array([e[0] for e in edges], dtype=int)
    b = np.array([e[1] for e in edges], dtype=int)
    lon_a, lat_a = lon[a], lat[a]
    lon_b, lat_b = lon[b], lat[b]
    length_km = haversine_km(lon_a, lat_a, lon_b, lat_b)
    lon_center = (lon_a + lon_b) / 2.0
    lat_center = (lat_a + lat_b) / 2.0

    df = pd.DataFrame(
        {
            "cml_id": np.arange(1, len(edges) + 1, dtype=int),
            "site_a_id": a,
            "site_b_id": b,
            "site_a_lon": lon_a,
            "site_a_lat": lat_a,
            "site_b_lon": lon_b,
            "site_b_lat": lat_b,
            "lon_center": lon_center,
            "lat_center": lat_center,
            "length": length_km,
        }
    )

    return df


def generate_network(
    spec: NetworkSpec, debug: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sites = generate_sites(spec)
    links = generate_links(sites, spec)
    if debug:
        print("---------------- sites info ----------------")
        print("n_sites returned:", len(sites))
        print("lon range:", sites.lon.min(), sites.lon.max())
        print("lat range:", sites.lat.min(), sites.lat.max())
        print("bbox:", spec.bbox)
        print("--------------------------------------------")

        # min pairwise distance in meters (should be >= site_min_dist_m-ish)
        from scipy.spatial import cKDTree

        x, y = lonlat_to_mercator_m(sites.lon.to_numpy(), sites.lat.to_numpy())
        tree = cKDTree(np.c_[x, y])
        d, _ = tree.query(
            np.c_[x, y], k=2
        )  # first neighbor is itself, second is nearest other
        print("min nn dist [m]:", float(np.min(d[:, 1])))

    return sites, links
