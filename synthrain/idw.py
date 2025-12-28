"""
IDW interpolation with a KDTree (SciPy).

Matches TelcoRain-style parameters:
- nnear: number of neighbors
- p: inverse distance power
- max_distance: maximum neighbor radius (same coordinate units as x/y)
- exclude_nan: ignore NaNs in input values

This module is intentionally small and dependency-light.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class IdwKdtree:
    nnear: int = 8
    p: float = 2.0
    max_distance: Optional[float] = None
    exclude_nan: bool = True
    eps: float = 1e-12  # avoid division by zero

    def fit(self, xy: np.ndarray, values: np.ndarray) -> "IdwKdtree":
        """
        xy: (N,2) array
        values: (N,) array
        """
        xy = np.asarray(xy, dtype=float)
        values = np.asarray(values, dtype=float)

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must be (N,2)")
        if values.ndim != 1 or values.shape[0] != xy.shape[0]:
            raise ValueError("values must be (N,) matching xy")

        if self.exclude_nan:
            mask = ~np.isnan(values)
            self._xy = xy[mask]
            self._values = values[mask]
        else:
            self._xy = xy
            self._values = values

        if self._xy.shape[0] == 0:
            raise ValueError("No valid points after NaN filtering.")

        self._tree = cKDTree(self._xy)
        return self

    def predict_grid(self, xg: np.ndarray, yg: np.ndarray) -> np.ndarray:
        """
        xg, yg are 2D meshgrid arrays.
        Returns z with same shape.
        """
        pts = np.column_stack([xg.ravel(), yg.ravel()])
        k = min(self.nnear, self._xy.shape[0])

        dists, idxs = self._tree.query(pts, k=k, workers=-1)

        # Ensure 2D
        if k == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]

        # Apply max_distance cutoff
        if self.max_distance is not None:
            bad = dists > float(self.max_distance)
        else:
            bad = np.zeros_like(dists, dtype=bool)

        # If any point lands exactly on a station (dist=0), return that station's value
        zero = dists <= self.eps
        out = np.full((pts.shape[0],), np.nan, dtype=float)

        any_zero = np.any(zero, axis=1)
        if np.any(any_zero):
            z_idx = np.argmax(zero[any_zero], axis=1)
            out[any_zero] = self._values[idxs[any_zero, z_idx]]

        # For remaining points do weighted average
        rem = ~any_zero
        if np.any(rem):
            d = dists[rem].copy()
            ii = idxs[rem].copy()
            b = bad[rem]

            # If all neighbors are beyond max_distance -> remain NaN
            all_bad = np.all(b, axis=1)
            ok = ~all_bad
            if np.any(ok):
                d = d[ok]
                ii = ii[ok]
                b = b[ok]

                w = 1.0 / np.maximum(d, self.eps) ** float(self.p)
                w[b] = 0.0
                wsum = np.sum(w, axis=1)
                # avoid /0
                wsum = np.where(wsum <= 0.0, np.nan, wsum)
                z = np.sum(w * self._values[ii], axis=1) / wsum

                # place back
                tmp = np.full((rem.sum(),), np.nan, dtype=float)
                tmp_idx = np.flatnonzero(ok)
                tmp[tmp_idx] = z
                out[rem] = tmp

        return out.reshape(xg.shape)
