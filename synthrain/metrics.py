from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FieldMetrics:
    rmse: float
    mae: float
    bias: float
    pearson_r: float | None
    valid_pixel_fraction: float
    wet_hit_rate: float | None
    wet_miss_rate: float | None
    dry_false_rain_rate: float | None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def compute_field_metrics(
    z_true: np.ndarray, z_est: np.ndarray, min_rain: float
) -> FieldMetrics:
    z_true = np.asarray(z_true, dtype=float)
    z_est = np.asarray(z_est, dtype=float)

    valid = np.isfinite(z_true) & np.isfinite(z_est)
    valid_fraction = float(np.mean(valid))

    if not np.any(valid):
        return FieldMetrics(
            rmse=float("nan"),
            mae=float("nan"),
            bias=float("nan"),
            pearson_r=None,
            valid_pixel_fraction=valid_fraction,
            wet_hit_rate=None,
            wet_miss_rate=None,
            dry_false_rain_rate=None,
        )

    true_v = z_true[valid]
    est_v = z_est[valid]
    err = est_v - true_v

    pearson_r: float | None
    if true_v.size < 2 or np.allclose(np.std(true_v), 0.0) or np.allclose(np.std(est_v), 0.0):
        pearson_r = None
    else:
        pearson_r = float(np.corrcoef(true_v, est_v)[0, 1])

    wet_true = true_v >= float(min_rain)
    wet_est = est_v >= float(min_rain)

    wet_hit_rate = float(np.mean(wet_est[wet_true])) if np.any(wet_true) else None
    wet_miss_rate = float(np.mean(~wet_est[wet_true])) if np.any(wet_true) else None
    dry_false_rain_rate = float(np.mean(wet_est[~wet_true])) if np.any(~wet_true) else None

    return FieldMetrics(
        rmse=float(np.sqrt(np.mean(err**2))),
        mae=float(np.mean(np.abs(err))),
        bias=float(np.mean(err)),
        pearson_r=pearson_r,
        valid_pixel_fraction=valid_fraction,
        wet_hit_rate=wet_hit_rate,
        wet_miss_rate=wet_miss_rate,
        dry_false_rain_rate=dry_false_rain_rate,
    )
