from __future__ import annotations

import numpy as np

from synthrain.metrics import compute_field_metrics


def test_metrics_basic_values():
    z_true = np.array([[0.0, 1.0], [2.0, 3.0]])
    z_est = np.array([[0.0, 2.0], [1.0, 3.0]])

    metrics = compute_field_metrics(z_true, z_est, min_rain=0.5)

    assert metrics.rmse > 0
    assert metrics.mae > 0
    assert metrics.valid_pixel_fraction == 1.0
    assert metrics.wet_hit_rate == 1.0
    assert metrics.dry_false_rain_rate == 0.0
