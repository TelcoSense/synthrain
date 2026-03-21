from __future__ import annotations

import numpy as np
import pandas as pd

from synthrain.simulate_rain import (
    ObservationSpec,
    WetDrySpec,
    build_link_observations,
)


def test_build_link_observations_adds_fault_metadata():
    x = np.linspace(0.0, 10.0, 6)
    y = np.linspace(0.0, 10.0, 6)
    xg, yg = np.meshgrid(x, y)
    z_true = xg + yg

    links = pd.DataFrame(
        {
            "cml_id": [1, 2, 3],
            "x_a": [0.0, 2.0, 4.0],
            "y_a": [0.0, 1.0, 2.0],
            "x_b": [10.0, 8.0, 6.0],
            "y_b": [10.0, 9.0, 8.0],
            "x_center": [5.0, 5.0, 5.0],
            "y_center": [5.0, 5.0, 5.0],
        }
    )

    obs = build_link_observations(
        links,
        xg,
        yg,
        z_true,
        WetDrySpec(wet_mode="threshold", wet_target=0.5, seed=3),
        ObservationSpec(
            noise_mmph=0.0,
            link_path_samples=5,
            outage_fraction=1 / 3,
            stuck_zero_fraction=1 / 3,
            seed=8,
        ),
    )

    assert {"R_true", "R_obs", "available", "fault_label", "obs_noise_mmph"} <= set(obs.columns)
    assert obs["available"].isin([True, False]).all()
