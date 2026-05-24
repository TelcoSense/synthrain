from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from synthrain.sweeps import (
    _add_score_columns,
    _build_parameter_robustness_rows,
    _idw_arg,
    _rank_rows,
)


def test_idw_preset_supplies_poster_grid_values():
    args = Namespace(preset="poster", powers=None, sheet_rows=None)

    assert _idw_arg(args, "powers") == "1,2.5,4"
    assert _idw_arg(args, "sheet_rows") == 2


def test_balanced_score_and_ranking_columns(tmp_path: Path):
    rows = [
        {
            "out_dir": str(tmp_path / "run_a"),
            "rmse": 1.0,
            "mae": 0.8,
            "bias": 0.1,
            "valid_pixel_fraction": 1.0,
            "wet_miss_rate": 0.1,
            "dry_false_rain_rate": 0.0,
        },
        {
            "out_dir": str(tmp_path / "run_b"),
            "rmse": 2.0,
            "mae": 1.2,
            "bias": -0.4,
            "valid_pixel_fraction": 0.5,
            "wet_miss_rate": 0.5,
            "dry_false_rain_rate": 0.2,
        },
    ]

    ranked = _rank_rows(rows, tmp_path, ranking_metric="balanced_score")

    assert ranked[0]["run_dir"] == "run_a"
    assert ranked[0]["rank_balanced"] == 1
    assert ranked[0]["balanced_score"] < ranked[1]["balanced_score"]


def test_parameter_robustness_aggregates_by_idw_tuple():
    rows = _add_score_columns(
        [
            {
                "idw_power": 2,
                "idw_near": 8,
                "idw_dist_m": 10000,
                "scenario_tag": "a",
                "rmse": 1.0,
                "mae": 0.8,
                "bias": 0.0,
                "valid_pixel_fraction": 1.0,
            },
            {
                "idw_power": 2,
                "idw_near": 8,
                "idw_dist_m": 10000,
                "scenario_tag": "b",
                "rmse": 3.0,
                "mae": 1.6,
                "bias": 0.2,
                "valid_pixel_fraction": 0.8,
            },
        ]
    )

    robustness = _build_parameter_robustness_rows(rows)

    assert len(robustness) == 1
    assert robustness[0]["n_runs"] == 2
    assert robustness[0]["rmse_mean"] == 2.0
    assert robustness[0]["rank_robust"] == 1
