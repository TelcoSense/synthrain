"""
Export a TelcoRain-like CSV snippet (synthetic).

This is only meant to resemble the 'calc_dataset' style so you can reuse downstream tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


@dataclass
class CsvSpec:
    start_time: str = "2025-06-20 22:50:00"
    n_steps: int = 24
    step_minutes: int = 10
    seed: int = 0


@dataclass
class CsvSpecMinimal:
    start_time: str = "2025-06-20 22:50:00"
    n_steps: int = 24
    step_minutes: int = 10
    seed: int = 0
    keep_link_wet_pattern: bool = True
    # if True, keep the original per-link wet pattern across time steps


def export_calc_dataset_csv(
    links_obs: pd.DataFrame, path: str, spec: CsvSpec
) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed)

    t0 = pd.to_datetime(spec.start_time)
    times = [
        t0 + pd.Timedelta(minutes=spec.step_minutes * i) for i in range(spec.n_steps)
    ]

    # Static per-link fields
    df_links = links_obs.copy()

    # Add "telcorain-like" channel fields
    df_links["channel_id"] = "A(rx)_B(tx)"
    df_links["frequency"] = rng.uniform(6.0, 18.0, size=len(df_links)).round(3)
    df_links["polarization"] = rng.choice(["H", "V"], size=len(df_links))
    df_links["dummy_channel"] = False

    # Fill "site" fields using lon/lat degrees
    df_links["site_a_latitude"] = df_links["site_a_lat"]
    df_links["site_b_latitude"] = df_links["site_b_lat"]
    df_links["site_a_longitude"] = df_links["site_a_lon"]
    df_links["site_b_longitude"] = df_links["site_b_lon"]
    df_links["lat_center"] = df_links["lat_center"]
    df_links["lon_center"] = df_links["lon_center"]

    # Dummy coords (small perturbation)
    df_links["dummy_a_latitude"] = df_links["site_a_latitude"] + rng.normal(
        0, 0.001, size=len(df_links)
    )
    df_links["dummy_b_latitude"] = df_links["site_b_latitude"] + rng.normal(
        0, 0.001, size=len(df_links)
    )
    df_links["dummy_a_longitude"] = df_links["site_a_longitude"] + rng.normal(
        0, 0.001, size=len(df_links)
    )
    df_links["dummy_b_longitude"] = df_links["site_b_longitude"] + rng.normal(
        0, 0.001, size=len(df_links)
    )

    # Static baseline/waa/A placeholders
    df_links["baseline"] = 0.0
    df_links["waa"] = 0.0
    df_links["A"] = 0.0

    # Temperatures and tsl/rsl/trsl are per-time (create plausible values)
    rows = []
    for t in times:
        temp_rx = 25 + 6 * rng.random(len(df_links))
        temp_tx = 25 + 6 * rng.random(len(df_links))

        # Create R per timestep: scale the per-link observed (for demo we add small time jitter)
        time_jitter = 0.8 + 0.4 * rng.random(len(df_links))
        R = df_links["R_obs"].to_numpy(float) * time_jitter
        wet = (R > 0).astype(bool)
        # wet_fraction = rng.uniform(0.5, 1.0, size=len(df_links))  # synthetic placeholder

        # rsl is negative; trsl positive; tsl not used (keep 0)
        trsl = 60 + 5 * rng.normal(0, 1, size=len(df_links)) + R  # increase with rain
        rsl = -trsl

        step = pd.DataFrame(
            {
                "cml_id": df_links["cml_id"].to_numpy(int),
                "channel_id": df_links["channel_id"],
                "time": t,
                "tsl": 0.0,
                "rsl": rsl,
                "temperature_rx": temp_rx,
                "temperature_tx": temp_tx,
                "site_a_latitude": df_links["site_a_latitude"],
                "site_b_latitude": df_links["site_b_latitude"],
                "site_a_longitude": df_links["site_a_longitude"],
                "site_b_longitude": df_links["site_b_longitude"],
                "frequency": df_links["frequency"],
                "polarization": df_links["polarization"],
                "length": df_links["length"],
                "dummy_channel": df_links["dummy_channel"],
                "dummy_a_latitude": df_links["dummy_a_latitude"],
                "dummy_b_latitude": df_links["dummy_b_latitude"],
                "dummy_a_longitude": df_links["dummy_a_longitude"],
                "dummy_b_longitude": df_links["dummy_b_longitude"],
                "trsl": trsl,
                "wet": wet,
                # "wet_fraction": wet_fraction,
                "baseline": df_links["baseline"],
                "waa": df_links["waa"],
                "A": df_links["A"],
                "R": R,
                "lat_center": df_links["lat_center"],
                "lon_center": df_links["lon_center"],
            }
        )
        rows.append(step)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def export_synth_minimal_csv(
    links_obs: pd.DataFrame, path: str, spec: CsvSpecMinimal
) -> pd.DataFrame:
    """
    Export a minimal CSV used by the synthetic pipeline:
    cml_id, time, lat_center, lon_center, R, wet

    Expects links_obs to contain:
      - cml_id (int-like)
      - lat_center, lon_center (degrees)
      - R_obs (mm/h-like), and optionally wet (bool)
    """
    rng = np.random.default_rng(spec.seed)

    t0 = pd.to_datetime(spec.start_time)
    times = [
        t0 + pd.Timedelta(minutes=spec.step_minutes * i) for i in range(spec.n_steps)
    ]

    df_links = links_obs.copy()

    # enforce integer-like ids
    df_links["cml_id"] = pd.to_numeric(df_links["cml_id"], errors="coerce").astype(
        "Int64"
    )
    df_links = df_links[df_links["cml_id"].notna()].copy()
    df_links["cml_id"] = df_links["cml_id"].astype(int)

    base_R = df_links["R_obs"].to_numpy(float)

    rows = []
    for t in times:
        # small time jitter so each 10-min step differs
        time_jitter = 0.8 + 0.4 * rng.random(len(df_links))
        R = base_R * time_jitter

        if spec.keep_link_wet_pattern and "wet" in df_links.columns:
            # preserve provided wet labels; enforce R=0 for dry if desired
            wet = df_links["wet"].to_numpy(bool)
            R = np.where(wet, R, 0.0)
        else:
            wet = (R > 0).astype(bool)

        step = pd.DataFrame(
            {
                "cml_id": df_links["cml_id"].to_numpy(int),
                "time": t,
                "lat_center": df_links["lat_center"].to_numpy(float),
                "lon_center": df_links["lon_center"].to_numpy(float),
                "R": R,
                "wet": wet,
            }
        )
        rows.append(step)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False)
    return out
