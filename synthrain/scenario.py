from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

from synthrain.config import ScenarioConfig
from synthrain.generate_network import NetworkSpec, generate_network
from synthrain.geo import GridSpec, lonlat_to_mercator_m, make_grid
from synthrain.idw import IdwKdtree
from synthrain.metrics import FieldMetrics, compute_field_metrics
from synthrain.run_logging import format_path, log_info
from synthrain.simulate_rain import (
    ObservationSpec,
    RainFieldSpec,
    WetDrySpec,
    build_link_observations,
    make_true_field_on_grid,
)


@dataclass
class ScenarioResult:
    config: ScenarioConfig
    bbox_ll: tuple[float, float, float, float]
    grid: dict[str, np.ndarray | tuple[float, float, float, float]]
    sites: pd.DataFrame
    links_obs: pd.DataFrame
    z_true: np.ndarray
    z_idw: np.ndarray
    diff: np.ndarray
    metrics: FieldMetrics

    def to_summary_row(self, extra: dict[str, object] | None = None) -> dict[str, object]:
        available = self.links_obs["available"].to_numpy(bool)
        stuck_zero = self.links_obs["stuck_zero"].to_numpy(bool)
        bias_factor = self.links_obs["bias_factor"].to_numpy(float)
        noise_sigma = self.links_obs["obs_noise_mmph"].to_numpy(float)
        row: dict[str, object] = {
            "out_dir": self.config.io.out,
            "n_sites": self.config.network.n_sites,
            "n_links": int(len(self.links_obs)),
            "seed": self.config.io.seed,
            "wet_target": self.config.wet.wet_target,
            "wet_fraction": float(np.mean(self.links_obs["wet"].to_numpy(bool))),
            "available_fraction": float(np.mean(available)),
            "idw_power": self.config.interp.idw_power,
            "idw_near": self.config.interp.idw_near,
            "idw_dist_m": self.config.interp.idw_dist_m,
            "interp_style": self.config.interp.interp_style,
            "path_samples": self.config.observation.link_path_samples,
            "fault_outage_fraction": self.config.observation.outage_fraction,
            "fault_clustered_outage_fraction": self.config.observation.clustered_outage_fraction,
            "fault_stuck_zero_fraction": self.config.observation.stuck_zero_fraction,
            "fault_bias_fraction": self.config.observation.bias_fraction,
            "fault_extra_noise_fraction": self.config.observation.extra_noise_fraction,
            "realized_unavailable_fraction": float(np.mean(~available)),
            "realized_stuck_zero_fraction": float(np.mean(stuck_zero)),
            "realized_bias_fraction": float(np.mean(~np.isclose(bias_factor, 1.0))),
            "realized_extra_noise_fraction": float(
                np.mean(noise_sigma > float(self.config.observation.noise_mmph))
            ),
        }
        row.update(self.metrics.to_dict())
        if extra:
            row.update(extra)
        return row


def _interpolate_pycomlink(
    links_obs: pd.DataFrame, cfg: ScenarioConfig, xg_m: np.ndarray, yg_m: np.ndarray
) -> np.ndarray:
    try:
        warnings.filterwarnings(
            "ignore",
            message=r".*pkg_resources is deprecated as an API.*",
            category=UserWarning,
            module=r"pycomlink\.io\.examples",
        )
        from pycomlink.spatial.interpolator import IdwKdtreeInterpolator
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pycomlink backend requested but pycomlink is not installed. "
            "Install the optional dependency or use '--interp-style custom'."
        ) from exc

    z_t = links_obs["R_obs"].to_numpy(float).copy()
    wet = links_obs["wet"].to_numpy(bool)
    available = links_obs["available"].to_numpy(bool)
    if not cfg.interp.dry_as_zero:
        z_t[~wet] = np.nan
    z_t[~available] = np.nan

    max_dist = None if cfg.interp.idw_dist_m <= 0 else float(cfg.interp.idw_dist_m)
    interpolator = IdwKdtreeInterpolator(
        nnear=cfg.interp.idw_near,
        p=cfg.interp.idw_power,
        exclude_nan=True,
        max_distance=max_dist,
    )
    return interpolator(
        x=links_obs["x_center"].to_numpy(float),
        y=links_obs["y_center"].to_numpy(float),
        z=z_t,
        xgrid=xg_m,
        ygrid=yg_m,
    )


def _interpolate_custom(
    links_obs: pd.DataFrame, cfg: ScenarioConfig, xg_m: np.ndarray, yg_m: np.ndarray
) -> np.ndarray:
    values = links_obs["R_obs"].to_numpy(float).copy()
    wet = links_obs["wet"].to_numpy(bool)
    available = links_obs["available"].to_numpy(bool)
    if not cfg.interp.dry_as_zero:
        values[~wet] = np.nan
    values[~available] = np.nan

    max_dist = None if cfg.interp.idw_dist_m <= 0 else float(cfg.interp.idw_dist_m)
    idw = IdwKdtree(
        nnear=cfg.interp.idw_near,
        p=cfg.interp.idw_power,
        max_distance=max_dist,
        exclude_nan=True,
    ).fit(links_obs[["x_center", "y_center"]].to_numpy(float), values)
    return idw.predict_grid(xg_m, yg_m)


def _postprocess_idw(z_idw: np.ndarray, cfg: ScenarioConfig) -> np.ndarray:
    out = np.asarray(z_idw, dtype=float).copy()
    out[out < float(cfg.rain.min_rain)] = 0.0
    return out


def run_scenario(cfg: ScenarioConfig, *, quiet_run: bool = False) -> ScenarioResult:
    bbox_ll = cfg.resolved_bbox()
    out_dir = cfg.io.out
    if not quiet_run:
        log_info("SCENARIO", f"output directory: {format_path(out_dir)}")

    net_spec = NetworkSpec(
        n_sites=cfg.network.n_sites,
        bbox=bbox_ll,
        mean_degree=cfg.network.mean_degree,
        min_length_km=cfg.network.min_length_km,
        max_length_km=cfg.network.max_length_km,
        seed=cfg.io.seed,
        site_sampling=cfg.network.site_sampling,
        site_min_dist_m=cfg.network.site_min_dist_m,
    )
    sites, links = generate_network(net_spec, cfg.io.debug)

    grid = make_grid(
        GridSpec(
            bbox_lonlat=bbox_ll,
            grid_step_m=cfg.interp.grid_step_m,
            use_mercator=True,
            grid_nx=cfg.interp.grid_nx,
            grid_ny=cfg.interp.grid_ny,
        )
    )
    xg_m = grid["xg_m"]
    yg_m = grid["yg_m"]

    if cfg.io.debug and not quiet_run:
        dx = float(np.median(np.diff(xg_m, axis=1))) if xg_m.shape[1] > 1 else 0.0
        dy = float(np.median(np.diff(yg_m, axis=0))) if yg_m.shape[0] > 1 else 0.0
        log_info("GRID", f"spacing: dx={dx:.3f} m, dy={dy:.3f} m")
        log_info("GRID", f"size: nx={xg_m.shape[1]}, ny={xg_m.shape[0]}")

    rain_spec = RainFieldSpec(
        n_blobs=cfg.rain.n_blobs,
        blob_sigma=cfg.rain.blob_sigma_m,
        peak_mmph=cfg.rain.peak_mmph,
        background_mmph=cfg.rain.background_mmph,
        seed=cfg.io.seed + 10,
    )
    z_true = make_true_field_on_grid(xg_m, yg_m, rain_spec)

    links_obs = links.copy()
    x_a, y_a = lonlat_to_mercator_m(
        links_obs["site_a_lon"].to_numpy(float), links_obs["site_a_lat"].to_numpy(float)
    )
    x_b, y_b = lonlat_to_mercator_m(
        links_obs["site_b_lon"].to_numpy(float), links_obs["site_b_lat"].to_numpy(float)
    )
    x_c, y_c = lonlat_to_mercator_m(
        links_obs["lon_center"].to_numpy(float), links_obs["lat_center"].to_numpy(float)
    )
    links_obs["x_a"] = x_a
    links_obs["y_a"] = y_a
    links_obs["x_b"] = x_b
    links_obs["y_b"] = y_b
    links_obs["x_center"] = x_c
    links_obs["y_center"] = y_c

    wet_spec = WetDrySpec(
        wet_mode=cfg.wet.wet_mode,
        wet_target=cfg.wet.wet_target,
        wet_min_mmph=cfg.wet.wet_min_mmph,
        strata_nx=cfg.wet.wet_strata_nx,
        strata_ny=cfg.wet.wet_strata_ny,
        flip_dry_to_wet=cfg.wet.flip_dry_to_wet,
        flip_wet_to_dry=cfg.wet.flip_wet_to_dry,
        seed=cfg.io.seed + 20,
    )
    observation_spec = ObservationSpec(
        noise_mmph=cfg.observation.noise_mmph,
        link_path_samples=cfg.observation.link_path_samples,
        outage_fraction=cfg.observation.outage_fraction,
        stuck_zero_fraction=cfg.observation.stuck_zero_fraction,
        bias_fraction=cfg.observation.bias_fraction,
        bias_low=cfg.observation.bias_low,
        bias_high=cfg.observation.bias_high,
        extra_noise_fraction=cfg.observation.extra_noise_fraction,
        extra_noise_mmph=cfg.observation.extra_noise_mmph,
        clustered_outage_fraction=cfg.observation.clustered_outage_fraction,
        seed=cfg.io.seed + 30,
    )
    links_obs = build_link_observations(
        links_obs,
        xg_m,
        yg_m,
        z_true,
        wet_spec,
        observation_spec,
    )

    if cfg.interp.interp_style == "pycomlink":
        z_idw = _interpolate_pycomlink(links_obs, cfg, xg_m, yg_m)
    else:
        z_idw = _interpolate_custom(links_obs, cfg, xg_m, yg_m)
    z_idw = _postprocess_idw(z_idw, cfg)

    diff = z_idw - z_true
    metrics = compute_field_metrics(z_true, z_idw, min_rain=cfg.rain.min_rain)

    return ScenarioResult(
        config=cfg,
        bbox_ll=bbox_ll,
        grid=grid,
        sites=sites,
        links_obs=links_obs,
        z_true=z_true,
        z_idw=z_idw,
        diff=diff,
        metrics=metrics,
    )
