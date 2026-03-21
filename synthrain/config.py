from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
import configparser


BBox = tuple[float, float, float, float]


def _parse_bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean: {v}")


def _parse_str(v: str) -> str:
    s = str(v).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {"'", '"'}:
        return s[1:-1].strip()
    return s


def _parse_bbox(v: str) -> BBox:
    parts = [float(p.strip()) for p in str(v).split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'lon_min,lon_max,lat_min,lat_max'")
    return (parts[0], parts[1], parts[2], parts[3])


def _parse_float_list(v: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in str(v).split(",") if x.strip())


@dataclass(frozen=True)
class IoConfig:
    out: str = "outputs"
    debug: bool = True
    seed: int = 0


@dataclass(frozen=True)
class NetworkConfig:
    city: bool = True
    city_bbox: BBox = (14.2, 14.8, 49.9, 50.2)
    noncity_bbox: BBox = (12.0, 19.0, 48.5, 51.2)
    bbox: BBox | None = None
    n_sites: int = 50
    mean_degree: int = 4
    min_length_km: float = 0.5
    max_length_km: float = 20.0
    site_sampling: str = "poisson"
    site_min_dist_m: float = 3000.0


@dataclass(frozen=True)
class InterpConfig:
    interp_style: str = "pycomlink"
    grid_step_m: float = 1000.0
    grid_nx: int | None = None
    grid_ny: int | None = None
    idw_power: float = 2.0
    idw_near: int = 8
    idw_dist_m: float = 10000.0
    dry_as_zero: bool = True


@dataclass(frozen=True)
class RainConfig:
    n_blobs: int = 6
    blob_sigma_m: float = 6000.0
    peak_mmph: float = 25.0
    background_mmph: float = 0.0
    min_rain: float = 0.1


@dataclass(frozen=True)
class WetConfig:
    wet_mode: str = "random"
    wet_target: float = 0.1
    wet_min_mmph: float = 0.2
    wet_strata_nx: int = 8
    wet_strata_ny: int = 8
    flip_dry_to_wet: float = 0.02
    flip_wet_to_dry: float = 0.10


@dataclass(frozen=True)
class ObservationConfig:
    noise_mmph: float = 1.0
    link_path_samples: int = 9
    outage_fraction: float = 0.0
    stuck_zero_fraction: float = 0.0
    bias_fraction: float = 0.0
    bias_low: float = 0.85
    bias_high: float = 1.15
    extra_noise_fraction: float = 0.0
    extra_noise_mmph: float = 2.0
    clustered_outage_fraction: float = 0.0


@dataclass(frozen=True)
class CsvConfig:
    export_csv: bool = True
    csv_steps: int = 1
    csv_step_min: int = 10
    csv_start: str = "2025-12-24 17:00:00"


@dataclass(frozen=True)
class PlotConfig:
    title_name: str = "IDW from links (mm/h)"


@dataclass(frozen=True)
class SweepConfig:
    wet_targets: tuple[float, ...] = (0.05, 0.10, 0.20, 0.35, 0.50)


@dataclass(frozen=True)
class ScenarioConfig:
    io: IoConfig = field(default_factory=IoConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    interp: InterpConfig = field(default_factory=InterpConfig)
    rain: RainConfig = field(default_factory=RainConfig)
    wet: WetConfig = field(default_factory=WetConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    csv: CsvConfig = field(default_factory=CsvConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)

    def resolved_bbox(self) -> BBox:
        if self.network.bbox is not None:
            return self.network.bbox
        return self.network.city_bbox if self.network.city else self.network.noncity_bbox

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get(cp: configparser.ConfigParser, section: str, key: str, cast, default: Any) -> Any:
    if section in cp and key in cp[section]:
        return cast(cp[section][key])
    return default


def load_scenario_config(path: str | Path | None = None) -> ScenarioConfig:
    cfg = ScenarioConfig()
    if path is None:
        return cfg

    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    read = cp.read(path)
    if not read:
        raise FileNotFoundError(path)

    cfg = replace(
        cfg,
        io=replace(
            cfg.io,
            out=_get(cp, "io", "out", _parse_str, cfg.io.out),
            debug=_get(cp, "io", "debug", _parse_bool, cfg.io.debug),
            seed=_get(cp, "io", "seed", int, cfg.io.seed),
        ),
        network=replace(
            cfg.network,
            city=_get(cp, "network", "city", _parse_bool, cfg.network.city),
            city_bbox=_get(cp, "network", "city_bbox", _parse_bbox, cfg.network.city_bbox),
            noncity_bbox=_get(cp, "network", "noncity_bbox", _parse_bbox, cfg.network.noncity_bbox),
            bbox=_get(cp, "network", "bbox", _parse_bbox, cfg.network.bbox),
            n_sites=_get(cp, "network", "n_sites", int, cfg.network.n_sites),
            mean_degree=_get(cp, "network", "mean_degree", int, cfg.network.mean_degree),
            min_length_km=_get(cp, "network", "min_length_km", float, cfg.network.min_length_km),
            max_length_km=_get(cp, "network", "max_length_km", float, cfg.network.max_length_km),
            site_sampling=_get(cp, "network", "site_sampling", _parse_str, cfg.network.site_sampling),
            site_min_dist_m=_get(cp, "network", "site_min_dist_m", float, cfg.network.site_min_dist_m),
        ),
        interp=replace(
            cfg.interp,
            interp_style=_get(cp, "interp", "interp_style", _parse_str, cfg.interp.interp_style),
            grid_step_m=_get(cp, "interp", "grid_step_m", float, cfg.interp.grid_step_m),
            grid_nx=_get(cp, "interp", "grid_nx", int, cfg.interp.grid_nx),
            grid_ny=_get(cp, "interp", "grid_ny", int, cfg.interp.grid_ny),
            idw_power=_get(cp, "interp", "idw_power", float, cfg.interp.idw_power),
            idw_near=_get(cp, "interp", "idw_near", int, cfg.interp.idw_near),
            idw_dist_m=_get(cp, "interp", "idw_dist_m", float, cfg.interp.idw_dist_m),
            dry_as_zero=_get(cp, "interp", "dry_as_zero", _parse_bool, cfg.interp.dry_as_zero),
        ),
        rain=replace(
            cfg.rain,
            n_blobs=_get(cp, "rain", "n_blobs", int, cfg.rain.n_blobs),
            blob_sigma_m=_get(cp, "rain", "blob_sigma_m", float, cfg.rain.blob_sigma_m),
            peak_mmph=_get(cp, "rain", "peak_mmph", float, cfg.rain.peak_mmph),
            background_mmph=_get(cp, "rain", "background_mmph", float, cfg.rain.background_mmph),
            min_rain=_get(cp, "rain", "min_rain", float, cfg.rain.min_rain),
        ),
        wet=replace(
            cfg.wet,
            wet_mode=_get(cp, "wet", "wet_mode", _parse_str, cfg.wet.wet_mode),
            wet_target=_get(cp, "wet", "wet_target", float, cfg.wet.wet_target),
            wet_min_mmph=_get(cp, "wet", "wet_min_mmph", float, cfg.wet.wet_min_mmph),
            wet_strata_nx=_get(cp, "wet", "wet_strata_nx", int, cfg.wet.wet_strata_nx),
            wet_strata_ny=_get(cp, "wet", "wet_strata_ny", int, cfg.wet.wet_strata_ny),
            flip_dry_to_wet=_get(cp, "wet", "flip_dry_to_wet", float, cfg.wet.flip_dry_to_wet),
            flip_wet_to_dry=_get(cp, "wet", "flip_wet_to_dry", float, cfg.wet.flip_wet_to_dry),
        ),
        observation=replace(
            cfg.observation,
            noise_mmph=_get(
                cp,
                "observation",
                "noise_mmph",
                float,
                _get(cp, "rain", "noise_mmph", float, cfg.observation.noise_mmph),
            ),
            link_path_samples=_get(cp, "observation", "link_path_samples", int, cfg.observation.link_path_samples),
            outage_fraction=_get(cp, "faults", "outage_fraction", float, cfg.observation.outage_fraction),
            stuck_zero_fraction=_get(cp, "faults", "stuck_zero_fraction", float, cfg.observation.stuck_zero_fraction),
            bias_fraction=_get(cp, "faults", "bias_fraction", float, cfg.observation.bias_fraction),
            bias_low=_get(cp, "faults", "bias_low", float, cfg.observation.bias_low),
            bias_high=_get(cp, "faults", "bias_high", float, cfg.observation.bias_high),
            extra_noise_fraction=_get(cp, "faults", "extra_noise_fraction", float, cfg.observation.extra_noise_fraction),
            extra_noise_mmph=_get(cp, "faults", "extra_noise_mmph", float, cfg.observation.extra_noise_mmph),
            clustered_outage_fraction=_get(cp, "faults", "clustered_outage_fraction", float, cfg.observation.clustered_outage_fraction),
        ),
        csv=replace(
            cfg.csv,
            export_csv=_get(cp, "csv", "export_csv", _parse_bool, cfg.csv.export_csv),
            csv_steps=_get(cp, "csv", "csv_steps", int, cfg.csv.csv_steps),
            csv_step_min=_get(cp, "csv", "csv_step_min", int, cfg.csv.csv_step_min),
            csv_start=_get(cp, "csv", "csv_start", _parse_str, cfg.csv.csv_start),
        ),
        plot=replace(
            cfg.plot,
            title_name=_get(cp, "plot", "title_name", _parse_str, cfg.plot.title_name),
        ),
        sweep=replace(
            cfg.sweep,
            wet_targets=_get(cp, "sweep", "wet_targets", _parse_float_list, cfg.sweep.wet_targets),
        ),
    )
    return cfg
