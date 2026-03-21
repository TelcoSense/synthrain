from __future__ import annotations

import argparse
from dataclasses import replace

from synthrain.config import load_scenario_config
from synthrain.run_logging import setup_tee_logging


def parse_bbox(s: str):
    parts = [float(p.strip()) for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be 'lon_min,lon_max,lat_min,lat_max'")
    return tuple(parts)


def _build_pre_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", type=str, default="configs/config.ini")
    ap.add_argument("--log-dir", default="logs", help="Directory for datetime log files")
    ap.add_argument("--log-to-file", action="store_true", default=True)
    ap.add_argument("--no-log-to-file", dest="log_to_file", action="store_false")
    ap.add_argument("--quiet-run", action="store_true", help="Reduce non-essential console output")
    return ap


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(parents=[_build_pre_parser()])

    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--debug", dest="debug", action=argparse.BooleanOptionalAction, default=None)

    ap.add_argument("--n-sites", dest="n_sites", type=int, default=None)
    ap.add_argument("--mean-degree", dest="mean_degree", type=int, default=None)
    ap.add_argument("--bbox", type=parse_bbox, default=None)
    ap.add_argument("--city", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--min-length-km", dest="min_length_km", type=float, default=None)
    ap.add_argument("--max-length-km", dest="max_length_km", type=float, default=None)
    ap.add_argument("--site-sampling", dest="site_sampling", type=str, default=None)
    ap.add_argument("--site-min-dist-m", dest="site_min_dist_m", type=float, default=None)

    ap.add_argument("--grid-step-m", dest="grid_step_m", type=float, default=None)
    ap.add_argument("--grid-nx", dest="grid_nx", type=int, default=None)
    ap.add_argument("--grid-ny", dest="grid_ny", type=int, default=None)
    ap.add_argument(
        "--interp-style",
        dest="interp_style",
        choices=["pycomlink", "custom"],
        default=None,
    )
    ap.add_argument("--idw-power", dest="idw_power", type=float, default=None)
    ap.add_argument("--idw-near", dest="idw_near", type=int, default=None)
    ap.add_argument("--idw-dist-m", dest="idw_dist_m", type=float, default=None)
    ap.add_argument(
        "--dry-as-zero",
        dest="dry_as_zero",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    ap.add_argument("--n-blobs", dest="n_blobs", type=int, default=None)
    ap.add_argument("--blob-sigma-m", dest="blob_sigma_m", type=float, default=None)
    ap.add_argument("--peak-mmph", dest="peak_mmph", type=float, default=None)
    ap.add_argument("--background-mmph", dest="background_mmph", type=float, default=None)
    ap.add_argument("--min-rain", dest="min_rain", type=float, default=None)

    ap.add_argument(
        "--wet-mode",
        dest="wet_mode",
        choices=["threshold", "random", "stratified"],
        default=None,
    )
    ap.add_argument("--wet-target", dest="wet_target", type=float, default=None)
    ap.add_argument("--wet-min-mmph", dest="wet_min_mmph", type=float, default=None)
    ap.add_argument("--flip-dry-to-wet", dest="flip_dry_to_wet", type=float, default=None)
    ap.add_argument("--flip-wet-to-dry", dest="flip_wet_to_dry", type=float, default=None)
    ap.add_argument("--wet-strata-nx", dest="wet_strata_nx", type=int, default=None)
    ap.add_argument("--wet-strata-ny", dest="wet_strata_ny", type=int, default=None)

    ap.add_argument("--noise-mmph", dest="noise_mmph", type=float, default=None)
    ap.add_argument("--link-path-samples", dest="link_path_samples", type=int, default=None)
    ap.add_argument("--outage-fraction", dest="outage_fraction", type=float, default=None)
    ap.add_argument(
        "--clustered-outage-fraction",
        dest="clustered_outage_fraction",
        type=float,
        default=None,
    )
    ap.add_argument(
        "--stuck-zero-fraction",
        dest="stuck_zero_fraction",
        type=float,
        default=None,
    )
    ap.add_argument("--bias-fraction", dest="bias_fraction", type=float, default=None)
    ap.add_argument("--bias-low", dest="bias_low", type=float, default=None)
    ap.add_argument("--bias-high", dest="bias_high", type=float, default=None)
    ap.add_argument(
        "--extra-noise-fraction",
        dest="extra_noise_fraction",
        type=float,
        default=None,
    )
    ap.add_argument("--extra-noise-mmph", dest="extra_noise_mmph", type=float, default=None)

    ap.add_argument(
        "--export-csv",
        dest="export_csv",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    ap.add_argument("--csv-steps", dest="csv_steps", type=int, default=None)
    ap.add_argument("--csv-step-min", dest="csv_step_min", type=int, default=None)
    ap.add_argument("--csv-start", dest="csv_start", type=str, default=None)
    ap.add_argument("--title-name", dest="title_name", type=str, default=None)
    return ap


def _apply_overrides(cfg, args):
    io = replace(
        cfg.io,
        out=args.out if args.out is not None else cfg.io.out,
        seed=args.seed if args.seed is not None else cfg.io.seed,
        debug=args.debug if args.debug is not None else cfg.io.debug,
    )
    network = replace(
        cfg.network,
        city=args.city if args.city is not None else cfg.network.city,
        bbox=args.bbox if args.bbox is not None else cfg.network.bbox,
        n_sites=args.n_sites if args.n_sites is not None else cfg.network.n_sites,
        mean_degree=args.mean_degree if args.mean_degree is not None else cfg.network.mean_degree,
        min_length_km=args.min_length_km if args.min_length_km is not None else cfg.network.min_length_km,
        max_length_km=args.max_length_km if args.max_length_km is not None else cfg.network.max_length_km,
        site_sampling=args.site_sampling if args.site_sampling is not None else cfg.network.site_sampling,
        site_min_dist_m=args.site_min_dist_m if args.site_min_dist_m is not None else cfg.network.site_min_dist_m,
    )
    interp = replace(
        cfg.interp,
        interp_style=args.interp_style if args.interp_style is not None else cfg.interp.interp_style,
        grid_step_m=args.grid_step_m if args.grid_step_m is not None else cfg.interp.grid_step_m,
        grid_nx=args.grid_nx if args.grid_nx is not None else cfg.interp.grid_nx,
        grid_ny=args.grid_ny if args.grid_ny is not None else cfg.interp.grid_ny,
        idw_power=args.idw_power if args.idw_power is not None else cfg.interp.idw_power,
        idw_near=args.idw_near if args.idw_near is not None else cfg.interp.idw_near,
        idw_dist_m=args.idw_dist_m if args.idw_dist_m is not None else cfg.interp.idw_dist_m,
        dry_as_zero=args.dry_as_zero if args.dry_as_zero is not None else cfg.interp.dry_as_zero,
    )
    rain = replace(
        cfg.rain,
        n_blobs=args.n_blobs if args.n_blobs is not None else cfg.rain.n_blobs,
        blob_sigma_m=args.blob_sigma_m if args.blob_sigma_m is not None else cfg.rain.blob_sigma_m,
        peak_mmph=args.peak_mmph if args.peak_mmph is not None else cfg.rain.peak_mmph,
        background_mmph=args.background_mmph if args.background_mmph is not None else cfg.rain.background_mmph,
        min_rain=args.min_rain if args.min_rain is not None else cfg.rain.min_rain,
    )
    wet = replace(
        cfg.wet,
        wet_mode=args.wet_mode if args.wet_mode is not None else cfg.wet.wet_mode,
        wet_target=args.wet_target if args.wet_target is not None else cfg.wet.wet_target,
        wet_min_mmph=args.wet_min_mmph if args.wet_min_mmph is not None else cfg.wet.wet_min_mmph,
        wet_strata_nx=args.wet_strata_nx if args.wet_strata_nx is not None else cfg.wet.wet_strata_nx,
        wet_strata_ny=args.wet_strata_ny if args.wet_strata_ny is not None else cfg.wet.wet_strata_ny,
        flip_dry_to_wet=args.flip_dry_to_wet if args.flip_dry_to_wet is not None else cfg.wet.flip_dry_to_wet,
        flip_wet_to_dry=args.flip_wet_to_dry if args.flip_wet_to_dry is not None else cfg.wet.flip_wet_to_dry,
    )
    observation = replace(
        cfg.observation,
        noise_mmph=args.noise_mmph if args.noise_mmph is not None else cfg.observation.noise_mmph,
        link_path_samples=args.link_path_samples if args.link_path_samples is not None else cfg.observation.link_path_samples,
        outage_fraction=args.outage_fraction if args.outage_fraction is not None else cfg.observation.outage_fraction,
        clustered_outage_fraction=args.clustered_outage_fraction if args.clustered_outage_fraction is not None else cfg.observation.clustered_outage_fraction,
        stuck_zero_fraction=args.stuck_zero_fraction if args.stuck_zero_fraction is not None else cfg.observation.stuck_zero_fraction,
        bias_fraction=args.bias_fraction if args.bias_fraction is not None else cfg.observation.bias_fraction,
        bias_low=args.bias_low if args.bias_low is not None else cfg.observation.bias_low,
        bias_high=args.bias_high if args.bias_high is not None else cfg.observation.bias_high,
        extra_noise_fraction=args.extra_noise_fraction if args.extra_noise_fraction is not None else cfg.observation.extra_noise_fraction,
        extra_noise_mmph=args.extra_noise_mmph if args.extra_noise_mmph is not None else cfg.observation.extra_noise_mmph,
    )
    csv = replace(
        cfg.csv,
        export_csv=args.export_csv if args.export_csv is not None else cfg.csv.export_csv,
        csv_steps=args.csv_steps if args.csv_steps is not None else cfg.csv.csv_steps,
        csv_step_min=args.csv_step_min if args.csv_step_min is not None else cfg.csv.csv_step_min,
        csv_start=args.csv_start if args.csv_start is not None else cfg.csv.csv_start,
    )
    plot = replace(
        cfg.plot,
        title_name=args.title_name if args.title_name is not None else cfg.plot.title_name,
    )
    return replace(
        cfg,
        io=io,
        network=network,
        interp=interp,
        rain=rain,
        wet=wet,
        observation=observation,
        csv=csv,
        plot=plot,
    )


def main() -> int:
    pre = _build_pre_parser()
    pre_args, _ = pre.parse_known_args()
    cfg = load_scenario_config(pre_args.config)

    ap = build_parser()
    args = ap.parse_args()
    setup_tee_logging(args.log_dir, enabled=bool(args.log_to_file))

    from synthrain.outputs import write_scenario_outputs
    from synthrain.scenario import run_scenario

    cfg = _apply_overrides(cfg, args)
    result = run_scenario(cfg, quiet_run=bool(args.quiet_run))
    write_scenario_outputs(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
