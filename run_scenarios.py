"""
Main CLI: generate synthetic network, simulate wet/dry, run IDW, save PNGs (+ optional CSV).

Example:
python run_scenarios.py --out out_demo --seed 0 --city --n-sites 250 --mean-degree 3 \
  --wet-target 0.35 --wet-mode threshold --flip-dry-to-wet 0.03 --flip-wet-to-dry 0.15 \
  --idw-near 12 --idw-power 2.0 --idw-dist 0.15 --dry-as-zero --export-csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

from pycomlink.spatial.interpolator import IdwKdtreeInterpolator
from synthrain.config_utils import load_ini_defaults
from synthrain.generate_network import NetworkSpec, generate_network
from synthrain.simulate_rain import (
    RainFieldSpec,
    WetDrySpec,
    make_true_field_on_grid,
    build_link_observations,
)
from synthrain.idw import IdwKdtree
from synthrain.render import save_field_image, save_links_image
from synthrain.csv_export import (
    CsvSpec,
    CsvSpecMinimal,
    export_calc_dataset_csv,
    export_synth_minimal_csv,
)
from synthrain.geo import GridSpec, make_grid, lonlat_to_mercator_m


def parse_bbox(s: str):
    parts = [float(p.strip()) for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be 'x_min,x_max,y_min,y_max'")
    return tuple(parts)


def main():
    # Two-stage parsing: first read --config (if provided), then apply INI values as defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        type=str,
        default="configs/config.ini",
        help="INI config file (TelcoRain-like). CLI flags override it.",
    )
    pre_args, _ = pre.parse_known_args()

    defaults = {}
    if pre_args.config:
        defaults = load_ini_defaults(pre_args.config)

    ap = argparse.ArgumentParser(parents=[pre])

    ap.add_argument("--out", required=("out" not in defaults), help="Output directory")
    ap.add_argument("--seed", type=int, default=0)

    # Network
    ap.add_argument("--n-sites", dest="n_sites", type=int, default=100)
    ap.add_argument("--mean-degree", dest="mean_degree", type=int, default=3)
    ap.add_argument(
        "--bbox", type=parse_bbox, default=None, help="lon_min,lon_max,lat_min,lat_max"
    )
    ap.add_argument(
        "--city",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a smaller bbox around a fictive city (dense network)",
    )

    ap.add_argument("--min-length-km", dest="min_length_km", type=float, default=0.5)
    ap.add_argument("--max-length-km", dest="max_length_km", type=float, default=30.0)

    # Interp grid (TelcoRain-like)
    ap.add_argument("--grid-step-m", dest="grid_step_m", type=float, default=1000.0)
    ap.add_argument("--grid-nx", dest="grid_nx", type=int, default=None)
    ap.add_argument("--grid-ny", dest="grid_ny", type=int, default=None)

    # True field (on the interpolation grid)
    ap.add_argument("--n-blobs", dest="n_blobs", type=int, default=4)
    ap.add_argument("--blob-sigma-m", dest="blob_sigma_m", type=float, default=8000.0)
    ap.add_argument("--peak-mmph", dest="peak_mmph", type=float, default=15.0)
    ap.add_argument("--noise-mmph", dest="noise_mmph", type=float, default=0.6)

    # Wet/dry
    ap.add_argument(
        "--wet-mode",
        dest="wet_mode",
        choices=["threshold", "random", "stratified"],
        default="random",
    )
    ap.add_argument("--wet-target", dest="wet_target", type=float, default=0.3)
    ap.add_argument(
        "--wet-min-mmph",
        dest="wet_min_mmph",
        type=float,
        default=0.2,
        help="Only used by some wet-modes; keep small if you want wide wet variation.",
    )
    ap.add_argument(
        "--flip-dry-to-wet", dest="flip_dry_to_wet", type=float, default=0.0
    )
    ap.add_argument(
        "--flip-wet-to-dry", dest="flip_wet_to_dry", type=float, default=0.0
    )
    ap.add_argument("--wet-strata-nx", dest="wet_strata_nx", type=int, default=6)
    ap.add_argument("--wet-strata-ny", dest="wet_strata_ny", type=int, default=6)

    # IDW
    ap.add_argument("--idw-power", dest="idw_power", type=float, default=2.0)
    ap.add_argument(
        "--idw-near",
        dest="idw_near",
        type=int,
        default=8,
        help="Max neighbours per grid point (like TelcoRain idw_near).",
    )
    ap.add_argument(
        "--idw-dist-m",
        dest="idw_dist_m",
        type=float,
        default=10000,
        help="Neighbour radius (meters); set <=0 for unlimited.",
    )
    ap.add_argument(
        "--dry-as-zero",
        dest="dry_as_zero",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, include dry links as 0 in IDW; else ignore dry links.",
    )

    # CSV export
    ap.add_argument(
        "--export-csv",
        dest="export_csv",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    ap.add_argument("--csv-steps", dest="csv_steps", type=int, default=24)
    ap.add_argument("--csv-step-min", dest="csv_step_min", type=int, default=10)
    ap.add_argument(
        "--csv-start", dest="csv_start", type=str, default="2025-06-20 22:50:00"
    )
    ap.add_argument(
        "--title-name", dest="title_name", type=str, default="IDW from links (mm/h)"
    )

    if defaults:
        ap.set_defaults(**defaults)

    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # bbox
    if args.bbox is not None:
        bbox_ll = args.bbox
    else:
        # Rough Czechia bbox by default (synthetic still)
        bbox_ll = (12.0, 19.0, 48.5, 51.2)
        if args.city:
            # Smaller area around Prague-like bbox (still synthetic)
            bbox_ll = (14.2, 14.8, 49.9, 50.2)

    # 1) Network
    net_spec = NetworkSpec(
        n_sites=args.n_sites,
        bbox=bbox_ll,
        mean_degree=args.mean_degree,
        min_length_km=args.min_length_km,
        max_length_km=args.max_length_km,
        seed=args.seed,
        site_sampling=getattr(args, "site_sampling", "uniform"),
        site_min_dist_m=getattr(args, "site_min_dist_m", 0),
    )
    sites, links = generate_network(net_spec, args.debug)

    # 1b) Build interpolation grid (meters + lon/lat for plotting)
    grid = make_grid(
        GridSpec(
            bbox_lonlat=bbox_ll,
            grid_step_m=args.grid_step_m,
            use_mercator=True,
            grid_nx=args.grid_nx,
            grid_ny=args.grid_ny,
        )
    )
    xg_m = grid["xg_m"]
    yg_m = grid["yg_m"]
    lon_g = grid["lon_g"]
    lat_g = grid["lat_g"]

    # 2) True field (ground truth)
    rf_spec = RainFieldSpec(
        n_blobs=args.n_blobs,
        blob_sigma=args.blob_sigma_m,
        peak_mmph=args.peak_mmph,
        seed=args.seed + 10,
    )
    z_true = make_true_field_on_grid(xg_m, yg_m, rf_spec)

    # 3) Link observations
    wet_spec = WetDrySpec(
        wet_mode=args.wet_mode,
        wet_target=args.wet_target,
        wet_min_mmph=args.wet_min_mmph,
        strata_nx=args.wet_strata_nx,
        strata_ny=args.wet_strata_ny,
        flip_dry_to_wet=args.flip_dry_to_wet,
        flip_wet_to_dry=args.flip_wet_to_dry,
        seed=args.seed + 20,
    )

    # Add projected center coordinates for IDW and sampling
    cx_m, cy_m = lonlat_to_mercator_m(
        links["lon_center"].to_numpy(float), links["lat_center"].to_numpy(float)
    )
    links_m = links.copy()
    links_m["x_center"] = cx_m
    links_m["y_center"] = cy_m
    links_obs = build_link_observations(
        links_m, xg_m, yg_m, z_true, wet_spec, noise_mmph=args.noise_mmph
    )

    # 4) IDW interpolation
    # either pycomlink or custom style (testing not done yet)
    if args.interp_style == "pycomlink":
        x_sites = links_obs["x_center"].to_numpy(float)
        y_sites = links_obs["y_center"].to_numpy(float)

        z_t = links_obs["R_obs"].to_numpy(float).copy()
        if not args.dry_as_zero:
            z_t[~links_obs["wet"].to_numpy(bool)] = np.nan

        max_dist = (
            None
            if (args.idw_dist_m is None or args.idw_dist_m <= 0)
            else float(args.idw_dist_m)
        )

        interpolator = IdwKdtreeInterpolator(
            nnear=args.idw_near,
            p=args.idw_power,
            exclude_nan=True,
            max_distance=max_dist,
        )

        z_idw = interpolator(
            x=x_sites,
            y=y_sites,
            z=z_t,
            xgrid=xg_m,
            ygrid=yg_m,
        )

        # TelcoRain post-threshold
        z_idw[z_idw < float(args.min_rain)] = 0.0
    else:
        values = links_obs["R_obs"].to_numpy(float).copy()
        if not args.dry_as_zero:
            values[~links_obs["wet"].to_numpy(bool)] = np.nan

        xy = links_obs[["x_center", "y_center"]].to_numpy(float)

        max_dist = (
            None
            if (args.idw_dist_m is None or float(args.idw_dist_m) <= 0)
            else float(args.idw_dist_m)
        )
        idw = IdwKdtree(
            nnear=args.idw_near,
            p=args.idw_power,
            max_distance=max_dist,
            exclude_nan=True,
        ).fit(xy, values)

        z_idw = idw.predict_grid(xg_m, yg_m)

    # 5) Render
    vmin = 0.0
    vmax = (
        max(float(np.nanmax(z_true)), float(np.nanmax(z_idw)))
        if np.isfinite(z_idw).any()
        else float(np.nanmax(z_true))
    )
    save_field_image(
        str(out_dir / "true_field"),
        lon_g,
        lat_g,
        z_true,
        title="True rainfall field (mm/h)",
        vmin=vmin,
        vmax=vmax,
        xlabel="lon",
        ylabel="lat",
        min_rain=args.min_rain,
        suffix="png",
    )
    save_links_image(
        str(out_dir / "links"),
        links_obs,
        bbox=bbox_ll,
        title="Link centers (wet/dry)",
        suffix="png",
    )
    save_field_image(
        str(out_dir / "idw_field"),
        lon_g,
        lat_g,
        z_idw,
        title=args.title_name,
        vmin=vmin,
        vmax=vmax,
        links=links_obs,
        show_links=True,
        xlabel="lon",
        ylabel="lat",
        min_rain=args.min_rain,
        suffix="pdf",
    )
    save_field_image(
        str(out_dir / "idw_field"),
        lon_g,
        lat_g,
        z_idw,
        title=args.title_name,
        vmin=vmin,
        vmax=vmax,
        links=links_obs,
        show_links=True,
        xlabel="lon",
        ylabel="lat",
        min_rain=args.min_rain,
        suffix="png",
    )

    diff = z_idw - z_true
    save_field_image(
        str(out_dir / "diff"),
        lon_g,
        lat_g,
        diff,
        title="IDW - True (mm/h)",
        xlabel="lon",
        ylabel="lat",
        min_rain=args.min_rain,
        suffix="png",
    )

    # 6) Optional CSV export
    if args.export_csv:
        csv_path = out_dir / "calc_dataset_synth.csv"
        _ = export_synth_minimal_csv(
            links_obs,
            str(csv_path),
            CsvSpecMinimal(
                start_time=args.csv_start,
                n_steps=args.csv_steps,
                step_minutes=args.csv_step_min,
                seed=args.seed + 30,
            ),
        )

    # Persist scenario metadata
    meta = {
        "bbox_ll": bbox_ll,
        "bbox_m": grid["bbox_m"],
        "n_sites": args.n_sites,
        "n_links": int(len(links_obs)),
        "wet_fraction": float(np.mean(links_obs["wet"].to_numpy(bool))),
        "idw": {
            "nnear": args.idw_near,
            "p": args.idw_power,
            "max_distance_m": max_dist,
        },
        "wet_spec": {
            "wet_mode": args.wet_mode,
            "wet_target": args.wet_target,
            "wet_min_mmph": args.wet_min_mmph,
            "flip_dry_to_wet": args.flip_dry_to_wet,
            "flip_wet_to_dry": args.flip_wet_to_dry,
            "strata_nx": args.wet_strata_nx,
            "strata_ny": args.wet_strata_ny,
        },
        "rain_field": {
            "n_blobs": args.n_blobs,
            "blob_sigma_m": args.blob_sigma_m,
            "peak_mmph": args.peak_mmph,
            "noise_mmph": args.noise_mmph,
            "grid_step_m": args.grid_step_m,
            "grid_nx": args.grid_nx,
            "grid_ny": args.grid_ny,
        },
        "dry_as_zero": bool(args.dry_as_zero),
        "seed": args.seed,
    }
    (out_dir / "scenario.json").write_text(
        __import__("json").dumps(meta, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
