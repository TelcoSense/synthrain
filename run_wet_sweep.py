"""
Batch generator: sweep wet fraction and/or dry_as_zero mode to compare artefacts.

Example:
python run_wet_sweep.py --out out_sweep --city --n-sites 350 --mean-degree 3 \
  --wet-targets 0.05,0.10,0.20,0.35,0.50 --idw-near 12 --idw-power 2 --idw-dist 0.15
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

# reuse run_scenarios.main-like logic by invoking it as a module would be messy; do direct imports
from synthrain.generate_network import NetworkSpec, generate_network
from synthrain.simulate_rain import (
    RainFieldSpec,
    WetDrySpec,
    make_true_field_on_grid,
    build_link_observations,
)
from synthrain.idw import IdwKdtree
from synthrain.render import save_field_png, save_links_png
from synthrain.geo import GridSpec, make_grid, lonlat_to_mercator_m


def parse_bbox(s: str):
    parts = [float(p.strip()) for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be 'x_min,x_max,y_min,y_max'")
    return tuple(parts)


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        type=str,
        default=None,
        help="INI config file. CLI flags override it.",
    )
    pre_args, _ = pre.parse_known_args()

    defaults = {}
    if pre_args.config:
        from synthrain.config_utils import load_ini_defaults

        defaults = load_ini_defaults(pre_args.config)

    ap = argparse.ArgumentParser(parents=[pre])
    ap.set_defaults(**defaults)

    ap.add_argument("--out", required=("out" not in defaults), help="Output directory")
    ap.add_argument("--seed", type=int, default=0)

    # Network
    ap.add_argument("--n-sites", dest="n_sites", type=int, default=350)
    ap.add_argument("--mean-degree", dest="mean_degree", type=int, default=3)
    ap.add_argument(
        "--bbox", type=parse_bbox, default=None, help="lon_min,lon_max,lat_min,lat_max"
    )
    ap.add_argument("--city", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--min-length-km", dest="min_length_km", type=float, default=0.5)
    ap.add_argument("--max-length-km", dest="max_length_km", type=float, default=30.0)

    # Interp/IDW (TelcoRain-like)
    ap.add_argument(
        "--use-mercator",
        dest="use_mercator",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--grid-step-m", dest="grid_step_m", type=float, default=1000.0)
    ap.add_argument("--grid-nx", dest="grid_nx", type=int, default=None)
    ap.add_argument("--grid-ny", dest="grid_ny", type=int, default=None)
    ap.add_argument("--idw-power", dest="idw_power", type=float, default=2.0)
    ap.add_argument("--idw-near", dest="idw_near", type=int, default=8)
    ap.add_argument("--idw-dist-m", dest="idw_dist_m", type=float, default=30000.0)

    # Rain field
    ap.add_argument("--n-blobs", dest="n_blobs", type=int, default=4)
    ap.add_argument("--blob-sigma-m", dest="blob_sigma_m", type=float, default=8000.0)
    ap.add_argument("--peak-mmph", dest="peak_mmph", type=float, default=15.0)
    ap.add_argument("--noise-mmph", dest="noise_mmph", type=float, default=0.6)

    # Wet sweep
    ap.add_argument(
        "--wet-mode",
        dest="wet_mode",
        choices=["threshold", "random", "stratified"],
        default="threshold",
    )
    ap.add_argument(
        "--wet-targets",
        dest="wet_targets",
        type=str,
        default="0.05,0.10,0.20,0.35,0.50",
    )
    ap.add_argument("--wet-min-mmph", dest="wet_min_mmph", type=float, default=0.2)
    ap.add_argument(
        "--flip-dry-to-wet", dest="flip_dry_to_wet", type=float, default=0.0
    )
    ap.add_argument(
        "--flip-wet-to-dry", dest="flip_wet_to_dry", type=float, default=0.0
    )
    ap.add_argument("--wet-strata-nx", dest="wet_strata_nx", type=int, default=6)
    ap.add_argument("--wet-strata-ny", dest="wet_strata_ny", type=int, default=6)

    # Compare dry handling
    ap.add_argument(
        "--compare-dry-modes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, run both 'ignore dry' and 'dry-as-zero' for each wet_target.",
    )

    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wet_targets = [float(x) for x in args.wet_targets.split(",") if x.strip() != ""]
    modes = [
        m.strip().lower() for m in args.dry_as_zero_modes.split(",") if m.strip() != ""
    ]
    for m in modes:
        if m not in ("ignore", "zero"):
            raise SystemExit("dry-as-zero-modes must be a subset of: ignore,zero")

    # bbox (lon_min,lon_max,lat_min,lat_max)
    if args.bbox is not None:
        bbox_ll = args.bbox
    else:
        bbox_ll = (12.0, 19.0, 48.5, 51.2)
        if args.city:
            bbox_ll = (14.2, 14.8, 49.9, 50.2)

    # Make one network and one true field for comparability across sweeps
    net_spec = NetworkSpec(
        n_sites=args.n_sites,
        bbox=bbox_ll,
        mean_degree=args.mean_degree,
        min_length_km=args.min_length_km,
        max_length_km=args.max_length_km,
        seed=args.seed,
    )
    sites, links = generate_network(net_spec)

    grid = make_grid(
        GridSpec(
            bbox_lonlat=bbox_ll,
            grid_step_m=args.grid_step_m,
            use_mercator=bool(args.use_mercator),
            grid_nx=args.grid_nx,
            grid_ny=args.grid_ny,
        )
    )
    xg_m, yg_m = grid["xg_m"], grid["yg_m"]
    lon_g, lat_g = grid["lon_g"], grid["lat_g"]

    rf_spec = RainFieldSpec(
        n_blobs=args.n_blobs,
        blob_sigma=args.blob_sigma_m,
        peak_mmph=args.peak_mmph,
        seed=args.seed + 10,
    )
    z_true = make_true_field_on_grid(xg_m, yg_m, rf_spec)
    vmax_true = float(np.nanmax(z_true))

    # Save true once
    save_field_png(
        str(out_dir / "true_field"),
        lon_g,
        lat_g,
        z_true,
        title="True rainfall field (mm/h)",
        xlabel="lon",
        ylabel="lat",
    )

    for wet_target in wet_targets:
        wet_spec = WetDrySpec(
            wet_mode=args.wet_mode,
            wet_target=wet_target,
            wet_min_mmph=args.wet_min_mmph,
            strata_nx=args.wet_strata_nx,
            strata_ny=args.wet_strata_ny,
            flip_dry_to_wet=args.flip_dry_to_wet,
            flip_wet_to_dry=args.flip_wet_to_dry,
            seed=args.seed + 20 + int(wet_target * 1000),
        )

        # Project link centers for sampling/IDW (meters)
        cx_m, cy_m = lonlat_to_mercator_m(
            links["lon_center"].to_numpy(float), links["lat_center"].to_numpy(float)
        )
        links_m = links.copy()
        links_m["x_center"] = cx_m
        links_m["y_center"] = cy_m
        links_obs = build_link_observations(
            links_m, xg_m, yg_m, z_true, wet_spec, noise_mmph=args.noise_mmph
        )

        # Links overview
        sub_base = out_dir / f"wet_{wet_target:.2f}"
        sub_base.mkdir(parents=True, exist_ok=True)
        save_links_png(
            str(sub_base / "links.png"),
            links_obs,
            bbox=bbox_ll,
            title=f"Link centers (wet_target={wet_target:.2f})",
        )

        for mode in modes:
            dry_as_zero = mode == "zero"
            values = links_obs["R_obs"].to_numpy(float).copy()
            if not dry_as_zero:
                values[~links_obs["wet"].to_numpy(bool)] = np.nan

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
            ).fit(links_obs[["x_center", "y_center"]].to_numpy(float), values)

            z_idw = idw.predict_grid(xg_m, yg_m)
            vmax = max(
                vmax_true,
                float(np.nanmax(z_idw)) if np.isfinite(z_idw).any() else vmax_true,
            )
            mode_dir = sub_base / ("dry_as_zero" if dry_as_zero else "ignore_dry")
            mode_dir.mkdir(parents=True, exist_ok=True)

            save_field_png(
                str(mode_dir / "idw_field"),
                lon_g,
                lat_g,
                z_idw,
                title=f"IDW (wet_target={wet_target:.2f}, mode={mode})",
                vmin=0.0,
                vmax=vmax,
                links=links_obs,
                show_links=True,
                xlabel="lon",
                ylabel="lat",
                suffix="pdf",
            )
            save_field_png(
                str(mode_dir / "diff"),
                lon_g,
                lat_g,
                z_idw - z_true,
                title=f"IDW - True (wet_target={wet_target:.2f}, mode={mode})",
                xlabel="lon",
                ylabel="lat",
            )

            # Minimal metadata
            (mode_dir / "meta.txt").write_text(
                f"wet_fraction={float(np.mean(links_obs['wet'].to_numpy(bool))):.4f}\n"
                f"dry_as_zero={dry_as_zero}\n"
                f"idw_near={args.idw_near}\n"
                f"idw_power={args.idw_power}\n"
                f"idw_dist_m={args.idw_dist_m}\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
