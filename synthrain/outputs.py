from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from synthrain.run_logging import format_path, log_info
from synthrain.scenario import ScenarioResult
from synthrain.csv_export import CsvSpecMinimal, export_synth_minimal_csv
from synthrain.render import save_field_image, save_links_image


def _fault_counts(links_obs) -> dict[str, int]:
    if "fault_label" not in links_obs.columns:
        return {}
    counts = links_obs["fault_label"].value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def build_metadata(result: ScenarioResult) -> dict[str, object]:
    return {
        "config": result.config.to_dict(),
        "bbox_ll": result.bbox_ll,
        "bbox_m": result.grid["bbox_m"],
        "n_sites": int(len(result.sites)),
        "n_links": int(len(result.links_obs)),
        "wet_fraction": float(np.mean(result.links_obs["wet"].to_numpy(bool))),
        "available_fraction": float(np.mean(result.links_obs["available"].to_numpy(bool))),
        "metrics": result.metrics.to_dict(),
        "fault_counts": _fault_counts(result.links_obs),
    }


def write_scenario_outputs(result: ScenarioResult) -> dict[str, object]:
    out_dir = Path(result.config.io.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_info("SCENARIO", f"writing outputs: {format_path(out_dir)}")

    lon_g = result.grid["lon_g"]
    lat_g = result.grid["lat_g"]
    z_true = result.z_true
    z_idw = result.z_idw
    diff = result.diff

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
        min_rain=result.config.rain.min_rain,
        suffix="png",
    )
    save_links_image(
        str(out_dir / "links"),
        result.links_obs,
        bbox=result.bbox_ll,
        title="Link centers (wet/dry/unavailable)",
        suffix="png",
    )
    for suffix in ("pdf", "png"):
        save_field_image(
            str(out_dir / "idw_field"),
            lon_g,
            lat_g,
            z_idw,
            title=result.config.plot.title_name,
            vmin=vmin,
            vmax=vmax,
            links=result.links_obs,
            show_links=True,
            xlabel="lon",
            ylabel="lat",
            min_rain=result.config.rain.min_rain,
            suffix=suffix,
        )
    save_field_image(
        str(out_dir / "diff"),
        lon_g,
        lat_g,
        diff,
        title="IDW - True (mm/h)",
        xlabel="lon",
        ylabel="lat",
        min_rain=result.config.rain.min_rain,
        suffix="png",
    )

    if result.config.csv.export_csv:
        export_synth_minimal_csv(
            result.links_obs,
            str(out_dir / "calc_dataset_synth.csv"),
            CsvSpecMinimal(
                start_time=result.config.csv.csv_start,
                n_steps=result.config.csv.csv_steps,
                step_minutes=result.config.csv.csv_step_min,
                seed=result.config.io.seed + 40,
            ),
        )

    metadata = build_metadata(result)
    (out_dir / "scenario.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
