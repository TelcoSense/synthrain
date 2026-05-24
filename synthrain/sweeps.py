from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from synthrain.run_logging import log_error, log_info, log_warn


DEFAULT_IDW_SWEEP = {
    "powers": "1,2,3",
    "nears": "4,8,12",
    "dists": "10000,30000,60000",
    "n_sites_list": "50,75,100",
    "seeds": "0,1,2",
    "wet_targets": "0.1,0.33,0.5",
}

IDW_SWEEP_PRESETS = {
    "quick": {
        "powers": "1,2,3",
        "nears": "4,8",
        "dists": "10000,30000",
        "n_sites_list": "50",
        "seeds": "0",
        "wet_targets": "0.2",
    },
    "poster": {
        "powers": "1,2.5,4",
        "nears": "6,12",
        "dists": "10000,30000",
        "n_sites_list": "50",
        "seeds": "0",
        "wet_targets": "0.2",
        "max_per_page": 12,
        "sheet_rows": 2,
        "sheet_cols": 6,
    },
    "robust": {
        "powers": "1,1.5,2,2.5,3,4",
        "nears": "4,6,8,12,16",
        "dists": "5000,10000,20000,30000,0",
        "n_sites_list": "40,50,75",
        "seeds": "0,1,2,3,4",
        "wet_targets": "0.1,0.2,0.35,0.5",
    },
}


@dataclass
class IdwReportState:
    all_rows: list[dict[str, object]]
    best_rows: list[dict[str, object]]
    global_all_pngs: list[Path]
    global_all_pdfs: list[Path]
    global_rep_pngs: list[Path]
    global_rep_pdfs: list[Path]


def _tagify_float(x: float) -> str:
    xf = float(x)
    if xf.is_integer():
        return str(int(xf))
    s = f"{xf:.12g}"
    return s.rstrip("0").rstrip(".")


def _apply_plot_style(plot_cfg=None) -> dict[str, float]:
    from synthrain.render import apply_plot_style

    return apply_plot_style(plot_cfg)


def _auto_grid(n: int, max_per_page: int = 25) -> tuple[int, int, int]:
    if n <= 0:
        return 1, 1, 1
    k = min(n, max_per_page)
    cols = math.ceil(math.sqrt(k))
    rows = math.ceil(k / cols)
    return rows, cols, rows * cols


def _sheet_grid(
    n: int,
    max_per_page: int = 25,
    sheet_rows: int | None = None,
    sheet_cols: int | None = None,
) -> tuple[int, int, int]:
    if sheet_rows is None and sheet_cols is None:
        return _auto_grid(n, max_per_page=max_per_page)
    if sheet_rows is None or sheet_cols is None:
        raise ValueError("--sheet-rows and --sheet-cols must be used together")
    if sheet_rows <= 0 or sheet_cols <= 0:
        raise ValueError("--sheet-rows and --sheet-cols must be positive")
    per_page = sheet_rows * sheet_cols
    return sheet_rows, sheet_cols, per_page


def _make_pdf_contact_sheet(
    images: list[Path],
    out_pdf: Path,
    title: str,
    max_per_page: int,
    show_titles: bool = True,
    show_subplot_titles: bool | None = None,
    show_figure_title: bool | None = None,
    plot_cfg=None,
    sheet_rows: int | None = None,
    sheet_cols: int | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image

    images = [Path(p) for p in images if Path(p).exists()]
    if not images:
        return

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    n = len(images)
    _, _, per_page = _sheet_grid(
        n,
        max_per_page=max_per_page,
        sheet_rows=sheet_rows,
        sheet_cols=sheet_cols,
    )
    n_pages = max(1, math.ceil(n / per_page))
    style = _apply_plot_style(plot_cfg)
    subplot_titles_enabled = (
        show_titles if show_subplot_titles is None else bool(show_subplot_titles)
    )
    figure_title_enabled = (
        show_titles if show_figure_title is None else bool(show_figure_title)
    )

    with PdfPages(out_pdf) as pdf:
        for page in range(n_pages):
            start = page * per_page
            chunk = images[start : start + per_page]
            rows, cols, _ = _sheet_grid(
                len(chunk),
                max_per_page=max_per_page,
                sheet_rows=sheet_rows,
                sheet_cols=sheet_cols,
            )
            fig = plt.figure(figsize=(cols * 4.0, rows * 3.0))
            for i, img_path in enumerate(chunk):
                ax = fig.add_subplot(rows, cols, i + 1)
                ax.imshow(Image.open(img_path))
                ax.set_axis_off()
                if subplot_titles_enabled:
                    ax.set_title(
                        _contact_sheet_caption(img_path),
                        fontsize=style["contact_sheet_title_size"],
                    )
            if figure_title_enabled:
                fig.suptitle(
                    f"{title} (page {page + 1}/{n_pages})",
                    fontsize=style["figure_title_size"],
                )
            fig.tight_layout(pad=0.2, w_pad=0.08, h_pad=0.12)
            pdf.savefig(fig)
            plt.close(fig)


def _merge_pdfs(pdfs: list[Path], out_pdf: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    try:
        from pypdf import PdfReader, PdfWriter
    except Exception:
        from PyPDF2 import PdfReader, PdfWriter  # type: ignore

    writer = PdfWriter()
    for p in pdfs:
        if not p.exists():
            continue
        reader = PdfReader(str(p))
        for page in reader.pages:
            writer.add_page(page)

    with out_pdf.open("wb") as f:
        writer.write(f)


def _make_pdf_contact_sheet_vector(
    pdfs: list[Path],
    out_pdf: Path,
    title: str,
    max_per_page: int,
    show_titles: bool = True,
    show_subplot_titles: bool | None = None,
    show_figure_title: bool | None = None,
    plot_cfg=None,
    cell_w_pt: float = 252.0,
    cell_h_pt: float = 192.0,
    margin_pt: float = 6.0,
    pad_pt: float = 2.5,
    sheet_rows: int | None = None,
    sheet_cols: int | None = None,
) -> None:
    import tempfile
    import matplotlib.pyplot as plt

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    try:
        from pypdf import PdfReader, PdfWriter, Transformation
        from pypdf._page import PageObject
    except Exception:
        from PyPDF2 import PdfReader, PdfWriter, Transformation  # type: ignore
        from PyPDF2._page import PageObject  # type: ignore

    pdfs = [Path(p) for p in pdfs if Path(p).exists()]
    if not pdfs:
        return

    n = len(pdfs)
    _, _, per_page = _sheet_grid(
        n,
        max_per_page=max_per_page,
        sheet_rows=sheet_rows,
        sheet_cols=sheet_cols,
    )
    n_pages = max(1, math.ceil(n / per_page))
    writer = PdfWriter()
    style = _apply_plot_style(plot_cfg)
    subplot_titles_enabled = (
        show_titles if show_subplot_titles is None else bool(show_subplot_titles)
    )
    figure_title_enabled = (
        show_titles if show_figure_title is None else bool(show_figure_title)
    )
    title_band_pt = (
        max(style["contact_sheet_title_size"] * 1.25, 10.0)
        if subplot_titles_enabled
        else 0.0
    )

    for page_i in range(n_pages):
        start = page_i * per_page
        chunk = pdfs[start : start + per_page]
        rows, cols, _ = _sheet_grid(
            len(chunk),
            max_per_page=max_per_page,
            sheet_rows=sheet_rows,
            sheet_cols=sheet_cols,
        )

        page_w = margin_pt * 2 + cols * cell_w_pt
        page_h = margin_pt * 2 + rows * cell_h_pt
        base = PageObject.create_blank_page(width=page_w, height=page_h)

        for i, pdf_path in enumerate(chunk):
            src = PdfReader(str(pdf_path)).pages[0]
            src_w = float(src.mediabox.width)
            src_h = float(src.mediabox.height)

            col = i % cols
            row_from_top = i // cols
            row = (rows - 1) - row_from_top
            cell_x0 = margin_pt + col * cell_w_pt
            cell_y0 = margin_pt + row * cell_h_pt

            avail_w = cell_w_pt - 2 * pad_pt
            avail_h = cell_h_pt - 2 * pad_pt - title_band_pt
            scale = min(avail_w / src_w, avail_h / src_h)
            dx = cell_x0 + pad_pt + (avail_w - src_w * scale) / 2.0
            dy = cell_y0 + pad_pt + (avail_h - src_h * scale) / 2.0

            t = Transformation().scale(scale, scale).translate(dx, dy)
            base.merge_transformed_page(src, t)

        if subplot_titles_enabled or figure_title_enabled:
            with tempfile.TemporaryDirectory(prefix="synthrain_sheet_") as tmpdir:
                overlay_path = Path(tmpdir) / "overlay.pdf"
                fig = plt.figure(figsize=(page_w / 72.0, page_h / 72.0))
                fig.patch.set_alpha(0.0)
                ax = fig.add_axes([0, 0, 1, 1])
                ax.set_axis_off()

                if figure_title_enabled:
                    fig.suptitle(
                        f"{title} (page {page_i + 1}/{n_pages})",
                        fontsize=style["figure_title_size"],
                        y=0.99,
                    )

                if subplot_titles_enabled:
                    for i, pdf_path in enumerate(chunk):
                        col = i % cols
                        row_from_top = i // cols
                        row = (rows - 1) - row_from_top
                        cell_x0 = margin_pt + col * cell_w_pt
                        cell_y0 = margin_pt + row * cell_h_pt
                        x_frac = (cell_x0 + cell_w_pt / 2.0) / page_w
                        y_frac = (
                            cell_y0 + cell_h_pt - pad_pt - style["contact_sheet_title_size"] * 0.15
                        ) / page_h
                        fig.text(
                            x_frac,
                            y_frac,
                            _contact_sheet_caption(pdf_path),
                            ha="center",
                            va="top",
                            fontsize=style["contact_sheet_title_size"],
                        )

                fig.savefig(overlay_path, transparent=True, pad_inches=0.0)
                plt.close(fig)

                overlay = PdfReader(str(overlay_path)).pages[0]
                base.merge_page(overlay)

        writer.add_page(base)

    with out_pdf.open("wb") as f:
        writer.write(f)


def _contact_sheet_caption(img_path: Path) -> str:
    img_path = Path(img_path)
    stem = img_path.stem
    parent = img_path.parent.name
    grandparent = (
        img_path.parent.parent.name if img_path.parent.parent != img_path.parent else ""
    )
    if stem == "idw_field":
        if parent and parent != "run":
            return parent
        if grandparent:
            return grandparent
    return stem


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _idw_arg(args: argparse.Namespace, name: str) -> object:
    value = getattr(args, name)
    if value is not None:
        return value
    preset_name = getattr(args, "preset", "") or ""
    if preset_name:
        return IDW_SWEEP_PRESETS[preset_name].get(name, DEFAULT_IDW_SWEEP.get(name))
    return DEFAULT_IDW_SWEEP.get(name)


def _normalize_idw_report_args(args: argparse.Namespace) -> None:
    args.max_per_page = int(_idw_arg(args, "max_per_page") or 25)
    if args.sheet_rows is None:
        args.sheet_rows = _idw_arg(args, "sheet_rows")
    if args.sheet_cols is None:
        args.sheet_cols = _idw_arg(args, "sheet_cols")
    if (args.sheet_rows is None) != (args.sheet_cols is None):
        raise ValueError("--sheet-rows and --sheet-cols must be used together")
    if args.sheet_rows is not None:
        args.sheet_rows = int(args.sheet_rows)
        args.sheet_cols = int(args.sheet_cols)
        if args.sheet_rows <= 0 or args.sheet_cols <= 0:
            raise ValueError("--sheet-rows and --sheet-cols must be positive")


def _parse_idw_grid_args(
    args: argparse.Namespace,
) -> tuple[list[float], list[int], list[float], list[int], list[int], list[float]]:
    powers = _parse_float_list(str(_idw_arg(args, "powers")))
    nears = _parse_int_list(str(_idw_arg(args, "nears")))
    dists = _parse_float_list(str(_idw_arg(args, "dists")))
    n_sites_list = _parse_int_list(str(_idw_arg(args, "n_sites_list")))
    seeds = _parse_int_list(str(_idw_arg(args, "seeds")))
    wet_targets = _parse_float_list(str(_idw_arg(args, "wet_targets")))
    return powers, nears, dists, n_sites_list, seeds, wet_targets


def _write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _bool_debug(base_cfg, force_debug: bool) -> bool:
    return True if force_debug else base_cfg.io.debug


def _finite_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _metric_values(rows: list[dict[str, object]], key: str) -> list[float]:
    return [v for row in rows if (v := _finite_float(row.get(key))) is not None]


def _normalize_low(value: object, values: list[float]) -> float | None:
    f = _finite_float(value)
    if f is None:
        return None
    if not values:
        return None
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return 0.0
    return (f - lo) / (hi - lo)


def _add_score_columns(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []

    rmse_values = _metric_values(rows, "rmse")
    mae_values = _metric_values(rows, "mae")
    bias_values = [abs(v) for row in rows if (v := _finite_float(row.get("bias"))) is not None]

    scored: list[dict[str, object]] = []
    for row in rows:
        enriched = dict(row)
        rmse_n = _normalize_low(row.get("rmse"), rmse_values)
        mae_n = _normalize_low(row.get("mae"), mae_values)
        bias = _finite_float(row.get("bias"))
        bias_n = _normalize_low(abs(bias) if bias is not None else None, bias_values)
        invalid = 1.0 - (_finite_float(row.get("valid_pixel_fraction")) or 0.0)
        wet_miss = _finite_float(row.get("wet_miss_rate"))
        dry_false = _finite_float(row.get("dry_false_rain_rate"))
        wet_miss_penalty = 0.0 if wet_miss is None else wet_miss
        dry_false_penalty = 0.0 if dry_false is None else dry_false

        parts = [
            (0.45, rmse_n),
            (0.20, mae_n),
            (0.15, invalid),
            (0.10, wet_miss_penalty),
            (0.05, dry_false_penalty),
            (0.05, bias_n),
        ]
        weighted = [(w, v) for w, v in parts if v is not None]
        enriched["balanced_score"] = (
            sum(w * float(v) for w, v in weighted) / sum(w for w, _ in weighted)
            if weighted
            else None
        )
        enriched["detection_score"] = wet_miss_penalty + dry_false_penalty + invalid
        scored.append(enriched)
    return scored


def _rank_sort_key(row: dict[str, object], metric: str) -> tuple[int, float]:
    if metric == "valid_pixel_fraction":
        value = _finite_float(row.get(metric))
        return (1, float("inf")) if value is None else (0, -value)
    value = _finite_float(row.get(metric))
    return (1, float("inf")) if value is None else (0, value)


def _add_rank_columns(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    enriched = [dict(row) for row in rows]
    by_id = {id(row): row for row in enriched}
    for col, metric in [
        ("rank_rmse", "rmse"),
        ("rank_balanced", "balanced_score"),
        ("rank_detection", "detection_score"),
    ]:
        for rank, row in enumerate(sorted(enriched, key=lambda r: _rank_sort_key(r, metric)), start=1):
            by_id[id(row)][col] = rank
    return enriched


def _rmse_key(row: dict[str, object]) -> tuple[int, float]:
    value = row.get("rmse")
    if value is None:
        return (1, float("inf"))
    try:
        f = float(value)
    except (TypeError, ValueError):
        return (1, float("inf"))
    if math.isnan(f):
        return (1, float("inf"))
    return (0, f)


def _rank_rows(
    rows: list[dict[str, object]], root: Path, ranking_metric: str = "rmse"
) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    scored_rows = _add_rank_columns(_add_score_columns(rows))
    for rank, row in enumerate(
        sorted(scored_rows, key=lambda r: _rank_sort_key(r, ranking_metric)), start=1
    ):
        out_dir = Path(str(row["out_dir"]))
        out_path = out_dir if out_dir.is_absolute() else root / out_dir
        enriched = dict(row)
        enriched["run_dir"] = str(out_path.resolve().relative_to(root.resolve()))
        enriched["rank"] = rank
        enriched["ranking_metric"] = ranking_metric
        enriched["is_best"] = rank == 1
        enriched["out_dir"] = enriched["run_dir"]
        ranked.append(enriched)
    return ranked


def _compact_leaderboard_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = [
        "scenario_tag",
        "run_tag",
        "rank",
        "rank_rmse",
        "rank_balanced",
        "rank_detection",
        "is_best",
        "ranking_metric",
        "balanced_score",
        "detection_score",
        "rmse",
        "mae",
        "bias",
        "pearson_r",
        "valid_pixel_fraction",
        "wet_hit_rate",
        "dry_false_rain_rate",
        "n_sites",
        "wet_target",
        "idw_power",
        "idw_near",
        "idw_dist_m",
        "interp_style",
        "run_dir",
    ]
    return [{k: row.get(k) for k in keys if k in row} for row in rows]


def _compact_comparison_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = [
        "scenario_tag",
        "run_tag",
        "rmse",
        "mae",
        "bias",
        "pearson_r",
        "valid_pixel_fraction",
        "wet_hit_rate",
        "wet_miss_rate",
        "dry_false_rain_rate",
        "n_sites",
        "wet_target",
        "idw_power",
        "idw_near",
        "idw_dist_m",
        "interp_style",
        "run_dir",
    ]
    return [{k: row.get(k) for k in keys if k in row} for row in rows]


def _best_run_payload(row: dict[str, object]) -> dict[str, object]:
    return {
        "scenario_tag": row.get("scenario_tag"),
        "run_tag": row.get("run_tag"),
        "rank": row.get("rank"),
        "ranking_metric": row.get("ranking_metric"),
        "metrics": {
            "balanced_score": row.get("balanced_score"),
            "detection_score": row.get("detection_score"),
            "rmse": row.get("rmse"),
            "mae": row.get("mae"),
            "bias": row.get("bias"),
            "pearson_r": row.get("pearson_r"),
            "valid_pixel_fraction": row.get("valid_pixel_fraction"),
            "wet_hit_rate": row.get("wet_hit_rate"),
            "wet_miss_rate": row.get("wet_miss_rate"),
            "dry_false_rain_rate": row.get("dry_false_rain_rate"),
        },
        "parameters": {
            "n_sites": row.get("n_sites"),
            "wet_target": row.get("wet_target"),
            "idw_power": row.get("idw_power"),
            "idw_near": row.get("idw_near"),
            "idw_dist_m": row.get("idw_dist_m"),
            "interp_style": row.get("interp_style"),
            "path_samples": row.get("path_samples"),
        },
        "artifacts": {
            "run_dir": row.get("run_dir"),
            "png": f"{row.get('run_dir')}/idw_field.png",
            "pdf": f"{row.get('run_dir')}/idw_field.pdf",
            "scenario_json": f"{row.get('run_dir')}/scenario.json",
        },
    }


def _select_mode_items(
    all_items: list[Path], rep_items: list[Path], mode: str, scenario_count: int
) -> list[Path]:
    if mode == "rep":
        return rep_items
    if mode == "all":
        return all_items
    return all_items if scenario_count == 1 else rep_items


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def _build_parameter_robustness_rows(
    rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    grouped: dict[tuple[float, int, float], list[dict[str, object]]] = {}
    for row in rows:
        power = _finite_float(row.get("idw_power"))
        near = _finite_float(row.get("idw_near"))
        dist = _finite_float(row.get("idw_dist_m"))
        if power is None or near is None or dist is None:
            continue
        grouped.setdefault((power, int(near), dist), []).append(row)

    out: list[dict[str, object]] = []
    for (power, near, dist), group in grouped.items():
        record: dict[str, object] = {
            "idw_power": power,
            "idw_near": near,
            "idw_dist_m": dist,
            "n_runs": len(group),
            "n_scenarios": len({str(r.get("scenario_tag", "")) for r in group}),
        }
        higher_is_better = {"valid_pixel_fraction", "wet_hit_rate"}
        for metric in [
            "rmse",
            "mae",
            "balanced_score",
            "detection_score",
            "valid_pixel_fraction",
            "wet_hit_rate",
            "wet_miss_rate",
            "dry_false_rain_rate",
        ]:
            values = _metric_values(group, metric)
            mean, std = _mean_std(values)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
            if metric in higher_is_better:
                record[f"{metric}_best"] = max(values) if values else None
                record[f"{metric}_worst"] = min(values) if values else None
            else:
                record[f"{metric}_best"] = min(values) if values else None
                record[f"{metric}_worst"] = max(values) if values else None
        out.append(record)

    out = sorted(
        out,
        key=lambda r: (
            _rank_sort_key(r, "balanced_score_mean"),
            _rank_sort_key(r, "rmse_mean"),
        ),
    )
    for rank, row in enumerate(out, start=1):
        row["rank_robust"] = rank
    return out


def _plot_wet_metric_lines(
    rows: list[dict[str, object]],
    out_path: Path,
    show_titles: bool = True,
    plot_cfg=None,
) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return

    style = _apply_plot_style(plot_cfg)
    rows = sorted(rows, key=lambda r: float(r["wet_target"]))
    wet_targets = np.array([float(r["wet_target"]) for r in rows], dtype=float)
    metrics = [
        ("rmse", "RMSE"),
        ("valid_pixel_fraction", "Valid Pixel Fraction"),
        ("wet_hit_rate", "Wet Hit Rate"),
        ("dry_false_rain_rate", "Dry False Rain Rate"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, title) in zip(axes.ravel(), metrics):
        y = np.array(
            [np.nan if r.get(key) is None else float(r[key]) for r in rows], dtype=float
        )
        ax.plot(wet_targets, y, marker="o", linewidth=2)
        if show_titles:
            ax.set_title(title)
        ax.set_xlabel("wet_target")
        ax.set_ylabel(key)
        ax.grid(alpha=0.3)

    if show_titles:
        fig.suptitle("Wet Sweep Metrics", fontsize=style["figure_title_size"])
    fig.tight_layout(pad=0.2, w_pad=0.08, h_pad=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_idw_metric_heatmap(
    rows: list[dict[str, object]],
    metric_key: str,
    title: str,
    out_path: Path,
    show_titles: bool = True,
    plot_cfg=None,
) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return

    style = _apply_plot_style(plot_cfg)
    powers = sorted({float(r["idw_power"]) for r in rows})
    nears = sorted({int(r["idw_near"]) for r in rows})
    dists = sorted({float(r["idw_dist_m"]) for r in rows})

    fig, axes = plt.subplots(
        1,
        len(powers),
        figsize=(max(5, 4 * len(powers)), 4.5),
        squeeze=False,
    )

    for ax, power in zip(axes.ravel(), powers):
        mat = np.full((len(nears), len(dists)), np.nan, dtype=float)
        for row in rows:
            if float(row["idw_power"]) != power:
                continue
            i = nears.index(int(row["idw_near"]))
            j = dists.index(float(row["idw_dist_m"]))
            value = row.get(metric_key)
            mat[i, j] = np.nan if value is None else float(value)

        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        if show_titles:
            ax.set_title(f"p={_tagify_float(power)}")
        ax.set_xticks(range(len(dists)))
        ax.set_xticklabels([_tagify_float(d) for d in dists], rotation=45, ha="right")
        ax.set_yticks(range(len(nears)))
        ax.set_yticklabels([str(n) for n in nears])
        ax.set_xlabel("idw_dist_m")
        ax.set_ylabel("idw_near")

        for i in range(len(nears)):
            for j in range(len(dists)):
                if np.isfinite(mat[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=max(
                            8.0 * float(getattr(plot_cfg, "font_scale", 1.0)),
                            1.0,
                        ),
                    )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if show_titles:
        fig.suptitle(title, fontsize=style["figure_title_size"])
    fig.tight_layout(pad=0.2, w_pad=0.08, h_pad=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _idw_paths_from_rows(rows: list[dict[str, object]], out_root: Path) -> tuple[list[Path], list[Path]]:
    png_paths: list[Path] = []
    pdf_paths: list[Path] = []
    for row in rows:
        run_dir = row.get("run_dir") or row.get("out_dir")
        if not run_dir:
            continue
        run_path = out_root / str(run_dir)
        png = run_path / "idw_field.png"
        pdf = run_path / "idw_field.pdf"
        if png.exists():
            png_paths.append(png)
        if pdf.exists():
            pdf_paths.append(pdf)
    return png_paths, pdf_paths


def _write_idw_parameter_robustness(
    rows: list[dict[str, object]], out_root: Path, global_reports: Path
) -> list[dict[str, object]]:
    robustness_rows = _build_parameter_robustness_rows(_add_score_columns(rows))
    if robustness_rows:
        _write_rows_csv(out_root / "parameter_robustness.csv", robustness_rows)
        _write_rows_csv(global_reports / "parameter_robustness.csv", robustness_rows)
    return robustness_rows


def _empty_idw_report_state() -> IdwReportState:
    return IdwReportState([], [], [], [], [], [])


def _add_idw_best_run(
    state: IdwReportState,
    best_row: dict[str, object],
    scenario_root: Path,
    out_root: Path,
) -> None:
    state.best_rows.append(best_row)
    _write_json(scenario_root / "best_run.json", _best_run_payload(best_row))
    best_pngs, best_pdfs = _idw_paths_from_rows([best_row], out_root)
    state.global_rep_pngs.extend(best_pngs)
    state.global_rep_pdfs.extend(best_pdfs)


def _write_idw_scenario_reports(
    *,
    rows: list[dict[str, object]],
    png_paths: list[Path],
    pdf_paths: list[Path],
    scenario_reports: Path,
    title_tag: str,
    args: argparse.Namespace,
    base_cfg,
    sort_by_parent: bool = False,
) -> None:
    show_titles = base_cfg.plot.show_titles
    path_key = (lambda p: p.parent.name) if sort_by_parent else (lambda p: p.name)
    if not args.skip_pdf and png_paths:
        _make_pdf_contact_sheet(
            images=sorted(png_paths, key=path_key),
            out_pdf=scenario_reports / args.pdf_name,
            title=f"IDW Sweep - {title_tag}",
            max_per_page=args.max_per_page,
            show_titles=show_titles,
            show_subplot_titles=True,
            show_figure_title=show_titles,
            plot_cfg=base_cfg.plot,
            sheet_rows=args.sheet_rows,
            sheet_cols=args.sheet_cols,
        )
    if not args.skip_vector_merge and pdf_paths:
        _merge_pdfs(
            sorted(pdf_paths, key=path_key),
            scenario_reports / args.vector_merge_name,
        )
    if not args.skip_vector_sheet and pdf_paths:
        _make_pdf_contact_sheet_vector(
            pdfs=sorted(pdf_paths, key=path_key),
            out_pdf=scenario_reports / args.vector_sheet_name,
            title=f"IDW Sweep - {title_tag}",
            max_per_page=args.max_per_page,
            show_titles=show_titles,
            show_subplot_titles=True,
            show_figure_title=show_titles,
            plot_cfg=base_cfg.plot,
            sheet_rows=args.sheet_rows,
            sheet_cols=args.sheet_cols,
        )
    if rows:
        for metric_key, label, filename in [
            ("rmse", "RMSE", "metric_heatmap_rmse.png"),
            (
                "valid_pixel_fraction",
                "Valid Pixel Fraction",
                "metric_heatmap_valid_pixel_fraction.png",
            ),
            ("balanced_score", "Balanced Score", "metric_heatmap_balanced_score.png"),
        ]:
            _plot_idw_metric_heatmap(
                rows,
                metric_key=metric_key,
                title=f"{title_tag} - {label} by IDW Parameters",
                out_path=scenario_reports / filename,
                show_titles=show_titles,
                plot_cfg=base_cfg.plot,
            )


def _write_idw_global_outputs(
    *,
    state: IdwReportState,
    out_root: Path,
    global_reports: Path,
    args: argparse.Namespace,
    base_cfg,
    scenario_count: int,
) -> None:
    show_titles = base_cfg.plot.show_titles
    sorted_best = sorted(
        state.best_rows, key=lambda r: _rank_sort_key(r, args.ranking_metric)
    )
    _write_rows_csv(out_root / "leaderboard.csv", _compact_leaderboard_rows(sorted_best))
    _write_rows_csv(out_root / args.summary_name, _compact_leaderboard_rows(sorted_best))
    _write_rows_csv(
        global_reports / "all_runs.csv",
        sorted(state.all_rows, key=lambda r: _rank_sort_key(r, args.ranking_metric)),
    )
    _write_rows_csv(global_reports / "best_per_scenario.csv", sorted_best)
    _write_idw_parameter_robustness(state.all_rows, out_root, global_reports)

    png_for_global = _select_mode_items(
        all_items=state.global_all_pngs,
        rep_items=state.global_rep_pngs,
        mode=args.global_png_sheet_mode,
        scenario_count=scenario_count,
    )
    vec_for_global = _select_mode_items(
        all_items=state.global_all_pdfs,
        rep_items=state.global_rep_pdfs,
        mode=args.global_vector_sheet_mode,
        scenario_count=scenario_count,
    )

    if not args.skip_pdf and png_for_global:
        global_png_name = (
            "all_runs_contact_sheet.pdf"
            if args.global_png_sheet_mode == "all"
            else "best_per_scenario_contact_sheet.pdf"
        )
        keep_subplot_titles = global_png_name == "best_per_scenario_contact_sheet.pdf"
        _make_pdf_contact_sheet(
            images=sorted(png_for_global, key=lambda p: str(p)),
            out_pdf=global_reports / global_png_name,
            title="IDW Sweep - Global Representative Fields",
            max_per_page=args.max_per_page,
            show_titles=show_titles,
            show_subplot_titles=keep_subplot_titles or show_titles,
            show_figure_title=show_titles,
            plot_cfg=base_cfg.plot,
            sheet_rows=args.sheet_rows,
            sheet_cols=args.sheet_cols,
        )
    if not args.skip_vector_sheet and vec_for_global:
        global_vec_name = (
            "all_runs_vector_sheet.pdf"
            if args.global_vector_sheet_mode == "all"
            else args.global_vector_sheet_name
        )
        _make_pdf_contact_sheet_vector(
            pdfs=sorted(vec_for_global, key=lambda p: str(p)),
            out_pdf=global_reports / global_vec_name,
            title="IDW Sweep - Global Representative Fields",
            max_per_page=args.max_per_page,
            show_titles=show_titles,
            show_subplot_titles=show_titles,
            show_figure_title=show_titles,
            plot_cfg=base_cfg.plot,
            sheet_rows=args.sheet_rows,
            sheet_cols=args.sheet_cols,
        )
    if not args.skip_vector_merge and state.global_all_pdfs:
        _merge_pdfs(
            sorted(state.global_all_pdfs, key=lambda p: str(p)),
            global_reports / args.global_vector_merge_name,
        )


def _rerender_idw_reports(args: argparse.Namespace, base_cfg, out_root: Path) -> int:
    scenarios_root = out_root / "scenarios"
    global_reports = out_root / "reports" / "global"
    state = _empty_idw_report_state()

    if not scenarios_root.exists():
        log_error("IDW-SWEEP", f"cannot rerender; missing scenarios directory: {scenarios_root}")
        return 2

    for scenario_root in sorted(p for p in scenarios_root.iterdir() if p.is_dir()):
        scenario_reports = scenario_root / "reports"
        rows = _read_rows_csv(scenario_root / args.summary_name)
        if not rows:
            log_warn("IDW-SWEEP", f"skipping rerender; missing summary: {scenario_root}")
            continue
        rows = _rank_rows(rows, out_root, ranking_metric=args.ranking_metric)
        _write_rows_csv(scenario_root / args.summary_name, rows)
        state.all_rows.extend(rows)

        png_paths, pdf_paths = _idw_paths_from_rows(rows, out_root)
        state.global_all_pngs.extend(png_paths)
        state.global_all_pdfs.extend(pdf_paths)
        if rows:
            _add_idw_best_run(state, rows[0], scenario_root, out_root)

        _write_idw_scenario_reports(
            rows=rows,
            png_paths=png_paths,
            pdf_paths=pdf_paths,
            scenario_reports=scenario_reports,
            title_tag=scenario_root.name,
            args=args,
            base_cfg=base_cfg,
            sort_by_parent=True,
        )
    _write_idw_global_outputs(
        state=state,
        out_root=out_root,
        global_reports=global_reports,
        args=args,
        base_cfg=base_cfg,
        scenario_count=len(state.best_rows),
    )
    log_info("IDW-SWEEP", f"rerendered reports from existing runs: {out_root}")
    return 0


def _build_manifest(
    sweep_type: str,
    out_root: Path,
    scenario_tags: list[str],
    total_runs: int,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "sweep_type": sweep_type,
        "root": str(out_root),
        "scenario_count": len(scenario_tags),
        "total_runs": total_runs,
        "layout": {
            "leaderboard_csv": "leaderboard.csv",
            "scenarios_dir": "scenarios",
            "reports_dir": "reports",
        },
        "scenarios": scenario_tags,
    }
    if extra:
        payload.update(extra)
    return payload


def build_wet_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="configs/config.ini", help="Base INI file")
    ap.add_argument(
        "--out-root", default="outputs_wet_sweep", help="Root folder for all runs"
    )
    ap.add_argument(
        "--wet-targets",
        default="",
        help="Comma-separated wet_target values. If empty, uses [sweep] wet_targets from base config.",
    )
    ap.add_argument("--seed", type=int, default=None, help="Override [io] seed")
    ap.add_argument(
        "--n-sites", type=int, default=None, help="Override [network] n_sites"
    )
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--idw-png-dirname", default="_idw_png", help="Deprecated; ignored")
    ap.add_argument("--idw-pdf-dirname", default="_idw_pdf", help="Deprecated; ignored")
    ap.add_argument("--png-sheet-name", default="wet_sweep_contact_sheet.pdf")
    ap.add_argument("--vector-sheet-name", default="wet_sweep_contact_sheet_vector.pdf")
    ap.add_argument("--comparison-name", default="comparison.csv")
    ap.add_argument("--summary-name", default="summary.csv")
    ap.add_argument("--skip-png-sheet", action="store_true")
    ap.add_argument("--skip-vector-sheet", action="store_true")
    ap.add_argument("--max-per-page", type=int, default=25)
    ap.add_argument("--keep-temp-configs", action="store_true")
    return ap


def run_wet_sweep(args: argparse.Namespace) -> int:
    from synthrain.config import load_scenario_config
    from synthrain.outputs import write_scenario_outputs
    from synthrain.scenario import run_scenario

    base_config = Path(args.base_config).resolve()
    if not base_config.exists():
        log_error("WET-SWEEP", f"base config not found: {base_config}")
        return 2

    if getattr(args, "keep_temp_configs", False):
        log_warn("WET-SWEEP", "--keep-temp-configs is deprecated and ignored")
    if args.idw_png_dirname != "_idw_png" or args.idw_pdf_dirname != "_idw_pdf":
        log_warn(
            "WET-SWEEP",
            "--idw-png-dirname/--idw-pdf-dirname are deprecated and ignored",
        )

    base_cfg = load_scenario_config(base_config)
    show_titles = base_cfg.plot.show_titles
    wet_targets = (
        _parse_float_list(args.wet_targets)
        if args.wet_targets.strip()
        else list(base_cfg.sweep.wet_targets)
    )
    if not wet_targets:
        log_error("WET-SWEEP", "no wet_targets provided and none found in config")
        return 2

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log_info("WET-SWEEP", f"output root: {out_root}")

    scenarios_root = out_root / "scenarios"
    reports_root = out_root / "reports"
    global_reports = reports_root / "global"
    scenario_tags: list[str] = []
    comparison_rows: list[dict[str, object]] = []
    png_paths: list[Path] = []
    pdf_paths: list[Path] = []

    for wt in wet_targets:
        tag = f"wet{_tagify_float(wt)}"
        scenario_tags.append(tag)
        scenario_root = scenarios_root / tag
        run_out = scenario_root / "run"
        scenario_cfg = replace(
            base_cfg,
            io=replace(
                base_cfg.io,
                out=str(run_out),
                seed=args.seed if args.seed is not None else base_cfg.io.seed,
                debug=_bool_debug(base_cfg, args.debug),
            ),
            network=replace(
                base_cfg.network,
                n_sites=(
                    args.n_sites
                    if args.n_sites is not None
                    else base_cfg.network.n_sites
                ),
            ),
            wet=replace(base_cfg.wet, wet_target=wt),
            plot=replace(base_cfg.plot, title_name=tag),
        )

        try:
            result = run_scenario(scenario_cfg)
            write_scenario_outputs(result)
        except Exception as exc:
            log_error("WET-SWEEP", f"run failed: {tag} ({exc})")
            continue

        row = result.to_summary_row(
            {"sweep_type": "wet", "scenario_tag": tag, "run_tag": "run"}
        )
        comparison_row = dict(row)
        comparison_row["run_dir"] = str(
            run_out.resolve().relative_to(out_root.resolve())
        )
        comparison_row["out_dir"] = comparison_row["run_dir"]
        comparison_rows.append(comparison_row)

        _write_rows_csv(scenario_root / args.summary_name, [comparison_row])

        png_path = run_out / "idw_field.png"
        pdf_path = run_out / "idw_field.pdf"
        if png_path.exists():
            png_paths.append(png_path)
        if pdf_path.exists():
            pdf_paths.append(pdf_path)

    comparison_rows = sorted(comparison_rows, key=lambda r: float(r["wet_target"]))
    compact_rows = _compact_comparison_rows(comparison_rows)
    _write_rows_csv(out_root / args.comparison_name, compact_rows)
    _write_rows_csv(out_root / args.summary_name, compact_rows)
    _write_rows_csv(global_reports / "all_runs.csv", comparison_rows)

    if not args.skip_png_sheet and png_paths:
        _make_pdf_contact_sheet(
            images=png_paths,
            out_pdf=global_reports / args.png_sheet_name,
            title="Wet Sweep - IDW Field",
            max_per_page=args.max_per_page,
            show_titles=show_titles,
            show_subplot_titles=True,
            show_figure_title=show_titles,
            plot_cfg=base_cfg.plot,
        )
    if not args.skip_vector_sheet and pdf_paths:
        _make_pdf_contact_sheet_vector(
            pdfs=pdf_paths,
            out_pdf=global_reports / args.vector_sheet_name,
            title="Wet Sweep - IDW Field",
            max_per_page=args.max_per_page,
            show_titles=show_titles,
            show_subplot_titles=True,
            show_figure_title=show_titles,
            plot_cfg=base_cfg.plot,
        )
    if comparison_rows:
        _plot_wet_metric_lines(
            comparison_rows,
            global_reports / "wet_sweep_metrics.png",
            show_titles=show_titles,
            plot_cfg=base_cfg.plot,
        )

    manifest = _build_manifest(
        "wet",
        out_root,
        scenario_tags,
        total_runs=len(comparison_rows),
        extra={
            "layout": {
                "comparison_csv": args.comparison_name,
                "summary_csv": args.summary_name,
                "scenarios_dir": "scenarios",
                "reports_dir": "reports",
            },
            "reports": {"global_dir": "reports/global"},
        },
    )
    _write_json(out_root / "manifest.json", manifest)
    return 0


def build_idw_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="configs/config.ini", help="Base INI file")
    ap.add_argument(
        "--out-root", default="outputs_sweep", help="Root folder for all runs"
    )
    ap.add_argument(
        "--debug", action="store_true", help="Enable debug mode in run_scenario"
    )
    ap.add_argument(
        "--preset",
        choices=sorted(IDW_SWEEP_PRESETS),
        default="",
        help="Named IDW sweep grid. Explicit parameter lists override preset values.",
    )
    ap.add_argument(
        "--powers", default=None, help="Comma-separated idw_power values"
    )
    ap.add_argument("--nears", default=None, help="Comma-separated idw_near values")
    ap.add_argument(
        "--dists",
        default=None,
        help="Comma-separated idw_dist_m values; use 0 for unlimited",
    )
    ap.add_argument(
        "--n-sites-list", default=None, help="Comma-separated n_sites values"
    )
    ap.add_argument("--seeds", default=None, help="Comma-separated seed values")
    ap.add_argument(
        "--wet-targets",
        default=None,
        help="Comma-separated wet_target values",
    )
    ap.add_argument("--idw-pdf-dirname", default="_idw_pdf", help="Deprecated; ignored")
    ap.add_argument("--idw-png-dirname", default="_idw_png", help="Deprecated; ignored")
    ap.add_argument("--pdf-name", default="idw_sweep_contact_sheet.pdf")
    ap.add_argument("--vector-merge-name", default="idw_fields_merged_vector.pdf")
    ap.add_argument(
        "--global-vector-merge-name",
        default="ALL_SCENARIOS_idw_fields_merged_vector.pdf",
    )
    ap.add_argument("--skip-vector-merge", action="store_true")
    ap.add_argument("--vector-sheet-name", default="idw_sweep_contact_sheet_vector.pdf")
    ap.add_argument(
        "--global-vector-sheet-name",
        default="ALL_SCENARIOS_idw_fields_contact_sheet_vector.pdf",
    )
    ap.add_argument("--skip-vector-sheet", action="store_true")
    ap.add_argument("--summary-name", default="summary.csv")
    ap.add_argument("--keep-temp-configs", action="store_true")
    ap.add_argument("--skip-pdf", action="store_true", help="Skip PNG contact sheets")
    ap.add_argument("--max-per-page", type=int, default=None)
    ap.add_argument("--sheet-rows", type=int, default=None)
    ap.add_argument("--sheet-cols", type=int, default=None)
    ap.add_argument(
        "--ranking-metric",
        choices=["rmse", "balanced_score", "detection_score", "valid_pixel_fraction"],
        default="rmse",
        help="Metric used to select best runs. Lower is better except valid_pixel_fraction.",
    )
    ap.add_argument(
        "--rerender-only",
        action="store_true",
        help="Rebuild reports from existing run outputs and summary CSV files.",
    )
    ap.add_argument(
        "--global-vector-sheet-mode",
        choices=["auto", "rep", "all"],
        default="auto",
        help="Global vector grid: auto=all runs if only 1 scenario else rep; rep=best per scenario; all=all runs.",
    )
    ap.add_argument(
        "--global-png-sheet-mode",
        choices=["auto", "rep", "all"],
        default="auto",
        help="Global PNG grid: auto=all runs if only 1 scenario else rep; rep=best per scenario; all=all runs.",
    )
    return ap


def run_idw_sweep(args: argparse.Namespace) -> int:
    from synthrain.config import load_scenario_config
    from synthrain.outputs import write_scenario_outputs
    from synthrain.scenario import run_scenario

    base_config = Path(args.base_config).resolve()
    if not base_config.exists():
        log_error("IDW-SWEEP", f"base config not found: {base_config}")
        return 2

    if getattr(args, "keep_temp_configs", False):
        log_warn("IDW-SWEEP", "--keep-temp-configs is deprecated and ignored")
    if args.idw_png_dirname != "_idw_png" or args.idw_pdf_dirname != "_idw_pdf":
        log_warn(
            "IDW-SWEEP",
            "--idw-png-dirname/--idw-pdf-dirname are deprecated and ignored",
        )
    try:
        _normalize_idw_report_args(args)
    except ValueError as exc:
        log_error("IDW-SWEEP", str(exc))
        return 2

    base_cfg = load_scenario_config(base_config)
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log_info("IDW-SWEEP", f"output root: {out_root}")
    if args.preset:
        log_info("IDW-SWEEP", f"using preset: {args.preset}")

    if args.rerender_only:
        return _rerender_idw_reports(args, base_cfg, out_root)

    powers, nears, dists, n_sites_list, seeds, wet_targets = _parse_idw_grid_args(args)

    idw_combos = list(itertools.product(powers, nears, dists))
    outer_combos = list(itertools.product(n_sites_list, wet_targets, seeds))
    if args.debug:
        log_info("IDW-SWEEP", f"outer scenarios: {len(outer_combos)}")
        log_info("IDW-SWEEP", f"idw combos per scenario: {len(idw_combos)}")
        log_info("IDW-SWEEP", f"total runs: {len(outer_combos) * len(idw_combos)}")

    scenarios_root = out_root / "scenarios"
    global_reports = out_root / "reports" / "global"

    state = _empty_idw_report_state()
    scenario_tags: list[str] = []

    for n_sites, wet_target, seed in outer_combos:
        scen_tag = f"nsites{n_sites}_wet{_tagify_float(wet_target)}_seed{seed}"
        scenario_tags.append(scen_tag)
        scenario_root = scenarios_root / scen_tag
        runs_root = scenario_root / "runs"
        scenario_reports = scenario_root / "reports"
        scenario_root.mkdir(parents=True, exist_ok=True)
        log_info("IDW-SWEEP", f"scenario directory ({scen_tag}): {scenario_root}")

        run_rows: list[dict[str, object]] = []
        png_paths: list[Path] = []
        pdf_paths: list[Path] = []

        for power, near, dist in idw_combos:
            tag = f"p{_tagify_float(power)}_n{near}_d{_tagify_float(dist)}"
            run_out = runs_root / tag
            scenario_cfg = replace(
                base_cfg,
                io=replace(
                    base_cfg.io,
                    out=str(run_out),
                    seed=seed,
                    debug=_bool_debug(base_cfg, args.debug),
                ),
                network=replace(base_cfg.network, n_sites=n_sites),
                wet=replace(base_cfg.wet, wet_target=wet_target),
                interp=replace(
                    base_cfg.interp,
                    idw_power=power,
                    idw_near=near,
                    idw_dist_m=dist,
                ),
                plot=replace(base_cfg.plot, title_name=f"{scen_tag}_{tag}"),
            )

            try:
                result = run_scenario(scenario_cfg)
                write_scenario_outputs(result)
            except Exception as exc:
                log_error("IDW-SWEEP", f"run failed: {scen_tag}/{tag} ({exc})")
                continue

            row = result.to_summary_row(
                {"scenario_tag": scen_tag, "run_tag": tag, "sweep_type": "idw"}
            )
            run_rows.append(row)

            png_path = run_out / "idw_field.png"
            pdf_path = run_out / "idw_field.pdf"
            if png_path.exists():
                png_paths.append(png_path)
                state.global_all_pngs.append(png_path)
            else:
                log_warn("IDW-SWEEP", f"missing {png_path}")
            if pdf_path.exists():
                pdf_paths.append(pdf_path)
                state.global_all_pdfs.append(pdf_path)
            else:
                log_warn("IDW-SWEEP", f"missing {pdf_path}")

        ranked_rows = _rank_rows(run_rows, out_root, ranking_metric=args.ranking_metric)
        state.all_rows.extend(ranked_rows)
        _write_rows_csv(scenario_root / args.summary_name, ranked_rows)

        if ranked_rows:
            _add_idw_best_run(state, ranked_rows[0], scenario_root, out_root)

        _write_idw_scenario_reports(
            rows=ranked_rows,
            png_paths=png_paths,
            pdf_paths=pdf_paths,
            scenario_reports=scenario_reports,
            title_tag=scen_tag,
            args=args,
            base_cfg=base_cfg,
        )

    _write_idw_global_outputs(
        state=state,
        out_root=out_root,
        global_reports=global_reports,
        args=args,
        base_cfg=base_cfg,
        scenario_count=len(outer_combos),
    )

    manifest = _build_manifest(
        "idw",
        out_root,
        scenario_tags,
        total_runs=len(state.all_rows),
        extra={
            "reports": {"global_dir": "reports/global"},
            "ranking_metric": args.ranking_metric,
            "preset": args.preset or None,
            "parameter_grid": {
                "powers": powers,
                "nears": nears,
                "dists": dists,
                "n_sites_list": n_sites_list,
                "seeds": seeds,
                "wet_targets": wet_targets,
            },
        },
    )
    _write_json(out_root / "manifest.json", manifest)
    return 0
