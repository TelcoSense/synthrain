from __future__ import annotations

import argparse
import itertools
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

from synthrain.run_logging import log_error, log_info, log_warn


def _fmt_ini_bool(v: bool) -> str:
    return "true" if bool(v) else "false"


def _fmt_ini_number(x: float) -> str:
    xf = float(x)
    if xf.is_integer():
        return str(int(xf))
    return f"{xf:.12g}"


def _tagify_float(x: float) -> str:
    xf = float(x)
    if xf.is_integer():
        return str(int(xf))
    s = f"{xf:.12g}"
    return s.rstrip("0").rstrip(".")


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def _set_ini_value(text: str, section: str, key: str, value: str) -> str:
    sec_pat = re.compile(rf"(?ms)^\[{re.escape(section)}\]\s*(.*?)(?=^\[|\Z)")
    m = sec_pat.search(text)
    if not m:
        return text.rstrip() + f"\n\n[{section}]\n{key} = {value}\n"

    body = m.group(1)
    key_pat = re.compile(rf"(?m)^(?P<prefix>\s*{re.escape(key)}\s*=\s*)(?P<val>.*)$")
    if key_pat.search(body):
        new_body = key_pat.sub(rf"\g<prefix>{value}", body, count=1)
    else:
        new_body = body.rstrip() + f"\n{key} = {value}\n"

    new_sec_block = f"[{section}]\n{new_body}"
    return text[: m.start()] + new_sec_block + text[m.end() :]


def _auto_grid(n: int, max_per_page: int = 25) -> tuple[int, int, int]:
    if n <= 0:
        return 1, 1, 1
    k = min(n, max_per_page)
    cols = math.ceil(math.sqrt(k))
    rows = math.ceil(k / cols)
    return rows, cols, rows * cols


def _make_pdf_contact_sheet(
    images: list[Path], out_pdf: Path, title: str, max_per_page: int
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    n = len(images)
    _, _, per_page = _auto_grid(n, max_per_page=max_per_page)
    n_pages = max(1, math.ceil(n / per_page))

    with PdfPages(out_pdf) as pdf:
        for page in range(n_pages):
            start = page * per_page
            chunk = images[start : start + per_page]

            r, c, _ = _auto_grid(len(chunk), max_per_page=max_per_page)
            fig = plt.figure(figsize=(c * 4.2, r * 3.2))

            for i, img_path in enumerate(chunk):
                ax = fig.add_subplot(r, c, i + 1)
                ax.imshow(Image.open(img_path))
                ax.set_axis_off()
                ax.set_title(img_path.stem, fontsize=9)

            fig.suptitle(f"{title} (page {page+1}/{n_pages})", fontsize=14)
            fig.tight_layout()
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
    cell_w_pt: float = 260.0,
    cell_h_pt: float = 200.0,
    margin_pt: float = 18.0,
    pad_pt: float = 8.0,
) -> None:
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
    _, _, per_page = _auto_grid(n, max_per_page=max_per_page)
    n_pages = max(1, math.ceil(n / per_page))

    writer = PdfWriter()

    for page_i in range(n_pages):
        start = page_i * per_page
        chunk = pdfs[start : start + per_page]

        rows, cols, _ = _auto_grid(len(chunk), max_per_page=max_per_page)

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
            avail_h = cell_h_pt - 2 * pad_pt
            scale = min(avail_w / src_w, avail_h / src_h)

            dx = cell_x0 + pad_pt + (avail_w - src_w * scale) / 2.0
            dy = cell_y0 + pad_pt + (avail_h - src_h * scale) / 2.0

            t = Transformation().scale(scale, scale).translate(dx, dy)
            base.merge_transformed_page(src, t)

        writer.add_page(base)

    with out_pdf.open("wb") as f:
        writer.write(f)


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _run_scenario(config_path: Path) -> int:
    cmd = [
        sys.executable,
        "run_scenarios.py",
        "--config",
        str(config_path),
        "--no-log-to-file",
    ]
    log_info("RUN", " ".join(cmd))
    r = subprocess.run(cmd)
    return int(r.returncode)


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

    ap.add_argument("--idw-png-dirname", default="_idw_png")
    ap.add_argument("--idw-pdf-dirname", default="_idw_pdf")

    ap.add_argument("--png-sheet-name", default="wet_sweep_contact_sheet.pdf")
    ap.add_argument("--vector-sheet-name", default="wet_sweep_contact_sheet_vector.pdf")
    ap.add_argument("--skip-png-sheet", action="store_true")
    ap.add_argument("--skip-vector-sheet", action="store_true")
    ap.add_argument("--max-per-page", type=int, default=25)
    ap.add_argument("--keep-temp-configs", action="store_true")
    return ap


def run_wet_sweep(args: argparse.Namespace) -> int:
    base_config = Path(args.base_config).resolve()
    if not base_config.exists():
        log_error("WET-SWEEP", f"base config not found: {base_config}")
        return 2

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log_info("WET-SWEEP", f"output root: {out_root}")

    base_text = _read_text(base_config)

    wet_targets: list[float] = []
    if args.wet_targets.strip():
        wet_targets = _parse_float_list(args.wet_targets)
    else:
        m = re.search(r"(?ms)^\[sweep\]\s*(.*?)(?=^\[|\Z)", base_text)
        if m:
            body = m.group(1)
            m2 = re.search(r"(?m)^\s*wet_targets\s*=\s*(.+)$", body)
            if m2:
                wet_targets = _parse_float_list(m2.group(1))
    if not wet_targets:
        log_error(
            "WET-SWEEP",
            "no wet_targets provided (use --wet-targets or set [sweep] wet_targets in config).",
        )
        return 2

    temp_dir = out_root / "_tmp_configs"
    temp_dir.mkdir(parents=True, exist_ok=True)

    idw_png_dir = out_root / args.idw_png_dirname
    idw_pdf_dir = out_root / args.idw_pdf_dirname
    idw_png_dir.mkdir(parents=True, exist_ok=True)
    idw_pdf_dir.mkdir(parents=True, exist_ok=True)

    collected_pngs: list[Path] = []
    collected_pdfs: list[Path] = []

    for wt in wet_targets:
        tag = f"wet{_tagify_float(wt)}"
        run_out = out_root / tag
        run_out.mkdir(parents=True, exist_ok=True)
        log_info("WET-SWEEP", f"run directory ({tag}): {run_out}")

        cfg = base_text
        cfg = _set_ini_value(cfg, "io", "debug", _fmt_ini_bool(args.debug))
        if args.seed is not None:
            cfg = _set_ini_value(cfg, "io", "seed", str(int(args.seed)))
        if args.n_sites is not None:
            cfg = _set_ini_value(cfg, "network", "n_sites", str(int(args.n_sites)))

        cfg = _set_ini_value(cfg, "wet", "wet_target", _fmt_ini_number(wt))
        cfg = _set_ini_value(cfg, "io", "out", str(run_out.as_posix()))
        cfg = _set_ini_value(cfg, "plot", "title_name", f"{tag}")

        cfg_path = temp_dir / f"{tag}.ini"
        _write_text(cfg_path, cfg)

        rc = _run_scenario(cfg_path)
        if rc != 0:
            log_error("WET-SWEEP", f"run failed: {tag} (exit {rc})")
            continue

        src_png = run_out / "idw_field.png"
        if src_png.exists():
            dst_png = idw_png_dir / f"{tag}_idw_field.png"
            shutil.copy2(src_png, dst_png)
            collected_pngs.append(dst_png)
        else:
            log_warn("WET-SWEEP", f"missing {src_png}")

        src_pdf = run_out / "idw_field.pdf"
        if src_pdf.exists():
            dst_pdf = idw_pdf_dir / f"{tag}_idw_field.pdf"
            shutil.copy2(src_pdf, dst_pdf)
            collected_pdfs.append(dst_pdf)
        else:
            log_warn("WET-SWEEP", f"missing {src_pdf}")

    if not args.skip_png_sheet and collected_pngs:
        _make_pdf_contact_sheet(
            images=sorted(collected_pngs, key=lambda p: p.name),
            out_pdf=out_root / args.png_sheet_name,
            title="Wet-target sweep -- IDW field",
            max_per_page=args.max_per_page,
        )

    if not args.skip_vector_sheet and collected_pdfs:
        _make_pdf_contact_sheet_vector(
            pdfs=sorted(collected_pdfs, key=lambda p: p.name),
            out_pdf=out_root / args.vector_sheet_name,
            title="Wet-target sweep -- IDW field",
            max_per_page=args.max_per_page,
        )

    if not args.keep_temp_configs:
        for p in temp_dir.glob("*.ini"):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    return 0


def build_idw_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="configs/config.ini", help="Base INI file")
    ap.add_argument(
        "--out-root", default="outputs_sweep", help="Root folder for all runs"
    )
    ap.add_argument(
        "--debug", action="store_true", help="Enable debug mode in run_scenarios"
    )

    ap.add_argument(
        "--powers", default="1,2,3", help="Comma-separated idw_power values (float)"
    )
    ap.add_argument(
        "--nears", default="4,8,12", help="Comma-separated idw_near values (int)"
    )
    ap.add_argument(
        "--dists",
        default="10000,30000,60000",
        help="Comma-separated idw_dist_m in meters (float); use 0 for unlimited",
    )

    ap.add_argument(
        "--n-sites-list",
        default="50,75,100",
        help="Comma-separated n_sites values (int)",
    )
    ap.add_argument(
        "--seeds", default="0,1,2", help="Comma-separated seed values (int)"
    )
    ap.add_argument(
        "--wet-targets",
        default="0.1,0.33,0.5",
        help="Comma-separated wet_target values (float)",
    )

    ap.add_argument("--idw-pdf-dirname", default="_idw_pdf")
    ap.add_argument("--idw-png-dirname", default="_idw_png")

    ap.add_argument(
        "--pdf-name",
        default="idw_sweep_contact_sheet.pdf",
        help="Per-scenario PNG contact sheet",
    )

    ap.add_argument(
        "--vector-merge-name",
        default="idw_fields_merged_vector.pdf",
        help="Per-scenario merged vector PDF",
    )
    ap.add_argument(
        "--global-vector-merge-name",
        default="ALL_SCENARIOS_idw_fields_merged_vector.pdf",
        help="Global merged vector PDF (all runs, 1 plot per page)",
    )
    ap.add_argument("--skip-vector-merge", action="store_true")

    ap.add_argument(
        "--vector-sheet-name",
        default="idw_sweep_contact_sheet_vector.pdf",
        help="Per-scenario vector grid",
    )
    ap.add_argument(
        "--global-vector-sheet-name",
        default="ALL_SCENARIOS_idw_fields_contact_sheet_vector.pdf",
        help="Global vector grid",
    )
    ap.add_argument("--skip-vector-sheet", action="store_true")

    ap.add_argument("--keep-temp-configs", action="store_true")
    ap.add_argument("--skip-pdf", action="store_true", help="Skip PNG contact sheets")
    ap.add_argument("--max-per-page", type=int, default=25)

    ap.add_argument(
        "--global-vector-sheet-mode",
        choices=["auto", "rep", "all"],
        default="auto",
        help="Global vector grid: auto=all runs if only 1 scenario else rep; rep=1 per scenario; all=all runs.",
    )
    ap.add_argument(
        "--global-png-sheet-mode",
        choices=["auto", "rep", "all"],
        default="auto",
        help="Global PNG grid: auto=all runs if only 1 scenario else rep; rep=1 per scenario; all=all runs.",
    )
    return ap


def run_idw_sweep(args: argparse.Namespace) -> int:
    base_config = Path(args.base_config).resolve()
    if not base_config.exists():
        log_error("IDW-SWEEP", f"base config not found: {base_config}")
        return 2

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log_info("IDW-SWEEP", f"output root: {out_root}")

    base_text = _read_text(base_config)

    powers = _parse_float_list(args.powers)
    nears = _parse_int_list(args.nears)
    dists = _parse_float_list(args.dists)

    n_sites_list = _parse_int_list(args.n_sites_list)
    seeds = _parse_int_list(args.seeds)
    wet_targets = _parse_float_list(args.wet_targets)

    temp_dir = out_root / "_tmp_configs"
    temp_dir.mkdir(parents=True, exist_ok=True)

    idw_combos = list(itertools.product(powers, nears, dists))
    outer_combos = list(itertools.product(n_sites_list, wet_targets, seeds))

    if args.debug:
        log_info(
            "IDW-SWEEP",
            f"outer scenarios: {len(outer_combos)} (n_sites x wet_target x seed)",
        )
        log_info(
            "IDW-SWEEP",
            f"idw combos per scenario: {len(idw_combos)} (power x near x dist)",
        )
        log_info("IDW-SWEEP", f"total runs: {len(outer_combos) * len(idw_combos)}")

    global_png_dir = out_root / "_GLOBAL_idw_png"
    global_png_dir.mkdir(parents=True, exist_ok=True)
    global_vec_dir = out_root / "_GLOBAL_idw_pdf"
    global_vec_dir.mkdir(parents=True, exist_ok=True)

    global_png_rep: list[Path] = []
    global_pdf_rep: list[Path] = []

    global_png_all: list[Path] = []
    global_pdf_all: list[Path] = []

    for n_sites, wet_target, seed in outer_combos:
        scen_tag = f"nsites{n_sites}_wet{_tagify_float(wet_target)}_seed{seed}"
        scen_root = out_root / scen_tag
        scen_root.mkdir(parents=True, exist_ok=True)
        log_info("IDW-SWEEP", f"scenario directory ({scen_tag}): {scen_root}")

        idw_pdf_dir = scen_root / args.idw_pdf_dirname
        idw_png_dir = scen_root / args.idw_png_dirname
        idw_pdf_dir.mkdir(parents=True, exist_ok=True)
        idw_png_dir.mkdir(parents=True, exist_ok=True)

        collected_pngs: list[Path] = []
        collected_pdfs: list[Path] = []

        log_info("IDW-SWEEP", f"scenario start: {scen_tag}")

        for p, nnear, dist in idw_combos:
            tag = f"p{_tagify_float(p)}_n{nnear}_d{_tagify_float(dist)}"
            run_out = scen_root / tag
            run_out.mkdir(parents=True, exist_ok=True)

            cfg = base_text
            cfg = _set_ini_value(cfg, "io", "debug", _fmt_ini_bool(args.debug))
            cfg = _set_ini_value(cfg, "io", "seed", str(seed))
            cfg = _set_ini_value(cfg, "network", "n_sites", str(n_sites))
            cfg = _set_ini_value(cfg, "wet", "wet_target", _fmt_ini_number(wet_target))

            cfg = _set_ini_value(cfg, "interp", "idw_power", _fmt_ini_number(p))
            cfg = _set_ini_value(cfg, "interp", "idw_near", str(nnear))
            cfg = _set_ini_value(cfg, "interp", "idw_dist_m", _fmt_ini_number(dist))

            cfg = _set_ini_value(cfg, "io", "out", str(run_out.as_posix()))
            cfg = _set_ini_value(cfg, "plot", "title_name", f"{scen_tag}_{tag}")

            cfg_path = temp_dir / f"{scen_tag}__{tag}.ini"
            _write_text(cfg_path, cfg)

            rc = _run_scenario(cfg_path)
            if rc != 0:
                log_error("IDW-SWEEP", f"run failed: {scen_tag}/{tag} (exit {rc})")
                continue

            src_png = run_out / "idw_field.png"
            if src_png.exists():
                dst_png = idw_png_dir / f"{tag}_idw_field.png"
                shutil.copy2(src_png, dst_png)
                collected_pngs.append(dst_png)
            else:
                log_warn("IDW-SWEEP", f"missing {src_png}")

            src_pdf = run_out / "idw_field.pdf"
            if src_pdf.exists():
                dst_pdf = idw_pdf_dir / f"{tag}_idw_field.pdf"
                shutil.copy2(src_pdf, dst_pdf)
                collected_pdfs.append(dst_pdf)
            else:
                log_warn("IDW-SWEEP", f"missing {src_pdf}")

        collected_pngs = sorted(collected_pngs, key=lambda p: p.name)
        collected_pdfs = sorted(collected_pdfs, key=lambda p: p.name)

        if not args.skip_pdf and collected_pngs:
            _make_pdf_contact_sheet(
                images=collected_pngs,
                out_pdf=scen_root / args.pdf_name,
                title=f"IDW sweep -- {scen_tag}",
                max_per_page=args.max_per_page,
            )

        if (not args.skip_vector_merge) and collected_pdfs:
            _merge_pdfs(collected_pdfs, scen_root / args.vector_merge_name)

        if (not args.skip_vector_sheet) and collected_pdfs:
            _make_pdf_contact_sheet_vector(
                pdfs=collected_pdfs,
                out_pdf=scen_root / args.vector_sheet_name,
                title=f"IDW sweep -- {scen_tag}",
                max_per_page=args.max_per_page,
            )

        if collected_pngs:
            rep_png = collected_pngs[0]
            gname = f"{scen_tag}__{rep_png.name}"
            gcopy = global_png_dir / gname
            shutil.copy2(rep_png, gcopy)
            global_png_rep.append(gcopy)

        if collected_pdfs:
            rep_pdf = collected_pdfs[0]
            gname = f"{scen_tag}__{rep_pdf.name}"
            gcopy = global_vec_dir / gname
            shutil.copy2(rep_pdf, gcopy)
            global_pdf_rep.append(gcopy)

        global_png_all.extend(collected_pngs)
        global_pdf_all.extend(collected_pdfs)

    if args.global_png_sheet_mode == "rep":
        png_for_global = global_png_rep
    elif args.global_png_sheet_mode == "all":
        png_for_global = global_png_all
    else:
        png_for_global = global_png_all if len(outer_combos) == 1 else global_png_rep

    if not args.skip_pdf and png_for_global:
        _make_pdf_contact_sheet(
            images=sorted(png_for_global, key=lambda p: p.name),
            out_pdf=out_root / "ALL_SCENARIOS_idw_fields_contact_sheet.pdf",
            title="All scenarios -- IDW field",
            max_per_page=args.max_per_page,
        )

    if args.global_vector_sheet_mode == "rep":
        vec_for_global = global_pdf_rep
    elif args.global_vector_sheet_mode == "all":
        vec_for_global = global_pdf_all
    else:
        vec_for_global = global_pdf_all if len(outer_combos) == 1 else global_pdf_rep

    if (not args.skip_vector_sheet) and vec_for_global:
        _make_pdf_contact_sheet_vector(
            pdfs=sorted(vec_for_global, key=lambda p: p.name),
            out_pdf=out_root / args.global_vector_sheet_name,
            title="All scenarios -- IDW field",
            max_per_page=args.max_per_page,
        )

    if (not args.skip_vector_merge) and global_pdf_all:
        _merge_pdfs(
            sorted(global_pdf_all, key=lambda p: p.name),
            out_root / args.global_vector_merge_name,
        )

    if not args.keep_temp_configs:
        for p in temp_dir.glob("*.ini"):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    return 0
