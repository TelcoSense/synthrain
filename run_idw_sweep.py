from __future__ import annotations

import argparse
import itertools
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _fmt_ini_number(x: float) -> str:
    # for writing into INI, keep numeric clean
    if float(x).is_integer():
        return str(int(x))
    return repr(float(x))  # stable-ish, avoids locale


def _tagify_float(x: float) -> str:
    # for folder/file tags (no trailing .0)
    if float(x).is_integer():
        return str(int(x))
    s = f"{float(x):.12g}"  # prevents long tails like 0.30000000000004
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


# def _tagify_float(x: float) -> str:
#     return str(x)


def _auto_grid(n: int, max_per_page: int = 25) -> tuple[int, int, int]:
    if n <= 0:
        return 1, 1, 1
    k = min(n, max_per_page)
    cols = math.ceil(math.sqrt(k))
    rows = math.ceil(k / cols)
    return rows, cols, rows * cols


def _make_pdf_contact_sheet(
    images: list[Path],
    out_pdf: Path,
    title: str = "IDW sweep",
    max_per_page: int = 25,
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
    """
    Concatenate PDFs (keeps vector content). Output is a single PDF (1 plot per page).
    Requires: pip install pypdf
    """
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
    title: str = "IDW sweep (vector contact sheet)",
    max_per_page: int = 25,
    # layout controls in PDF points (1 pt = 1/72 inch)
    cell_w_pt: float = 260.0,
    cell_h_pt: float = 200.0,
    margin_pt: float = 18.0,
    pad_pt: float = 8.0,
) -> None:
    """
    N-up contact sheet from single-page PDFs WITHOUT rasterizing.
    Each input PDF page is placed into a grid cell (scaled to fit) on output pages.
    Requires: pip install pypdf
    """
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    try:
        from pypdf import PdfReader, PdfWriter, Transformation
        from pypdf._page import PageObject
    except Exception as e:
        raise RuntimeError(
            "Vector contact sheet requires 'pypdf'. Install with: pip install pypdf"
        ) from e

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
            reader = PdfReader(str(pdf_path))
            src = reader.pages[0]

            src_w = float(src.mediabox.width)
            src_h = float(src.mediabox.height)

            col = i % cols
            row_from_top = i // cols
            row = (rows - 1) - row_from_top  # PDF origin bottom-left

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="configs/config.ini", help="Base INI file")
    ap.add_argument("--debug", default=False, help="Enable additional console prints")
    ap.add_argument(
        "--out-root", default="outputs_sweep", help="Root folder for all runs"
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
        "--seeds",
        default="0,1,2",
        help="Comma-separated seed values (int) to write into [io] seed",
    )
    ap.add_argument(
        "--wet-targets",
        default="0.1,0.33,0.5",
        help="Comma-separated wet_target values (float) to write into [wet] wet_target",
    )

    ap.add_argument(
        "--idw-pdf-dirname",
        default="_idw_pdf",
        help="Folder (inside EACH scenario root) collecting all idw_field.pdf",
    )
    ap.add_argument(
        "--idw-png-dirname",
        default="_idw_png",
        help="Folder (inside EACH scenario root) collecting all idw_field.png",
    )

    ap.add_argument(
        "--pdf-name",
        default="idw_sweep_contact_sheet.pdf",
        help="PNG contact-sheet PDF filename (inside EACH scenario root)",
    )

    ap.add_argument(
        "--vector-merge-name",
        default="idw_fields_merged_vector.pdf",
        help="Per-scenario merged vector PDF filename (inside EACH scenario root).",
    )
    ap.add_argument(
        "--global-vector-merge-name",
        default="ALL_SCENARIOS_idw_fields_merged_vector.pdf",
        help="Global merged vector PDF filename (inside out-root).",
    )
    ap.add_argument(
        "--skip-vector-merge",
        action="store_true",
        help="Do not merge vector PDFs (idw_field.pdf).",
    )

    ap.add_argument(
        "--vector-sheet-name",
        default="idw_sweep_contact_sheet_vector.pdf",
        help="Per-scenario VECTOR contact-sheet (grid) PDF filename.",
    )
    ap.add_argument(
        "--global-vector-sheet-name",
        default="ALL_SCENARIOS_idw_fields_contact_sheet_vector.pdf",
        help="Global VECTOR contact-sheet (grid) PDF filename (inside out-root).",
    )
    ap.add_argument(
        "--skip-vector-sheet",
        action="store_true",
        help="Do not generate VECTOR contact-sheet PDFs (grid).",
    )

    ap.add_argument("--keep-temp-configs", action="store_true")
    ap.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Only collect images, do not generate PNG contact-sheet PDFs",
    )
    ap.add_argument(
        "--max-per-page", type=int, default=25, help="Max images per PDF page"
    )

    ap.add_argument(
        "--global-sheet-mode",
        choices=["auto", "rep", "all"],
        default="auto",
        help="Global vector grid: auto=all runs if only 1 scenario else rep; rep=1 per scenario; all=all runs.",
    )

    args = ap.parse_args()

    base_config = Path(args.base_config).resolve()
    if not base_config.exists():
        print(f"ERROR: base config not found: {base_config}", file=sys.stderr)
        return 2

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

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
        print(f"Outer scenarios: {len(outer_combos)} (n_sites x wet_target x seed)")
        print(f"IDW combos per scenario: {len(idw_combos)} (power x near x dist)")
        print(f"Total runs: {len(outer_combos) * len(idw_combos)}")

    # global PNG contact sheet (one PNG per scenario)
    global_png_dir = out_root / "_GLOBAL_idw_png"
    global_png_dir.mkdir(parents=True, exist_ok=True)
    global_collected_png: list[Path] = []

    # global VECTOR contact sheet sources (one PDF per scenario; representative)
    global_vec_dir = out_root / "_GLOBAL_idw_pdf"
    global_vec_dir.mkdir(parents=True, exist_ok=True)
    global_collected_vec_rep: list[Path] = []

    # global merge (all runs across all scenarios)
    global_collected_vec_all: list[Path] = []
    global_collected_png_all: list[Path] = []

    for n_sites, wet_target, seed in outer_combos:
        scen_tag = f"nsites{n_sites}_wet{_tagify_float(wet_target)}_seed{seed}"
        scen_root = out_root / scen_tag
        scen_root.mkdir(parents=True, exist_ok=True)

        idw_pdf_dir = scen_root / args.idw_pdf_dirname
        idw_pdf_dir.mkdir(parents=True, exist_ok=True)
        idw_png_dir = scen_root / args.idw_png_dirname
        idw_png_dir.mkdir(parents=True, exist_ok=True)

        collected_pngs: list[Path] = []
        collected_vec_pdfs: list[Path] = []

        print(f"\n=== Scenario: {scen_tag} ===")

        for p, nnear, dist in idw_combos:
            tag = f"p{p}_n{nnear}_d{_tagify_float(dist)}"
            run_out = scen_root / tag
            run_out.mkdir(parents=True, exist_ok=True)

            cfg = base_text
            cfg = _set_ini_value(cfg, "io", "debug", bool(args.debug))
            cfg = _set_ini_value(cfg, "io", "seed", str(seed))
            cfg = _set_ini_value(cfg, "network", "n_sites", str(n_sites))
            cfg = _set_ini_value(cfg, "wet", "wet_target", str(wet_target))

            cfg = _set_ini_value(cfg, "interp", "idw_power", str(p))
            cfg = _set_ini_value(cfg, "interp", "idw_near", str(nnear))
            cfg = _set_ini_value(cfg, "interp", "idw_dist_m", _fmt_ini_number(dist))

            cfg = _set_ini_value(cfg, "io", "out", str(run_out.as_posix()))
            cfg = _set_ini_value(cfg, "plot", "title_name", f"{scen_tag}_{tag}")

            cfg_path = temp_dir / f"{scen_tag}__{tag}.ini"
            _write_text(cfg_path, cfg)

            cmd = [sys.executable, "run_scenarios.py", "--config", str(cfg_path)]
            if args.debug:
                print(">", " ".join(cmd))
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(
                    f"FAILED: {scen_tag}/{tag} (exit {r.returncode})", file=sys.stderr
                )
                continue

            # collect PNG
            src_png = run_out / "idw_field.png"
            if src_png.exists():
                dst_png = idw_png_dir / f"{tag}_idw_field.png"
                shutil.copy2(src_png, dst_png)
                collected_pngs.append(dst_png)
                global_collected_png_all.append(dst_png)
            else:
                print(f"WARNING: missing {src_png}", file=sys.stderr)

            # collect PDF (vector)
            src_pdf = run_out / "idw_field.pdf"
            if src_pdf.exists():
                dst_pdf = idw_pdf_dir / f"{tag}_idw_field.pdf"
                shutil.copy2(src_pdf, dst_pdf)
                collected_vec_pdfs.append(dst_pdf)
                global_collected_vec_all.append(dst_pdf)
            else:
                print(f"WARNING: missing {src_pdf}", file=sys.stderr)

        collected_sorted = sorted(collected_pngs, key=lambda pth: pth.name)
        collected_vec_sorted = sorted(collected_vec_pdfs, key=lambda pth: pth.name)

        if args.debug:
            print(
                f"Collected {len(collected_sorted)} idw_field PNGs into: {idw_png_dir}"
            )
            print(
                f"Collected {len(collected_vec_sorted)} idw_field PDFs into: {idw_pdf_dir}"
            )

        # per-scenario PNG contact sheet
        if not args.skip_pdf and collected_sorted:
            out_pdf = scen_root / args.pdf_name
            _make_pdf_contact_sheet(
                images=collected_sorted,
                out_pdf=out_pdf,
                title=f"IDW sweep (idw_field) — {scen_tag}",
                max_per_page=args.max_per_page,
            )

        # per-scenario VECTOR merge (1 plot per page)
        if (not args.skip_vector_merge) and collected_vec_sorted:
            out_vec = scen_root / args.vector_merge_name
            _merge_pdfs(collected_vec_sorted, out_vec)

        # per-scenario VECTOR contact sheet (grid, vector)
        if (not args.skip_vector_sheet) and collected_vec_sorted:
            out_vec_sheet = scen_root / args.vector_sheet_name
            _make_pdf_contact_sheet_vector(
                pdfs=collected_vec_sorted,
                out_pdf=out_vec_sheet,
                title=f"IDW sweep (vector) — {scen_tag}",
                max_per_page=args.max_per_page,
            )

        # add one representative PNG (for global PNG sheet)
        if collected_sorted:
            rep_png = collected_sorted[0]
            global_name = f"{scen_tag}__{rep_png.name}"
            global_copy = global_png_dir / global_name
            global_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rep_png, global_copy)
            global_collected_png.append(global_copy)

        # add one representative PDF (for global VECTOR grid sheet)
        if collected_vec_sorted:
            rep_pdf = collected_vec_sorted[0]
            global_pdf_name = f"{scen_tag}__{rep_pdf.name}"
            global_pdf_copy = global_vec_dir / global_pdf_name
            global_pdf_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rep_pdf, global_pdf_copy)
            global_collected_vec_rep.append(global_pdf_copy)

    # global PNG contact sheet (grid)  <-- dynamic
    if not args.skip_pdf:
        if args.global_sheet_mode == "auto":
            use_all = len(outer_combos) == 1
        elif args.global_sheet_mode == "all":
            use_all = True
        else:  # "rep"
            use_all = False

        if use_all:
            src_pngs = sorted(global_collected_png_all, key=lambda p: p.as_posix())
            title = "All runs — IDW field (PNG)"
        else:
            src_pngs = sorted(global_collected_png, key=lambda p: p.as_posix())
            title = "All scenarios — IDW field (1 PNG per scenario)"

        if src_pngs:
            global_pdf = out_root / "ALL_SCENARIOS_idw_fields_contact_sheet.pdf"
            _make_pdf_contact_sheet(
                images=src_pngs,
                out_pdf=global_pdf,
                title=title,
                max_per_page=args.max_per_page,
            )

    # global VECTOR contact sheet (grid of PDFs, vector)  <-- dynamic
    if not args.skip_vector_sheet:
        # decide which PDFs to use
        if args.global_sheet_mode == "auto":
            # if only 1 scenario, show ALL runs in a grid; otherwise 1 per scenario
            use_all = len(outer_combos) == 1
        elif args.global_sheet_mode == "all":
            use_all = True
        else:  # "rep"
            use_all = False

        if use_all:
            src_pdfs = sorted(global_collected_vec_all, key=lambda p: p.as_posix())
            title = "All runs — IDW field (vector)"
        else:
            src_pdfs = sorted(global_collected_vec_rep, key=lambda p: p.as_posix())
            title = "All scenarios — IDW field (1 vector PDF per scenario)"

        if src_pdfs:
            global_vec_sheet = out_root / args.global_vector_sheet_name
            _make_pdf_contact_sheet_vector(
                pdfs=src_pdfs,
                out_pdf=global_vec_sheet,
                title=title,
                max_per_page=args.max_per_page,
            )

    # global VECTOR merge (all idw_field.pdf across all scenarios; 1 plot per page)
    if (not args.skip_vector_merge) and global_collected_vec_all:
        global_vec = out_root / args.global_vector_merge_name
        _merge_pdfs(
            sorted(global_collected_vec_all, key=lambda pth: pth.name), global_vec
        )

    # cleanup temp configs
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


if __name__ == "__main__":
    raise SystemExit(main())


# Example:
# python run_idw_sweep.py --out-root outputs_sweep/nsites_grid --n-sites-list 50,75,100 --seeds 0,1,2 --wet-targets 0.1,0.33,0.5 --powers 2,2.5,3,3.5,4 --nears 4,8,12,16 --dists 20000,30000,50000
# python run_idw_sweep.py --out-root outputs_sweep/interp_power --n-sites-list 50 --seeds 0 --wet-targets 0.1 --powers 1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0 --nears 12 --dists 10000
# python run_idw_sweep.py --out-root outputs_sweep/interp_wet_targets --n-sites-list 50 --seeds 0 --wet-targets 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --powers 2 --nears 12 --dists 10000
