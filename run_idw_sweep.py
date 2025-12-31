from __future__ import annotations

import argparse
import itertools
import re
import shutil
import subprocess
import sys
from pathlib import Path

import math


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


def _tagify_float(x: float) -> str:
    # return f"{x}".replace(".", "p")
    return f"{x}"


def _auto_grid(n: int, max_per_page: int = 25) -> tuple[int, int, int]:
    """
    Returns (rows, cols, per_page) for a near-square layout.
    per_page is capped by max_per_page to keep pages readable.
    """
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
    max_per_page: int = 25,  # <= change if you want 36 (6x6), etc.
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    n = len(images)
    rows, cols, per_page = _auto_grid(n, max_per_page=max_per_page)
    n_pages = max(1, math.ceil(n / per_page))

    with PdfPages(out_pdf) as pdf:
        for page in range(n_pages):
            start = page * per_page
            chunk = images[start : start + per_page]

            # recompute per-page grid so the last page is also near-square
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="configs/config.ini", help="Base INI file")
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
        "--idw-pdf-dirname",
        default="_idw_pdf",
        help="Folder (inside out-root) collecting all idw_field.pdf",
    )
    ap.add_argument(
        "--idw-png-dirname",
        default="_idw_png",
        help="Folder (inside out-root) collecting all idw_field.png",
    )

    ap.add_argument(
        "--pdf-name",
        default="idw_sweep_contact_sheet.pdf",
        help="PDF filename (inside out-root)",
    )

    ap.add_argument("--keep-temp-configs", action="store_true")
    ap.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Only collect images, do not generate PDF",
    )

    args = ap.parse_args()

    base_config = Path(args.base_config).resolve()
    if not base_config.exists():
        print(f"ERROR: base config not found: {base_config}", file=sys.stderr)
        return 2

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    idw_pdf_dir = out_root / args.idw_pdf_dirname
    idw_pdf_dir.mkdir(parents=True, exist_ok=True)
    idw_png_dir = out_root / args.idw_png_dirname
    idw_png_dir.mkdir(parents=True, exist_ok=True)

    base_text = _read_text(base_config)

    powers = [float(x.strip()) for x in args.powers.split(",") if x.strip()]
    nears = [int(x.strip()) for x in args.nears.split(",") if x.strip()]
    dists = [float(x.strip()) for x in args.dists.split(",") if x.strip()]

    temp_dir = out_root / "_tmp_configs"
    temp_dir.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(powers, nears, dists))
    print(f"Running {len(combos)} combinations...")

    collected: list[Path] = []

    for p, nnear, dist in combos:
        tag = f"p{_tagify_float(p)}_n{nnear}_d{_tagify_float(dist)}"
        run_out = out_root / tag
        run_out.mkdir(parents=True, exist_ok=True)

        cfg = base_text
        cfg = _set_ini_value(cfg, "interp", "idw_power", str(p))
        cfg = _set_ini_value(cfg, "interp", "idw_near", str(nnear))
        cfg = _set_ini_value(cfg, "interp", "idw_dist_m", str(dist))
        cfg = _set_ini_value(cfg, "io", "out", str(run_out.as_posix()))

        cfg_path = temp_dir / f"{tag}.ini"
        _write_text(cfg_path, cfg)

        cmd = [sys.executable, "run_scenarios.py", "--config", str(cfg_path)]
        print(">", " ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"FAILED: {tag} (exit {r.returncode})", file=sys.stderr)
            continue

        # Collect idw_field imgs
        src_img = run_out / "idw_field.png"
        if src_img.exists():
            dst_img = idw_png_dir / f"{tag}_idw_field.png"
            shutil.copy2(src_img, dst_img)
            collected.append(dst_img)
        else:
            print(f"WARNING: missing {src_img}", file=sys.stderr)

        src_img = run_out / "idw_field.pdf"
        if src_img.exists():
            dst_img = idw_pdf_dir / f"{tag}_idw_field.pdf"
            shutil.copy2(src_img, dst_img)
        else:
            print(f"WARNING: missing {src_img}", file=sys.stderr)

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

    collected_sorted = sorted(collected, key=lambda p: p.name)
    print(f"Collected {len(collected_sorted)} idw_field imges into: {idw_png_dir}")

    if not args.skip_pdf and collected_sorted:
        out_pdf = out_root / args.pdf_name
        _make_pdf_contact_sheet(
            images=collected_sorted,
            out_pdf=out_pdf,
            title="IDW sweep (idw_field)",
            max_per_page=25,
        )
        print(f"Wrote PDF contact sheet: {out_pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# python run_idw_sweep.py --base-config configs/config.ini --out-root outputs_sweep --powers 1,2,2.5,3,3.5,4 --nears 4,8,12,16 --dists 10000,20000,30000,50000
