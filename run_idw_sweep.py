from __future__ import annotations

from synthrain.run_logging import setup_tee_logging
from synthrain.sweeps import build_idw_parser, run_idw_sweep


def main() -> int:
    ap = build_idw_parser()
    ap.add_argument("--log-dir", default="logs", help="Directory for datetime log files")
    ap.add_argument("--log-to-file", action="store_true", default=True)
    ap.add_argument("--no-log-to-file", dest="log_to_file", action="store_false")
    args = ap.parse_args()

    setup_tee_logging(args.log_dir, enabled=bool(args.log_to_file))
    return run_idw_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
