from __future__ import annotations

import datetime as dt
import io
import sys
from pathlib import Path


class _TeeStream(io.TextIOBase):
    def __init__(self, console: io.TextIOBase, file_stream: io.TextIOBase) -> None:
        self._console = console
        self._file = file_stream

    def write(self, s: str) -> int:
        n = self._console.write(s)
        self._file.write(s)
        return n

    def flush(self) -> None:
        self._console.flush()
        self._file.flush()


def setup_tee_logging(log_dir: str | Path = "logs", enabled: bool = True) -> Path | None:
    """Mirror stdout/stderr to logs/<YYYYmmdd_HHMMSS>.log while keeping console output."""
    if not enabled:
        return None

    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    log_path = path / f"{dt.datetime.now():%Y%m%d_%H%M%S}.log"

    file_stream = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, file_stream)
    sys.stderr = _TeeStream(sys.stderr, file_stream)
    log_info("LOG", f"writing console output to {log_path}")
    return log_path


def _log(level: str, scope: str, message: str, *, err: bool = False) -> None:
    stream = sys.stderr if err else sys.stdout
    print(f"[{level}][{scope}] {message}", file=stream)


def log_info(scope: str, message: str) -> None:
    _log("INFO", scope, message)


def log_warn(scope: str, message: str) -> None:
    _log("WARN", scope, message, err=True)


def log_error(scope: str, message: str) -> None:
    _log("ERROR", scope, message, err=True)


def format_path(path: str | Path) -> str:
    """
    Prefer repo/workdir-relative path display for logs.
    Falls back to absolute path when relative conversion is not possible.
    """
    p = Path(path)
    try:
        p_abs = p.resolve()
        cwd = Path.cwd().resolve()
        return str(p_abs.relative_to(cwd))
    except Exception:
        return str(p)
