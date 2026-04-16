"""Logging configuration and progress bar utilities."""

from __future__ import annotations

import logging
import sys
from typing import Optional

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> None:
    """
    Configure the root logger.

    Parameters
    ----------
    level    : logging level (e.g. logging.DEBUG)
    log_file : optional path to write logs to a file
    fmt      : log message format string
    """
    handlers: list = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("trimesh").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)


class ProgressBar:
    """
    Thin wrapper around tqdm with graceful fallback when tqdm is not installed.
    """

    def __init__(
        self,
        total: int,
        desc: str = "",
        unit: str = "it",
    ) -> None:
        self._total   = total
        self._current = 0
        self._desc    = desc

        if _TQDM_AVAILABLE:
            self._bar = _tqdm(total=total, desc=desc, unit=unit, file=sys.stdout)
        else:
            self._bar = None
            print(f"{desc}: 0/{total}", end="", flush=True)

    def update(self, n: int = 1) -> None:
        """Advance the progress bar by *n* steps."""
        self._current += n
        if self._bar is not None:
            self._bar.update(n)
        else:
            pct = int(100 * self._current / max(1, self._total))
            print(f"\r{self._desc}: {self._current}/{self._total} ({pct}%)",
                  end="", flush=True)

    def close(self) -> None:
        """Finalise and close the progress bar."""
        if self._bar is not None:
            self._bar.close()
        else:
            print()

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
