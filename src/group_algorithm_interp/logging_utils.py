"""Stdlib logging setup: console + per-run file. The trainer and scripts call
``setup_logging`` instead of using ``print`` so runs leave a readable
``runs/<id>/run.log`` and the console output is leveled and timestamped.

``rich`` is an optional dependency; when it is not installed (or ``use_rich`` is
False) a plain ``StreamHandler`` is used. Logging is intentionally separate from
metrics -- metrics go to W&B, this is human-readable progress only.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER_NAME = "group_algorithm_interp"
_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"
# Tagged so repeat calls can find and clear only the handlers we added.
_OWNED = "_group_algorithm_interp_owned"


def _console_handler(use_rich: bool) -> logging.Handler:
    if use_rich:
        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
            handler.setFormatter(logging.Formatter("%(message)s", datefmt=_DATEFMT))
            return handler
        except ImportError:
            pass
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    return handler


def setup_logging(
    run_dir: Path | None = None,
    level: int = logging.INFO,
    use_rich: bool = True,
) -> logging.Logger:
    """Configure and return the package logger with a console handler and,
    when ``run_dir`` is given, a ``run_dir/run.log`` file handler. Idempotent:
    handlers added by a previous call are removed first."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, _OWNED, False):
            logger.removeHandler(handler)
            handler.close()

    console = _console_handler(use_rich)
    setattr(console, _OWNED, True)
    logger.addHandler(console)

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(run_dir / "run.log")
        file_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        setattr(file_handler, _OWNED, True)
        logger.addHandler(file_handler)

    return logger
