#!/usr/bin/env python3
"""Backfill every campaign run into W&B, without the saturation flaw that
stalled ``scripts/stream_runs.py``.

That sidecar opens a *persistent* ``wandb.init`` for every run it discovers
and holds all of them open at once. At ~350 concurrent runs it saturated
wandb-core (hundreds of HTTPS connections and file descriptors), its
synchronous calls started blocking forever, the poll loop never completed a
cycle, and it never discovered another run again -- permanently stuck at
~350 of 1,702.

This script never holds more than a small bounded number of W&B runs open at
once. Each sweep visits every run directory once; for each one it opens the
run, logs whatever new epochs exist, and **finishes (closes) it before
touching the next**, either strictly sequentially or via a small
``concurrent.futures`` pool capped at ``--max-concurrent`` (default 4). No run
is ever held open across sweeps, and no amount of runs under ``runs/`` can
grow the number of simultaneously open W&B sessions past that cap -- the
regression this exists to prevent.

Discovery does not require ``checkpoints/``: a run pruned by
``scripts/ship_runs.py`` keeps ``run.log``, ``manifest.yaml``, and
``resolved_config.yaml`` (only its trajectory snapshots are deleted, once
shipped and verified), and is backfilled exactly like any other run.

A local ledger (``<runs-root>/.wandb_backfill_ledger.json``) records each
run's ``last_logged_epoch`` and a ``done`` flag once its manifest is
finalised and fully logged -- a finalised+done run is skipped on every future
sweep without even opening W&B. An in-progress run has no ``done`` flag and is
revisited every sweep, picking up whatever new epochs ``run.log`` grew.

**Reconciling the sidecar's ~350 half-open runs**: on resume
(``wandb.init(id=<run_id>, resume="allow")``), W&B restores the run's own
summary, including whatever epoch the stuck sidecar last logged before it
stalled (``run.summary["epoch"]``). This script reads that back
(``WandbSink.last_epoch``) and starts from
``max(ledger_last_logged_epoch, wandb_last_epoch)`` -- so a half-open run left
by the old sidecar picks up exactly where it stopped, never re-sends an
already-logged epoch, and (if its manifest is finalised) gets its summary
written and is properly ``finish()``-ed for the first time. The local ledger
does not even need to have seen that run before; the W&B server is the source
of truth for what has already gone out, so a lost or stale local ledger can
never cause a duplicate.

Reuses ``scripts/stream_runs.py``'s naming/config extraction, ``WandbSink``
seam, and grok-summary tracker rather than reimplementing them (loaded by
file path, matching that module's own test-loading convention, since
``scripts/`` has no ``__init__.py``); ``stream_runs.py`` itself is untouched.
``run.log`` parsing reuses
:func:`group_algorithm_interp.instruments.checkpoints.parse_run_log`.

``wandb`` is never imported directly here -- ``WandbSink`` imports it lazily,
so this module (and its test suite) import and run with no network and no
``wandb`` install.

Examples::

    uv run python scripts/backfill_runs.py --once          # one full backfill pass
    uv run python scripts/backfill_runs.py                 # keep sweeping (live-updating)
    uv run python scripts/backfill_runs.py --interval 60 --max-concurrent 2
    uv run python scripts/backfill_runs.py --complete-sentinel runs/../CAMPAIGN_COMPLETE
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from group_algorithm_interp.instruments.checkpoints import (  # noqa: E402
    EpochRow,
    parse_run_log,
)
from group_algorithm_interp.manifest import (  # noqa: E402
    MANIFEST_NAME,
    RESOLVED_CONFIG_NAME,
)

RUN_LOG_NAME = "run.log"
LEDGER_FILENAME = ".wandb_backfill_ledger.json"
DEFAULT_LOG_EVERY = 50
DEFAULT_INTERVAL_S = 120.0
DEFAULT_MAX_CONCURRENT = 4


def _warn(message: str) -> None:
    print(f"[backfill_runs] {message}", file=sys.stderr)


def _load_sibling_module(name: str, filename: str) -> ModuleType:
    """Load ``scripts/<filename>`` by file path under module name ``name``,
    reusing an already-loaded instance from ``sys.modules`` if one exists
    (e.g. a test that loaded ``stream_runs.py`` first under that name).
    ``scripts/`` has no ``__init__.py``, so a plain ``import stream_runs``
    cannot be resolved as a package import regardless of caller -- this
    mirrors ``tests/test_stream_runs.py``'s own loading convention."""
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Typed as Any: it is a dynamically loaded module, and mypy cannot see
# stream_runs.py's attributes through a static ModuleType.
stream_runs: Any = _load_sibling_module("stream_runs", "stream_runs.py")


def is_finalised(manifest: dict[str, Any]) -> bool:
    """A manifest counts as finalised once its ``status`` is terminal AND
    ``completed_at`` is set -- the same two-part check
    ``scripts/ship_runs.py`` uses (``status`` alone briefly lags
    ``completed_at`` during ``finalize_manifest``'s write)."""
    status = manifest.get("status")
    return status in stream_runs.TERMINAL_STATUSES and manifest.get("completed_at") is not None


# --- ledger --------------------------------------------------------------------


def default_ledger_path(runs_root: Path) -> Path:
    return runs_root / LEDGER_FILENAME


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"runs": {}}
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("runs"), dict):
        raise ValueError(f"ledger at {path} is not a mapping with a 'runs' key")
    return data


def save_ledger(path: Path, ledger: dict[str, Any]) -> None:
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")


def ledger_entry(ledger: dict[str, Any], run_id: str) -> dict[str, Any]:
    """This run's ledger state, or the fresh-run default (never seen,
    nothing logged)."""
    entry = ledger.get("runs", {}).get(run_id)
    if entry is None:
        return {"last_logged_epoch": -1, "done": False}
    return entry


def set_ledger_entry(
    ledger: dict[str, Any], run_id: str, *, last_logged_epoch: int, done: bool
) -> None:
    ledger.setdefault("runs", {})[run_id] = {
        "last_logged_epoch": last_logged_epoch,
        "done": done,
    }


# --- discovery -------------------------------------------------------------------


def discover_run_dirs(runs_root: Path) -> list[Path]:
    """Every immediate child of ``runs_root`` with its own ``manifest.yaml``,
    ``resolved_config.yaml``, and ``run.log`` -- deliberately NOT gated on
    ``checkpoints/``, since a pruned run (``scripts/ship_runs.py``) keeps
    exactly those three files and loses only its trajectory snapshots."""
    if not runs_root.is_dir():
        return []
    found = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        if not (run_dir / MANIFEST_NAME).is_file():
            continue
        if not (run_dir / RESOLVED_CONFIG_NAME).is_file():
            continue
        if not (run_dir / RUN_LOG_NAME).is_file():
            continue
        found.append(run_dir)
    return found


# --- per-run backfill --------------------------------------------------------------


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    last_logged_epoch: int
    done: bool
    logged: int
    error: str | None = None


def _select_rows_to_log(rows_to_send: Sequence[EpochRow], log_every: int) -> list[EpochRow]:
    """Every Nth new epoch row, plus the newest new row (the frontier) even
    when it does not land on the stride -- mirrors ``stream_runs.py``'s own
    throttle so a resumed run's chart density does not change mid-history."""
    selected = [row for row in rows_to_send if row.epoch % log_every == 0]
    if rows_to_send and (not selected or selected[-1].epoch != rows_to_send[-1].epoch):
        selected.append(rows_to_send[-1])
    return selected


def _final_metrics(run_dir: Path, rows: Sequence[EpochRow]) -> dict[str, float]:
    """The ``completed <run_id> | {...}`` trailer's metrics if one was
    written (the last such line wins), else the last parsed epoch row's."""
    completed: dict[str, float] | None = None
    log_path = run_dir / RUN_LOG_NAME
    try:
        with log_path.open() as handle:
            for line in handle:
                parsed = stream_runs.parse_completed_line(line)
                if parsed is not None:
                    completed = parsed
    except OSError:
        pass
    if completed is not None:
        return completed
    return dict(rows[-1].metrics) if rows else {}


def backfill_one(
    run_dir: Path,
    *,
    campaign_lookup: dict[tuple[int, int, int], dict[str, Any]],
    sink: Any,
    log_every: int,
    last_logged_epoch: int,
    done: bool,
    warn: Callable[[str], None] = _warn,
) -> RunOutcome:
    """Backfill one run for this sweep: open its W&B run (if there is
    anything new to send), log any new rows, close the terminal-run loop with
    a summary, and **always finish() before returning** -- the one invariant
    that keeps this script from ever repeating ``stream_runs.py``'s
    saturation failure, even on an error mid-flush."""
    run_id = run_dir.name
    if done:
        return RunOutcome(run_id, last_logged_epoch, True, 0)

    try:
        manifest = yaml.safe_load((run_dir / MANIFEST_NAME).read_text()) or {}
    except Exception as exc:  # noqa: BLE001 -- one bad manifest must not stop the sweep
        warn(f"{run_id}: manifest unreadable ({type(exc).__name__}: {exc})")
        return RunOutcome(run_id, last_logged_epoch, done, 0, error=str(exc))
    terminal = is_finalised(manifest)

    try:
        rows = parse_run_log(run_dir / RUN_LOG_NAME)
    except Exception as exc:  # noqa: BLE001 -- retried next sweep
        warn(f"{run_id}: run.log unreadable ({type(exc).__name__}: {exc})")
        return RunOutcome(run_id, last_logged_epoch, done, 0, error=str(exc))

    new_rows = [row for row in rows if row.epoch > last_logged_epoch]
    if not new_rows and not (terminal and not done):
        return RunOutcome(run_id, last_logged_epoch, done, 0)

    try:
        spec = stream_runs.build_run_spec(run_dir, campaign_lookup)
    except Exception as exc:  # noqa: BLE001 -- retried next sweep (e.g. config not written yet)
        warn(f"{run_id}: cannot read run metadata yet ({exc})")
        return RunOutcome(run_id, last_logged_epoch, done, 0, error=str(exc))

    try:
        handle = sink.init_run(spec)
    except Exception as exc:  # noqa: BLE001 -- no handle to close; retried next sweep
        warn(f"{run_id}: cannot open W&B run ({type(exc).__name__}: {exc}); retrying next sweep")
        return RunOutcome(run_id, last_logged_epoch, done, 0, error=str(exc))

    new_last_logged = last_logged_epoch
    new_done: bool = done
    logged = 0
    error: str | None = None
    try:
        # Reconcile a half-open run left by a stalled prior sidecar: its
        # summary already carries the last epoch that made it out, even
        # though the process that logged it never called finish().
        already = sink.last_epoch(handle)
        start_from = max(last_logged_epoch, already)
        rows_to_send = [row for row in rows if row.epoch > start_from]
        selected = _select_rows_to_log(rows_to_send, log_every)
        for row in selected:
            sink.log(handle, {"epoch": float(row.epoch), **row.metrics})
            new_last_logged = row.epoch
            logged += 1
        if not selected:
            new_last_logged = start_from

        if terminal:
            final_metrics = _final_metrics(run_dir, rows)
            grok = stream_runs.GrokTracker(
                metric_key=spec.generalize_metric_key,
                threshold=spec.generalize_test_acc,
                patience=spec.generalize_patience,
            )
            for row in rows:
                grok.observe(row.epoch, row.metrics)
            sink.set_summary(handle, {**final_metrics, **grok.summary(), "stream/complete": True})
            new_done = True
    except Exception as exc:  # noqa: BLE001 -- rows already sent stay sent; retried next sweep
        error = str(exc)
        warn(f"{run_id}: W&B error ({type(exc).__name__}: {exc}); retrying next sweep")
    finally:
        # ALWAYS close -- the anti-saturation invariant. Never left open even
        # when logging/summary above raised.
        try:
            sink.finish(handle)
        except Exception as exc:  # noqa: BLE001 -- best-effort; nothing more to do with it
            warn(f"{run_id}: error finishing W&B run ({type(exc).__name__}: {exc})")

    return RunOutcome(run_id, new_last_logged, new_done, logged, error=error)


# --- the sweep loop --------------------------------------------------------------


class Backfiller:
    """Sweeps ``runs_root`` once per :meth:`poll_once`, backfilling every
    non-``done`` run either sequentially or via a bounded thread pool. Never
    holds more than ``max_concurrent`` W&B runs open at any instant -- each
    pool worker opens, logs, and finishes one run before taking the next."""

    def __init__(
        self,
        runs_root: Path,
        sink: Any,
        *,
        campaign_lookup: dict[tuple[int, int, int], dict[str, Any]] | None = None,
        log_every: int = DEFAULT_LOG_EVERY,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        ledger_path: Path | None = None,
        warn: Callable[[str], None] = _warn,
    ):
        self.runs_root = runs_root
        self.sink = sink
        self.campaign_lookup = campaign_lookup or {}
        self.log_every = log_every
        self.max_concurrent = max(1, max_concurrent)
        self.warn = warn
        self.ledger_path = ledger_path or default_ledger_path(runs_root)
        self.ledger = load_ledger(self.ledger_path)

    def _targets(self) -> list[Path]:
        return [
            run_dir
            for run_dir in discover_run_dirs(self.runs_root)
            if not ledger_entry(self.ledger, run_dir.name)["done"]
        ]

    def poll_once(self) -> list[RunOutcome]:
        targets = self._targets()
        if not targets:
            return []

        outcomes: list[RunOutcome] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            futures = {
                pool.submit(
                    backfill_one,
                    run_dir,
                    campaign_lookup=self.campaign_lookup,
                    sink=self.sink,
                    log_every=self.log_every,
                    last_logged_epoch=ledger_entry(self.ledger, run_dir.name)["last_logged_epoch"],
                    done=ledger_entry(self.ledger, run_dir.name)["done"],
                    warn=self.warn,
                ): run_dir
                for run_dir in targets
            }
            for future in concurrent.futures.as_completed(futures):
                outcome = future.result()
                outcomes.append(outcome)
                set_ledger_entry(
                    self.ledger,
                    outcome.run_id,
                    last_logged_epoch=outcome.last_logged_epoch,
                    done=outcome.done,
                )

        save_ledger(self.ledger_path, self.ledger)
        return outcomes

    def run_forever(
        self,
        interval: float,
        *,
        once: bool = False,
        complete_sentinel: Path | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        while True:
            self.poll_once()
            if once:
                return
            if complete_sentinel is not None and complete_sentinel.is_file():
                self.warn(
                    f"campaign-complete sentinel found ({complete_sentinel}); final sweep, then exit"
                )
                self.poll_once()
                return
            sleep(interval)


# --- CLI ---------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--project", default=os.environ.get("WANDB_PROJECT", stream_runs.DEFAULT_PROJECT)
    )
    parser.add_argument("--entity", default=None)
    parser.add_argument(
        "--campaign-config",
        type=Path,
        default=stream_runs.DEFAULT_CAMPAIGN_CONFIG,
        help="cell list used for run naming (cosmetic metadata only)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
        help="stream every Nth epoch row (the newest new row always goes out too)",
    )
    parser.add_argument(
        "--interval", type=float, default=DEFAULT_INTERVAL_S, help="seconds between sweeps"
    )
    parser.add_argument(
        "--once", action="store_true", help="one full sweep over every run, then exit"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help="upper bound on simultaneously open W&B runs (never exceeded)",
    )
    parser.add_argument(
        "--complete-sentinel",
        type=Path,
        default=None,
        help="a file whose appearance ends the loop after one final sweep "
        "(e.g. the campaign's CAMPAIGN_COMPLETE marker)",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="override the ledger path (default: <runs-root>/.wandb_backfill_ledger.json)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    skip = stream_runs.streaming_skip_reason()
    if skip is not None:
        _warn(f"not backfilling: {skip}")
        return 2
    if args.log_every < 1:
        _warn("--log-every must be at least 1")
        return 2
    if args.max_concurrent < 1:
        _warn("--max-concurrent must be at least 1")
        return 2

    sink = stream_runs.WandbSink(args.project, args.entity)
    backfiller = Backfiller(
        args.runs_root,
        sink,
        campaign_lookup=stream_runs.load_campaign_lookup(args.campaign_config),
        log_every=args.log_every,
        max_concurrent=args.max_concurrent,
        ledger_path=args.ledger,
    )
    try:
        backfiller.run_forever(
            args.interval, once=args.once, complete_sentinel=args.complete_sentinel
        )
    except KeyboardInterrupt:
        _warn(
            "interrupted; the ledger + W&B's own resumed summary mean the next sweep "
            "picks up exactly where this one left off"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
