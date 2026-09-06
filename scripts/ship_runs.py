#!/usr/bin/env python3
"""Durability sidecar: ship a curated snapshot subset per run to S3, then
prune the local trajectory -- so a pod's local disk never fills and results
are continuously durable off-pod.

A completed run's ``checkpoints/`` directory is huge (~0.8 GB/run, ~600
files); most of that is never read by any analysis. This daemon watches
``runs/`` for finalised runs, picks a small curated subset per run (the dip-
aware selection primitives already used by post-hoc analysis --
:mod:`group_algorithm_interp.instruments.checkpoints` -- are reused, never
reimplemented), uploads the curated files plus the provenance files, and only
then deletes the local ``checkpoints/`` directory. ``manifest.yaml``,
``resolved_config.yaml``, and ``run.log`` are never deleted (tiny, and the
campaign's skip-if-complete logic and any post-hoc analysis both depend on
them staying local).

The curated set per run, deduplicated by file (categories in
:data:`CATEGORIES`):

1. **first** -- the earliest saved snapshot.
2. **leaked_onset** -- the first saved snapshot at/after the first epoch
   ``val/accuracy`` clears the stability threshold.
3. **unleaked_onset** -- the same, for the metric key whose name contains
   "unleak" (discovered from a run.log row -- this project's two training
   paths agree on ``val/unleaked_accuracy``, but the key is never hardcoded).
4. **intermediate** -- 1-2 saved snapshots strictly between the leaked- and
   unleaked-onset epochs (skipped when there is no such interval).
5. **last** -- ``final.pt``, or the latest saved snapshot if ``final.pt`` is
   missing (a forced-interrupted run).
6. **stable_end** -- :func:`~group_algorithm_interp.instruments.checkpoints.
   select_checkpoint`'s dip-aware pick, including its ``substitution``
   (``None``/``"window"``/``"trajectory"``) and ``rejected`` trail so a
   post-hoc reconstruction from ``selection.json`` alone never has to
   fabricate "no substitution"; ``None`` for a run with no stable checkpoint
   anywhere (recorded, not an error).

A run with ``val/accuracy`` never clearing the threshold is **censored**:
categories 2 and 4 are skipped (never populated), 1/5/6 still ship. Every
selection, its metric value, and every anomaly (a missing onset snapshot, a
degenerate onset ordering, a forced-interrupted "last") is written to
``selection.json`` and shipped alongside the checkpoints -- reported as data,
never hidden.

Verify-before-prune is a HARD INVARIANT (see the comment at the prune site in
:func:`ship_run`): local files are deleted only after *every* curated object
for that run has been confirmed present in S3 with a byte-matching
``head_object``. A previous campaign lost 50 runs to a prune that ran ahead
of verification; this ordering is deliberately not negotiable.

Resumable and idempotent: a local ledger (``<runs-root>/.ship_ledger.json``)
records every run that shipped, verified, and (if not empty) pruned; a
restart skips ledgered runs. A run only ever ships once its own manifest is
finalised (mirrors the completion check `scripts/stream_runs.py` and
`scripts/sync_runs.py` both use), so this daemon can run alongside training
without ever touching a run still in progress.

boto3 is a runtime-only dependency (imported lazily inside :class:`S3Sink`),
so this module -- and its test suite -- import and run with no network access
and no boto3 install, exactly like a plugin.

Examples::

    uv run python scripts/ship_runs.py --dry-run
    uv run python scripts/ship_runs.py --once
    uv run python scripts/ship_runs.py --interval 120
    uv run python scripts/ship_runs.py --complete-sentinel runs/../CAMPAIGN_COMPLETE
    uv run python scripts/ship_runs.py --force-interrupted --once
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from group_algorithm_interp.instruments.checkpoints import (  # noqa: E402
    CheckpointSelection,
    EpochRow,
    parse_run_log,
    select_checkpoint,
)
from group_algorithm_interp.manifest import (  # noqa: E402
    MANIFEST_NAME,
    RESOLVED_CONFIG_NAME,
    read_manifest,
)

RUN_LOG_NAME = "run.log"
CHECKPOINTS_DIRNAME = "checkpoints"
FINAL_CHECKPOINT_NAME = "final.pt"
SELECTION_FILENAME = "selection.json"
LEDGER_FILENAME = ".ship_ledger.json"

# A run's manifest reaching one of these (with `completed_at` set) is what
# "finalised" means; a run still `running` is never a ship candidate (same
# contract scripts/stream_runs.py and scripts/sync_runs.py use).
TERMINAL_STATUSES = frozenset({"completed", "failed", "aborted"})

LEAKED_METRIC_KEY = "val/accuracy"
DEFAULT_STABLE_THRESHOLD = 0.99
DEFAULT_KEY_PREFIX = "curated"
DEFAULT_INTERVAL_S = 60.0
DEFAULT_RETRY_ATTEMPTS = 4
DEFAULT_RETRY_BASE_DELAY_S = 2.0

CATEGORIES = (
    "first",
    "leaked_onset",
    "unleaked_onset",
    "intermediate",
    "last",
    "stable_end",
)

# Snapshot filename families, epoch encoded in the name -- the same families
# instruments/checkpoints.py and scripts/stream_runs.py glob for. Not
# exported by checkpoints.py (it only exposes the parser + selector, which
# this module reuses rather than reimplementing), so enumerated locally.
_SNAPSHOT_PREFIXES = ("final_epoch_", "generalized_step_", "step_")


def _warn(message: str) -> None:
    print(f"[ship_runs] {message}", file=sys.stderr)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- snapshot discovery --------------------------------------------------------


def snapshot_epochs(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    """``(epoch, path)`` for every epoch-named snapshot, ascending by epoch.
    ``final.pt`` carries no epoch in its name and is excluded."""
    found: dict[int, Path] = {}
    if not checkpoint_dir.is_dir():
        return []
    for path in checkpoint_dir.glob("*.pt"):
        for prefix in _SNAPSHOT_PREFIXES:
            if path.name.startswith(prefix):
                tail = path.name[len(prefix) : -len(".pt")]
                if tail.isdigit():
                    found.setdefault(int(tail), path)
                break
    return sorted(found.items())


def _snapshot_at_or_after(
    snapshots: Sequence[tuple[int, Path]], epoch: int
) -> tuple[int, Path] | None:
    for snap_epoch, path in snapshots:
        if snap_epoch >= epoch:
            return snap_epoch, path
    return None


def _pick_intermediate(
    candidates: Sequence[tuple[int, Path]],
) -> list[tuple[int, Path]]:
    """1-2 evenly spaced picks from ``candidates`` (already ascending)."""
    if not candidates:
        return []
    if len(candidates) <= 2:
        return list(candidates)
    first_idx = len(candidates) // 3
    second_idx = (2 * len(candidates)) // 3
    if second_idx == first_idx:
        second_idx = min(first_idx + 1, len(candidates) - 1)
    picks = [candidates[first_idx]]
    if second_idx != first_idx:
        picks.append(candidates[second_idx])
    return picks


def find_unleaked_metric_key(rows: Sequence[EpochRow]) -> str | None:
    """The metric key whose name contains "unleak" (case-insensitive),
    discovered from the first row's keys. ``None`` when there are no rows or
    none of its keys mention "unleak"."""
    if not rows:
        return None
    for key in rows[0].metrics:
        if "unleak" in key.lower():
            return key
    return None


def _first_onset(rows: Sequence[EpochRow], metric_key: str | None, threshold: float) -> int | None:
    """The first (ascending) epoch whose row clears ``metric_key >=
    threshold``, ignoring NaN rows (a degenerate empty-subset run records
    NaN, never "clear")."""
    if metric_key is None:
        return None
    for row in rows:
        value = row.metrics.get(metric_key)
        if value is not None and not math.isnan(value) and value >= threshold:
            return row.epoch
    return None


# --- curated selection ----------------------------------------------------------


@dataclass(frozen=True)
class CuratedEntry:
    category: str
    epoch: int | None
    metric_key: str | None
    metric_value: float | None
    path: Path
    # Only meaningful for "stable_end" (the dip-aware pick): whether the rule
    # substituted an earlier window epoch or a trajectory snapshot for a
    # dipped primary choice, and every candidate it passed over first. Absent
    # (None) for every other category, whose picks carry no substitution
    # semantics. Recording these mirrors what the flat selection.json shape
    # already preserves (CheckpointSelection.to_record), so a curated-shape
    # reconstruction is never left fabricating "no substitution" for a run
    # that genuinely dipped (see instruments/checkpoints.py's
    # _recompute_curated_substitution for the reconstruction this feeds).
    substitution: str | None = None
    rejected: list[dict[str, Any]] | None = None

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "epoch": self.epoch,
            "metric_key": self.metric_key,
            "metric_value": self.metric_value,
            "filename": self.path.name,
        }
        if self.category == "stable_end":
            record["substitution"] = self.substitution
            record["rejected"] = list(self.rejected) if self.rejected is not None else []
        return record


@dataclass
class CuratedSelection:
    run_id: str
    leaked_metric_key: str
    unleaked_metric_key: str | None
    threshold: float
    censored: bool
    interrupted: bool
    entries: list[CuratedEntry] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)

    def unique_files(self) -> list[Path]:
        """Every curated file, deduplicated by resolved local path (several
        categories can legitimately point at the same physical snapshot),
        first-seen order preserved."""
        seen: dict[Path, None] = {}
        for entry in self.entries:
            seen.setdefault(entry.path, None)
        return list(seen.keys())

    def to_record(self) -> dict[str, Any]:
        categories: dict[str, Any] = {name: None for name in CATEGORIES}
        categories["intermediate"] = []
        for entry in self.entries:
            if entry.category == "intermediate":
                categories["intermediate"].append(entry.to_record())
            else:
                categories[entry.category] = entry.to_record()
        return {
            "run_id": self.run_id,
            "leaked_metric_key": self.leaked_metric_key,
            "unleaked_metric_key": self.unleaked_metric_key,
            "threshold": self.threshold,
            "censored": self.censored,
            "interrupted": self.interrupted,
            "categories": categories,
            "anomalies": list(self.anomalies),
        }


def build_curated_selection(
    run_dir: Path,
    *,
    threshold: float = DEFAULT_STABLE_THRESHOLD,
    manifest_status: str | None = None,
) -> CuratedSelection:
    """The curated file selection for one finalised run (module docstring).
    Never mutates or reads ``checkpoints/`` beyond listing/stat-ing files."""
    run_id = run_dir.name
    log_path = run_dir / RUN_LOG_NAME
    ckpt_dir = run_dir / CHECKPOINTS_DIRNAME
    final_path = ckpt_dir / FINAL_CHECKPOINT_NAME
    anomalies: list[str] = []

    rows: list[EpochRow] = []
    if log_path.is_file():
        rows = parse_run_log(log_path)
        if not rows:
            anomalies.append("run.log has no epoch rows")
    else:
        anomalies.append("run.log not found")
    rows_by_epoch = {row.epoch: row.metrics for row in rows}

    unleaked_key = find_unleaked_metric_key(rows)
    if rows and unleaked_key is None:
        anomalies.append("no metric key containing 'unleak' found in run.log")

    snapshots = snapshot_epochs(ckpt_dir)
    if not snapshots:
        anomalies.append("no saved trajectory snapshots found")

    leaked_onset_epoch = _first_onset(rows, LEAKED_METRIC_KEY, threshold)
    unleaked_onset_epoch = _first_onset(rows, unleaked_key, threshold)
    censored = leaked_onset_epoch is None

    interrupted = (not final_path.is_file()) or (manifest_status in ("failed", "aborted"))

    entries: list[CuratedEntry] = []

    # 1. first
    if snapshots:
        epoch, path = snapshots[0]
        entries.append(CuratedEntry("first", epoch, None, None, path))

    # 2. leaked_onset
    leaked_snap: tuple[int, Path] | None = None
    if leaked_onset_epoch is not None:
        leaked_snap = _snapshot_at_or_after(snapshots, leaked_onset_epoch)
        if leaked_snap is None:
            anomalies.append(
                f"leaked onset at epoch {leaked_onset_epoch} has no saved snapshot at/after it"
            )
        else:
            epoch, path = leaked_snap
            entries.append(
                CuratedEntry(
                    "leaked_onset",
                    epoch,
                    LEAKED_METRIC_KEY,
                    rows_by_epoch.get(epoch, {}).get(LEAKED_METRIC_KEY),
                    path,
                )
            )

    # 3. unleaked_onset
    unleaked_snap: tuple[int, Path] | None = None
    if unleaked_onset_epoch is not None:
        unleaked_snap = _snapshot_at_or_after(snapshots, unleaked_onset_epoch)
        if unleaked_snap is None:
            anomalies.append(
                f"unleaked onset at epoch {unleaked_onset_epoch} has no saved snapshot at/after it"
            )
        else:
            epoch, path = unleaked_snap
            entries.append(
                CuratedEntry(
                    "unleaked_onset",
                    epoch,
                    unleaked_key,
                    rows_by_epoch.get(epoch, {}).get(unleaked_key) if unleaked_key else None,
                    path,
                )
            )

    # 4. intermediate -- only when both onsets landed on a real snapshot and
    # are properly ordered; a censored run (no leaked_snap) skips this too.
    if leaked_snap is not None and unleaked_snap is not None:
        lo, hi = leaked_onset_epoch, unleaked_onset_epoch
        assert lo is not None and hi is not None
        if hi <= lo:
            anomalies.append(
                f"unleaked-onset epoch ({hi}) <= leaked-onset epoch ({lo}); "
                "skipping intermediate selection"
            )
        else:
            between = [(e, p) for e, p in snapshots if lo < e < hi]
            for epoch, path in _pick_intermediate(between):
                entries.append(
                    CuratedEntry(
                        "intermediate",
                        epoch,
                        LEAKED_METRIC_KEY,
                        rows_by_epoch.get(epoch, {}).get(LEAKED_METRIC_KEY),
                        path,
                    )
                )
    elif not censored and unleaked_onset_epoch is None:
        anomalies.append(
            "no unleaked-onset epoch found (metric never reached threshold); "
            "skipping unleaked-onset/intermediate"
        )

    # 5. last
    if final_path.is_file():
        last_epoch = rows[-1].epoch if rows else None
        entries.append(CuratedEntry("last", last_epoch, None, None, final_path))
    elif snapshots:
        epoch, path = snapshots[-1]
        anomalies.append("final.pt missing; 'last' falls back to the latest saved snapshot")
        entries.append(CuratedEntry("last", epoch, None, None, path))
    else:
        anomalies.append("neither final.pt nor any saved snapshot found for 'last'")

    # 6. stable_end
    stable: CheckpointSelection | None = None
    if (run_dir / RESOLVED_CONFIG_NAME).is_file() and log_path.is_file():
        try:
            stable = select_checkpoint(run_dir, metric=LEAKED_METRIC_KEY, threshold=threshold)
        except Exception as exc:  # noqa: BLE001 -- recorded as an anomaly, never fatal
            anomalies.append(f"select_checkpoint failed: {type(exc).__name__}: {exc}")
    else:
        anomalies.append("stable_end: resolved_config.yaml or run.log missing")
    if stable is not None:
        if stable.path is not None:
            entries.append(
                CuratedEntry(
                    "stable_end",
                    stable.epoch,
                    stable.metric,
                    stable.metric_value,
                    stable.path,
                    substitution=stable.substitution,
                    rejected=list(stable.rejected),
                )
            )
        else:
            anomalies.append(f"stable_end: {stable.reason}")

    return CuratedSelection(
        run_id=run_id,
        leaked_metric_key=LEAKED_METRIC_KEY,
        unleaked_metric_key=unleaked_key,
        threshold=threshold,
        censored=censored,
        interrupted=interrupted,
        entries=entries,
        anomalies=anomalies,
    )


# --- completion detection --------------------------------------------------------


def discover_run_dirs(runs_root: Path) -> list[Path]:
    """Every immediate child of ``runs_root`` that looks like a run directory
    (holds its own ``manifest.yaml``); naturally excludes the ledger file and
    any ``schedules/``-style sibling."""
    if not runs_root.is_dir():
        return []
    return sorted(
        path for path in runs_root.iterdir() if path.is_dir() and (path / MANIFEST_NAME).is_file()
    )


def is_shippable(run_dir: Path, *, force_interrupted: bool = False) -> tuple[bool, str | None]:
    """Whether ``run_dir`` may ship now, and if not, why.

    Default: the manifest must be finalised (a terminal ``status`` with
    ``completed_at`` set) AND ``checkpoints/final.pt`` must exist -- a run
    still training, or one that crashed before writing its final checkpoint,
    is never a candidate. ``force_interrupted=True`` also allows a finalised
    run with no ``final.pt`` (its curated 'last' category then falls back to
    the latest saved trajectory snapshot, and ``selection.json`` records
    ``interrupted: true``)."""
    try:
        manifest = read_manifest(run_dir)
    except Exception as exc:  # noqa: BLE001 -- a corrupt manifest blocks shipping, not the loop
        return False, f"manifest unreadable ({type(exc).__name__}: {exc})"
    status = manifest.get("status")
    finalised = status in TERMINAL_STATUSES and manifest.get("completed_at") is not None
    if not finalised:
        return False, f"manifest not finalised (status={status!r})"
    if (run_dir / CHECKPOINTS_DIRNAME / FINAL_CHECKPOINT_NAME).is_file():
        return True, None
    if force_interrupted:
        return True, None
    return (
        False,
        "manifest finalised but final.pt missing (pass --force-interrupted to ship anyway)",
    )


# --- ledger -----------------------------------------------------------------------


def ledger_path_for(runs_root: Path) -> Path:
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


def is_ledgered(ledger: dict[str, Any], run_id: str) -> bool:
    return run_id in ledger.get("runs", {})


def record_shipped(
    ledger: dict[str, Any], run_id: str, *, pruned: bool, files: Sequence[str], when: str
) -> None:
    ledger.setdefault("runs", {})[run_id] = {
        "shipped_at": when,
        "pruned": pruned,
        "files": sorted(files),
    }


# --- the S3 seam --------------------------------------------------------------


class S3Sink:
    """The one boundary that talks to S3. ``boto3`` is imported lazily here
    (a runtime-only dependency, like a plugin) so the module -- and its test
    suite -- import fine with no network and no boto3 install. Every method
    may raise; the caller (:func:`ship_run`'s retry wrapper) decides whether
    to retry. Tests inject a fake with the same surface."""

    def __init__(self, bucket: str, *, endpoint_url: str | None, region_name: str | None):
        import boto3  # type: ignore[import-not-found]

        # Credentials are never read or logged here -- boto3's default chain
        # picks up AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY from the
        # environment on its own.
        self._client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region_name)
        self.bucket = bucket

    def put_file(self, key: str, local_path: Path) -> None:
        self._client.upload_file(str(local_path), self.bucket, key)

    def head(self, key: str) -> int | None:
        """``ContentLength`` of the object at ``key``, or ``None`` if it does
        not exist (a 404/NoSuchKey ``head_object``)."""
        import botocore.exceptions  # type: ignore[import-not-found]

        try:
            response = self._client.head_object(Bucket=self.bucket, Key=key)
        except botocore.exceptions.ClientError as exc:
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = exc.response.get("Error", {}).get("Code")
            if status == 404 or code in ("404", "NoSuchKey", "NotFound"):
                return None
            raise
        return int(response["ContentLength"])


def _retry(
    fn: Callable[[], Any],
    *,
    attempts: int,
    base_delay: float,
    warn: Callable[[str], None],
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Call ``fn`` with bounded exponential-backoff retry. The final
    attempt's exception propagates uncaught -- the caller treats that as a
    hard failure for this run's shipment (never a reason to prune)."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 -- retried; re-raised on the last attempt
            if attempt == attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            warn(
                f"transient error ({type(exc).__name__}: {exc}); retrying in {delay:.1f}s "
                f"(attempt {attempt}/{attempts})"
            )
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


# --- shipping one run -----------------------------------------------------------


@dataclass(frozen=True)
class UploadItem:
    key_suffix: str  # filename under <prefix>/<run_id>/
    local_path: Path


@dataclass
class ShipOutcome:
    run_id: str
    ok: bool
    pruned: bool
    reason: str | None = None
    uploaded: list[str] = field(default_factory=list)


def build_upload_items(
    run_dir: Path, selection: CuratedSelection, tmp_dir: Path
) -> list[UploadItem]:
    """Every file this run ships: the deduplicated curated checkpoints, the
    provenance files verbatim, ``run.log`` gzipped, and ``selection.json``.
    The gzip and selection files are materialised under ``tmp_dir`` (never
    written into the run directory) so a killed pass leaves no stray files
    for training or the campaign's skip-if-complete logic to trip over."""
    items = [UploadItem(path.name, path) for path in selection.unique_files()]

    manifest_path = run_dir / MANIFEST_NAME
    if manifest_path.is_file():
        items.append(UploadItem(MANIFEST_NAME, manifest_path))
    resolved_path = run_dir / RESOLVED_CONFIG_NAME
    if resolved_path.is_file():
        items.append(UploadItem(RESOLVED_CONFIG_NAME, resolved_path))

    log_path = run_dir / RUN_LOG_NAME
    if log_path.is_file():
        gz_path = tmp_dir / "run.log.gz"
        with log_path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        items.append(UploadItem("run.log.gz", gz_path))

    selection_path = tmp_dir / SELECTION_FILENAME
    selection_path.write_text(json.dumps(selection.to_record(), indent=2, sort_keys=True) + "\n")
    items.append(UploadItem(SELECTION_FILENAME, selection_path))

    return items


def ship_run(
    run_dir: Path,
    *,
    sink: Any,
    key_prefix: str,
    threshold: float,
    manifest_status: str | None,
    dry_run: bool,
    warn: Callable[[str], None] = _warn,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY_S,
) -> ShipOutcome:
    """Ship one finalised run's curated files and, only once every one of
    them verifies, prune its local ``checkpoints/``. Caller must already have
    confirmed :func:`is_shippable`."""
    run_id = run_dir.name
    selection = build_curated_selection(
        run_dir, threshold=threshold, manifest_status=manifest_status
    )
    for anomaly in selection.anomalies:
        warn(f"{run_id}: {anomaly}")

    with tempfile.TemporaryDirectory(prefix="ship-runs-") as tmp:
        items = build_upload_items(run_dir, selection, Path(tmp))

        if dry_run:
            for item in items:
                warn(
                    f"[dry-run] {run_id}: would upload {item.key_suffix} "
                    f"({item.local_path.stat().st_size} bytes)"
                )
            warn(f"[dry-run] {run_id}: would prune checkpoints/ once every upload verifies")
            return ShipOutcome(
                run_id, ok=True, pruned=False, uploaded=[item.key_suffix for item in items]
            )

        uploaded: list[str] = []
        for item in items:
            key = f"{key_prefix}/{run_id}/{item.key_suffix}"
            try:
                _retry(
                    lambda: sink.put_file(key, item.local_path),
                    attempts=retry_attempts,
                    base_delay=retry_base_delay,
                    warn=warn,
                )
            except Exception as exc:  # noqa: BLE001 -- reported; local files are untouched
                return ShipOutcome(
                    run_id,
                    ok=False,
                    pruned=False,
                    reason=f"upload failed for {item.key_suffix}: {type(exc).__name__}: {exc}",
                    uploaded=uploaded,
                )
            uploaded.append(item.key_suffix)

        # -------------------------------------------------------------------
        # HARD INVARIANT -- verify before prune. Do not weaken this.
        #
        # Every curated object just uploaded for this run MUST be confirmed
        # present in S3, via head_object, with a ContentLength that matches
        # the local file's size, BEFORE any local file is deleted. This is
        # not an optimisation to skip under time pressure: a previous
        # campaign lost 50 runs' trajectories to a prune that ran ahead of
        # verification. If any item fails to verify, this function returns
        # immediately with ok=False and NOTHING local is touched -- the run
        # stays off the ledger and is retried on the next poll.
        # -------------------------------------------------------------------
        for item in items:
            key = f"{key_prefix}/{run_id}/{item.key_suffix}"
            try:
                remote_size = _retry(
                    lambda: sink.head(key),
                    attempts=retry_attempts,
                    base_delay=retry_base_delay,
                    warn=warn,
                )
            except Exception as exc:  # noqa: BLE001 -- verify failed; local files are untouched
                return ShipOutcome(
                    run_id,
                    ok=False,
                    pruned=False,
                    reason=f"verify failed for {item.key_suffix}: {type(exc).__name__}: {exc}",
                    uploaded=uploaded,
                )
            local_size = item.local_path.stat().st_size
            if remote_size != local_size:
                return ShipOutcome(
                    run_id,
                    ok=False,
                    pruned=False,
                    reason=(
                        f"verify mismatch for {item.key_suffix}: "
                        f"remote={remote_size!r} local={local_size}"
                    ),
                    uploaded=uploaded,
                )

    # Every curated object verified present with a matching size. Only now
    # may the local trajectory be pruned.
    checkpoints_dir = run_dir / CHECKPOINTS_DIRNAME
    if checkpoints_dir.is_dir():
        shutil.rmtree(checkpoints_dir)
    return ShipOutcome(run_id, ok=True, pruned=True, uploaded=uploaded)


# --- the daemon loop --------------------------------------------------------------


class Daemon:
    """Discovers finalised runs under ``runs_root``, ships each one's curated
    subset, and prunes once verified. Safe to run alongside training (only
    ever touches a run whose manifest is finalised) and safe to restart mid-
    campaign (ledgered runs are skipped; an interrupted shipment is retried
    from scratch on the next poll -- uploads are idempotent overwrites)."""

    def __init__(
        self,
        runs_root: Path,
        sink: Any,
        *,
        key_prefix: str = DEFAULT_KEY_PREFIX,
        threshold: float = DEFAULT_STABLE_THRESHOLD,
        force_interrupted: bool = False,
        dry_run: bool = False,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY_S,
        warn: Callable[[str], None] = _warn,
    ):
        self.runs_root = runs_root
        self.sink = sink
        self.key_prefix = key_prefix
        self.threshold = threshold
        self.force_interrupted = force_interrupted
        self.dry_run = dry_run
        self.retry_attempts = retry_attempts
        self.retry_base_delay = retry_base_delay
        self.warn = warn
        self.ledger_path = ledger_path_for(runs_root)
        self.ledger = load_ledger(self.ledger_path)

    def poll_once(self) -> list[ShipOutcome]:
        outcomes: list[ShipOutcome] = []
        for run_dir in discover_run_dirs(self.runs_root):
            run_id = run_dir.name
            if is_ledgered(self.ledger, run_id):
                continue
            shippable, reason = is_shippable(run_dir, force_interrupted=self.force_interrupted)
            if not shippable:
                continue  # still training, or waiting on --force-interrupted; retried next poll
            try:
                manifest = read_manifest(run_dir)
            except Exception as exc:  # noqa: BLE001 -- unreadable manifest, skip this pass
                self.warn(f"{run_id}: manifest unreadable ({type(exc).__name__}: {exc})")
                continue
            try:
                outcome = ship_run(
                    run_dir,
                    sink=self.sink,
                    key_prefix=self.key_prefix,
                    threshold=self.threshold,
                    manifest_status=manifest.get("status"),
                    dry_run=self.dry_run,
                    warn=self.warn,
                    retry_attempts=self.retry_attempts,
                    retry_base_delay=self.retry_base_delay,
                )
            except Exception as exc:  # noqa: BLE001 -- one run's failure must not stop the loop
                self.warn(f"{run_id}: ship failed unexpectedly ({type(exc).__name__}: {exc})")
                continue
            outcomes.append(outcome)
            if outcome.ok:
                if not self.dry_run:
                    record_shipped(
                        self.ledger,
                        run_id,
                        pruned=outcome.pruned,
                        files=outcome.uploaded,
                        when=_utcnow(),
                    )
                    save_ledger(self.ledger_path, self.ledger)
                    self.warn(f"{run_id}: shipped and verified ({len(outcome.uploaded)} file(s))")
            else:
                self.warn(
                    f"{run_id}: shipment incomplete ({outcome.reason}); "
                    "local checkpoints preserved, will retry next poll"
                )
        return outcomes

    def run_forever(
        self,
        poll_seconds: float,
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
            sleep(poll_seconds)


# --- CLI -----------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--interval", type=float, default=DEFAULT_INTERVAL_S, help="seconds between polls"
    )
    parser.add_argument(
        "--once", action="store_true", help="process the current backlog once, then exit"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="log what would ship/prune; touch nothing"
    )
    parser.add_argument(
        "--key-prefix", default=DEFAULT_KEY_PREFIX, help="S3 key prefix under the bucket"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_STABLE_THRESHOLD,
        help="generalisation bar used for onset/stable-end selection",
    )
    parser.add_argument(
        "--force-interrupted",
        action="store_true",
        help="also ship finalised runs whose final.pt is missing (falls back to the "
        "latest saved snapshot for 'last'; flagged interrupted in selection.json)",
    )
    parser.add_argument(
        "--complete-sentinel",
        type=Path,
        default=None,
        help="a file whose appearance ends the loop after one final sweep "
        "(e.g. the campaign's CAMPAIGN_COMPLETE marker)",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("GAI_S3_BUCKET"),
        help="S3 bucket (default: $GAI_S3_BUCKET)",
    )
    parser.add_argument(
        "--endpoint-url",
        default=os.environ.get("GAI_S3_ENDPOINT"),
        help="S3 endpoint URL (default: $GAI_S3_ENDPOINT)",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("GAI_S3_REGION"),
        help="S3 region (default: $GAI_S3_REGION)",
    )
    parser.add_argument("--retry-attempts", type=int, default=DEFAULT_RETRY_ATTEMPTS)
    parser.add_argument("--retry-base-delay", type=float, default=DEFAULT_RETRY_BASE_DELAY_S)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.interval <= 0:
        parser.error("--interval must be positive")
    if not args.dry_run and not args.bucket:
        parser.error("no S3 bucket configured: pass --bucket or set GAI_S3_BUCKET")

    sink: Any = None
    if not args.dry_run:
        sink = S3Sink(args.bucket, endpoint_url=args.endpoint_url, region_name=args.region)

    daemon = Daemon(
        args.runs_root,
        sink,
        key_prefix=args.key_prefix,
        threshold=args.threshold,
        force_interrupted=args.force_interrupted,
        dry_run=args.dry_run,
        retry_attempts=args.retry_attempts,
        retry_base_delay=args.retry_base_delay,
    )
    try:
        daemon.run_forever(args.interval, once=args.once, complete_sentinel=args.complete_sentinel)
    except KeyboardInterrupt:
        _warn("interrupted; ledgered runs stay skipped on restart, unfinished ones retry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
