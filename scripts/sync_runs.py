#!/usr/bin/env python3
"""Get completed runs off a pod before it is destroyed.

``runs/`` is gitignored and a rented pod's disk does not survive teardown, so
this is the only path a checkpoint, manifest, or trajectory snapshot has off
the machine that produced it.

Two independent transports, either or both in one invocation:

* **W&B artifacts** (default, unless ``--no-wandb``): each completed run
  becomes one artifact named ``run-<run_id>``, with the run's manifest
  attached as artifact metadata.
* **An object store via ``rclone``** (``--rclone-target``): copies the same
  files to ``<target>/<run_id>/``. The bucket/host is never hardcoded here --
  it comes from ``--rclone-target`` or ``GAI_RCLONE_TARGET``, because the repo
  cannot know which pod's target is live.

A run is a candidate only once its manifest reaches a **terminal** status
(``completed``/``failed``/``aborted``) -- a run still ``running`` is never
touched, so a mid-flight checkpoint is never shipped half-written.

Idempotent: a local ledger (``<runs-dir>/.sync_ledger.json``) records which
run/transport/category-set combinations have already gone out, so a repeated
invocation (e.g. from cron) costs nothing once a run's requested files are
already synced. ``--force`` bypasses the ledger.

Examples::

    uv run python scripts/sync_runs.py --dry-run
    uv run python scripts/sync_runs.py
    uv run python scripts/sync_runs.py --light
    uv run python scripts/sync_runs.py --no-wandb --rclone-target s3remote:my-bucket/gai-runs
    uv run python scripts/sync_runs.py --include manifest,config,checkpoint

``WANDB_MODE=offline`` or ``WANDB_MODE=disabled`` in the environment makes the
W&B transport print why and exit cleanly rather than attempt any network call
-- this is what keeps ``just gate`` (which forces ``WANDB_MODE=disabled``)
offline-green.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from group_algorithm_interp.manifest import (  # noqa: E402
    MANIFEST_NAME,
    RESOLVED_CONFIG_NAME,
    read_manifest,
)

ROOT = Path(__file__).resolve().parent.parent

CHECKPOINTS_DIRNAME = "checkpoints"
FINAL_CHECKPOINT_NAME = "final.pt"
RUN_LOG_NAME = "run.log"  # written by logging_utils.setup_logging
LEDGER_FILENAME = ".sync_ledger.json"

# What "complete" means: a run's own manifest has reached one of these. A
# `running` run (or a manifest with any other/missing status) is never a sync
# candidate -- uploading it would ship a checkpoint mid-write.
TERMINAL_STATUSES = frozenset({"completed", "failed", "aborted"})

# manifest+config+summary+checkpoint+snapshots -- the five file groups a run
# directory can contribute. "summary" is run.log, the per-epoch eval-history
# text log; the manifest's own numeric summary block travels as artifact
# metadata regardless of which categories are selected (see sync_wandb).
ALL_CATEGORIES: tuple[str, ...] = ("manifest", "config", "summary", "checkpoint", "snapshots")
LIGHT_CATEGORIES: tuple[str, ...] = ("manifest", "summary")

DEFAULT_WANDB_PROJECT = "group-algorithm-interp"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_terminal_status(status: object) -> bool:
    return isinstance(status, str) and status in TERMINAL_STATUSES


def discover_runs(runs_dir: Path) -> list[Path]:
    """Every immediate child of ``runs_dir`` that looks like a run directory
    (holds its own ``manifest.yaml``). Naturally excludes ``runs/schedules/``
    (campaign manifests, not per-seed run directories) and the ledger file."""
    if not runs_dir.is_dir():
        return []
    return sorted(
        path for path in runs_dir.iterdir() if path.is_dir() and (path / MANIFEST_NAME).is_file()
    )


def _existing(path: Path) -> list[Path]:
    return [path] if path.is_file() else []


def _snapshot_files(checkpoints_dir: Path) -> list[Path]:
    """Every trajectory snapshot (``step_N.pt``, ``generalized_step_N.pt``)
    written under ``checkpoints/`` -- everything except ``final.pt``, which is
    its own category."""
    if not checkpoints_dir.is_dir():
        return []
    return sorted(p for p in checkpoints_dir.glob("*.pt") if p.name != FINAL_CHECKPOINT_NAME)


def resolve_included_files(run_dir: Path, categories: Iterable[str]) -> dict[str, list[Path]]:
    """Map each requested category to the files it actually resolves to on
    disk for this run. A category resolves to an empty list (not an error)
    when the file was never written -- e.g. ``snapshot.enabled=false`` leaves
    no trajectory snapshots, and that is a legitimate run, not a broken one."""
    checkpoints_dir = run_dir / CHECKPOINTS_DIRNAME
    resolvers: dict[str, Callable[[], list[Path]]] = {
        "manifest": lambda: _existing(run_dir / MANIFEST_NAME),
        "config": lambda: _existing(run_dir / RESOLVED_CONFIG_NAME),
        "summary": lambda: _existing(run_dir / RUN_LOG_NAME),
        "checkpoint": lambda: _existing(checkpoints_dir / FINAL_CHECKPOINT_NAME),
        "snapshots": lambda: _snapshot_files(checkpoints_dir),
    }
    return {category: resolvers[category]() for category in categories if category in resolvers}


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    run_dir: Path
    status: str
    manifest: dict[str, Any]
    files: dict[str, list[Path]]

    def file_count(self) -> int:
        return sum(len(paths) for paths in self.files.values())


def collect_run_records(
    runs_dir: Path,
    categories: Sequence[str],
    warn: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr),
) -> list[RunRecord]:
    """Every completed (terminal-status) run under ``runs_dir``, with its
    included files already resolved. A run whose manifest cannot be read is
    skipped with a warning rather than aborting the whole scan -- one corrupt
    directory should not block syncing every other run."""
    records: list[RunRecord] = []
    for run_dir in discover_runs(runs_dir):
        try:
            manifest = read_manifest(run_dir)
        except Exception as exc:
            warn(f"skipping {run_dir.name}: manifest unreadable ({type(exc).__name__}: {exc})")
            continue
        status = manifest.get("status")
        if not is_terminal_status(status):
            continue  # includes `running` -- never a sync candidate
        assert isinstance(status, str)  # narrowed by is_terminal_status, for mypy
        records.append(
            RunRecord(
                run_id=run_dir.name,
                run_dir=run_dir,
                status=status,
                manifest=manifest,
                files=resolve_included_files(run_dir, categories),
            )
        )
    return records


# --- ledger ------------------------------------------------------------------


def ledger_path_for(runs_dir: Path) -> Path:
    return runs_dir / LEDGER_FILENAME


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"runs": {}}
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("runs"), dict):
        raise ValueError(f"ledger at {path} is not a mapping with a 'runs' key")
    return data


def save_ledger(path: Path, ledger: dict[str, Any]) -> None:
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")


def transport_key(transport: str, discriminator: str) -> str:
    """Namespaces a ledger entry by transport AND destination (W&B
    project/entity, or the rclone target). A run already synced to one rclone
    target must still upload when pointed at a different one."""
    return f"{transport}:{discriminator}"


def already_synced(
    ledger: dict[str, Any], run_id: str, key: str, categories: Iterable[str]
) -> bool:
    """True iff every requested category was already recorded for this
    run/transport-destination. Requesting a *superset* of what was previously
    shipped (e.g. asking for snapshots after only ever running ``--light``)
    still counts as needing a sync."""
    entry = ledger.get("runs", {}).get(run_id, {}).get(key)
    if entry is None:
        return False
    already = set(entry.get("categories", []))
    return set(categories) <= already


def record_synced(
    ledger: dict[str, Any], run_id: str, key: str, categories: Iterable[str], when: str
) -> None:
    runs = ledger.setdefault("runs", {})
    run_entry = runs.setdefault(run_id, {})
    existing = set(run_entry.get(key, {}).get("categories", []))
    run_entry[key] = {"categories": sorted(existing | set(categories)), "synced_at": when}


# --- transports ----------------------------------------------------------------


@dataclass
class TransportOutcome:
    synced: list[str] = field(default_factory=list)
    # True when the transport did nothing FOR A GLOBAL REASON (e.g.
    # WANDB_MODE=offline/disabled) rather than a per-run failure -- callers
    # must not treat this as an error or write a ledger entry for it.
    skipped: bool = False


def wandb_skip_reason() -> str | None:
    """``WANDB_MODE`` as the real ``wandb`` SDK reads it. 'disabled' and
    'offline' both mean "do not attempt a network call" here -- offline
    artifact queuing (for a later `wandb sync`) is deliberately not
    implemented, since this script is meant to run unattended (cron) and
    nothing else would ever flush an offline queue."""
    mode = os.environ.get("WANDB_MODE", "online").lower()
    return mode if mode in ("disabled", "offline") else None


def verify_artifact_bytes(api: Any, qualified_name: str) -> None:
    """Refuse to call an upload done until a byte comes back.

    ``run.log_artifact`` is asynchronous, and even a completed session can
    lie: on 2026-07-15 every upload in a session 403'd at the storage layer
    (GCS ``SignatureDoesNotMatch``) while the SDK still printed a "Synced N
    artifact file(s)" summary, so the ledger recorded runs whose artifacts
    hold no fetchable bytes -- and the snapshot pruner then deleted the only
    other copy. Fetch the freshly-committed artifact and download its
    smallest entry; loading the artifact manifest is itself a second,
    independent byte read (``wandb_manifest.json``)."""
    artifact = api.artifact(qualified_name, type="run-output")
    entries = artifact.manifest.entries
    if not entries:
        return
    smallest = min(entries, key=lambda name: entries[name].size or 0)
    with tempfile.TemporaryDirectory(prefix="sync-runs-verify-") as tmp:
        artifact.get_entry(smallest).download(root=tmp)


def sync_wandb(
    records: Sequence[RunRecord],
    *,
    project: str,
    entity: str | None,
    dry_run: bool,
) -> TransportOutcome:
    """Upload each record as a W&B artifact named ``run-<run_id>``, with the
    run's manifest as artifact metadata (attached regardless of which file
    categories were selected -- it is cheap and it is the provenance record).
    A run counts as synced only once its artifact is *committed*
    (``Artifact.wait()``) **and** a byte fetch of the committed artifact
    succeeds (``verify_artifact_bytes``) -- ``log_artifact`` alone is
    asynchronous and proves nothing. A per-run failure is logged and
    skipped; it does not abort the batch."""
    skip = wandb_skip_reason()
    if skip is not None:
        print(f"wandb: WANDB_MODE={skip}; skipping ({len(records)} run(s) not uploaded)")
        return TransportOutcome(skipped=True)
    if not records:
        return TransportOutcome()
    if dry_run:
        for record in records:
            print(
                f"[dry-run] wandb: would upload run-{record.run_id} ({record.file_count()} file(s))"
            )
        return TransportOutcome(synced=[r.run_id for r in records])

    import wandb

    run = wandb.init(project=project, entity=entity, job_type="sync-runs")
    api = wandb.Api()
    synced: list[str] = []
    try:
        for record in records:
            try:
                artifact = wandb.Artifact(
                    name=f"run-{record.run_id}", type="run-output", metadata=record.manifest
                )
                for paths in record.files.values():
                    for path in paths:
                        artifact.add_file(str(path), name=str(path.relative_to(record.run_dir)))
                run.log_artifact(artifact)
                artifact.wait()  # block until committed; raises if any upload failed
                verify_artifact_bytes(api, artifact.qualified_name)
                print(
                    f"wandb: uploaded run-{record.run_id} "
                    f"({record.file_count()} file(s), committed and byte-verified)"
                )
                synced.append(record.run_id)
            except Exception as exc:
                print(
                    f"wandb: FAILED for {record.run_id}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
    finally:
        run.finish()
    return TransportOutcome(synced=synced)


def sync_rclone(
    records: Sequence[RunRecord],
    *,
    target: str,
    dry_run: bool,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> TransportOutcome:
    """``rclone copy`` each record's included files to ``<target>/<run_id>/``,
    preserving each file's path relative to the run directory. One rclone
    invocation per run, driven by ``--files-from`` so only the requested
    categories move (not the whole run directory). A per-run failure is
    logged and skipped; it does not abort the batch."""
    if not records:
        return TransportOutcome()
    synced: list[str] = []
    for record in records:
        relative_paths = sorted(
            str(path.relative_to(record.run_dir))
            for paths in record.files.values()
            for path in paths
        )
        if not relative_paths:
            continue
        destination = f"{target.rstrip('/')}/{record.run_id}"
        if dry_run:
            print(f"[dry-run] rclone: would copy {len(relative_paths)} file(s) to {destination}")
            synced.append(record.run_id)
            continue
        fd, list_name = tempfile.mkstemp(suffix=".txt", prefix="sync-runs-files-")
        list_path = Path(list_name)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write("\n".join(relative_paths) + "\n")
            result = run(
                [
                    "rclone",
                    "copy",
                    str(record.run_dir),
                    destination,
                    "--files-from",
                    str(list_path),
                ],
                capture_output=True,
                text=True,
            )
        finally:
            list_path.unlink(missing_ok=True)
        if result.returncode != 0:
            print(f"rclone: FAILED for {record.run_id}: {result.stderr.strip()}", file=sys.stderr)
            continue
        print(f"rclone: copied {record.run_id} ({len(relative_paths)} file(s)) -> {destination}")
        synced.append(record.run_id)
    return TransportOutcome(synced=synced)


# --- CLI -------------------------------------------------------------------


def _parse_categories(value: str) -> tuple[str, ...]:
    categories = tuple(sorted({piece.strip() for piece in value.split(",") if piece.strip()}))
    unknown = [c for c in categories if c not in ALL_CATEGORIES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown --include categor{'y' if len(unknown) == 1 else 'ies'}: "
            f"{', '.join(unknown)}; choose from {', '.join(ALL_CATEGORIES)}"
        )
    return categories


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=ROOT / "runs", help="Default: <repo root>/runs"
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--include",
        type=_parse_categories,
        metavar="CATEGORY,...",
        help=(
            "Comma-separated subset of {manifest,config,summary,checkpoint,snapshots} to ship "
            "(default: all five). 'summary' is run.log, the eval-history text log."
        ),
    )
    selection.add_argument(
        "--light",
        action="store_true",
        help="Shorthand for --include manifest,summary -- cheap, frequent-sync mode.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be synced; upload nothing."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-sync even if the ledger already has this run/transport/category-set.",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        help="Override the ledger path (default: <runs-dir>/.sync_ledger.json).",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Skip the W&B artifact transport entirely."
    )
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--rclone-target",
        help=(
            "rclone remote:path prefix to sync to, e.g. s3remote:my-bucket/gai-runs. Never "
            "hardcoded here -- the repo cannot know a pod's target -- so this or "
            "GAI_RCLONE_TARGET must supply it."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    categories: tuple[str, ...] = args.include or (
        LIGHT_CATEGORIES if args.light else ALL_CATEGORIES
    )
    rclone_target = args.rclone_target or os.environ.get("GAI_RCLONE_TARGET")
    wandb_enabled = not args.no_wandb
    if not wandb_enabled and not rclone_target:
        parser.error(
            "nothing to do: pass --rclone-target (or set GAI_RCLONE_TARGET), or drop --no-wandb"
        )

    runs_dir: Path = args.runs_dir
    records = collect_run_records(runs_dir, categories)
    if not records:
        print(f"no completed runs (status in {sorted(TERMINAL_STATUSES)}) found under {runs_dir}")
        return 0

    ledger_path = args.ledger or ledger_path_for(runs_dir)
    ledger = load_ledger(ledger_path)
    now = _utcnow()
    any_partial_failure = False
    ledger_dirty = False

    if wandb_enabled:
        key = transport_key("wandb", f"{args.wandb_entity or ''}/{args.wandb_project}")
        pending = [
            r
            for r in records
            if args.force or not already_synced(ledger, r.run_id, key, categories)
        ]
        if not pending:
            print("wandb: nothing to sync (every candidate run already recorded in the ledger)")
        else:
            outcome = sync_wandb(
                pending, project=args.wandb_project, entity=args.wandb_entity, dry_run=args.dry_run
            )
            if not args.dry_run and not outcome.skipped:
                for run_id in outcome.synced:
                    record_synced(ledger, run_id, key, categories, now)
                ledger_dirty = ledger_dirty or bool(outcome.synced)
                if len(outcome.synced) < len(pending):
                    any_partial_failure = True

    if rclone_target:
        key = transport_key("rclone", rclone_target)
        pending = [
            r
            for r in records
            if args.force or not already_synced(ledger, r.run_id, key, categories)
        ]
        if not pending:
            print("rclone: nothing to sync (every candidate run already recorded in the ledger)")
        else:
            outcome = sync_rclone(pending, target=rclone_target, dry_run=args.dry_run)
            if not args.dry_run:
                for run_id in outcome.synced:
                    record_synced(ledger, run_id, key, categories, now)
                ledger_dirty = ledger_dirty or bool(outcome.synced)
                if len(outcome.synced) < len(pending):
                    any_partial_failure = True

    if ledger_dirty:
        save_ledger(ledger_path, ledger)

    return 1 if any_partial_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
