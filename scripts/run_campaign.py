#!/usr/bin/env python3
"""Run (or resume) the pre-registered core-study campaign.

Reads the ordered cell list from a campaign file (default
``configs/campaign/core.yaml`` -- see that file's header for what a "cell"
is and why the cells are ordered the way they are), skips any cell whose
full seed range already has
a ``completed`` manifest for every seed under ``runs/``, and invokes
``scripts/run_batch.py`` once per remaining cell.

``scripts/run_batch.py`` is built and owned by a concurrent effort; this
module only depends on its documented contract, never its internals:

* Hydra-composed like ``scripts/run.py`` -- ``experiment=core``,
  ``data.group.order=...``, ``data.group.index=...``, ``model.d_model=...``
  are plain ``key=value`` overrides, composed the same way ``scripts/run.py``
  takes them on its command line.
* ``--seeds START:STOP`` (Python slice convention, ``STOP`` exclusive)
  selects the seed range for the whole cell in one invocation.
* Exit code ``0`` iff every seed in that range reached a ``completed``
  manifest; non-zero otherwise.
* One ``runs/<run_id>/manifest.yaml`` + ``resolved_config.yaml`` per trained
  model, in the same shape ``scripts/run.py``/``manifest.py`` write -- which
  is what the completion scan below reads. This script never imports
  ``run_batch`` itself, only shells out to it.

The bonus phases (``bonus-e6``/``bonus-e2`` in the campaign file -- extension
supply, not part of the pre-registered core) exist only under
``--include-bonus``: without the flag they are excluded entirely, including
from ``--dry-run`` listings.

``--shard i/n`` partitions the cell list deterministically across ``n``
workers (see :func:`shard_cells`), so eight GPUs can each run
``CUDA_VISIBLE_DEVICES=$i uv run python scripts/run_campaign.py --shard $i/8``
against the same campaign file and together cover every cell exactly once.

Examples::

    uv run python scripts/run_campaign.py --dry-run
    uv run python scripts/run_campaign.py --only-phase pilot
    uv run python scripts/run_campaign.py --keep-going
    uv run python scripts/run_campaign.py --override logging.mode=online
    uv run python scripts/run_campaign.py --include-bonus --dry-run
    CUDA_VISIBLE_DEVICES=3 uv run python scripts/run_campaign.py --shard 3/8

Exit codes: ``0`` every cell ended ``skip`` or ``complete`` (a dry run always
returns ``0``, since nothing was actually run); ``1`` the campaign file
failed validation, or at least one cell failed.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from group_algorithm_interp.manifest import (  # noqa: E402
    MANIFEST_NAME,
    RESOLVED_CONFIG_NAME,
    read_manifest,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CAMPAIGN = ROOT / "configs" / "campaign" / "core.yaml"
DEFAULT_RUN_BATCH = ROOT / "scripts" / "run_batch.py"
DEFAULT_GROUP_PROPERTIES = ROOT / "data" / "group_properties_full.jsonl"

# Execution order: earlier phases always run before later ones, both in the
# campaign file and within every shard. The bonus phases are opt-in
# (--include-bonus) and are not part of the pre-registered core study.
PHASES: tuple[str, ...] = ("pilot", "core", "bonus-e6", "bonus-e2")
BONUS_PHASES: frozenset[str] = frozenset({"bonus-e6", "bonus-e2"})


# --- the cell -----------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One (group, width, seed-range) block of the campaign, run through one
    ``run_batch.py`` invocation."""

    phase: str
    order: int
    index: int
    name: str
    width: int
    seeds: str  # "START:STOP", run_batch.py's own --seeds syntax
    experiment: str
    note: str | None = None

    @property
    def group_key(self) -> tuple[int, int]:
        return (self.order, self.index)

    @property
    def seed_range(self) -> range:
        return parse_seed_range(self.seeds)

    def cell_id(self) -> str:
        return f"({self.order},{self.index})@w{self.width}"

    def label(self) -> str:
        return (
            f"{self.phase:<5} SmallGroup({self.order},{self.index}) {self.name} "
            f"width={self.width} seeds={self.seeds}"
        )


def parse_seed_range(spec: str) -> range:
    """``"0:50"`` -> ``range(0, 50)``. The same slice convention
    ``scripts/run_batch.py``'s own ``--seeds`` takes -- one canonical parser
    so a campaign cell and a ``run_batch.py`` invocation cannot disagree
    about which seeds a cell means."""
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"seeds {spec!r} must be START:STOP (e.g. '0:50')")
    try:
        start, stop = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"seeds {spec!r} must be START:STOP with integer bounds") from exc
    if stop <= start:
        raise ValueError(f"seeds {spec!r}: STOP must be greater than START")
    return range(start, stop)


# --- campaign file parsing ------------------------------------------------------

_REQUIRED_CELL_FIELDS = ("phase", "order", "index", "name", "width", "seeds", "experiment")


def parse_campaign(path: Path) -> list[Cell]:
    """Load and validate the campaign file's shape (not its scientific
    content -- see :func:`validate_cells` for the group-catalogue check)."""
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("cells"), list):
        raise ValueError(f"{path} must be a mapping with a top-level 'cells' list")
    cells: list[Cell] = []
    for position, raw in enumerate(data["cells"]):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: cell #{position} is not a mapping")
        missing = [field for field in _REQUIRED_CELL_FIELDS if field not in raw]
        if missing:
            raise ValueError(f"{path}: cell #{position} is missing field(s): {', '.join(missing)}")
        if raw["phase"] not in PHASES:
            raise ValueError(
                f"{path}: cell #{position} has phase={raw['phase']!r}, expected one of {PHASES}"
            )
        seeds = str(raw["seeds"])
        try:
            parse_seed_range(seeds)
        except ValueError as exc:
            raise ValueError(f"{path}: cell #{position} ({raw.get('name')!r}): {exc}") from exc
        cells.append(
            Cell(
                phase=str(raw["phase"]),
                order=int(raw["order"]),
                index=int(raw["index"]),
                name=str(raw["name"]),
                width=int(raw["width"]),
                seeds=seeds,
                experiment=str(raw["experiment"]),
                note=str(raw["note"]) if raw.get("note") is not None else None,
            )
        )
    return cells


# --- pre-registration validation -------------------------------------------------


def load_known_groups(path: Path) -> set[tuple[int, int]]:
    """Every ``(order, index)`` present in the group-properties ground truth."""
    known: set[tuple[int, int]] = set()
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            known.add((int(record["order"]), int(record["index"])))
    return known


def validate_cells(cells: Sequence[Cell], known_groups: set[tuple[int, int]]) -> list[str]:
    """Every cell's group must be a real, catalogued group -- this is what
    keeps the campaign file honest as the pre-registered run list: a
    typo'd order/index would otherwise train (and bill for) a group nobody
    pre-registered, silently."""
    return [
        f"{cell.cell_id()} {cell.name!r}: SmallGroup({cell.order},{cell.index}) is not in "
        "the group-properties catalogue"
        for cell in cells
        if cell.group_key not in known_groups
    ]


# --- sharding ----------------------------------------------------------------------

# The cost heuristic, a calibrated cost
# law: sec/seed at width 128 scales ~ order^2.119, and doubling the width
# costs ~1.76x (so the width factor is 1.76^log2(width/128)). A cell's cost
# multiplies that per-seed figure by its seed count. This is a RELATIVE
# load-balancing estimate only -- constants of proportionality, the epoch
# ceiling, and the card all cancel out of a partition -- never a billing
# number.
ORDER_COST_EXPONENT = 2.119
WIDTH_DOUBLING_FACTOR = 1.76
REFERENCE_WIDTH = 128


def estimated_cost(cell: Cell) -> float:
    """Relative wall-clock estimate for one cell (see the constants above)."""
    width_factor = WIDTH_DOUBLING_FACTOR ** math.log2(cell.width / REFERENCE_WIDTH)
    return (cell.order**ORDER_COST_EXPONENT) * len(cell.seed_range) * width_factor


def parse_shard(spec: str) -> tuple[int, int]:
    """``"3/8"`` -> ``(3, 8)``: this worker's index and the worker count.

    Zero-based, matching ``CUDA_VISIBLE_DEVICES=$i ... --shard $i/8``.
    """
    parts = spec.split("/")
    if len(parts) != 2:
        raise ValueError(f"shard {spec!r} must be I/N (e.g. '3/8')")
    try:
        index, count = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"shard {spec!r} must be I/N with integer parts") from exc
    if count < 1:
        raise ValueError(f"shard {spec!r}: N must be at least 1")
    if not 0 <= index < count:
        raise ValueError(f"shard {spec!r}: I must satisfy 0 <= I < N (zero-based)")
    return index, count


def shard_cells(cells: Sequence[Cell], shard_index: int, shard_count: int) -> list[Cell]:
    """This worker's deterministic slice of ``cells``.

    The partition: sort all cells by DESCENDING estimated cost (ties broken
    by original campaign position, so the result is deterministic), then
    deal them round-robin -- shard ``i`` takes positions ``i, i+n, i+2n, ...``
    of that costed ordering. Dealing from most- to least-expensive puts each
    of the ``n`` costliest cells on a different worker and fills in with
    progressively cheaper ones, which keeps the shards roughly cost-balanced
    without bin-packing machinery.

    Each shard is then re-sorted for execution: phase order first (pilot
    before core before the bonus phases -- a shard must not start a core cell
    while its pilot cells wait), original campaign position second (which
    restores cheapest-first within each phase). The ``n`` shards of one cell
    list are always disjoint and together cover it exactly.
    """
    by_cost = sorted(enumerate(cells), key=lambda pair: (-estimated_cost(pair[1]), pair[0]))
    mine = by_cost[shard_index::shard_count]
    mine.sort(key=lambda pair: (PHASES.index(pair[1].phase), pair[0]))
    return [cell for _, cell in mine]


# --- completion scan --------------------------------------------------------------


def discover_run_dirs(runs_dir: Path) -> list[Path]:
    """Every immediate child of ``runs_dir`` that holds its own manifest.

    Mirrors ``scripts/sync_runs.py``'s ``discover_runs`` (naturally excludes
    ``runs/schedules/`` -- the campaign plan/status manifests, not per-seed
    run directories -- and the sync ledger)."""
    if not runs_dir.is_dir():
        return []
    return sorted(
        path for path in runs_dir.iterdir() if path.is_dir() and (path / MANIFEST_NAME).is_file()
    )


CompletionIndex = dict[tuple[int, int, int], set[int]]


def build_completion_index(runs_dir: Path) -> CompletionIndex:
    """``(order, index, width) -> {seeds whose run reached a completed
    manifest}``, read once from every run directory under ``runs_dir``.

    Reads ``manifest.yaml`` for the terminal status (mirroring
    ``sync_runs.py``: a manifest that cannot be read, or is not
    ``completed``, is simply not counted -- a run still ``running`` or one
    that ``failed`` must not make a cell look done) and
    ``resolved_config.yaml`` for the group/width/seed a manifest's status
    block alone does not carry (the manifest records only the group's
    canonical *name*, e.g. ``SmallGroup(32,18)``, not its order/index
    broken out)."""
    index: CompletionIndex = {}
    for run_dir in discover_run_dirs(runs_dir):
        try:
            manifest = read_manifest(run_dir)
        except Exception:
            continue
        if manifest.get("status") != "completed":
            continue
        config_path = run_dir / RESOLVED_CONFIG_NAME
        if not config_path.is_file():
            continue
        try:
            config = yaml.safe_load(config_path.read_text())
        except Exception:
            continue
        if not isinstance(config, dict):
            continue
        data = config.get("data")
        model = config.get("model")
        group = data.get("group") if isinstance(data, dict) else None
        order = group.get("order") if isinstance(group, dict) else None
        idx = group.get("index") if isinstance(group, dict) else None
        width = model.get("d_model") if isinstance(model, dict) else None
        seed = config.get("seed")
        if (
            isinstance(order, int)
            and isinstance(idx, int)
            and isinstance(width, int)
            and isinstance(seed, int)
            and not any(isinstance(value, bool) for value in (order, idx, width, seed))
        ):
            index.setdefault((order, idx, width), set()).add(seed)
    return index


def cell_is_complete(cell: Cell, index: CompletionIndex) -> bool:
    completed = index.get((cell.order, cell.index, cell.width), set())
    return set(cell.seed_range) <= completed


# --- run_batch.py invocation --------------------------------------------------


def build_run_batch_command(
    run_batch_path: Path, cell: Cell, extra_overrides: Sequence[str] = ()
) -> list[str]:
    """One cell's complete ``run_batch.py`` invocation: the Hydra overrides
    (bare ``key=value``, exactly as ``scripts/run.py`` takes them) first,
    then ``--seeds``. ``extra_overrides`` (the campaign's own ``--override``
    flags, e.g. ``logging.mode=online``) are appended after the cell's own
    overrides and before ``--seeds``. Hydra's last-wins precedence means an
    operator override on a key a cell also sets (unusual, but not forbidden
    here) wins."""
    return [
        "uv",
        "run",
        "python",
        str(run_batch_path),
        f"experiment={cell.experiment}",
        f"data.group.order={cell.order}",
        f"data.group.index={cell.index}",
        f"model.d_model={cell.width}",
        *extra_overrides,
        "--seeds",
        cell.seeds,
    ]


# --- the campaign loop ----------------------------------------------------------

CellStatus = Literal["skip", "complete", "failed", "would_run", "not_run"]


@dataclass(frozen=True)
class CellOutcome:
    cell: Cell
    status: CellStatus
    command: list[str] | None
    returncode: int | None


def run_campaign(
    cells: Sequence[Cell],
    *,
    completion_index: CompletionIndex,
    run_batch_path: Path,
    extra_overrides: Sequence[str],
    keep_going: bool,
    dry_run: bool,
    cwd: Path,
    run: Callable[..., subprocess.CompletedProcess[Any]] | None = None,
    log: Callable[[str], None] = print,
) -> list[CellOutcome]:
    """Iterate ``cells`` in order, skipping any already complete and invoking
    ``run_batch.py`` for the rest.

    ``run`` defaults to ``subprocess.run`` -- resolved at call time (not in
    the signature), so a test that monkeypatches ``subprocess.run`` on this
    module actually intercepts the launch.

    Continue-past-failure convention: a failed cell
    does not raise, and every cell -- run, skipped, or never attempted --
    ends up in exactly one :class:`CellOutcome`. Without ``--keep-going`` the
    campaign stops launching *new* cells after the first failure (remaining
    cells are recorded ``not_run``, not silently dropped) but still finishes
    the loop and returns a complete accounting, rather than raising out from
    under the caller.
    """
    launch = run if run is not None else subprocess.run
    outcomes: list[CellOutcome] = []
    stop = False
    total = len(cells)
    for position, cell in enumerate(cells, start=1):
        prefix = f"[{position}/{total}] {cell.label()}"
        if cell_is_complete(cell, completion_index):
            log(f"{prefix} -> SKIP (already complete)")
            outcomes.append(CellOutcome(cell, "skip", None, None))
            continue

        command = build_run_batch_command(run_batch_path, cell, extra_overrides)

        if dry_run:
            log(f"{prefix} -> WOULD RUN: {' '.join(command)}")
            outcomes.append(CellOutcome(cell, "would_run", command, None))
            continue

        if stop:
            log(
                f"{prefix} -> NOT RUN (campaign stopped after an earlier failure; pass --keep-going to continue past it)"
            )
            outcomes.append(CellOutcome(cell, "not_run", command, None))
            continue

        log(f"{prefix} -> running: {' '.join(command)}")
        result = launch(command, cwd=cwd)
        returncode = result.returncode
        if returncode == 0:
            log(f"{prefix} -> COMPLETE")
            outcomes.append(CellOutcome(cell, "complete", command, returncode))
        else:
            log(f"{prefix} -> FAILED (exit {returncode})")
            outcomes.append(CellOutcome(cell, "failed", command, returncode))
            if not keep_going:
                stop = True

    return outcomes


def _summarize(outcomes: Sequence[CellOutcome]) -> str:
    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    return "campaign summary: " + ", ".join(
        f"{status}={count}" for status, count in sorted(counts.items())
    )


# --- CLI -----------------------------------------------------------------------


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--run-batch", type=Path, default=DEFAULT_RUN_BATCH, dest="run_batch")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs", dest="runs_dir")
    parser.add_argument(
        "--group-properties",
        type=Path,
        default=DEFAULT_GROUP_PROPERTIES,
        dest="group_properties",
        help="The group-properties ground truth every cell's (order, index) is checked "
        "against. Missing entirely (e.g. on a pod that only copied data/group_artifacts/) "
        "is a warning, not a failure -- see this script's module docstring.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the group-catalogue check entirely, even if --group-properties exists.",
    )
    parser.add_argument(
        "--only-phase", choices=PHASES, default=None, dest="only_phase", help="Run only this phase."
    )
    parser.add_argument(
        "--include-bonus",
        action="store_true",
        dest="include_bonus",
        help="Include the opt-in bonus phases (bonus-e6/bonus-e2 -- extension supply, "
        "not part of the pre-registered core study). Without this flag bonus cells are "
        "excluded entirely, including from --dry-run listings.",
    )
    parser.add_argument(
        "--shard",
        metavar="I/N",
        default=None,
        help="Run only this worker's deterministic slice of the cell list, e.g. "
        "CUDA_VISIBLE_DEVICES=$i ... --shard $i/8 on an 8-GPU pod. Cells are dealt "
        "round-robin over descending estimated cost (cost ~ order^2.119 x seeds x "
        "width factor), so the N shards are disjoint, cover every cell, and are "
        "roughly cost-balanced; each shard still runs pilot before core before bonus.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan and each cell's status; run nothing."
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue launching remaining cells after a failed cell instead of stopping; "
        "the campaign still exits non-zero if any cell failed.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="override",
        help="A Hydra override forwarded to every run_batch.py invocation this run makes "
        "(e.g. --override logging.mode=online); repeatable.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)

    try:
        cells = parse_campaign(args.campaign)
    except (ValueError, OSError) as exc:
        print(f"campaign file error: {exc}", file=sys.stderr)
        return 1

    if args.skip_validation:
        pass
    elif args.group_properties.is_file():
        known_groups = load_known_groups(args.group_properties)
        errors = validate_cells(cells, known_groups)
        if errors:
            print(f"{len(errors)} cell(s) reference an unknown group:", file=sys.stderr)
            for error in errors:
                print(f"  {error}", file=sys.stderr)
            return 1
    else:
        print(
            f"note: {args.group_properties} not found; skipping the group-catalogue check "
            "(pass --skip-validation to silence this)",
            file=sys.stderr,
        )

    # Without --include-bonus the bonus phases do not exist for this run at
    # all -- not in the loop, not in --dry-run listings, not in a shard's
    # slice (the partition below is of the *included* cell list, so all N
    # workers must agree on --include-bonus for their shards to be a
    # partition of the same list).
    if not args.include_bonus:
        cells = [cell for cell in cells if cell.phase not in BONUS_PHASES]

    if args.only_phase is not None:
        cells = [cell for cell in cells if cell.phase == args.only_phase]
        if not cells:
            hint = (
                " (bonus phases require --include-bonus)"
                if args.only_phase in BONUS_PHASES and not args.include_bonus
                else ""
            )
            print(f"no cells in phase {args.only_phase!r}{hint}", file=sys.stderr)
            return 1

    if args.shard is not None:
        try:
            shard_index, shard_count = parse_shard(args.shard)
        except ValueError as exc:
            print(f"--shard error: {exc}", file=sys.stderr)
            return 1
        cells = shard_cells(cells, shard_index, shard_count)
        print(f"shard {shard_index}/{shard_count}: {len(cells)} cell(s)")
        if not cells:
            print("this shard is empty (more workers than cells); nothing to do")
            return 0

    completion_index = build_completion_index(args.runs_dir)
    outcomes = run_campaign(
        cells,
        completion_index=completion_index,
        run_batch_path=args.run_batch,
        extra_overrides=args.override,
        keep_going=args.keep_going,
        dry_run=args.dry_run,
        cwd=ROOT,
    )

    print(_summarize(outcomes))
    if args.dry_run:
        return 0
    return 0 if all(outcome.status in ("skip", "complete") for outcome in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
