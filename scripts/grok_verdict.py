#!/usr/bin/env python3
"""Per-cell grok/censoring verdict for the completed core-study campaign.

Offline and read-only: for every run directory under ``runs_root`` this reads
``resolved_config.yaml`` (to place the run in its CELL -- ``(order, index,
width, epochs)``) and its durable ``selection.json`` (to decide grokked vs
censored). No training, no network, no checkpoints touched, and no
``run.log`` dependency -- the campaign's ``run.log`` files are gone, so
``selection.json`` is the only source, per the study's own prune/ship design.

Grok derivation is **not** reimplemented here: it imports
``group_algorithm_interp.instruments.publish``'s
``_grok_fields_from_selection`` -- the same canonical, unleaked-first,
two-selection-json-schema derivation the W&B publisher already uses, so this
script's verdicts and the publisher's stay in lockstep. That helper is
private (leading underscore) but already reused this way by
``tests/test_publish_occupancy.py``; nothing in ``publish.py`` is modified.

Usage::

    uv run python scripts/grok_verdict.py [runs_root] [--runs-root runs/]
        [--campaign-config configs/campaign/core.yaml] [--pair-floor 25]
        [--out results/grok_verdict.json]

Prints a per-cell table (seeds, grokked, censored, grok fraction, and the
epochs-to-grok median/10th/90th percentile over grokked seeds) and writes the
same records, plus the C1 tier-1 pair floor-adequacy flags, to ``--out`` as
JSON.

**Every run directory counts once, as found -- this is not deduplicated by
seed.** This campaign's ``runs/`` carries a small number of seed reruns
(the same nominal seed of the same cell, completed more than once across
restarted pods/campaign ids) and a handful of ad hoc epoch-extension
follow-ups (``optim.epochs`` overridden well past the pinned 60,000 ceiling,
outside ``configs/campaign/core.yaml``'s committed cell list). Both surface
in the output rather than being silently collapsed: each cell record reports
``n_distinct_seeds`` alongside ``n_total`` (the raw run-directory count), and
``in_pinned_cell_list`` marks whether a cell matches a committed
``(order, index, width)`` entry at the campaign's 60,000-epoch ceiling.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from group_algorithm_interp.instruments import publish as publish_module  # noqa: E402

DEFAULT_RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs"
DEFAULT_CAMPAIGN_CONFIG = (
    Path(__file__).resolve().parent.parent / "configs" / "campaign" / "core.yaml"
)
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "results" / "grok_verdict.json"

# The pinned ceiling every committed cell in configs/campaign/core.yaml runs to
# (docs/core-study.md, "Every cell runs to the 60,000-epoch ceiling"). A run
# whose resolved epochs differ from this is an ad hoc extension, not a
# committed cell -- flagged via `in_pinned_cell_list`, never folded into it.
PINNED_CEILING_EPOCHS = 60_000

# The working statistical floor for a C1 tier-1 pair member: below this many
# grokked seeds, a pair is "characterised boundary" rather than dropped.
DEFAULT_PAIR_FLOOR = 25

# The note substring that marks a campaign cell as one member of a C1 tier-1
# pair, with its partner named "with (<order>,<index>)" immediately after --
# see configs/campaign/core.yaml's notes for (27,3)/(27,4) and the six
# order-64 cells, cross-checked against docs/core-study.md's group table.
_TIER1_MARKER = "C1 tier-1 pair"
_EMBEDDING_PROBE_MARKER = "C1 embedding probe"

CellKey = tuple[int, int, int, int]  # (order, index, width, epochs)
GroupKey = tuple[int, int]  # (order, index)

# Minimum grokked seeds before an epochs-to-grok percentile interval is
# reported at all (below this, a 10th/90th percentile over so few points is
# noise dressed up as an interval).
MIN_SEEDS_FOR_INTERVAL = 5


def _warn(message: str) -> None:
    print(f"[grok_verdict] {message}", file=sys.stderr)


# --- discovery and per-run reads --------------------------------------------------


def discover_run_dirs(runs_root: Path) -> list[Path]:
    """Every immediate child of ``runs_root`` with its own
    ``resolved_config.yaml`` -- the minimum needed to place a run in its
    cell. A run missing ``selection.json`` is still discovered (and counted
    in ``n_total``) but contributes no grok verdict; see
    :func:`run_grok_fields`."""
    if not runs_root.is_dir():
        return []
    return [
        p
        for p in sorted(runs_root.iterdir())
        if p.is_dir() and (p / "resolved_config.yaml").is_file()
    ]


def run_cell_key(run_dir: Path) -> tuple[CellKey, Any] | None:
    """``((order, index, width, epochs), seed)`` from a run's
    ``resolved_config.yaml``, or ``None`` if it is missing or unreadable."""
    path = run_dir / "resolved_config.yaml"
    try:
        cfg = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(cfg, dict):
        return None
    try:
        order = int(cfg["data"]["group"]["order"])
        index = int(cfg["data"]["group"]["index"])
        width = int(cfg["model"]["d_model"])
        epochs = int(cfg["optim"]["epochs"])
    except (KeyError, TypeError, ValueError):
        return None
    seed = cfg.get("seed")
    return (order, index, width, epochs), seed


def run_grok_fields(run_dir: Path) -> dict[str, Any] | None:
    """``grokked``/``censored``/``epochs_to_grok`` for one run, via
    ``publish._grok_fields_from_selection`` -- ``selection.json`` only, no
    ``run.log`` fallback (this campaign's ``run.log`` files do not survive).
    ``None`` when ``selection.json`` is absent, unreadable, or unrecognised."""
    return publish_module._grok_fields_from_selection(run_dir)


# --- campaign config: cell names + C1 tier-1 pairs --------------------------------


def load_campaign_cells(path: Path) -> list[dict[str, Any]]:
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        _warn(f"cannot read campaign config {path}: {exc}")
        return []
    if not isinstance(data, dict):
        return []
    cells = data.get("cells")
    return list(cells) if isinstance(cells, list) else []


def build_cell_metadata(
    cells: list[dict[str, Any]],
) -> dict[tuple[int, int, int], dict[str, Any]]:
    """``(order, index, width) -> {"name", "phases"}`` for every committed
    campaign cell (phases collected as a set, since pilot and core commit the
    same ``(order, index, width)`` for the smoke cell, D32 w128)."""
    meta: dict[tuple[int, int, int], dict[str, Any]] = {}
    for cell in cells:
        try:
            key = (int(cell["order"]), int(cell["index"]), int(cell["width"]))
        except (KeyError, TypeError, ValueError):
            continue
        entry = meta.setdefault(key, {"name": cell.get("name"), "phases": set()})
        entry["phases"].add(cell.get("phase"))
        if entry.get("name") is None:
            entry["name"] = cell.get("name")
    return meta


def _extract_partner(note: str) -> GroupKey | None:
    """The ``(order, index)`` named immediately after a tier-1 marker in a
    cell's ``note``, e.g. ``"...C1 tier-1 pair with (27,4); ..."`` -> ``(27,
    4)``. Documentation only (free text), same caveat as
    ``publish.py``'s own ``_parse_pair_partner``."""
    idx = note.find("(", note.find(_TIER1_MARKER))
    if idx == -1:
        return None
    end = note.find(")", idx)
    if end == -1:
        return None
    parts = note[idx + 1 : end].split(",")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except ValueError:
        return None


def find_tier1_pairs(cells: list[dict[str, Any]]) -> list[tuple[GroupKey, GroupKey]]:
    """Every C1 tier-1 pair declared in the campaign config's cell notes, as
    a de-duplicated list of ``((order_a, index_a), (order_b, index_b))`` with
    ``a < b``. Each pair's note names its own partner, so both members are
    ordinarily found; a partner that cannot be parsed is silently dropped
    (nothing here is authoritative over ``docs/core-study.md``)."""
    pairs: set[tuple[GroupKey, GroupKey]] = set()
    for cell in cells:
        note = cell.get("note") or ""
        if _TIER1_MARKER not in note:
            continue
        partner = _extract_partner(note)
        if partner is None:
            continue
        try:
            this_group = (int(cell["order"]), int(cell["index"]))
        except (KeyError, TypeError, ValueError):
            continue
        pairs.add(tuple(sorted((this_group, partner))))  # type: ignore[arg-type]
    return sorted(pairs)


def find_embedding_probe_pairs(cells: list[dict[str, Any]]) -> list[tuple[GroupKey, GroupKey]]:
    """Same idea as :func:`find_tier1_pairs` but for the ``"C1 embedding
    probe"`` marker -- (54,10)/(54,11), the tier-1 construction's C2-bystander
    lift. Reported separately: it is C1-relevant but is not itself a tier-1
    pair, so it is never folded into the tier-1 floor check."""
    pairs: set[tuple[GroupKey, GroupKey]] = set()
    for cell in cells:
        note = cell.get("note") or ""
        if _EMBEDDING_PROBE_MARKER not in note:
            continue
        idx = note.find("(", note.find(_EMBEDDING_PROBE_MARKER))
        end = note.find(")", idx) if idx != -1 else -1
        partner: GroupKey | None = None
        if idx != -1 and end != -1:
            parts = note[idx + 1 : end].split(",")
            if len(parts) == 2:
                try:
                    partner = (int(parts[0].strip()), int(parts[1].strip()))
                except ValueError:
                    partner = None
        if partner is None:
            continue
        try:
            this_group = (int(cell["order"]), int(cell["index"]))
        except (KeyError, TypeError, ValueError):
            continue
        pairs.add(tuple(sorted((this_group, partner))))  # type: ignore[arg-type]
    return sorted(pairs)


# --- aggregation -------------------------------------------------------------------


@dataclass
class CellAggregate:
    key: CellKey
    seeds_seen: list[Any] = field(default_factory=list)
    n_grokked: int = 0
    n_censored: int = 0
    n_missing: int = 0
    epochs_to_grok: list[float] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)


def aggregate_runs(run_dirs: list[Path]) -> tuple[dict[CellKey, CellAggregate], int, int]:
    """Groups every discovered run into its cell. Returns the per-cell
    aggregates plus the count of runs whose cell key could not be determined
    (bad ``resolved_config.yaml``) -- callers should treat that as a data
    problem, not silently drop it -- and the count of runs measured overall."""
    cells: dict[CellKey, CellAggregate] = {}
    n_unresolvable = 0
    n_total = 0
    for run_dir in run_dirs:
        n_total += 1
        resolved = run_cell_key(run_dir)
        if resolved is None:
            n_unresolvable += 1
            _warn(f"{run_dir.name}: cannot determine cell from resolved_config.yaml; skipped")
            continue
        key, seed = resolved
        agg = cells.setdefault(key, CellAggregate(key=key))
        agg.seeds_seen.append(seed)
        agg.run_ids.append(run_dir.name)

        grok = run_grok_fields(run_dir)
        if grok is None:
            agg.n_missing += 1
            _warn(f"{run_dir.name}: no usable selection.json; excluded from grok counts")
            continue
        if grok["censored"]:
            agg.n_censored += 1
        else:
            agg.n_grokked += 1
            if grok["epochs_to_grok"] is not None:
                agg.epochs_to_grok.append(float(grok["epochs_to_grok"]))
    return cells, n_unresolvable, n_total


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (numpy's default convention) over an
    already-sorted sequence. ``pct`` in [0, 100]."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (pct / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_values[int(rank)]
    return sorted_values[lo] * (hi - rank) + sorted_values[hi] * (rank - lo)


def epochs_to_grok_summary(values: list[float]) -> dict[str, Any]:
    """``{n, median, p10, p90}`` over grokked seeds' epochs-to-grok. ``p10``/
    ``p90`` are ``None`` below :data:`MIN_SEEDS_FOR_INTERVAL` seeds -- a
    percentile interval over a handful of points is noise dressed up as
    precision -- but ``median`` is reported for any ``n >= 1``."""
    if not values:
        return {"n": 0, "median": None, "p10": None, "p90": None}
    ordered = sorted(values)
    summary: dict[str, Any] = {"n": len(ordered), "median": statistics.median(ordered)}
    if len(ordered) >= MIN_SEEDS_FOR_INTERVAL:
        summary["p10"] = _percentile(ordered, 10.0)
        summary["p90"] = _percentile(ordered, 90.0)
    else:
        summary["p10"] = None
        summary["p90"] = None
    return summary


def cell_record(
    agg: CellAggregate,
    cell_meta: dict[tuple[int, int, int], dict[str, Any]],
) -> dict[str, Any]:
    order, index, width, epochs = agg.key
    n_total = len(agg.seeds_seen)
    n_distinct_seeds = len(set(agg.seeds_seen))
    n_resolved = agg.n_grokked + agg.n_censored
    meta = cell_meta.get((order, index, width))
    return {
        "order": order,
        "index": index,
        "width": width,
        "epochs": epochs,
        "cell_name": meta["name"] if meta else None,
        "phases": sorted(p for p in meta["phases"] if p is not None) if meta else [],
        "in_pinned_cell_list": meta is not None and epochs == PINNED_CEILING_EPOCHS,
        "n_total": n_total,
        "n_distinct_seeds": n_distinct_seeds,
        "n_reruns": n_total - n_distinct_seeds,
        "n_grokked": agg.n_grokked,
        "n_censored": agg.n_censored,
        "n_missing_grok_fields": agg.n_missing,
        "grok_fraction": (agg.n_grokked / n_resolved) if n_resolved else None,
        "epochs_to_grok": epochs_to_grok_summary(agg.epochs_to_grok),
        "run_ids": sorted(agg.run_ids),
    }


# --- C1 tier-1 pair floor check -----------------------------------------------------


def _pinned_cell_for_group(
    cells: dict[CellKey, CellAggregate], group: GroupKey
) -> CellAggregate | None:
    """The pinned-ceiling cell aggregate for ``(order, index)`` -- the one
    used for the pair floor check -- preferring the widest-seeded match at
    :data:`PINNED_CEILING_EPOCHS` if a group happens to run at more than one
    width in the campaign (none of the tier-1 pairs currently do)."""
    order, index = group
    candidates = [
        agg
        for key, agg in cells.items()
        if key[0] == order and key[1] == index and key[3] == PINNED_CEILING_EPOCHS
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda agg: len(agg.seeds_seen))


def pair_records(
    pairs: list[tuple[GroupKey, GroupKey]],
    cells: dict[CellKey, CellAggregate],
    cell_meta: dict[tuple[int, int, int], dict[str, Any]],
    *,
    pair_floor: int,
    label: str,
) -> list[dict[str, Any]]:
    records = []
    for group_a, group_b in pairs:
        agg_a = _pinned_cell_for_group(cells, group_a)
        agg_b = _pinned_cell_for_group(cells, group_b)
        member_a = cell_record(agg_a, cell_meta) if agg_a else None
        member_b = cell_record(agg_b, cell_meta) if agg_b else None
        if member_a is None or member_b is None:
            status = "missing_member"
            meets_floor: bool | None = None
        else:
            meets_floor = (
                member_a["n_grokked"] >= pair_floor and member_b["n_grokked"] >= pair_floor
            )
            status = "meets_floor" if meets_floor else "characterised_boundary"
        records.append(
            {
                "label": label,
                "pair": [list(group_a), list(group_b)],
                "member_a": member_a,
                "member_b": member_b,
                "pair_floor": pair_floor,
                "both_meet_floor": meets_floor,
                "status": status,
            }
        )
    return records


# --- CLI -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "runs_root_positional",
        nargs="?",
        default=None,
        type=Path,
        help="run directory root (positional form of --runs-root)",
    )
    parser.add_argument("--runs-root", type=Path, default=None, help="run directory root")
    parser.add_argument("--campaign-config", type=Path, default=DEFAULT_CAMPAIGN_CONFIG)
    parser.add_argument(
        "--pair-floor",
        type=int,
        default=DEFAULT_PAIR_FLOOR,
        help="minimum grokked seeds per C1 tier-1 pair member (default 25)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--no-table", action="store_true", help="skip the human-readable table on stdout"
    )
    return parser


def _print_table(cell_records: list[dict[str, Any]]) -> None:
    header = (
        f"{'order':>6} {'index':>6} {'width':>5} {'epochs':>7} "
        f"{'n':>5} {'grok':>5} {'cens':>5} {'miss':>5} {'frac':>6} "
        f"{'median':>8} {'p10':>8} {'p90':>8}  name"
    )
    print(header)
    print("-" * len(header))
    for rec in cell_records:
        etg = rec["epochs_to_grok"]
        frac = f"{rec['grok_fraction']:.2f}" if rec["grok_fraction"] is not None else "  --"
        median = f"{etg['median']:.0f}" if etg["median"] is not None else "--"
        p10 = f"{etg['p10']:.0f}" if etg["p10"] is not None else "--"
        p90 = f"{etg['p90']:.0f}" if etg["p90"] is not None else "--"
        flag = "" if rec["in_pinned_cell_list"] else "  [not in pinned cell list]"
        rerun = f"  [+{rec['n_reruns']} rerun]" if rec["n_reruns"] else ""
        name = rec["cell_name"] or ""
        print(
            f"{rec['order']:>6} {rec['index']:>6} {rec['width']:>5} {rec['epochs']:>7} "
            f"{rec['n_total']:>5} {rec['n_grokked']:>5} {rec['n_censored']:>5} "
            f"{rec['n_missing_grok_fields']:>5} {frac:>6} "
            f"{median:>8} {p10:>8} {p90:>8}  {name}{flag}{rerun}"
        )


def _print_pairs(pairs: list[dict[str, Any]]) -> None:
    for rec in pairs:
        a, b = rec["pair"]
        print(f"\n{rec['label']}: ({a[0]},{a[1]}) / ({b[0]},{b[1]})  floor={rec['pair_floor']}")
        for tag, member in (("  a", rec["member_a"]), ("  b", rec["member_b"])):
            if member is None:
                print(f"{tag}: NOT FOUND in runs_root")
            else:
                print(
                    f"{tag}: ({member['order']},{member['index']}) w{member['width']} "
                    f"grokked={member['n_grokked']} censored={member['n_censored']} "
                    f"total={member['n_total']}"
                )
        print(f"  status: {rec['status']}")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    runs_root = args.runs_root or args.runs_root_positional or DEFAULT_RUNS_ROOT
    if not runs_root.is_dir():
        _warn(f"runs_root {runs_root} is not a directory")
        return 1

    run_dirs = discover_run_dirs(runs_root)
    cells, n_unresolvable, n_total_runs = aggregate_runs(run_dirs)

    campaign_cells = load_campaign_cells(args.campaign_config)
    cell_meta = build_cell_metadata(campaign_cells)
    tier1_pairs = find_tier1_pairs(campaign_cells)
    embedding_pairs = find_embedding_probe_pairs(campaign_cells)

    cell_records = [cell_record(agg, cell_meta) for agg in cells.values()]
    cell_records.sort(key=lambda r: (r["order"], r["index"], r["width"], r["epochs"]))

    total_grokked = sum(r["n_grokked"] for r in cell_records)
    total_censored = sum(r["n_censored"] for r in cell_records)
    total_missing = sum(r["n_missing_grok_fields"] for r in cell_records)

    expected_grokked = 1626
    expected_censored = 664

    tier1_records = pair_records(
        tier1_pairs, cells, cell_meta, pair_floor=args.pair_floor, label="C1 tier-1 pair"
    )
    embedding_records = pair_records(
        embedding_pairs,
        cells,
        cell_meta,
        pair_floor=args.pair_floor,
        label="C1 embedding probe (not tier-1)",
    )

    payload = {
        "runs_root": str(runs_root),
        "campaign_config": str(args.campaign_config),
        "pair_floor": args.pair_floor,
        "n_run_dirs": n_total_runs,
        "n_run_dirs_unresolvable": n_unresolvable,
        "totals": {
            "n_grokked": total_grokked,
            "n_censored": total_censored,
            "n_missing_grok_fields": total_missing,
        },
        "reconciliation": {
            "expected_grokked": expected_grokked,
            "expected_censored": expected_censored,
            "actual_grokked": total_grokked,
            "actual_censored": total_censored,
            "grokked_diff": total_grokked - expected_grokked,
            "censored_diff": total_censored - expected_censored,
        },
        "cells": cell_records,
        "tier1_pairs": tier1_records,
        "embedding_probe_pairs": embedding_records,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"[grok_verdict] wrote {args.out}")

    if not args.no_table:
        _print_table(cell_records)
        _print_pairs(tier1_records)
        _print_pairs(embedding_records)
        print(
            f"\ntotals: grokked={total_grokked} censored={total_censored} "
            f"missing={total_missing} (expected ~{expected_grokked}/{expected_censored})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
