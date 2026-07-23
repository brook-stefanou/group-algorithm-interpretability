#!/usr/bin/env python3
"""Non-destructive canonical-run index for the core-study campaign.

``runs/`` carries a small number of seed reruns: the same nominal
``(order, index, width, epochs, seed)`` cell+seed key, trained to completion
more than once, across restarted pods and campaign ids during the
crash-recovery. This script never touches ``runs/`` -- it only reads
``resolved_config.yaml``, ``manifest.yaml``, and ``selection.json`` from every
run directory and writes a lookup (``--out``, default
``results/canonical_runs.json``) that picks exactly one *canonical* run
directory per cell+seed key, so downstream analysis can count each seed once
without deleting or moving any duplicate.

Canonical rule (deterministic, documented rather than "correct" -- see
below): prefer ``manifest.status == "completed"`` over any other status;
among the preferred group, the earliest ``manifest.completed_at`` wins; if
that is missing or tied, the lexicographically earliest ``run_id`` wins (this
campaign's run ids are ``YYYY-MM-DD_HHMMSS_ffffff_...``, so string order is
timestamp order). Every run directory in this campaign is ``completed``, so
in practice the rule reduces to "first completed attempt is canonical" --
the rerun exists only because the crash-recovery restart did not know the
first attempt had already succeeded, not because the rerun corrected
anything. Every run's ``resolved_config.yaml`` also declares
``deterministic: false``, which means there is no principled way to call one
rerun's *outcome* more correct than another's: the choice is procedural (pick
one, consistently), not epistemic.

Grok derivation is not reimplemented: it imports
``group_algorithm_interp.instruments.publish``'s
``_grok_fields_from_selection`` -- the same canonical, unleaked-first,
two-selection-json-schema derivation ``scripts/grok_verdict.py`` already
reuses -- so this script's grok/censor labels stay in lockstep with both.

Usage::

    uv run python scripts/dedup_runs.py [runs_root] [--runs-root runs/]
        [--out results/canonical_runs.json] [--epoch-tolerance 1000]

Prints a summary of the raw-vs-canonical headline counts and the rerun
agreement check, and writes the full index plus the agreement detail to
``--out`` as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from group_algorithm_interp.instruments import publish as publish_module  # noqa: E402

DEFAULT_RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "results" / "canonical_runs.json"

# A "reasonable tolerance" for epochs-to-grok agreement between reruns of the
# same seed, given onset-snapshot granularity: this campaign's densest
# snapshot spacing near onset is ~100-300 epochs (configs/campaign/core.yaml's
# snapshot.interval), so 1000 is roughly an order of magnitude above that --
# generous enough that snapshot-alignment alone should not cause a "false"
# disagreement, while still flagging genuine divergence from run-to-run
# stochasticity (training is not bit-deterministic; see module docstring).
DEFAULT_EPOCH_TOLERANCE = 1000

SeedKey = tuple[int, int, int, int, Any]  # (order, index, width, epochs, seed)

_STATUS_RANK_COMPLETED = 0
_STATUS_RANK_OTHER = 1


def _warn(message: str) -> None:
    print(f"[dedup_runs] {message}", file=sys.stderr)


def _key_str(key: SeedKey) -> str:
    order, index, width, epochs, seed = key
    return f"{order}-{index}-{width}-{epochs}-seed{seed}"


# --- discovery and per-run reads --------------------------------------------------


def discover_run_dirs(runs_root: Path) -> list[Path]:
    """Every immediate child of ``runs_root`` with its own
    ``resolved_config.yaml`` -- mirrors ``grok_verdict.discover_run_dirs``."""
    if not runs_root.is_dir():
        return []
    return [
        p
        for p in sorted(runs_root.iterdir())
        if p.is_dir() and (p / "resolved_config.yaml").is_file()
    ]


def run_seed_key(run_dir: Path) -> SeedKey | None:
    """``(order, index, width, epochs, seed)`` from a run's
    ``resolved_config.yaml``, or ``None`` if it is missing, unreadable, or
    missing a field this key needs."""
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
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            pass
    return order, index, width, epochs, seed


def run_status_info(run_dir: Path) -> dict[str, Any]:
    """``{"run_id", "status", "completed_at"}`` from a run's
    ``manifest.yaml``. Falls back to the directory name as ``run_id`` and
    ``None``/``status=None`` for the rest when the manifest is missing or
    unreadable -- such a run still sorts (last, deterministically) rather
    than crashing the index."""
    run_id = run_dir.name
    path = run_dir / "manifest.yaml"
    try:
        manifest = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        manifest = None
    if not isinstance(manifest, dict):
        return {"run_id": run_id, "status": None, "completed_at": None}
    return {
        "run_id": manifest.get("run_id") or run_id,
        "status": manifest.get("status"),
        "completed_at": manifest.get("completed_at"),
    }


def run_grok_fields(run_dir: Path) -> dict[str, Any] | None:
    """``grokked``/``censored``/``epochs_to_grok`` for one run, via
    ``publish._grok_fields_from_selection``. ``None`` when ``selection.json``
    is absent, unreadable, or unrecognised."""
    return publish_module._grok_fields_from_selection(run_dir)


# --- canonical selection -----------------------------------------------------------


def _tiebreak_sort_key(info: dict[str, Any]) -> tuple[int, str, str]:
    """Sort key for choosing the canonical run among duplicates: completed
    status first, then earliest ``completed_at`` (missing sorts last within
    its status group via an empty string only when compared to a present
    timestamp -- ISO 8601 strings never sort before ``""``), then earliest
    ``run_id`` as a final, always-available tiebreak."""
    status_rank = _STATUS_RANK_COMPLETED if info["status"] == "completed" else _STATUS_RANK_OTHER
    completed_at = info["completed_at"] or ""
    return status_rank, completed_at, info["run_id"]


def choose_canonical(infos: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """``(canonical_run_id, dropped_run_ids)`` for one cell+seed key's run
    directories, ``dropped_run_ids`` sorted for determinism. A singleton
    list returns itself with no drops."""
    ordered = sorted(infos, key=_tiebreak_sort_key)
    canonical = ordered[0]["run_id"]
    dropped = sorted(info["run_id"] for info in ordered[1:])
    return canonical, dropped


# --- rerun agreement ---------------------------------------------------------------


def classify_agreement(
    run_ids: list[str],
    grok_by_run: dict[str, dict[str, Any] | None],
    *,
    epoch_tolerance: int,
) -> dict[str, Any]:
    """Agreement classification for one duplicated cell+seed key's reruns.

    ``outcome`` is one of:

    * ``"missing_data"`` -- at least one rerun has no usable grok fields, so
      no agreement judgement can be made.
    * ``"label_disagreement"`` -- the reruns disagree on grokked-vs-censored
      itself (the concerning case: the grok/censor count for this seed
      depends on which rerun is kept).
    * ``"epoch_disagreement"`` -- all reruns agree on grokked-vs-censored,
      all grokked, but their epochs-to-grok values span more than
      ``epoch_tolerance``.
    * ``"agreement"`` -- same label, and (if grokked) epochs-to-grok values
      all within ``epoch_tolerance`` of each other (censored reruns agree
      trivially, having no epoch to compare)."""
    fields = [grok_by_run.get(rid) for rid in run_ids]
    if any(f is None for f in fields):
        return {"outcome": "missing_data", "epoch_spread": None}
    resolved: list[dict[str, Any]] = [f for f in fields if f is not None]

    labels = {f["censored"] for f in resolved}
    if len(labels) > 1:
        return {"outcome": "label_disagreement", "epoch_spread": None}

    censored = next(iter(labels))
    if censored:
        return {"outcome": "agreement", "epoch_spread": None}

    epochs = [f["epochs_to_grok"] for f in resolved]
    epochs = [e for e in epochs if e is not None]
    if not epochs:
        return {"outcome": "agreement", "epoch_spread": None}
    spread = max(epochs) - min(epochs)
    outcome = "agreement" if spread <= epoch_tolerance else "epoch_disagreement"
    return {"outcome": outcome, "epoch_spread": spread}


# --- aggregation and reporting ------------------------------------------------------


def build_index(
    run_dirs: list[Path], *, epoch_tolerance: int
) -> tuple[dict[SeedKey, dict[str, Any]], dict[str, Any], int]:
    """Groups every discovered run by its cell+seed key, chooses a canonical
    run per key, and classifies rerun agreement for every key with more than
    one run directory.

    Returns ``(index, rerun_agreement, n_unresolvable)`` where ``index`` maps
    each ``SeedKey`` to ``{"canonical_run_id", "dropped_run_ids", "n_total"}``
    and ``rerun_agreement`` collects the disagreement detail."""
    grouped: dict[SeedKey, list[dict[str, Any]]] = {}
    n_unresolvable = 0
    for run_dir in run_dirs:
        key = run_seed_key(run_dir)
        if key is None:
            n_unresolvable += 1
            _warn(f"{run_dir.name}: cannot determine cell+seed key; skipped")
            continue
        info = run_status_info(run_dir)
        info["_run_dir"] = run_dir
        grouped.setdefault(key, []).append(info)

    index: dict[SeedKey, dict[str, Any]] = {}
    agreement_counts = {
        "agreement": 0,
        "epoch_disagreement": 0,
        "label_disagreement": 0,
        "missing_data": 0,
    }
    label_disagreements: list[dict[str, Any]] = []
    epoch_disagreements: list[dict[str, Any]] = []

    for key, infos in grouped.items():
        canonical, dropped = choose_canonical(infos)
        index[key] = {
            "canonical_run_id": canonical,
            "dropped_run_ids": dropped,
            "n_total": len(infos),
        }
        if len(infos) <= 1:
            continue

        run_ids = [info["run_id"] for info in infos]
        grok_by_run = {info["run_id"]: run_grok_fields(info["_run_dir"]) for info in infos}
        classified = classify_agreement(run_ids, grok_by_run, epoch_tolerance=epoch_tolerance)
        agreement_counts[classified["outcome"]] += 1

        detail = {
            "key": {
                "order": key[0],
                "index": key[1],
                "width": key[2],
                "epochs": key[3],
                "seed": key[4],
            },
            "runs": [
                {
                    "run_id": rid,
                    "grokked": (grok_by_run[rid] or {}).get("grokked"),
                    "censored": (grok_by_run[rid] or {}).get("censored"),
                    "epochs_to_grok": (grok_by_run[rid] or {}).get("epochs_to_grok"),
                }
                for rid in run_ids
            ],
            "epoch_spread": classified["epoch_spread"],
        }
        if classified["outcome"] == "label_disagreement":
            label_disagreements.append(detail)
        elif classified["outcome"] == "epoch_disagreement":
            epoch_disagreements.append(detail)

    label_disagreements.sort(key=lambda d: (d["key"]["order"], d["key"]["index"], d["key"]["seed"]))
    epoch_disagreements.sort(key=lambda d: -(d["epoch_spread"] or 0))

    rerun_agreement = {
        "epoch_tolerance": epoch_tolerance,
        "n_duplicate_keys": sum(1 for infos in grouped.values() if len(infos) > 1),
        "n_full_agreement": agreement_counts["agreement"],
        "n_epoch_disagreement": agreement_counts["epoch_disagreement"],
        "n_label_disagreement": agreement_counts["label_disagreement"],
        "n_missing_data": agreement_counts["missing_data"],
        "label_disagreements": label_disagreements,
        "epoch_disagreements": epoch_disagreements,
    }
    return index, rerun_agreement, n_unresolvable


def grok_totals(run_dirs_by_id: dict[str, Path]) -> dict[str, int]:
    """``{"grokked", "censored", "missing"}`` over exactly the run
    directories passed in -- used for both the raw (every dir) and canonical
    (deduped) headline counts, so the two totals are computed the same way."""
    totals = {"grokked": 0, "censored": 0, "missing": 0}
    for run_dir in run_dirs_by_id.values():
        grok = run_grok_fields(run_dir)
        if grok is None:
            totals["missing"] += 1
        elif grok["censored"]:
            totals["censored"] += 1
        else:
            totals["grokked"] += 1
    return totals


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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--epoch-tolerance",
        type=int,
        default=DEFAULT_EPOCH_TOLERANCE,
        help="max epochs-to-grok spread between reruns still called 'agreement' (default 1000)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    runs_root = args.runs_root or args.runs_root_positional or DEFAULT_RUNS_ROOT
    if not runs_root.is_dir():
        _warn(f"runs_root {runs_root} is not a directory")
        return 1

    run_dirs = discover_run_dirs(runs_root)
    index, rerun_agreement, n_unresolvable = build_index(
        run_dirs, epoch_tolerance=args.epoch_tolerance
    )

    all_dirs_by_id = {run_dir.name: run_dir for run_dir in run_dirs}
    canonical_dirs_by_id = {
        entry["canonical_run_id"]: all_dirs_by_id[entry["canonical_run_id"]]
        for entry in index.values()
        if entry["canonical_run_id"] in all_dirs_by_id
    }

    raw_totals = grok_totals(all_dirs_by_id)
    canonical_totals = grok_totals(canonical_dirs_by_id)

    n_duplicate_keys = sum(1 for entry in index.values() if entry["n_total"] > 1)
    n_dropped_dirs = sum(len(entry["dropped_run_ids"]) for entry in index.values())

    payload = {
        "runs_root": str(runs_root),
        "dedup_rule": (
            "prefer manifest.status == 'completed'; tie-break by earliest "
            "manifest.completed_at; final tie-break by lexicographically "
            "earliest run_id"
        ),
        "n_run_dirs": len(run_dirs),
        "n_run_dirs_unresolvable": n_unresolvable,
        "n_distinct_keys": len(index),
        "n_duplicate_keys": n_duplicate_keys,
        "n_dropped_duplicate_dirs": n_dropped_dirs,
        "grok_totals_raw": raw_totals,
        "grok_totals_canonical": canonical_totals,
        "rerun_agreement": rerun_agreement,
        "canonical_runs": {
            _key_str(key): {
                "order": key[0],
                "index": key[1],
                "width": key[2],
                "epochs": key[3],
                "seed": key[4],
                "canonical_run_id": entry["canonical_run_id"],
                "dropped_run_ids": entry["dropped_run_ids"],
                "n_total": entry["n_total"],
            }
            for key, entry in sorted(index.items())
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"[dedup_runs] wrote {args.out}")

    print(
        f"\nrun dirs: {len(run_dirs)}  distinct keys: {len(index)}  "
        f"duplicate keys: {n_duplicate_keys}  dropped dirs: {n_dropped_dirs}"
    )
    print(
        f"grok totals, raw (every dir):       "
        f"grokked={raw_totals['grokked']} censored={raw_totals['censored']} "
        f"missing={raw_totals['missing']}"
    )
    print(
        f"grok totals, canonical (one/seed):  "
        f"grokked={canonical_totals['grokked']} censored={canonical_totals['censored']} "
        f"missing={canonical_totals['missing']}"
    )
    print(
        f"\nrerun agreement over {rerun_agreement['n_duplicate_keys']} duplicated seeds "
        f"(epoch tolerance {rerun_agreement['epoch_tolerance']}):"
    )
    print(f"  full agreement:      {rerun_agreement['n_full_agreement']}")
    print(f"  epoch disagreement:  {rerun_agreement['n_epoch_disagreement']}")
    print(
        f"  LABEL disagreement:  {rerun_agreement['n_label_disagreement']}  (grok vs censor differs)"
    )
    print(f"  missing data:        {rerun_agreement['n_missing_data']}")
    if rerun_agreement["label_disagreements"]:
        print("\nlabel disagreements:")
        for d in rerun_agreement["label_disagreements"]:
            k = d["key"]
            outcomes = ", ".join(
                f"{r['run_id']}={'grok' if not r['censored'] else 'censored'}"
                f"({r['epochs_to_grok']})"
                for r in d["runs"]
            )
            print(f"  ({k['order']},{k['index']}) w{k['width']} seed={k['seed']}: {outcomes}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
