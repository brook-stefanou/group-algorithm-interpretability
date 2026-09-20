"""I-15 isotypic-block ablation across many checkpoints per cell.

``coset.isotypic_block_ablation`` (I-15, rung 3) is the instrument that
upgrades "block j is occupied" (I-10/I-14) to "the model uses block j": it
ablates one isotypic block from the shared input embedding and reports the
behavioural flip fraction against a matched-rank matched-norm random
subspace. Unlike the I-17/I-18/I-19 coset arm it is NOT gated on a
core-free-subgroup existing, so it runs on any group -- the C1/panel cells
included, not only the D32/QD32/Q32 case study ``measure_coset.py`` was built
for.

This driver runs I-15 across every seed of one (order, index, width) cell and
aggregates the per-seed, per-block, per-mode flip fractions with bootstrap
confidence intervals, so occupancy (a per-model measurement) can be compared
against usage (a per-cell measurement) directly. It reuses
``coset.isotypic_block_ablation`` and mirrors ``coset.measure_coset_run``'s
per-run scaffolding (manifest/config read, dip-aware checkpoint selection,
provenance hashes) without paying for the coset arm's core-free-subgroup
machinery, which this cell-wide sweep has no use for.

Cell resolution: by default, seeds are looked up in
``results/canonical_runs.json`` (``scripts/dedup_runs.py``'s canonical
run-per-seed index), filtered to the requested ``(order, index, width)`` and
capped at ``--max-seeds``. Explicit run directories may be passed instead
(positionally), bypassing the canonical lookup -- used by the test suite and
for ad hoc reruns.

    uv run python scripts/measure_isotypic_ablation.py measure \\
        --order 32 --index 18 --width 128 \\
        --out results/coset/iso_ablation_32_18_w128.json \\
        [--canonical-runs results/canonical_runs.json] [--runs-root runs] \\
        [--max-seeds 50] [--metric val/accuracy] [--threshold 0.99] \\
        [--n-random 16] [--artifacts-dir data/group_artifacts]

Offline (no network, no W&B); run with ``WANDB_MODE=disabled``. Exit code 0
iff every resolved run was measured; 1 when any seed was skipped for want of
a stable checkpoint (a censored seed), so a caller notices.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from group_algorithm_interp import stats  # noqa: E402
from group_algorithm_interp.instruments.coset import ABLATION_MODES  # noqa: E402
from group_algorithm_interp.instruments.device import resolve_device  # noqa: E402

# ---------------------------------------------------------------------------
# Local-checkpoint fallback
#
# This local checkout's ``runs/`` tree keeps only ``final.pt`` per run (the
# dense final-window snapshots named ``final_epoch_<E>.pt`` that
# ``checkpoints.select_checkpoint``'s dip-aware rule names are not present on
# disk for essentially any run in the corpus -- shipped/curated or freshly
# trained). Rather than reporting every seed as unmeasurable, this driver
# substitutes ``final.pt`` for the named checkpoint ONLY when it is provably
# the same trained state: the dip-aware rule's own verdict (however it is
# read) shows no substitution occurred and the selected epoch is exactly the
# run's configured ceiling epoch -- i.e. the rule's first choice (the literal
# last training step) already cleared the stability bar, so ``final.pt`` (the
# training loop's terminal dump) and the missing ``final_epoch_<last>.pt`` are
# the same weights, not a stand-in for a different, dip-avoided epoch. A
# censored run, or one whose stable pick is an earlier window epoch, is never
# affected: neither signal below fires when a real substitution or censoring
# occurred. This is a data-availability workaround local to this driver, not a
# change to ``checkpoints.py``'s shared rule; it is recorded verbatim in every
# affected record's ``checkpoint_selection.local_final_pt_fallback``.


_FINAL_EPOCH_RE = re.compile(r"final_epoch_\d+\.pt")


def _stable_end_hint(run_dir: Path) -> tuple[str | None, int | None, str | None] | None:
    """Best-effort read of the raw ``selection.json`` (either schema
    ``checkpoints.selection_from_json`` recognises) for the ``stable_end``
    candidate's ``(filename, epoch, substitution)``, used only to decide
    whether the local-``final.pt`` fallback applies -- ``select_checkpoint``
    itself nulls these fields out once path resolution fails, so they are
    read here directly. ``None`` when the file is absent/unreadable or
    carries no recognised ``stable_end`` shape; also ``None`` (not a fallback
    trigger) when a run recorded no stable checkpoint at all (a censored
    seed), since ``stable_end`` is then JSON ``null``, not a dict."""
    path = run_dir / "selection.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    selections = data.get("selections")
    if isinstance(selections, dict) and "stable_end" in selections:
        entry = selections.get("stable_end")
        if not isinstance(entry, dict):
            return None
        return entry.get("checkpoint"), entry.get("epoch"), entry.get("substitution")
    categories = data.get("categories")
    if isinstance(categories, dict) and "stable_end" in categories:
        entry = categories.get("stable_end")
        if not isinstance(entry, dict):
            return None
        return entry.get("filename"), entry.get("epoch"), entry.get("substitution")
    return None


def _final_pt_fallback(
    run_dir: Path,
    *,
    last_epoch: int,
    metric: str,
    threshold: float,
) -> tuple[Path, dict[str, Any]] | None:
    """``(final.pt path, note)`` when ``final.pt`` is provably the same
    trained state as the dip-aware rule's missing pick (module-level comment),
    else ``None``. Two independent signals, either sufficient: the recorded
    ``stable_end`` verdict (works with no local training log at all -- the
    common case for a curated run), and a direct ``run.log``/``run.log.gz``
    read at the ceiling epoch (works for a freshly trained local run with no
    ``selection.json``)."""
    from group_algorithm_interp.instruments.checkpoints import parse_run_log, resolve_run_log

    final_pt = run_dir / "final.pt"
    if not final_pt.is_file():
        return None

    hint = _stable_end_hint(run_dir)
    if hint is not None:
        filename, epoch, substitution = hint
        if (
            filename
            and _FINAL_EPOCH_RE.fullmatch(str(filename))
            and substitution is None
            and epoch == last_epoch
        ):
            return final_pt, {
                "signal": "selection.json stable_end: no substitution, ceiling epoch",
                "named_checkpoint": filename,
            }

    log_path = resolve_run_log(run_dir)
    if log_path is not None:
        rows = {row.epoch: row.metrics for row in parse_run_log(log_path)}
        metrics_at_last = rows.get(last_epoch)
        if metrics_at_last is not None and metric in metrics_at_last:
            value = metrics_at_last[metric]
            if not math.isnan(value) and value >= threshold:
                return final_pt, {
                    "signal": "run.log at the ceiling epoch clears the stability bar",
                    "metric": metric,
                    "metric_value": value,
                }
    return None


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _measure_run(
    run_dir: Path,
    *,
    metric: str,
    threshold: float,
    n_random: int,
    device: Any = None,
) -> dict[str, Any]:
    """The I-15 measurement for one run: manifest/config read, dip-aware
    checkpoint selection and provenance exactly as
    ``coset.measure_coset_run``, but computing only
    ``isotypic_block_ablation`` -- not the core-free-subgroup-gated coset arm,
    which this cell-wide sweep (run on every panel cell, not only the
    D32/QD32/Q32 case study) has no use for."""
    import torch
    import yaml

    from group_algorithm_interp.config import validate_config
    from group_algorithm_interp.groups.catalog import resolve_group
    from group_algorithm_interp.groups.data import artifact_path
    from group_algorithm_interp.instruments.checkpoints import select_checkpoint
    from group_algorithm_interp.instruments.coset import isotypic_block_ablation, unleaked_heldout
    from group_algorithm_interp.instruments.report import file_sha256, instrument_code_hashes
    from group_algorithm_interp.manifest import get_git_commit, read_manifest
    from group_algorithm_interp.training.trainer import build_model

    manifest = read_manifest(run_dir)
    config = validate_config(yaml.safe_load((run_dir / "resolved_config.yaml").read_text()))
    selection = select_checkpoint(run_dir, metric=metric, threshold=threshold)
    # Same artifact-hashing convention as measure_coset_run: pin the exact
    # group-artifact file the run is analysed against.
    art_path = artifact_path(config.data.group.order, config.data.group.index)
    repo_root = Path(__file__).resolve().parent.parent
    try:
        art_rel = str(art_path.relative_to(repo_root))
    except ValueError:
        art_rel = art_path.name
    record: dict[str, Any] = {
        "instrument": "isotypic-block-ablation-run",
        "run_id": manifest.get("run_id", run_dir.name),
        "seed": config.seed,
        "group": {
            "order": config.data.group.order,
            "index": config.data.group.index,
            "name": config.data.group.canonical_name,
        },
        "model": {
            "arch": config.model.arch,
            "d_model": config.model.d_model,
            "d_mlp": config.model.d_mlp,
            "activation": config.model.activation,
        },
        "checkpoint_selection": selection.to_record(),
        "provenance": {
            "git_commit": manifest.get("provenance", {}).get("git_commit"),
            "config_hash": manifest.get("provenance", {}).get("config_hash"),
            "config_group_hash": manifest.get("provenance", {}).get("config_group_hash"),
            "campaign_id": manifest.get("provenance", {}).get("campaign_id"),
            "dataset_spec_hash": manifest.get("dataset", {}).get("spec_hash"),
            "analysis_git_commit": get_git_commit(),
            "instrument_code_sha256": instrument_code_hashes(),
            "group_artifact": art_rel,
            "group_artifact_sha256": (file_sha256(art_path) if art_path.is_file() else None),
        },
    }
    checkpoint_path = selection.path
    if checkpoint_path is None:
        fallback = _final_pt_fallback(
            run_dir,
            last_epoch=config.optim.epochs - 1,
            metric=metric,
            threshold=threshold,
        )
        if fallback is None:
            record["status"] = "skipped"
            return record
        checkpoint_path, note = fallback
        record["checkpoint_selection"]["local_final_pt_fallback"] = note

    group = resolve_group(config.data.group)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(config, group)
    model.load_state_dict(checkpoint["model_state_dict"])
    tokens, targets = unleaked_heldout(
        group, train_frac=config.data.train_frac, split_seed=config.effective_split_seed
    )

    record["status"] = "measured"
    record["provenance"]["checkpoint_sha256"] = file_sha256(checkpoint_path)
    record["n_test"] = int(tokens.shape[0])
    record["isotypic_block_ablation"] = isotypic_block_ablation(
        model,
        group,
        tokens,
        targets,
        n_random=n_random,
        seed=config.seed,
        device=device if device is not None else torch.device("cpu"),
    )
    # Bound this process's memory to one seed at a time -- the checkpoint
    # state dict and the model built from it are never needed again once
    # ``record`` (scalars only) is built, so drop them explicitly rather than
    # relying on the caller's next loop iteration to overwrite the locals.
    del checkpoint, model, tokens, targets
    gc.collect()
    return record


def _summary(values: list[float]) -> dict[str, Any]:
    """Mean and, when there are at least two finite seed values, a 95%
    bootstrap CI across seeds -- the same convention as
    ``coset._distribution_summary``, applied here across seeds rather than
    across random-subspace draws."""
    vals = [float(v) for v in values if np.isfinite(v)]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "bootstrap_ci_95": None, "values": []}
    mean = float(np.mean(vals))
    ci: list[float] | None = None
    if n >= 2 and len(set(vals)) > 1:
        low, high = stats.bootstrap_ci(vals)
        ci = [low, high]
    elif n >= 2:
        ci = [vals[0], vals[0]]
    return {"n": n, "mean": mean, "bootstrap_ci_95": ci, "values": vals}


def aggregate_cell(
    records: list[dict[str, Any]],
    *,
    order: int,
    index: int,
    width: int,
    name: str | None = None,
) -> dict[str, Any]:
    """Aggregate I-15 across every seed of one cell: per (block, mode), the
    seed-level mean and bootstrap CI of the flip fraction, the matched-random
    baseline, and ``flip_fraction_over_random`` (the block-specific effect
    that licenses "used", not merely "occupied"). Skipped seeds (no stable
    checkpoint -- a censored seed) are counted but excluded from the
    aggregate. Full per-run records are kept under ``runs`` so every
    aggregated number traces back to the per-seed measurement and its
    provenance hashes."""
    measured = [r for r in records if r["status"] == "measured"]
    skipped = [r for r in records if r["status"] != "measured"]
    group_name = name if name is not None else (measured[0]["group"]["name"] if measured else None)

    clean_accuracies = [r["isotypic_block_ablation"]["clean_accuracy"] for r in measured]

    block_data: dict[int, dict[str, Any]] = {}
    for r in measured:
        for block in r["isotypic_block_ablation"]["blocks"]:
            idx = int(block["block_index"])
            entry = block_data.setdefault(
                idx,
                {
                    "block_index": idx,
                    "irrep_degree": block["irrep_degree"],
                    "block_rank": block["block_rank"],
                    "is_trivial": block["is_trivial"],
                    "modes": {
                        mode: {
                            "flip_fraction": [],
                            "random_baseline": [],
                            "flip_fraction_over_random": [],
                        }
                        for mode in ABLATION_MODES
                    },
                },
            )
            for mode, mode_stats in block["modes"].items():
                entry["modes"][mode]["flip_fraction"].append(mode_stats["flip_fraction"])
                entry["modes"][mode]["random_baseline"].append(
                    mode_stats["random_subspace"]["mean"]
                )
                entry["modes"][mode]["flip_fraction_over_random"].append(
                    mode_stats["flip_fraction_over_random"]
                )

    blocks_out = []
    for idx in sorted(block_data):
        entry = block_data[idx]
        modes_out = {}
        for mode in ABLATION_MODES:
            per_mode = entry["modes"][mode]
            modes_out[mode] = {
                "flip_fraction": _summary(per_mode["flip_fraction"]),
                "random_baseline": _summary(per_mode["random_baseline"]),
                "flip_fraction_over_random": _summary(per_mode["flip_fraction_over_random"]),
            }
        blocks_out.append(
            {
                "block_index": entry["block_index"],
                "irrep_degree": entry["irrep_degree"],
                "block_rank": entry["block_rank"],
                "is_trivial": entry["is_trivial"],
                "modes": modes_out,
            }
        )

    return {
        "instrument": "isotypic-block-ablation-panel",
        "cell": {"order": order, "index": index, "width": width, "name": group_name},
        "n_seeds_requested": len(records),
        "n_seeds_measured": len(measured),
        "n_seeds_skipped": len(skipped),
        "measured_seeds": sorted(int(r["seed"]) for r in measured),
        "skipped_seeds": sorted(int(r["seed"]) for r in skipped),
        "clean_accuracy": _summary(clean_accuracies),
        "blocks": blocks_out,
        "runs": records,
    }


def _resolve_cell_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dirs:
        return [Path(p) for p in args.run_dirs]
    canonical = json.loads(Path(args.canonical_runs).read_text())
    entries = [
        entry
        for entry in canonical["canonical_runs"].values()
        if entry["order"] == args.order
        and entry["index"] == args.index
        and entry["width"] == args.width
    ]
    entries.sort(key=lambda entry: entry["seed"])
    entries = entries[: args.max_seeds]
    if not entries:
        print(
            f"[iso-ablation] WARNING: no canonical runs found for "
            f"order={args.order} index={args.index} width={args.width} in {args.canonical_runs}",
            file=sys.stderr,
        )
    return [Path(args.runs_root) / entry["canonical_run_id"] for entry in entries]


def _cmd_measure(args: argparse.Namespace) -> int:
    if args.artifacts_dir is not None:
        os.environ["GROUP_ARTIFACTS_DIR"] = str(args.artifacts_dir)
    device = resolve_device(args.device)

    run_dirs = _resolve_cell_run_dirs(args)
    records: list[dict[str, Any]] = []
    n_skipped = 0
    for run_dir in run_dirs:
        record = _measure_run(
            run_dir,
            metric=args.metric,
            threshold=args.threshold,
            n_random=args.n_random,
            device=device,
        )
        records.append(record)
        if record["status"] == "measured":
            print(
                f"[iso-ablation] {record['run_id']}: seed={record['seed']} "
                f"checkpoint={record['checkpoint_selection']['checkpoint']}"
            )
        else:
            n_skipped += 1
            print(
                f"[iso-ablation] {record['run_id']}: SKIPPED -- "
                f"{record['checkpoint_selection']['reason']}"
            )

    aggregate = aggregate_cell(
        records, order=args.order, index=args.index, width=args.width, name=args.name
    )
    if args.out is not None:
        _write_json(Path(args.out), aggregate)
        print(
            f"[iso-ablation] wrote cell (order={args.order}, index={args.index}, width={args.width}) "
            f"with {aggregate['n_seeds_measured']}/{len(records)} measured seed(s) -> {args.out}"
        )
    return 0 if n_skipped == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser("measure", help="measure I-15 across a cell's seeds and aggregate")
    measure.add_argument(
        "run_dirs", nargs="*", help="explicit run directories (bypasses canonical lookup)"
    )
    measure.add_argument("--order", type=int, required=True, help="group order")
    measure.add_argument("--index", type=int, required=True, help="SmallGroup index")
    measure.add_argument(
        "--width", type=int, required=True, help="d_model for cell resolution/labelling"
    )
    measure.add_argument("--name", default=None, help="group name override for the output record")
    measure.add_argument("--out", default=None, help="write the aggregated cell record here")
    measure.add_argument(
        "--canonical-runs",
        default="results/canonical_runs.json",
        help="canonical run-per-seed index (scripts/dedup_runs.py output)",
    )
    measure.add_argument(
        "--runs-root", default="runs", help="root directory holding run directories"
    )
    measure.add_argument("--max-seeds", type=int, default=50, help="cap on seeds measured per cell")
    measure.add_argument("--metric", default="val/accuracy", help="stability metric in run.log")
    measure.add_argument(
        "--threshold", type=float, default=0.99, help="stability bar on --metric (default 0.99)"
    )
    measure.add_argument(
        "--n-random", type=int, default=16, help="matched random subspaces per block ablation"
    )
    measure.add_argument(
        "--artifacts-dir", default=None, help="override the group-artifact directory"
    )
    measure.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cpu"],
        help="device for the ablation forward passes; 'auto' is MPS when available, else CPU "
        "(default: auto). Downstream analysis always stays CPU float64.",
    )
    measure.set_defaults(func=_cmd_measure)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
