"""GCR character-readout entry point (the ``gcr_character_readout_instrument``
driver, ``instruments/probes.py``).

Runs the GCR character-readout instrument across the trained campaign's panel
cells, entirely offline -- no network, no W&B, no Sage/GAP::

    WANDB_MODE=disabled uv run python scripts/measure_gcr_readout.py measure \\
        --runs-dir runs --out-dir results/gcr_readout \\
        [--metric val/accuracy] [--threshold 0.99] [--max-seeds-per-cell 50] \\
        [--artifacts-dir data/group_artifacts]

A "cell" is a ``(order, index, width)`` triple: every trained run directory
under ``--runs-dir`` is grouped into its cell by reading ``resolved_config.yaml``
(mirrors ``scripts/run_campaign.py``'s ``build_completion_index`` parse), and
each cell is measured over up to ``--max-seeds-per-cell`` of its completed
seeds (sorted by run id, so the selection is deterministic). This includes
cells with no nontrivial one-dimensional irrep (perfect groups): the
instrument reports ``UNDEFINED`` for the nested comparison there rather than
being compared against an empty Fourier-only design -- a valid, recorded
outcome, not a reason to skip the cell.

Each run is loaded through the same dip-aware checkpoint rule
``instruments/checkpoints.py`` applies elsewhere (recorded, substitutions
included); a run with no stable checkpoint is recorded ``status: "skipped"``.
For a measured run, an untrained random-init model of the same shape (seeded
exactly as training would seed it) is passed as ``null_model``, so the
rule-1-regression null column travels in every record.

The instrument's load-bearing outputs are the nested full-vs-Fourier held-out
FVE gain and the minimal key-irrep set -- raw FVE is secondary context only
(see ``gcr_character_readout_instrument``'s docstring). One JSON is written per
cell to ``<out-dir>/gcr_readout_<order>_<index>_w<width>.json``, holding every
seed's full record plus per-cell aggregates (mean and 95% bootstrap CI) of the
nested gain, the raw FVEs, the minimal-set size, and the null column.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402

from group_algorithm_interp import stats  # noqa: E402
from group_algorithm_interp.config import validate_config  # noqa: E402
from group_algorithm_interp.groups.catalog import resolve_group  # noqa: E402
from group_algorithm_interp.groups.data import artifact_path  # noqa: E402
from group_algorithm_interp.instruments.checkpoints import select_checkpoint  # noqa: E402
from group_algorithm_interp.instruments.device import resolve_device  # noqa: E402
from group_algorithm_interp.instruments.probes import (  # noqa: E402
    UNDEFINED,
    gcr_character_readout_instrument,
)
from group_algorithm_interp.instruments.report import (  # noqa: E402
    file_sha256,
    instrument_code_hashes,
)
from group_algorithm_interp.manifest import get_git_commit, read_manifest  # noqa: E402
from group_algorithm_interp.seed import set_seed  # noqa: E402
from group_algorithm_interp.training.trainer import build_model  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

Cell = tuple[int, int, int]  # (order, index, width)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative_to_repo(path: Path) -> str:
    """``path`` relative to the repo root when possible, else its absolute
    form (e.g. an ``--artifacts-dir`` override or a test's tmp directory
    outside the repo) -- the same convention ``measure_probes.py`` follows."""
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path.resolve())


# ---------------------------------------------------------------------------
# Cell discovery.
# ---------------------------------------------------------------------------


def discover_cells(runs_dir: Path) -> dict[Cell, list[Path]]:
    """Every completed run directory under ``runs_dir``, grouped by its
    ``(order, index, width)`` panel cell.

    Reads ``manifest.yaml`` for the terminal status and ``resolved_config.yaml``
    for the group/width a manifest's status block does not carry -- the same
    parse ``scripts/run_campaign.py``'s ``build_completion_index`` uses. A run
    directory that fails to parse (missing file, malformed YAML, an
    unexpected shape) is silently excluded, not raised on: this is a
    discovery pass over whatever the campaign actually produced, not a
    validator of it. Run directories within a cell are sorted by run id
    (the timestamp-prefixed directory name), giving a deterministic,
    seed-ascending order to cap against.
    """
    cells: dict[Cell, list[Path]] = {}
    if not runs_dir.is_dir():
        return cells
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        manifest_path = run_dir / "manifest.yaml"
        config_path = run_dir / "resolved_config.yaml"
        if not manifest_path.is_file() or not config_path.is_file():
            continue
        try:
            manifest = yaml.safe_load(manifest_path.read_text())
        except Exception:
            continue
        if not isinstance(manifest, dict) or manifest.get("status") != "completed":
            continue
        try:
            raw_config = yaml.safe_load(config_path.read_text())
        except Exception:
            continue
        if not isinstance(raw_config, dict):
            continue
        data = raw_config.get("data")
        model = raw_config.get("model")
        group = data.get("group") if isinstance(data, dict) else None
        order = group.get("order") if isinstance(group, dict) else None
        index = group.get("index") if isinstance(group, dict) else None
        width = model.get("d_model") if isinstance(model, dict) else None
        if not (
            isinstance(order, int)
            and isinstance(index, int)
            and isinstance(width, int)
            and not any(isinstance(value, bool) for value in (order, index, width))
        ):
            continue
        cells.setdefault((order, index, width), []).append(run_dir)
    return cells


# ---------------------------------------------------------------------------
# Per-run measurement (mirrors measure_probes.py's probe_run).
# ---------------------------------------------------------------------------


def gcr_readout_run(
    run_dir: Path,
    *,
    metric: str = "val/accuracy",
    threshold: float = 0.99,
    train_frac: float = 0.7,
    target_fve: float = 0.95,
    improvement_tol: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    """One run's GCR character-readout record: dip-aware checkpoint selection
    (recorded, substitutions included), then :func:`gcr_character_readout_instrument`
    against a random-init null of the same shape. A run with no stable
    checkpoint comes back ``status: "skipped"`` with the full selection
    record -- reported as data, never silently dropped."""
    manifest = read_manifest(run_dir)
    config = validate_config(yaml.safe_load((run_dir / "resolved_config.yaml").read_text()))
    selection = select_checkpoint(run_dir, metric=metric, threshold=threshold)
    record: dict[str, Any] = {
        "instrument": "gcr-character-readout-driver",
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
            "analysed_at": _utcnow(),
            "instrument_code_sha256": instrument_code_hashes(),
        },
    }
    if selection.path is None:
        record["status"] = "skipped"
        return record

    group = resolve_group(config.data.group)
    checkpoint = torch.load(selection.path, map_location="cpu", weights_only=False)
    model = build_model(config, group)
    model.load_state_dict(checkpoint["model_state_dict"])

    # The random-init null: the model training would have started from (same
    # seeding as the training paths), for the rule-1 regression -- the same
    # convention measure_probes.py's probe_run follows.
    set_seed(config.seed, deterministic=False)
    null_model = build_model(config, group)

    group_artifact = artifact_path(config.data.group.order, config.data.group.index)
    record["status"] = "measured"
    record["provenance"]["checkpoint_sha256"] = file_sha256(selection.path)
    record["provenance"]["group_artifact_path"] = _relative_to_repo(group_artifact)
    record["provenance"]["group_artifact_sha256"] = file_sha256(group_artifact)
    record["gcr_readout"] = gcr_character_readout_instrument(
        model,
        group,
        train_frac=train_frac,
        seed=config.seed,
        null_model=null_model,
        target_fve=target_fve,
        improvement_tol=improvement_tol,
        device=device,
    )
    # Bound this process's memory to one seed at a time -- both models (the
    # trained checkpoint and the random-init null) and the checkpoint state
    # dict are never needed again once ``record`` (scalars only) is built.
    del checkpoint, model, null_model
    gc.collect()
    return record


# ---------------------------------------------------------------------------
# Per-cell aggregation.
# ---------------------------------------------------------------------------


def _summarise(values: list[float]) -> dict[str, Any]:
    """Mean and, when there are at least two distinct draws, a 95% bootstrap
    CI -- the same convention ``coset.py``'s ``_distribution_summary`` and
    ``audit.py`` follow."""
    if not values:
        return {"n": 0, "mean": None, "bootstrap_ci_95": None, "values": []}
    mean = sum(values) / len(values)
    ci: list[float] | None = None
    if len(values) >= 2 and len(set(values)) > 1:
        low, high = stats.bootstrap_ci(values)
        ci = [low, high]
    elif len(values) >= 2:
        ci = [values[0], values[0]]
    return {"n": len(values), "mean": mean, "bootstrap_ci_95": ci, "values": values}


def summarise_cell(
    order: int, index: int, width: int, records: list[dict[str, Any]]
) -> dict[str, Any]:
    """One cell's aggregate over its measured seeds: the nested full-vs-Fourier
    gain and raw FVEs (:func:`_summarise` each), the minimal-irrep-set size
    and the most frequently selected irrep set, and the null column -- built
    only from ``status: "measured"`` records. A cell whose group is perfect
    (no nontrivial one-dimensional irrep) reports ``nested_comparison_gain``
    ``UNDEFINED`` -- a recorded outcome, not a missing one -- while the raw
    FVE and minimal-set aggregates, which do not depend on the Fourier rival,
    are unaffected.
    """
    measured = [r for r in records if r.get("status") == "measured"]
    n_skipped = len(records) - len(measured)

    gains: list[float] = []
    full_fves: list[float] = []
    fourier_fves: list[float] = []
    n_selected: list[float] = []
    selected_sets: Counter[tuple[int, ...]] = Counter()
    null_full: list[float] = []
    null_fourier: list[float] = []
    n_undefined_nested = 0

    for r in measured:
        gcr = r["gcr_readout"]
        nested = gcr["primary"]["nested_comparison"]
        full_fves.append(float(nested["full_held_out_fve"]))
        if nested.get("status") == UNDEFINED:
            n_undefined_nested += 1
        else:
            gains.append(float(nested["full_vs_fourier_held_out_fve_gain"]))
            fourier_fves.append(float(nested["fourier_only_held_out_fve"]))
        minimal = gcr["primary"]["minimal_irrep_set"]
        n_selected.append(float(minimal["n_selected"]))
        selected_sets[tuple(sorted(minimal["selected_irrep_indices"]))] += 1
        null = gcr.get("null")
        if null is not None:
            if null.get("full_null_held_out_fve") is not None:
                null_full.append(float(null["full_null_held_out_fve"]))
            if null.get("fourier_only_null_held_out_fve") is not None:
                null_fourier.append(float(null["fourier_only_null_held_out_fve"]))

    if gains:
        nested_summary: dict[str, Any] = {
            "full_vs_fourier_held_out_fve_gain": _summarise(gains),
            "full_held_out_fve": _summarise(full_fves),
            "fourier_only_held_out_fve": _summarise(fourier_fves),
            "n_undefined": n_undefined_nested,
        }
    else:
        reason = (
            "no seed in this cell had a stable checkpoint (0 measured)"
            if not measured
            else "no measured seed has a Fourier-only rival (perfect group)"
        )
        nested_summary = {
            "status": UNDEFINED,
            "reason": reason,
            "full_held_out_fve": _summarise(full_fves),
            "n_undefined": n_undefined_nested,
        }

    most_common = selected_sets.most_common(1)
    return {
        "order": order,
        "index": index,
        "width": width,
        "n_seeds_attempted": len(records),
        "n_measured": len(measured),
        "n_skipped": n_skipped,
        "nested_comparison": nested_summary,
        "minimal_irrep_set": {
            "n_selected": _summarise(n_selected),
            "most_common_selected_irrep_indices": (
                list(most_common[0][0]) if most_common else None
            ),
            "most_common_count": most_common[0][1] if most_common else 0,
            "distinct_sets_observed": len(selected_sets),
        },
        "null": {
            "full_null_held_out_fve": _summarise(null_full),
            "fourier_only_null_held_out_fve": _summarise(null_fourier),
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _cmd_measure(args: argparse.Namespace) -> int:
    if args.artifacts_dir is not None:
        os.environ["GROUP_ARTIFACTS_DIR"] = str(args.artifacts_dir)
    device = resolve_device(args.device)

    cells = discover_cells(Path(args.runs_dir))
    if args.cell:
        wanted = {tuple(int(x) for x in spec.split(",")) for spec in args.cell}
        cells = {key: value for key, value in cells.items() if key in wanted}

    n_skipped_runs = 0
    for order, index, width in sorted(cells):
        run_dirs = cells[(order, index, width)][: args.max_seeds_per_cell]
        records = [
            gcr_readout_run(
                run_dir,
                metric=args.metric,
                threshold=args.threshold,
                train_frac=args.train_frac,
                target_fve=args.target_fve,
                improvement_tol=args.improvement_tol,
                device=device,
            )
            for run_dir in run_dirs
        ]
        n_skipped_runs += sum(1 for r in records if r["status"] == "skipped")
        summary = summarise_cell(order, index, width, records)
        payload = {
            "cell": {"order": order, "index": index, "width": width},
            "n_seeds_available": len(cells[(order, index, width)]),
            "max_seeds_per_cell": args.max_seeds_per_cell,
            "summary": summary,
            "runs": records,
        }
        out_path = Path(args.out_dir) / f"gcr_readout_{order}_{index}_w{width}.json"
        _write_json(out_path, payload)
        nested = summary["nested_comparison"]
        gain_desc = (
            "UNDEFINED"
            if nested.get("status") == UNDEFINED
            else f"gain={nested['full_vs_fourier_held_out_fve_gain']['mean']:.4f}"
        )
        print(
            f"[gcr_readout] ({order},{index}) w{width}: "
            f"n_measured={summary['n_measured']}/{summary['n_seeds_attempted']} "
            f"{gain_desc} -> {out_path}"
        )
    print(f"[gcr_readout] wrote {len(cells)} cell record(s) to {args.out_dir}")
    return 0 if n_skipped_runs == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser(
        "measure", help="run the GCR character-readout instrument across panel cells"
    )
    measure.add_argument(
        "--runs-dir", default="runs", help="root directory of trained run directories"
    )
    measure.add_argument(
        "--out-dir", default="results/gcr_readout", help="directory to write one JSON per cell"
    )
    measure.add_argument("--metric", default="val/accuracy", help="stability metric in run.log")
    measure.add_argument(
        "--threshold", type=float, default=0.99, help="stability bar on --metric (default 0.99)"
    )
    measure.add_argument(
        "--max-seeds-per-cell",
        type=int,
        default=50,
        dest="max_seeds_per_cell",
        help="cap on seeds measured per cell (default 50)",
    )
    measure.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        dest="train_frac",
        help="held-out pair split fraction (default 0.7, the instrument's default)",
    )
    measure.add_argument(
        "--target-fve",
        type=float,
        default=0.95,
        dest="target_fve",
        help="minimal-irrep-set search target held-out FVE (default 0.95)",
    )
    measure.add_argument(
        "--improvement-tol",
        type=float,
        default=0.01,
        dest="improvement_tol",
        help="minimal-irrep-set search stopping tolerance (default 0.01)",
    )
    measure.add_argument(
        "--artifacts-dir", default=None, help="override the group-artifact directory"
    )
    measure.add_argument(
        "--cell",
        action="append",
        default=[],
        metavar="ORDER,INDEX,WIDTH",
        help="restrict to this cell (repeatable); default is every discovered cell",
    )
    measure.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cpu"],
        help="device for the read-position forward pass; 'auto' is MPS when available, "
        "else CPU (default: auto). The fit itself always stays CPU float64.",
    )
    measure.set_defaults(func=_cmd_measure)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
