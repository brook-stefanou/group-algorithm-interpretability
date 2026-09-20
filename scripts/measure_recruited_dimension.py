"""Recruited-dimension entry point (``instruments/recruited_dimension.py``'s
``measure_recruited_dimension`` driver).

Runs the recruited-dimension instrument across the trained campaign's panel
cells, entirely offline -- no network, no W&B, no Sage/GAP::

    WANDB_MODE=disabled uv run python scripts/measure_recruited_dimension.py measure \\
        --runs-dir runs --out-dir results/recruited_dimension \\
        [--metric val/accuracy] [--threshold 0.99] [--max-seeds-per-cell 50] \\
        [--artifacts-dir data/group_artifacts] [--cell 32,18,128] [--device cpu]

A "cell" is a ``(order, index, width)`` triple: every trained run directory
under ``--runs-dir`` is grouped into its cell by reading ``resolved_config.yaml``
(mirrors ``scripts/measure_gcr_readout.py``'s ``discover_cells``), and each
cell is measured over up to ``--max-seeds-per-cell`` of its completed seeds
(sorted by run id, so the selection is deterministic).

Each run is loaded through the same dip-aware checkpoint rule
``instruments/checkpoints.py`` applies elsewhere (recorded, substitutions
included); a run with no stable checkpoint is recorded ``status: "skipped"``.
For a measured run, :func:`instruments.recruited_dimension.measure_recruited_dimension`
is called against the checkpoint alone -- there is no null-model comparison
here (the theoretical anchors, not a random-init baseline, are the
comparison).

One JSON is written per cell to
``<out-dir>/recruited_dimension_<order>_<index>_w<width>.json``, holding every
seed's full record plus per-cell aggregates (mean and 95% bootstrap CI) of the
primary recruited dimension, the discrimination ratios against each
theoretical anchor, and a count of which anchor each measured seed sat
closest to.

Exit code: ``0`` iff every requested run in a cell was measured; ``1`` when
any run was skipped for want of a stable checkpoint, so a campaign wrapper
notices missing measurements loudly.
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
from group_algorithm_interp.instruments.recruited_dimension import (  # noqa: E402
    measure_recruited_dimension,
)
from group_algorithm_interp.instruments.report import (  # noqa: E402
    file_sha256,
    instrument_code_hashes,
)
from group_algorithm_interp.manifest import get_git_commit, read_manifest  # noqa: E402
from group_algorithm_interp.training.trainer import build_model  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

Cell = tuple[int, int, int]  # (order, index, width)

#: The anchors ``discriminate()`` always reports; aggregated per cell in the
#: order recorded, so the aggregate's key set never depends on which anchor
#: happened to be closest for a particular seed.
_ANCHOR_NAMES = ("minimal_faithful_real", "tensor_rank_lower", "regular_rep")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative_to_repo(path: Path) -> str:
    """``path`` relative to the repo root when possible, else its absolute
    form (an ``--artifacts-dir`` override or a test's tmp directory outside
    the repo) -- the same convention ``measure_gcr_readout.py`` follows."""
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path.resolve())


# ---------------------------------------------------------------------------
# Cell discovery (mirrors scripts/measure_gcr_readout.py's discover_cells).
# ---------------------------------------------------------------------------


def discover_cells(runs_dir: Path) -> dict[Cell, list[Path]]:
    """Every completed run directory under ``runs_dir``, grouped by its
    ``(order, index, width)`` panel cell. See
    ``scripts/measure_gcr_readout.py::discover_cells`` (identical parse); a
    run directory that fails to parse is silently excluded rather than raised
    on, since this is a discovery pass over whatever a campaign actually
    produced. Run directories within a cell are sorted by run id."""
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
# Per-run measurement.
# ---------------------------------------------------------------------------


def recruited_dimension_run(
    run_dir: Path,
    *,
    metric: str = "val/accuracy",
    threshold: float = 0.99,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    """One run's recruited-dimension record: dip-aware checkpoint selection
    (recorded, substitutions included), then
    :func:`instruments.recruited_dimension.measure_recruited_dimension`. A run
    with no stable checkpoint comes back ``status: "skipped"`` with the full
    selection record -- reported as data, never silently dropped."""
    manifest = read_manifest(run_dir)
    config = validate_config(yaml.safe_load((run_dir / "resolved_config.yaml").read_text()))
    selection = select_checkpoint(run_dir, metric=metric, threshold=threshold)
    record: dict[str, Any] = {
        "instrument": "recruited-dimension-driver",
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

    group_artifact = artifact_path(config.data.group.order, config.data.group.index)
    record["status"] = "measured"
    record["provenance"]["checkpoint_sha256"] = file_sha256(selection.path)
    record["provenance"]["group_artifact_path"] = _relative_to_repo(group_artifact)
    record["provenance"]["group_artifact_sha256"] = file_sha256(group_artifact)
    record["recruited_dimension"] = measure_recruited_dimension(
        model, group, device=device
    ).to_record()
    # Bound this process's memory to one seed at a time -- the checkpoint
    # state dict and the model built from it are never needed again once
    # ``record`` (scalars only) is built.
    del checkpoint, model
    gc.collect()
    return record


# ---------------------------------------------------------------------------
# Per-cell aggregation.
# ---------------------------------------------------------------------------


def _summarise(values: list[float]) -> dict[str, Any]:
    """Mean and, when there are at least two distinct draws, a 95% bootstrap
    CI -- the same convention ``measure_gcr_readout.py``'s ``_summarise``
    follows."""
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
    """One cell's aggregate over its measured seeds: the primary recruited
    dimension, the discrimination ratio against each theoretical anchor, and a
    count of which anchor each measured seed sat closest to. Built only from
    ``status: "measured"`` records. The theoretical anchors are deterministic
    per group (independent of the seed), so they are carried once from the
    first measured record rather than re-aggregated."""
    measured = [r for r in records if r.get("status") == "measured"]
    n_skipped = len(records) - len(measured)

    primary: list[float] = []
    ratios: dict[str, list[float]] = {name: [] for name in _ANCHOR_NAMES}
    closest_counts: Counter[str] = Counter()
    primary_measures: Counter[str] = Counter()
    activation_available_counts: Counter[bool] = Counter()

    for r in measured:
        rd = r["recruited_dimension"]
        primary.append(float(rd["primary_recruited"]))
        primary_measures[rd["primary_measure"]] += 1
        activation_available_counts[bool(rd["activation_available"])] += 1
        discrimination = rd["discrimination"]
        for name in _ANCHOR_NAMES:
            if name in discrimination["ratios"]:
                ratios[name].append(float(discrimination["ratios"][name]))
        closest_counts[discrimination["closest"]] += 1

    theoretical = measured[0]["recruited_dimension"]["theoretical"] if measured else None

    return {
        "order": order,
        "index": index,
        "width": width,
        "n_seeds_attempted": len(records),
        "n_measured": len(measured),
        "n_skipped": n_skipped,
        "theoretical": theoretical,
        "primary_recruited": _summarise(primary),
        "primary_measure_counts": dict(primary_measures),
        "activation_available_counts": {str(k): v for k, v in activation_available_counts.items()},
        "discrimination_ratios": {name: _summarise(values) for name, values in ratios.items()},
        "closest_anchor_counts": dict(closest_counts),
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
            recruited_dimension_run(
                run_dir,
                metric=args.metric,
                threshold=args.threshold,
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
        out_path = Path(args.out_dir) / f"recruited_dimension_{order}_{index}_w{width}.json"
        _write_json(out_path, payload)
        primary_desc = (
            f"primary={summary['primary_recruited']['mean']:.3f}"
            if summary["primary_recruited"]["mean"] is not None
            else "primary=n/a"
        )
        print(
            f"[recruited_dimension] ({order},{index}) w{width}: "
            f"n_measured={summary['n_measured']}/{summary['n_seeds_attempted']} "
            f"{primary_desc} -> {out_path}"
        )
    print(f"[recruited_dimension] wrote {len(cells)} cell record(s) to {args.out_dir}")
    return 0 if n_skipped_runs == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser(
        "measure", help="run the recruited-dimension instrument across panel cells"
    )
    measure.add_argument(
        "--runs-dir", default="runs", help="root directory of trained run directories"
    )
    measure.add_argument(
        "--out-dir",
        default="results/recruited_dimension",
        help="directory to write one JSON per cell",
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
        default="cpu",
        choices=["auto", "mps", "cpu"],
        help="device for the activation forward pass; MPS lacks float64 so the "
        "effective-rank SVDs always stay CPU regardless (default: cpu -- MPS is "
        "slower here than CPU for this instrument's small forward passes)",
    )
    measure.set_defaults(func=_cmd_measure)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
