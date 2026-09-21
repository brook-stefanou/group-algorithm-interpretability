"""Power-map-attribution entry point (the
``power_map_attribution_instrument`` driver,
``instruments/power_map_attribution.py``).

Runs the power-map attribution instrument across the trained campaign's
panel cells, entirely offline -- no network, no W&B, no Sage/GAP::

    WANDB_MODE=disabled uv run python scripts/measure_power_map_attribution.py measure \\
        --runs-dir runs --out-dir results/power_map_attribution \\
        [--metric val/accuracy] [--threshold 0.99] [--max-seeds-per-cell 50] \\
        [--artifacts-dir data/group_artifacts] [--cell 216,106,128] [--power-k 2]

A "cell" is a ``(order, index, width)`` triple, discovered the same way
``scripts/measure_readout_characterisation.py`` discovers them (completed
manifests under ``--runs-dir``, grouped by ``resolved_config.yaml``'s group
and ``model.d_model``). ``--cell`` restricts to specific cells; the default
is every discovered cell.

Each run is loaded through the same dip-aware checkpoint rule
``instruments/checkpoints.py`` applies elsewhere (recorded, substitutions
included); a run with no stable checkpoint is recorded ``status:
"skipped"``. For a measured run, an untrained random-init model of the same
shape (seeded exactly as training would seed it) is passed as ``null_model``
-- the same rule-1-regression null convention ``measure_readout_characterisation.py``
follows.

The instrument's load-bearing output is
``held_out_fve_gain_headline`` (does the model's raw embedding, added to the
conjugacy-class one-hot, explain the headline power-map target's held-out
variance any better than conjugacy class alone); raw FVEs are secondary
context (see ``power_map_attribution_instrument``'s docstring -- a raw FVE
alone is not evidence, only the held-out gain is). One JSON is written per
cell to ``<out-dir>/power_map_attribution_<order>_<index>_w<width>.json``,
holding every seed's full record plus per-cell aggregates (mean and 95%
bootstrap CI) of the headline gain, the raw FVEs, and the null column.

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
from group_algorithm_interp.instruments.power_map_attribution import (  # noqa: E402
    UNDEFINED,
    power_map_attribution_instrument,
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
    form (an ``--artifacts-dir`` override or a test's tmp directory outside
    the repo) -- the same convention ``measure_readout_characterisation.py``
    follows."""
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path.resolve())


# ---------------------------------------------------------------------------
# Cell discovery (mirrors scripts/measure_readout_characterisation.py's
# discover_cells).
# ---------------------------------------------------------------------------


def discover_cells(runs_dir: Path) -> dict[Cell, list[Path]]:
    """Every completed run directory under ``runs_dir``, grouped by its
    ``(order, index, width)`` panel cell. A run directory that fails to parse
    is silently excluded rather than raised on, since this is a discovery
    pass over whatever a campaign actually produced. Run directories within a
    cell are sorted by run id."""
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
# Per-run measurement (mirrors measure_readout_characterisation.py's
# readout_characterisation_run).
# ---------------------------------------------------------------------------


def power_map_attribution_run(
    run_dir: Path,
    *,
    metric: str = "val/accuracy",
    threshold: float = 0.99,
    power_ks: tuple[int, ...] = (2,),
    n_folds: int = 5,
    tie_tol: float = 0.01,
) -> dict[str, Any]:
    """One run's power-map-attribution record: dip-aware checkpoint selection
    (recorded, substitutions included), then
    :func:`power_map_attribution_instrument` against a random-init null of
    the same shape. A run with no stable checkpoint comes back ``status:
    "skipped"`` with the full selection record -- reported as data, never
    silently dropped."""
    manifest = read_manifest(run_dir)
    config = validate_config(yaml.safe_load((run_dir / "resolved_config.yaml").read_text()))
    selection = select_checkpoint(run_dir, metric=metric, threshold=threshold)
    record: dict[str, Any] = {
        "instrument": "power-map-attribution-driver",
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
    # convention measure_readout_characterisation.py's driver follows.
    set_seed(config.seed, deterministic=False)
    null_model = build_model(config, group)

    group_artifact = artifact_path(config.data.group.order, config.data.group.index)
    record["status"] = "measured"
    record["provenance"]["checkpoint_sha256"] = file_sha256(selection.path)
    record["provenance"]["group_artifact_path"] = _relative_to_repo(group_artifact)
    record["provenance"]["group_artifact_sha256"] = file_sha256(group_artifact)
    record["power_map_attribution"] = power_map_attribution_instrument(
        model,
        group,
        power_ks=power_ks,
        n_folds=n_folds,
        seed=config.seed,
        tie_tol=tie_tol,
        null_model=null_model,
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
    CI -- the same convention ``measure_readout_characterisation.py``'s
    ``_summarise`` follows."""
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
    """One cell's aggregate over its measured seeds: the headline held-out
    gain, the raw restricted/full FVEs, and the null column. Built only from
    ``status: "measured"`` records. A cell whose headline target is
    ``UNDEFINED`` for every measured seed reports that explicitly rather
    than a fabricated number."""
    measured = [r for r in records if r.get("status") == "measured"]
    n_skipped = len(records) - len(measured)

    gains: list[float] = []
    restricted_fves: list[float] = []
    full_fves: list[float] = []
    n_undefined = 0
    null_gains: list[float] = []
    headline_key: str | None = None

    for r in measured:
        pm = r["power_map_attribution"]
        headline_key = pm["headline_target"]
        headline = pm["nested_beyond_class"].get(headline_key)
        if headline == UNDEFINED or headline is None:
            n_undefined += 1
            continue
        gains.append(float(pm["held_out_fve_gain_headline"]))
        restricted_fves.append(float(headline["restricted_held_out_fve"]))
        full_fves.append(float(headline["full_held_out_fve"]))
        null = pm.get("null")
        if null is not None:
            null_headline = null["nested_beyond_class"].get(headline_key)
            if isinstance(null_headline, dict):
                null_gains.append(float(null_headline["held_out_fve_gain_full_minus_restricted"]))

    n_defined = len(measured) - n_undefined

    if gains:
        gain_summary: dict[str, Any] = {
            "headline_target": headline_key,
            "held_out_fve_gain_headline": _summarise(gains),
            "restricted_held_out_fve": _summarise(restricted_fves),
            "full_held_out_fve": _summarise(full_fves),
            "n_undefined": n_undefined,
            "n_defined": n_defined,
        }
    else:
        reason = (
            "no seed in this cell had a stable checkpoint (0 measured)"
            if not measured
            else "every measured seed's headline target was UNDEFINED"
        )
        gain_summary = {"status": UNDEFINED, "reason": reason, "n_undefined": n_undefined}

    return {
        "order": order,
        "index": index,
        "width": width,
        "n_seeds_attempted": len(records),
        "n_measured": len(measured),
        "n_skipped": n_skipped,
        "gain": gain_summary,
        "null": {"held_out_fve_gain_headline": _summarise(null_gains)},
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _cmd_measure(args: argparse.Namespace) -> int:
    if args.artifacts_dir is not None:
        os.environ["GROUP_ARTIFACTS_DIR"] = str(args.artifacts_dir)

    cells = discover_cells(Path(args.runs_dir))
    if args.cell:
        wanted = {tuple(int(x) for x in spec.split(",")) for spec in args.cell}
        cells = {key: value for key, value in cells.items() if key in wanted}

    power_ks = tuple(args.power_k)
    n_skipped_runs = 0
    for order, index, width in sorted(cells):
        run_dirs = cells[(order, index, width)][: args.max_seeds_per_cell]
        records = [
            power_map_attribution_run(
                run_dir,
                metric=args.metric,
                threshold=args.threshold,
                power_ks=power_ks,
                n_folds=args.n_folds,
                tie_tol=args.tie_tol,
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
        out_path = Path(args.out_dir) / f"power_map_attribution_{order}_{index}_w{width}.json"
        _write_json(out_path, payload)
        gain = summary["gain"]
        gain_desc = (
            "UNDEFINED"
            if gain.get("status") == UNDEFINED
            else f"gain={gain['held_out_fve_gain_headline']['mean']:.4f}"
        )
        print(
            f"[power_map_attribution] ({order},{index}) w{width}: "
            f"n_measured={summary['n_measured']}/{summary['n_seeds_attempted']} "
            f"{gain_desc} -> {out_path}"
        )
    print(f"[power_map_attribution] wrote {len(cells)} cell record(s) to {args.out_dir}")
    return 0 if n_skipped_runs == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser(
        "measure", help="run the power-map-attribution instrument across panel cells"
    )
    measure.add_argument(
        "--runs-dir", default="runs", help="root directory of trained run directories"
    )
    measure.add_argument(
        "--out-dir",
        default="results/power_map_attribution",
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
        "--power-k",
        type=int,
        action="append",
        dest="power_k",
        default=None,
        help="power(s) k probed as the headline residual axis (repeatable); default is just "
        "squaring (k=2) -- see instruments/power_map_attribution.py's module docstring for why "
        "cubing is dropped from the default",
    )
    measure.add_argument(
        "--n-folds",
        type=int,
        default=5,
        dest="n_folds",
        help="K-fold count for the nested held-out-FVE test (default 5, capped at the group order)",
    )
    measure.add_argument(
        "--tie-tol",
        type=float,
        default=0.01,
        dest="tie_tol",
        help="held-out FVE gain above which power_map_beyond_class is True (default 0.01)",
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
    measure.set_defaults(func=_cmd_measure)

    args = parser.parse_args(argv)
    if getattr(args, "power_k", None) is None:
        args.power_k = [2]
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
