"""I-20 coset-quotient-route entry point (``instruments/coset_quotient.py``'s
``coset_quotient_route`` driver).

The non-degenerate coset-route decider for the D32/QD32 case study (and any
other group carrying a proper nontrivial normal subgroup): restricts/ablates
the rank-``[G:N]`` coset-mean subspace of the shared element embedding and
scores output-*coset* accuracy against a rank- and norm-matched random
control -- see ``instruments/coset_quotient.py``'s module docstring for why
this decides coset-route *organisation* but not coset-vs-GCR (the
:data:`DISCRIMINATION_NOTE` travels with every record). Runs entirely offline
from run-directory artifacts -- no network, no W&B, no Sage/GAP::

    WANDB_MODE=disabled uv run python scripts/measure_coset_quotient.py measure \\
        --runs-dir runs --out-dir results/coset_quotient \\
        [--metric val/accuracy] [--threshold 0.99] [--max-seeds-per-cell 50] \\
        [--n-random 16] [--artifacts-dir data/group_artifacts] \\
        [--cell 32,18,128] [--device cpu]

A "cell" is a ``(order, index, width)`` triple, discovered the same way
``scripts/measure_gcr_readout.py`` discovers them. **The screen comes
first, checkpoint-free**: :func:`instruments.coset_quotient.normal_quotients`
is read from the group artifact alone; a group exporting no proper nontrivial
normal subgroup has no coset-quotient route to measure at all
(``coset_quotient_route`` would just report ``defined: False`` for every
seed), so the whole cell is skipped before any checkpoint is touched, and no
output file is written for it. This is general -- it runs on any group the
campaign covers -- but the primary target is the D32/QD32 cells
((32,18)/(32,19)/(32,20)), the only case study I-18's rank-30 core-free-subgroup
test could not resolve.

Each measured run applies the same dip-aware checkpoint rule
(``instruments/checkpoints.py``) every other instrument in this project uses
(recorded, substitutions included); a run with no stable checkpoint is
recorded ``status: "skipped"``. Scored on the transpose-unleaked held-out set
(``instruments.coset.unleaked_heldout``), the same behavioural substrate the
rest of the coset arm uses.

One JSON is written per cell with at least one exported normal quotient to
``<out-dir>/coset_quotient_<order>_<index>_w<width>.json``, holding every
seed's full record plus per-quotient (keyed by ``subgroup_index``) aggregates
(mean and 95% bootstrap CI) of the sufficiency/necessity effects over measured
seeds, and a count of which quotient was decisive per seed.

Exit code: ``0`` iff every attempted run in a screened-in cell was measured;
``1`` when any such run was skipped for want of a stable checkpoint. A cell
with no exported normal quotient never counts against this -- it was never
attempted.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import Counter, defaultdict
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
from group_algorithm_interp.instruments.coset import unleaked_heldout  # noqa: E402
from group_algorithm_interp.instruments.coset_quotient import (  # noqa: E402
    DISCRIMINATION_NOTE,
    coset_quotient_route,
    normal_quotients,
)
from group_algorithm_interp.instruments.device import resolve_device  # noqa: E402
from group_algorithm_interp.instruments.report import (  # noqa: E402
    file_sha256,
    instrument_code_hashes,
)
from group_algorithm_interp.manifest import get_git_commit, read_manifest  # noqa: E402
from group_algorithm_interp.training.trainer import build_model  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

Cell = tuple[int, int, int]  # (order, index, width)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative_to_repo(path: Path) -> str:
    """``path`` relative to the repo root when possible, else its absolute
    form -- the same convention ``measure_gcr_readout.py``/``report.py``
    follow."""
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
    on. Run directories within a cell are sorted by run id."""
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
# Screen (group-level, no checkpoint needed).
# ---------------------------------------------------------------------------


def screen_cell(order: int, index: int) -> dict[str, Any]:
    """Whether ``(order, index)`` has a usable nontrivial normal subgroup to
    run I-20 on, from the group artifact alone. ``status`` is
    ``"screened"`` on a successful load (whatever the verdict), or
    ``"artifact_missing"`` when the group artifact is not on disk under the
    configured ``GROUP_ARTIFACTS_DIR`` -- reported as data rather than
    letting a missing artifact crash a cell sweep."""
    try:
        group = resolve_group((order, index))
    except FileNotFoundError as error:
        return {"status": "artifact_missing", "reason": str(error)}
    quotients = normal_quotients(group)
    return {
        "status": "screened",
        "usable": len(quotients) > 0,
        "n_quotients": len(quotients),
        "quotient_subgroup_indices": [q.subgroup_index for q in quotients],
    }


# ---------------------------------------------------------------------------
# Per-run measurement.
# ---------------------------------------------------------------------------


def coset_quotient_run(
    run_dir: Path,
    *,
    metric: str = "val/accuracy",
    threshold: float = 0.99,
    n_random: int = 16,
) -> dict[str, Any]:
    """One run's I-20 record: dip-aware checkpoint selection (recorded,
    substitutions included), then :func:`coset_quotient_route` on the
    transpose-unleaked held-out set. A run with no stable checkpoint comes
    back ``status: "skipped"`` with the full selection record -- reported as
    data, never silently dropped."""
    manifest = read_manifest(run_dir)
    config = validate_config(yaml.safe_load((run_dir / "resolved_config.yaml").read_text()))
    selection = select_checkpoint(run_dir, metric=metric, threshold=threshold)
    record: dict[str, Any] = {
        "instrument": "coset-quotient-driver",
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
    tokens, targets = unleaked_heldout(
        group, train_frac=config.data.train_frac, split_seed=config.effective_split_seed
    )

    group_artifact = artifact_path(config.data.group.order, config.data.group.index)
    record["status"] = "measured"
    record["provenance"]["checkpoint_sha256"] = file_sha256(selection.path)
    record["provenance"]["group_artifact_path"] = _relative_to_repo(group_artifact)
    record["provenance"]["group_artifact_sha256"] = file_sha256(group_artifact)
    record["n_test"] = int(tokens.shape[0])
    record["coset_quotient_route"] = coset_quotient_route(
        model, group, tokens, targets, n_random=n_random, seed=config.seed
    )
    # Bound this process's memory to one seed at a time.
    del checkpoint, model, tokens, targets
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
    """One cell's aggregate over its measured seeds: per quotient (keyed by
    ``subgroup_index``), the sufficiency/necessity effect sizes over seeds,
    plus a count of which quotient was decisive per seed. Built only from
    ``status: "measured"`` records whose group actually exported a quotient
    (``defined: True`` -- every measured record in a screened-in cell, but
    guarded rather than assumed)."""
    measured = [r for r in records if r.get("status") == "measured"]
    n_skipped = len(records) - len(measured)

    by_subgroup: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {
            "coset_accuracy_over_random": [],
            "coset_accuracy_drop_over_random": [],
            "restricted_coset_accuracy": [],
            "ablated_coset_accuracy": [],
        }
    )
    static_by_subgroup: dict[int, dict[str, Any]] = {}
    decisive_counts: Counter[int] = Counter()
    n_undefined = 0

    for r in measured:
        route = r["coset_quotient_route"]
        if not route.get("defined", False):
            n_undefined += 1
            continue
        decisive_counts[route["decisive_subgroup_index"]] += 1
        for probe in route["quotients"]:
            sub_idx = probe["subgroup_index"]
            static_by_subgroup.setdefault(
                sub_idx,
                {
                    "subgroup_index": sub_idx,
                    "subgroup_order": probe["subgroup_order"],
                    "coset_index": probe["coset_index"],
                    "quotient_rank": probe["quotient_rank"],
                    "chance_coset_accuracy": probe["chance_coset_accuracy"],
                },
            )
            bucket = by_subgroup[sub_idx]
            bucket["coset_accuracy_over_random"].append(
                float(probe["sufficiency"]["coset_accuracy_over_random"])
            )
            bucket["coset_accuracy_drop_over_random"].append(
                float(probe["necessity"]["coset_accuracy_drop_over_random"])
            )
            bucket["restricted_coset_accuracy"].append(
                float(probe["sufficiency"]["restricted_coset_accuracy"])
            )
            bucket["ablated_coset_accuracy"].append(
                float(probe["necessity"]["ablated_coset_accuracy"])
            )

    quotients_out = []
    for sub_idx in sorted(by_subgroup):
        bucket = by_subgroup[sub_idx]
        quotients_out.append(
            {
                **static_by_subgroup[sub_idx],
                "coset_accuracy_over_random": _summarise(bucket["coset_accuracy_over_random"]),
                "coset_accuracy_drop_over_random": _summarise(
                    bucket["coset_accuracy_drop_over_random"]
                ),
                "restricted_coset_accuracy": _summarise(bucket["restricted_coset_accuracy"]),
                "ablated_coset_accuracy": _summarise(bucket["ablated_coset_accuracy"]),
                "n_decisive": decisive_counts.get(sub_idx, 0),
            }
        )

    return {
        "order": order,
        "index": index,
        "width": width,
        "n_seeds_attempted": len(records),
        "n_measured": len(measured),
        "n_skipped": n_skipped,
        "n_undefined": n_undefined,
        "quotients": quotients_out,
        "discrimination_note": DISCRIMINATION_NOTE,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _cmd_measure(args: argparse.Namespace) -> int:
    if args.artifacts_dir is not None:
        os.environ["GROUP_ARTIFACTS_DIR"] = str(args.artifacts_dir)
    # coset_quotient_route has no device-moving forward pass to route to (it
    # always runs on wherever the loaded model's parameters already live,
    # CPU here) -- resolve_device is still called so an explicit "--device
    # mps" on a machine without MPS fails loudly, the same as every other
    # driver, rather than silently doing nothing.
    resolve_device(args.device)

    cells = discover_cells(Path(args.runs_dir))
    if args.cell:
        wanted = {tuple(int(x) for x in spec.split(",")) for spec in args.cell}
        cells = {key: value for key, value in cells.items() if key in wanted}

    n_skipped_runs = 0
    n_written = 0
    for order, index, width in sorted(cells):
        screen = screen_cell(order, index)
        if screen["status"] != "screened" or not screen["usable"]:
            reason = screen.get("reason") or "no proper nontrivial normal subgroup exported"
            print(
                f"[coset_quotient] ({order},{index}) w{width}: "
                f"screen={screen['status']} -- not attempting seeds ({reason})"
            )
            continue

        run_dirs = cells[(order, index, width)][: args.max_seeds_per_cell]
        records = [
            coset_quotient_run(
                run_dir,
                metric=args.metric,
                threshold=args.threshold,
                n_random=args.n_random,
            )
            for run_dir in run_dirs
        ]
        n_skipped_runs += sum(1 for r in records if r["status"] == "skipped")
        summary = summarise_cell(order, index, width, records)
        payload = {
            "cell": {"order": order, "index": index, "width": width},
            "n_seeds_available": len(cells[(order, index, width)]),
            "max_seeds_per_cell": args.max_seeds_per_cell,
            "screen": screen,
            "summary": summary,
            "runs": records,
        }
        out_path = Path(args.out_dir) / f"coset_quotient_{order}_{index}_w{width}.json"
        _write_json(out_path, payload)
        n_written += 1
        decisive = summary["quotients"][0] if summary["quotients"] else None
        headline = (
            f"decisive subgroup {decisive['subgroup_index']}: "
            f"sufficiency={decisive['coset_accuracy_over_random']['mean']:.4f}"
            if decisive is not None
            else "no measured seed"
        )
        print(
            f"[coset_quotient] ({order},{index}) w{width}: usable, "
            f"n_measured={summary['n_measured']}/{summary['n_seeds_attempted']} {headline} "
            f"-> {out_path}"
        )
    print(f"[coset_quotient] wrote {n_written} usable cell record(s) to {args.out_dir}")
    return 0 if n_skipped_runs == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser(
        "measure", help="run I-20 (coset-quotient route) across panel cells with a usable quotient"
    )
    measure.add_argument(
        "--runs-dir", default="runs", help="root directory of trained run directories"
    )
    measure.add_argument(
        "--out-dir", default="results/coset_quotient", help="directory to write one JSON per cell"
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
        "--n-random", type=int, default=16, help="matched random subspaces per quotient probe"
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
        help="validated for parity with the other drivers, but coset_quotient_route "
        "has no device-moving forward pass: the loaded model always stays on CPU, "
        "so this only rejects an unavailable '--device mps' request (default: cpu)",
    )
    measure.set_defaults(func=_cmd_measure)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
