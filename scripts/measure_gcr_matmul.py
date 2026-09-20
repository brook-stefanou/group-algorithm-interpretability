"""GCR matrix-product driver (``instruments/gcr_matmul.py``) entry point.

Fits the matrix-product-vs-generic-bilinear-vs-function-of-``ab`` comparison
against trained checkpoints, entirely offline from run-directory artifacts; no
network, no W&B, no Sage/GAP::

    # Measure one or more runs and aggregate seeds within each panel cell
    # (one group at one model width).
    uv run python scripts/measure_gcr_matmul.py measure runs/<id> [runs/<id> ...] \
        [--out-dir results/gcr_matmul] [--metric val/accuracy] [--threshold 0.99] \
        [--n-splits 5] [--fit-seed 0] [--ridge 0.0] [--min-coverage 0.9] \
        [--min-abs-fve 0.5] [--tie-tol 0.01] [--artifacts-dir data/group_artifacts]

``measure`` applies the identical dip-aware checkpoint rule
``scripts/measure_occupancy.py`` uses (``instruments/checkpoints.py``) to each
run, then runs
:func:`group_algorithm_interp.instruments.gcr_matmul.measure_gcr_matmul`
against the selected checkpoint. This instrument is only decisive for groups
with an irrep of degree ``>= 3`` (degree exactly 2 is flagged ``low_power`` by
the instrument itself; degree-1-only groups return ``UNDEFINED``) -- the
caller chooses which run directories to pass; this script does not filter by
group.

Every run's full record -- including a ``status: "skipped"`` record naming why
when no stable checkpoint exists -- is written into its cell's output file.
Runs are pooled into cells by ``(group order, group index, model d_model)``,
which is the panel's actual unit of replication (a group measured at more than
one width is two cells, e.g. the C1 width-rescue campaign's ``w512`` reruns);
one JSON is written per cell to
``<out-dir>/gcr_matmul_<order>_<index>_w<d_model>.json``, holding every
measured seed's full ``to_record()`` output plus, per occupied degree-``>= 2``
irrep, aggregates (mean/std/bootstrap CI over seeds where >= 2 seeds measured)
of the numeric fields.

Exit code: ``0`` iff every requested run was measured, ``1`` when a run was
skipped for lack of a stable checkpoint -- mirroring ``measure_occupancy.py``
so a campaign wrapper notices missing measurements loudly.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from group_algorithm_interp import stats  # noqa: E402
from group_algorithm_interp.config import validate_config  # noqa: E402
from group_algorithm_interp.groups.catalog import resolve_group  # noqa: E402
from group_algorithm_interp.groups.data import artifact_path  # noqa: E402
from group_algorithm_interp.instruments.checkpoints import select_checkpoint  # noqa: E402
from group_algorithm_interp.instruments.device import resolve_device  # noqa: E402
from group_algorithm_interp.instruments.gcr_matmul import measure_gcr_matmul  # noqa: E402
from group_algorithm_interp.instruments.report import (  # noqa: E402
    file_sha256,
    instrument_code_hashes,
    sanitise_nonfinite,
)
from group_algorithm_interp.manifest import (  # noqa: E402
    get_git_commit,
    get_git_dirty,
    read_manifest,
)
from group_algorithm_interp.training.trainer import build_model  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

# The IrrepMatmulFit.to_record() fields this driver aggregates numerically
# over seeds (mean/std/bootstrap CI); the boolean verdict fields
# (mp_favoured, credits_gcr) are aggregated separately as a fraction, and
# low_power/irrep_degree/irrep_index are constant across seeds for a fixed
# group so are simply carried through from the first fit.
_NUMERIC_FIT_FIELDS = (
    "occupancy",
    "mp_fve_heldout",
    "bilinear_fve_heldout",
    "ab_fve_heldout",
    "mp_fraction_of_ab",
    "mp_minus_bilinear_heldout",
    "bic_mp",
    "bic_bilinear",
)


def _relative_to_repo(path: Path) -> str:
    """``path`` relative to the repo root when possible, else its absolute
    form -- matching ``instruments.report``'s provenance convention."""
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _group_artifact_provenance(order: int, index: int) -> dict[str, Any]:
    path = artifact_path(order, index)
    return {
        "group_artifact_path": _relative_to_repo(path),
        "group_artifact_sha256": file_sha256(path) if path.is_file() else None,
    }


def _analysis_provenance() -> dict[str, Any]:
    return {
        "analysis_git_commit": get_git_commit(),
        "analysis_git_dirty": get_git_dirty(),
        "instrument_code_sha256": instrument_code_hashes(),
    }


def _local_final_pt_fallback(run_dir: Path, selection: Any) -> tuple[Path | None, dict[str, Any]]:
    """A last-resort local substitution for a curated run whose recorded
    stable-checkpoint filename is absent from this machine's restored corpus.

    This corpus's restore keeps only a fixed per-run file set (``final.pt``,
    ``manifest.yaml``, ``resolved_config.yaml``, ``run.log.gz``,
    ``selection.json``), so a ``stable_end`` pick named e.g.
    ``final_epoch_59999.pt`` -- genuinely selected at ship time, per
    ``selection.json`` -- is provably absent even when it was the correct
    pick. When the named checkpoint's epoch is identical to the run's
    ``last`` category epoch, both names refer to the run's true final-epoch
    weights (``final.pt`` is written unconditionally at the end of training;
    see ``internal/docs/handoff.md``'s final-window note), so ``final.pt`` on
    disk is byte-identical to the missing file and substituting it invents no
    checkpoint the dip-aware rule did not already select. A genuine
    dip-window substitution to an *earlier*, still-missing snapshot (a
    mismatched epoch) is never covered by this fallback and stays skipped,
    honestly, since no locally-available file is provably equivalent.
    """
    if selection.path is not None:
        return None, {}
    selection_json = run_dir / "selection.json"
    if not selection_json.is_file():
        return None, {}
    try:
        data = json.loads(selection_json.read_text())
    except (OSError, ValueError):
        return None, {}
    categories = data.get("categories") if isinstance(data, dict) else None
    stable_end = categories.get("stable_end") if isinstance(categories, dict) else None
    last = categories.get("last") if isinstance(categories, dict) else None
    if not isinstance(stable_end, dict) or not isinstance(last, dict):
        return None, {}
    stable_epoch, last_epoch = stable_end.get("epoch"), last.get("epoch")
    if stable_epoch is None or stable_epoch != last_epoch:
        return None, {}
    final_path = run_dir / "final.pt"
    if not final_path.is_file():
        return None, {}
    fallback_record = {
        "local_checkpoint_fallback": "final.pt",
        "local_checkpoint_fallback_reason": (
            f"selection.json's stable_end names {stable_end.get('filename')!r} "
            "(absent from this machine's restored corpus, which keeps only a "
            f"fixed per-run file set), but its epoch ({stable_epoch}) equals "
            "the 'last' category's epoch, so final.pt on disk holds "
            "byte-identical weights"
        ),
    }
    return final_path, fallback_record


def measure_gcr_matmul_run(
    run_dir: Path,
    *,
    metric: str = "val/accuracy",
    threshold: float = 0.99,
    device: torch.device = torch.device("cpu"),
    **screen_kwargs: Any,
) -> dict[str, Any]:
    """One run's GCR matrix-product record.

    Mirrors ``instruments.report.measure_run``'s shape and dip-aware
    checkpoint rule for the ``gcr_matmul`` instrument in place of
    ``occupancy``: a run whose selection finds no stable checkpoint is
    returned with ``status: "skipped"`` and the full selection record,
    reported as data rather than silently dropped. A run whose selection
    names a checkpoint absent from the restored corpus but provably
    byte-identical to ``final.pt`` (:func:`_local_final_pt_fallback`) is
    measured against ``final.pt``, with the substitution recorded plainly in
    ``checkpoint_selection`` rather than silently applied.
    """
    manifest = read_manifest(run_dir)
    config = validate_config(yaml.safe_load((run_dir / "resolved_config.yaml").read_text()))
    selection = select_checkpoint(run_dir, metric=metric, threshold=threshold)
    record: dict[str, Any] = {
        "instrument": "gcr_matmul",
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
            **_analysis_provenance(),
            **_group_artifact_provenance(config.data.group.order, config.data.group.index),
        },
    }
    checkpoint_path = selection.path
    if checkpoint_path is None:
        checkpoint_path, fallback_record = _local_final_pt_fallback(run_dir, selection)
        if checkpoint_path is None:
            record["status"] = "skipped"
            return record
        record["checkpoint_selection"] = {**record["checkpoint_selection"], **fallback_record}

    group = resolve_group(config.data.group)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(config, group)
    model.load_state_dict(checkpoint["model_state_dict"])
    result = measure_gcr_matmul(model, group, device=device, **screen_kwargs)

    record["status"] = "measured"
    record["provenance"]["checkpoint_sha256"] = file_sha256(checkpoint_path)
    record["gcr_matmul"] = result.to_record()
    # Bound this process's memory to one seed at a time: the checkpoint state
    # dict, the model built from it, and the per-irrep fit result (whose
    # activation grids and design matrices are the largest transient arrays
    # this driver touches) are never needed again once ``record`` is built,
    # so drop them explicitly rather than relying on them falling out of
    # scope only when the caller's next loop iteration reassigns locals.
    del checkpoint, model, result
    gc.collect()
    return record


def _cell_key(record: dict[str, Any]) -> tuple[int, int, int]:
    """The panel-cell identity a record pools into: group + model width. A
    group re-run at a second width (the C1 width-rescue campaign) is a
    distinct cell, never merged with the original width."""
    return (record["group"]["order"], record["group"]["index"], record["model"]["d_model"])


def _bootstrap_field(values: list[float]) -> dict[str, Any]:
    """Mean/std/bootstrap-CI summary of one numeric field over a cell's
    measured seeds. ``bootstrap_ci_95`` is ``None`` for a single-seed cell
    (``stats.bootstrap_ci`` is undefined below 2 values), reported as such
    rather than a fabricated point interval."""
    mean, std = stats.mean_std(values)
    entry: dict[str, Any] = {"mean": mean, "std": std, "n": len(values), "values": values}
    entry["bootstrap_ci_95"] = list(stats.bootstrap_ci(values)) if len(values) >= 2 else None
    return entry


def _aggregate_cell(members: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-irrep (``block_index``) aggregates over one cell's measured seeds:
    mean/std/bootstrap CI of each numeric field, and the fraction of seeds for
    which each boolean verdict field holds. Irreps not occupied at all in a
    given seed's fit list simply contribute no observation to that irrep's
    aggregate (``fit_matmul_gcr`` is only run for occupied blocks)."""
    measured = [m for m in members if m["status"] == "measured"]
    by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for member in measured:
        for fit in member["gcr_matmul"]["fits"]:
            by_block[fit["block_index"]].append(fit)

    irreps = []
    for block_index in sorted(by_block):
        fits = by_block[block_index]
        agg: dict[str, Any] = {
            "block_index": block_index,
            "irrep_index": fits[0]["irrep_index"],
            "irrep_degree": fits[0]["irrep_degree"],
            "low_power": fits[0]["low_power"],
            "n_seeds": len(fits),
        }
        for field_name in _NUMERIC_FIT_FIELDS:
            agg[field_name] = _bootstrap_field([f[field_name] for f in fits])
        agg["mp_favoured_fraction"] = sum(1 for f in fits if f["mp_favoured"]) / len(fits)
        agg["credits_gcr_fraction"] = sum(1 for f in fits if f["credits_gcr"]) / len(fits)
        irreps.append(agg)

    return {
        "n_seeds_measured": len(measured),
        "n_seeds_skipped": len(members) - len(measured),
        "seeds_measured": sorted(m["seed"] for m in measured),
        "irreps": irreps,
    }


def _write_json(path: Path, payload: object) -> None:
    """Write ``payload`` as strict RFC-8259 JSON, non-finite floats sanitised
    to ``null`` first -- matches ``measure_occupancy.py``'s convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(sanitise_nonfinite(payload), indent=2, sort_keys=False, allow_nan=False)
    path.write_text(text + "\n")


def _cmd_measure(args: argparse.Namespace) -> int:
    if args.artifacts_dir is not None:
        os.environ["GROUP_ARTIFACTS_DIR"] = str(args.artifacts_dir)
    device = resolve_device(args.device)
    screen_kwargs = dict(
        n_splits=args.n_splits,
        seed=args.fit_seed,
        ridge=args.ridge,
        min_coverage=args.min_coverage,
        min_abs_fve=args.min_abs_fve,
        tie_tol=args.tie_tol,
        fit_device=resolve_device(args.fit_device),
        max_rows=args.max_rows,
        sample_seed=args.sample_seed,
    )
    cells: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    n_skipped = 0
    for run_dir in args.run_dirs:
        record = measure_gcr_matmul_run(
            Path(run_dir),
            metric=args.metric,
            threshold=args.threshold,
            device=device,
            **screen_kwargs,
        )
        cells[_cell_key(record)].append(record)
        if record["status"] == "measured":
            fits = record["gcr_matmul"]["fits"]
            summary = ", ".join(
                f"block{fit['block_index']}(d={fit['irrep_degree']}): "
                f"mp={fit['mp_fve_heldout']:.3f} credits_gcr={fit['credits_gcr']}"
                for fit in fits
            )
            print(f"[gcr_matmul] {record['run_id']}: {record['group']['name']} {summary}")
        else:
            n_skipped += 1
            print(
                f"[gcr_matmul] {record['run_id']}: SKIPPED -- "
                f"{record['checkpoint_selection']['reason']}"
            )

    out_dir = Path(args.out_dir)
    for (order, index, d_model), members in sorted(cells.items()):
        payload = {
            "instrument": "gcr_matmul",
            "group": {"order": order, "index": index, "name": members[0]["group"]["name"]},
            "model": members[0]["model"],
            "runs": members,
            "aggregate": _aggregate_cell(members),
        }
        out_path = out_dir / f"gcr_matmul_{order}_{index}_w{d_model}.json"
        _write_json(out_path, payload)
        print(f"[gcr_matmul] wrote {len(members)} run record(s) to {out_path}")
    return 0 if n_skipped == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser("measure", help="measure GCR matrix-product fits for one or more runs")
    measure.add_argument("run_dirs", nargs="+", help="run directories to measure")
    measure.add_argument(
        "--out-dir", default="results/gcr_matmul", help="directory to write per-cell JSON files"
    )
    measure.add_argument("--metric", default="val/accuracy", help="stability metric in run.log")
    measure.add_argument(
        "--threshold", type=float, default=0.99, help="stability bar on --metric (default 0.99)"
    )
    measure.add_argument("--n-splits", type=int, default=5, help="cross-validation folds")
    measure.add_argument("--fit-seed", type=int, default=0, help="fold-assignment RNG seed")
    measure.add_argument("--ridge", type=float, default=0.0, help="ridge penalty for the CV fit")
    measure.add_argument("--min-coverage", type=float, default=0.9)
    measure.add_argument("--min-abs-fve", type=float, default=0.5)
    measure.add_argument("--tie-tol", type=float, default=0.01)
    measure.add_argument(
        "--max-rows",
        type=int,
        default=0,
        dest="max_rows",
        help="cap on the number of (a, b) grid cells the fit sees; 0 or unset means no cap "
        "(the full order**2 grid, unchanged). Over the cap a fixed-seed uniform subsample of "
        "cells is drawn before the designs are built, so the big-group cells no longer stall in "
        "the CPU-bound design build.",
    )
    measure.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        dest="sample_seed",
        help="seed for the --max-rows row subsample (default 0); distinct from --fit-seed, "
        "which seeds the cross-validation fold assignment.",
    )
    measure.add_argument(
        "--artifacts-dir", default=None, help="override the group-artifact directory"
    )
    measure.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="device for the activation-extraction forward pass; 'auto' is MPS when "
        "available, else CPU (default: auto); 'cuda' is explicit-only.",
    )
    measure.add_argument(
        "--fit-device",
        default="cpu",
        choices=["auto", "mps", "cuda", "cpu"],
        help="device for the held-out-FVE matrix-product/bilinear fit; 'cpu' (default) is "
        "the numpy-float64 reference, 'mps'/'cuda' run the float32 normal-equations backend "
        "on the GPU for the big-group cells (see instruments/fit_backend.py). Opt-in: "
        "the default keeps the pre-registered CPU-float64 fit.",
    )
    measure.set_defaults(func=_cmd_measure)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
