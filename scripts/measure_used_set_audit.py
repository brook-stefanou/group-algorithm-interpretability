"""Used-set causal-sufficiency-audit entry point (``instruments/used_set_audit.py``'s
``used_set_audit`` driver, the rung-5 completion of the near-minimal-faithful-
subset thesis).

Runs the used-set audit across the trained campaign's panel cells, entirely
offline -- no network, no W&B, no Sage/GAP::

    WANDB_MODE=disabled uv run python scripts/measure_used_set_audit.py measure \\
        --runs-dir runs --out-dir results/used_set_audit \\
        [--metric val/accuracy] [--threshold 0.99] [--max-seeds-per-cell 50] \\
        [--n-random 16] [--artifacts-dir data/group_artifacts] \\
        [--coset-dir results/coset] [--cell 32,18,128] [--device cpu]

**Dependency on I-15 (isotypic-block ablation).** The "used" block set this
instrument audits is not re-derived here: it is
:func:`instruments.template_divergence.causally_used_blocks` applied to the
MEASURED I-15 per-cell aggregate
(``scripts/measure_isotypic_ablation.py``'s
``<coset-dir>/iso_ablation_<order>_<index>_w<width>.json``), read once per
cell and reused for every seed's audit. When that file is not yet on disk --
I-15 is a separate, possibly-not-yet-run instrument -- the cell's output
record is ``{"status": "pending", "missing": [...]}`` (the same convention
``measure_template_divergence.py``'s data-driven refinement uses for its own
upstream dependencies) rather than a failed run: no checkpoint is touched for
a pending cell.

A "cell" is a ``(order, index, width)`` triple, discovered the same way
``scripts/measure_gcr_readout.py`` discovers them. Each measured run applies
the same dip-aware checkpoint rule (``instruments/checkpoints.py``) every
other instrument in this project uses (recorded, substitutions included); a
run with no stable checkpoint is recorded ``status: "skipped"``. Scored on
the transpose-unleaked held-out set (``instruments.coset.unleaked_heldout``).

One JSON is written per cell to
``<out-dir>/used_set_audit_<order>_<index>_w<width>.json``: either the
pending record above, or every seed's full audit record plus per-cell
aggregates (mean and 95% bootstrap CI where >= 2 seeds measured) of the
completeness accuracy-retention fraction per ablation mode, the necessity/
minimality necessary-block count, the faithfulness KL and argmax agreement,
and a count of which descriptive verdict each measured seed landed on.

Exit code: ``0`` iff every attempted run in a non-pending cell was measured;
``1`` when any such run was skipped for want of a stable checkpoint. A
pending cell never counts against this -- it was never attempted.
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
from group_algorithm_interp.instruments.coset import ABLATION_MODES, unleaked_heldout  # noqa: E402
from group_algorithm_interp.instruments.device import resolve_device  # noqa: E402
from group_algorithm_interp.instruments.report import (  # noqa: E402
    file_sha256,
    instrument_code_hashes,
)
from group_algorithm_interp.instruments.template_divergence import (  # noqa: E402
    MIN_FLIP_OVER_RANDOM,
    causally_used_blocks,
)
from group_algorithm_interp.instruments.used_set_audit import used_set_audit  # noqa: E402
from group_algorithm_interp.manifest import get_git_commit, read_manifest  # noqa: E402
from group_algorithm_interp.training.trainer import build_model  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

Cell = tuple[int, int, int]  # (order, index, width)

#: The I-15 per-cell aggregated record's default directory --
#: ``scripts/measure_isotypic_ablation.py``'s own output directory
#: (``results/coset/iso_ablation_<order>_<index>_w<width>.json``). Named and
#: overridable (``--coset-dir``) rather than hard-coded, matching
#: ``measure_template_divergence.py``'s ``DEFAULT_COSET_DIR`` convention.
DEFAULT_COSET_DIR = Path("results/coset")


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
# I-15 dependency: the cell-level used-block set.
# ---------------------------------------------------------------------------


def _iso_ablation_path(order: int, index: int, width: int, coset_dir: Path) -> Path:
    """The I-15 per-cell aggregated record's path
    (``scripts/measure_isotypic_ablation.py``'s own naming convention)."""
    return Path(coset_dir) / f"iso_ablation_{order}_{index}_w{width}.json"


def cell_used_blocks(
    order: int,
    index: int,
    width: int,
    *,
    coset_dir: Path = DEFAULT_COSET_DIR,
    min_flip_over_random: float = MIN_FLIP_OVER_RANDOM,
) -> dict[str, Any]:
    """The cell-level used-block set this driver audits, derived from I-15.

    Returns ``{"status": "pending", "missing": [...]}`` -- never raising --
    when I-15's per-cell record is not yet on disk under ``coset_dir``, since
    I-15 is a separate instrument that may not have been run yet. On success,
    returns ``{"status": "measured", "iso_ablation_path": ...,
    "used_blocks": [...]}``: the MEASURED causally-used block set
    (:func:`instruments.template_divergence.causally_used_blocks` applied to
    the I-15 aggregate), computed once per cell and reused for every seed's
    audit."""
    iso_path = _iso_ablation_path(order, index, width, coset_dir)
    if not iso_path.is_file():
        return {"status": "pending", "missing": [_relative_to_repo(iso_path)]}
    iso_record = json.loads(iso_path.read_text())
    used_blocks = causally_used_blocks(iso_record, min_flip_over_random=min_flip_over_random)
    return {
        "status": "measured",
        "iso_ablation_path": _relative_to_repo(iso_path),
        "used_blocks": list(used_blocks),
    }


# ---------------------------------------------------------------------------
# Per-run measurement.
# ---------------------------------------------------------------------------


def used_set_audit_run(
    run_dir: Path,
    used_blocks: list[int],
    *,
    metric: str = "val/accuracy",
    threshold: float = 0.99,
    n_random: int = 16,
    near_full_rank_fraction: float,
    min_flip_over_random: float,
    sufficiency_retention_fraction: float,
    minimality_mode: str,
) -> dict[str, Any]:
    """One run's used-set-audit record: dip-aware checkpoint selection
    (recorded, substitutions included), then :func:`used_set_audit` against
    the cell-level ``used_blocks`` (:func:`cell_used_blocks`, never re-derived
    per seed). A run with no stable checkpoint comes back ``status:
    "skipped"`` with the full selection record -- reported as data, never
    silently dropped."""
    manifest = read_manifest(run_dir)
    config = validate_config(yaml.safe_load((run_dir / "resolved_config.yaml").read_text()))
    selection = select_checkpoint(run_dir, metric=metric, threshold=threshold)
    record: dict[str, Any] = {
        "instrument": "used-set-audit-driver",
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
        "used_blocks": list(used_blocks),
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
    record["used_set_audit"] = used_set_audit(
        model,
        group,
        tokens,
        targets,
        used_blocks,
        n_random=n_random,
        seed=config.seed,
        near_full_rank_fraction=near_full_rank_fraction,
        min_flip_over_random=min_flip_over_random,
        sufficiency_retention_fraction=sufficiency_retention_fraction,
        minimality_mode=minimality_mode,
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
    """One cell's aggregate over its measured seeds: the completeness
    accuracy-retention fraction per ablation mode, the necessary-block count,
    the faithfulness KL/argmax-agreement, the used/complement rank fractions,
    and a count of which descriptive verdict each measured seed landed on.
    Built only from ``status: "measured"`` records."""
    measured = [r for r in records if r.get("status") == "measured"]
    n_skipped = len(records) - len(measured)

    retention_by_mode: dict[str, list[float]] = {mode: [] for mode in ABLATION_MODES}
    n_necessary: list[float] = []
    n_used: list[float] = []
    used_rank_fraction: list[float] = []
    kl_means: list[float] = []
    agreement_means: list[float] = []
    verdict_counts: Counter[str] = Counter()
    degenerate_count = 0

    for r in measured:
        usa = r["used_set_audit"]
        verdict_counts[usa["verdict"]] += 1
        if usa["degenerate"]:
            degenerate_count += 1
        used_rank_fraction.append(float(usa["used_rank_fraction"]))
        n_necessary.append(float(usa["n_necessary_blocks"]))
        n_used.append(float(usa["n_used_blocks"]))
        kl_means.append(float(usa["faithfulness"]["kl_model_to_restricted_circuit"]["mean"]))
        agreement_means.append(float(usa["faithfulness"]["argmax_agreement"]["mean"]))
        for mode in ABLATION_MODES:
            value = usa["completeness"]["accuracy_retention_fraction"].get(mode)
            if value is not None and value == value:  # exclude NaN (clean_accuracy == 0)
                retention_by_mode[mode].append(float(value))

    return {
        "order": order,
        "index": index,
        "width": width,
        "n_seeds_attempted": len(records),
        "n_measured": len(measured),
        "n_skipped": n_skipped,
        "used_blocks": records[0]["used_blocks"] if records else [],
        "n_used_blocks": _summarise(n_used),
        "used_rank_fraction": _summarise(used_rank_fraction),
        "degenerate_seed_count": degenerate_count,
        "completeness_accuracy_retention_fraction": {
            mode: _summarise(values) for mode, values in retention_by_mode.items()
        },
        "n_necessary_blocks": _summarise(n_necessary),
        "faithfulness_kl_mean": _summarise(kl_means),
        "faithfulness_argmax_agreement_mean": _summarise(agreement_means),
        "verdict_counts": dict(verdict_counts),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _cmd_measure(args: argparse.Namespace) -> int:
    if args.artifacts_dir is not None:
        os.environ["GROUP_ARTIFACTS_DIR"] = str(args.artifacts_dir)
    # used_set_audit has no device-moving forward pass to route to (it always
    # runs on wherever the loaded model's parameters already live, CPU here)
    # -- resolve_device is still called so an explicit "--device mps" on a
    # machine without MPS fails loudly, the same as every other driver.
    resolve_device(args.device)

    cells = discover_cells(Path(args.runs_dir))
    if args.cell:
        wanted = {tuple(int(x) for x in spec.split(",")) for spec in args.cell}
        cells = {key: value for key, value in cells.items() if key in wanted}

    n_skipped_runs = 0
    n_written = 0
    coset_dir = Path(args.coset_dir)
    for order, index, width in sorted(cells):
        dependency = cell_used_blocks(
            order, index, width, coset_dir=coset_dir, min_flip_over_random=args.min_flip_over_random
        )
        if dependency["status"] == "pending":
            payload = {
                "cell": {"order": order, "index": index, "width": width},
                "status": "pending",
                "missing": dependency["missing"],
            }
            out_path = Path(args.out_dir) / f"used_set_audit_{order}_{index}_w{width}.json"
            _write_json(out_path, payload)
            print(
                f"[used_set_audit] ({order},{index}) w{width}: PENDING -- "
                f"missing {dependency['missing']} -- not attempting seeds"
            )
            continue

        used_blocks = dependency["used_blocks"]
        run_dirs = cells[(order, index, width)][: args.max_seeds_per_cell]
        records = [
            used_set_audit_run(
                run_dir,
                used_blocks,
                metric=args.metric,
                threshold=args.threshold,
                n_random=args.n_random,
                near_full_rank_fraction=args.near_full_rank_fraction,
                min_flip_over_random=args.min_flip_over_random,
                sufficiency_retention_fraction=args.sufficiency_retention_fraction,
                minimality_mode=args.minimality_mode,
            )
            for run_dir in run_dirs
        ]
        n_skipped_runs += sum(1 for r in records if r["status"] == "skipped")
        summary = summarise_cell(order, index, width, records)
        payload = {
            "cell": {"order": order, "index": index, "width": width},
            "status": "measured",
            "n_seeds_available": len(cells[(order, index, width)]),
            "max_seeds_per_cell": args.max_seeds_per_cell,
            "dependency": dependency,
            "summary": summary,
            "runs": records,
        }
        out_path = Path(args.out_dir) / f"used_set_audit_{order}_{index}_w{width}.json"
        _write_json(out_path, payload)
        n_written += 1
        print(
            f"[used_set_audit] ({order},{index}) w{width}: used_blocks={used_blocks} "
            f"n_measured={summary['n_measured']}/{summary['n_seeds_attempted']} "
            f"verdicts={summary['verdict_counts']} -> {out_path}"
        )
    print(f"[used_set_audit] wrote {n_written} non-pending cell record(s) to {args.out_dir}")
    return 0 if n_skipped_runs == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser(
        "measure", help="run the used-set causal-sufficiency audit across panel cells"
    )
    measure.add_argument(
        "--runs-dir", default="runs", help="root directory of trained run directories"
    )
    measure.add_argument(
        "--out-dir", default="results/used_set_audit", help="directory to write one JSON per cell"
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
        "--n-random", type=int, default=16, help="matched random subspaces per ablation"
    )
    measure.add_argument(
        "--near-full-rank-fraction",
        type=float,
        default=0.9,
        dest="near_full_rank_fraction",
        help="degeneracy-guard threshold on the used set's rank fraction (default 0.9)",
    )
    measure.add_argument(
        "--min-flip-over-random",
        type=float,
        default=MIN_FLIP_OVER_RANDOM,
        dest="min_flip_over_random",
        help=(
            "used both to derive the used-block set from I-15 "
            "(causally_used_blocks) and as the minimality-verdict bar "
            f"(default {MIN_FLIP_OVER_RANDOM})"
        ),
    )
    measure.add_argument(
        "--sufficiency-retention-fraction",
        type=float,
        default=0.9,
        dest="sufficiency_retention_fraction",
        help="completeness-verdict accuracy-retention bar (default 0.9)",
    )
    measure.add_argument(
        "--minimality-mode",
        default="zero",
        dest="minimality_mode",
        choices=list(ABLATION_MODES),
        help="ablation mode the verdict's necessity count and retention fraction read (default zero)",
    )
    measure.add_argument(
        "--artifacts-dir", default=None, help="override the group-artifact directory"
    )
    measure.add_argument(
        "--coset-dir",
        default=str(DEFAULT_COSET_DIR),
        dest="coset_dir",
        help=f"directory to read I-15's per-cell records from (default {DEFAULT_COSET_DIR})",
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
        help="validated for parity with the other drivers, but used_set_audit has no "
        "device-moving forward pass: the loaded model always stays on CPU, so this "
        "only rejects an unavailable '--device mps' request (default: cpu)",
    )
    measure.set_defaults(func=_cmd_measure)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
