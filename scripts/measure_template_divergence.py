"""Template-divergence instrument entry point (GCR-vs-coset, screened).

Drives ``instruments/template_divergence.py`` (already built and tested)
across trained-run cells, entirely offline -- no network, no W&B, no
Sage/GAP::

    WANDB_MODE=disabled uv run python scripts/measure_template_divergence.py measure \\
        --runs-dir runs --out-dir results/template_divergence \\
        --cell 216,106,128 --cell 81,7,128 \\
        [--metric val/accuracy] [--threshold 0.99] [--max-seeds-per-cell 50] \\
        [--artifacts-dir data/group_artifacts] [--max-support-fraction 0.5] \\
        [--datadriven/--no-datadriven] [--coset-dir results/coset] \\
        [--gcr-matmul-dir results/gcr_matmul]

**Data-driven refinement (on by default).** Alongside the template-only
comparison above, each measured run also gets a ``data_driven`` block nested
in ``occupancy[argument]`` (see
``instruments/template_divergence.py``'s ``causally_used_blocks``/
``product_carrying_blocks``/``data_driven_comparison`` for the underlying
functions): the MEASURED causally-used block set (I-15,
``results/coset/iso_ablation_<order>_<index>_w<width>.json``) and the
MEASURED product-carrying block set (the GCR matrix-product fit,
``results/gcr_matmul/gcr_matmul_<order>_<index>_w<width>.json``) compared
against each account's predicted set. Both upstream instruments are
per-cell, not per-argument or per-seed, so the same used/predicted sets are
compared against each argument's own occupancy. Either result file may not
exist yet (I-15/the matmul fit are separate, possibly-not-yet-run
instruments); when either is missing the block is recorded
``{"status": "pending", "missing": [...]}`` rather than failing the run.
Disable with ``--no-datadriven``.

A "cell" is a ``(order, index, width)`` triple, discovered the same way
``scripts/measure_gcr_readout.py`` discovers them (completed manifests under
``--runs-dir``, grouped by ``resolved_config.yaml``'s group and
``model.d_model``). ``--cell`` restricts to specific cells; the default is
every discovered cell, which is only cheap because the screen below is
checked once per cell *before* any checkpoint is touched.

**The screen comes first.** Comparing a model's occupancy to the coset
account's ``Ind_H^G 1`` template is only a meaningful GCR-vs-coset
discrimination when :func:`instruments.template_divergence.degeneracy_screen`
returns ``DEFINED`` for that group (see that module's docstring for why: a
degenerate coset target forces "closest to coset" regardless of which
account the model implements). This driver runs the screen once per cell,
from the group artifact alone -- no checkpoint load needed -- and only
proceeds to load checkpoints and compute occupancy for cells that screen
``DEFINED``. A cell whose screen is not ``DEFINED`` is recorded with its
screen verdict and reason but no per-run measurement is attempted, and no
output file is written for it: the whole point of this instrument is a
question that an undefined coset target cannot answer.

Each measured run applies the same dip-aware checkpoint rule
(``instruments/checkpoints.py``) every other instrument in this project
uses (recorded, substitutions included); a run with no stable checkpoint is
recorded ``status: "skipped"``. Occupancy is the same energy-weighted
population occupancy (``instruments/occupancy.py``) every other occupancy-
consuming instrument reads, computed separately for the left and right
Cayley-grid argument.

One JSON is written per ``DEFINED`` cell to
``<out-dir>/template_divergence_<order>_<index>_w<width>.json``, holding
every seed's full record (screen, checkpoint selection, per-argument
template-divergence comparison) plus per-cell aggregates (mean and 95%
bootstrap CI of each TV distance) over measured seeds.

Exit code: ``0`` iff every attempted run in a ``DEFINED`` cell was measured;
``1`` when any such run was skipped for want of a stable checkpoint, so a
campaign wrapper notices missing measurements loudly. A cell that screens
non-``DEFINED`` never counts against this -- it was never attempted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402

from group_algorithm_interp import stats  # noqa: E402
from group_algorithm_interp.config import ProjectConfig, validate_config  # noqa: E402
from group_algorithm_interp.groups.catalog import resolve_group  # noqa: E402
from group_algorithm_interp.groups.data import GroupData, artifact_path  # noqa: E402
from group_algorithm_interp.instruments.checkpoints import (  # noqa: E402
    CheckpointSelection,
    select_checkpoint,
)
from group_algorithm_interp.instruments.occupancy import (  # noqa: E402
    isotypic_energies,
    neuron_activations,
    population_occupancy,
)
from group_algorithm_interp.instruments.report import (  # noqa: E402
    ARGUMENTS,
    file_sha256,
    instrument_code_hashes,
)
from group_algorithm_interp.instruments.template_divergence import (  # noqa: E402
    DEFINED,
    MAX_COSET_SUPPORT_FRACTION,
    causally_used_blocks,
    compare_to_templates,
    data_driven_comparison,
    degeneracy_screen,
    product_carrying_blocks,
)
from group_algorithm_interp.manifest import get_git_commit, read_manifest  # noqa: E402
from group_algorithm_interp.training.trainer import build_model  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

Cell = tuple[int, int, int]  # (order, index, width)

#: Default directories the data-driven refinement reads I-15 and the GCR
#: matrix-product fit from -- the same output directories
#: ``scripts/measure_isotypic_ablation.py`` and ``scripts/measure_gcr_matmul.py``
#: default to. Named and overridable (``--coset-dir``/``--gcr-matmul-dir``)
#: rather than hard-coded, since a caller may point at a different results
#: tree (e.g. a test's tmp directory).
DEFAULT_COSET_DIR = Path("results/coset")
DEFAULT_GCR_MATMUL_DIR = Path("results/gcr_matmul")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative_to_repo(path: Path) -> str:
    """``path`` relative to the repo root when possible, else its absolute
    form (an ``--artifacts-dir`` override or a test's tmp directory outside
    the repo) -- the same convention ``measure_gcr_readout.py``/``report.py``
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
    run directory that fails to parse is silently excluded rather than
    raised on, since this is a discovery pass over whatever a campaign
    actually produced. Run directories within a cell are sorted by run id."""
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


def screen_cell(
    order: int, index: int, *, max_support_fraction: float = MAX_COSET_SUPPORT_FRACTION
) -> dict[str, Any]:
    """The degeneracy-screen verdict for one group, from the artifact alone.

    ``status`` is ``"screened"`` on a successful load (whatever the verdict),
    or ``"artifact_missing"`` when the group artifact is not on disk under
    the configured ``GROUP_ARTIFACTS_DIR`` -- reported as data rather than
    letting a missing artifact crash the whole cell sweep, since a caller
    sweeping every discovered cell may not have exported every group's
    artifact."""
    try:
        group: GroupData = resolve_group((order, index))
    except FileNotFoundError as error:
        return {"status": "artifact_missing", "reason": str(error)}
    screen = degeneracy_screen(group, max_support_fraction=max_support_fraction)
    return {"status": "screened", "screen": screen.to_record(), "verdict": screen.verdict}


_MISSING_FINAL_EPOCH_RE = re.compile(
    r"^selection\.json names 'final_epoch_(\d+)\.pt' but the checkpoint is not on disk$"
)


def _curated_final_fallback(
    run_dir: Path, selection: CheckpointSelection, config: ProjectConfig
) -> tuple[Path, str] | None:
    """A narrow, documented fallback for one specific curated-shipping gap:
    the campaign's bulk "core" panel ships only ``final.pt`` (no
    ``final_epoch_*.pt``/``step_*.pt``/``checkpoints/``), but a run's
    ``selection.json`` -- computed before that pruning -- may still name the
    un-substituted final-window pick ``final_epoch_<ceiling-1>.pt`` as
    ``stable_end``. That file and ``final.pt`` are the *same* checkpoint by
    construction, not merely by coincidence: ``training/ensemble.py``'s
    per-epoch loop writes ``final_epoch_<epoch>.pt`` for the last window
    epochs (including the ceiling) from the in-memory model state, and
    immediately after the loop ``_finalize`` writes ``final.pt`` from that
    same still-unmodified state (no optimiser step runs in between) --
    bit-identical ``model_state_dict`` contents, verified against several
    runs that (unlike this panel) still carry both files.

    Only triggers when every one of these holds, so a genuinely dipped or
    substituted pick is never silently reassigned:
    * the dip-aware rule's primary pick is exactly the un-substituted
      final-window epoch (``selection.substitution is None``);
    * :func:`checkpoints.select_checkpoint` failed to resolve it because
      only that specific filename is missing (the regex match), not some
      other reason;
    * the named epoch is exactly the run's own epoch ceiling minus one
      (``config.optim.epochs - 1``) -- the one epoch ``_finalize``'s
      ``final.pt`` write provably shares no intervening training with;
    * ``final.pt`` is actually present on disk.

    Returns ``(path, reason)`` when the fallback applies, else ``None``."""
    if selection.path is not None or selection.substitution is not None or selection.reason is None:
        return None
    match = _MISSING_FINAL_EPOCH_RE.match(selection.reason)
    if match is None:
        return None
    epoch = int(match.group(1))
    if epoch != config.optim.epochs - 1:
        return None
    final_path = run_dir / "final.pt"
    if not final_path.is_file():
        final_path = run_dir / "checkpoints" / "final.pt"
        if not final_path.is_file():
            return None
    reason = (
        f"selection.json named final_epoch_{epoch}.pt (the un-substituted final-window "
        "pick, epoch == optim.epochs - 1) but the curated layout kept only final.pt; "
        "substituted final.pt, which training saves from the identical in-memory model "
        "state at the same epoch with no optimiser step in between (see "
        "_curated_final_fallback's docstring) -- not a guess, a documented equivalence"
    )
    return final_path, reason


# ---------------------------------------------------------------------------
# Data-driven refinement: locate and load I-15 / the GCR matrix-product fit.
# ---------------------------------------------------------------------------


def _iso_ablation_path(order: int, index: int, width: int, coset_dir: Path) -> Path:
    """The I-15 per-cell aggregated record's path
    (``scripts/measure_isotypic_ablation.py``'s own naming convention)."""
    return Path(coset_dir) / f"iso_ablation_{order}_{index}_w{width}.json"


def _gcr_matmul_path(order: int, index: int, width: int, gcr_matmul_dir: Path) -> Path:
    """The GCR matrix-product fit's per-cell aggregated record's path
    (``scripts/measure_gcr_matmul.py``'s own naming convention)."""
    return Path(gcr_matmul_dir) / f"gcr_matmul_{order}_{index}_w{width}.json"


def data_driven_used_blocks(
    order: int,
    index: int,
    width: int,
    *,
    coset_dir: Path = DEFAULT_COSET_DIR,
    gcr_matmul_dir: Path = DEFAULT_GCR_MATMUL_DIR,
) -> dict[str, Any]:
    """The cell-level inputs the data-driven refinement needs: the MEASURED
    causally-used block set (I-15, :func:`causally_used_blocks`) and the
    MEASURED product-carrying block set (the GCR matrix-product fit,
    :func:`product_carrying_blocks`), located by this cell's ``(order,
    index, width)`` under ``coset_dir``/``gcr_matmul_dir``.

    Both upstream instruments are per-cell (not per-seed or per-argument),
    so this is computed once per cell and reused against every run's and
    every argument's occupancy. Returns ``{"status": "pending", "missing":
    [...]}`` -- never raising -- when either result file is not yet on
    disk, since I-15 and the matrix-product fit are separate instruments
    that may not have been run yet. On success, returns
    ``{"status": "measured", "iso_ablation_path": ..., "gcr_matmul_path":
    ..., "used_blocks_causal": [...], "used_blocks_product_carrying": [...],
    "used_blocks_intersection": [...]}`` -- both measured sets are reported
    explicitly (not just their intersection), so a reviewer can see where
    the two independent instruments agree and disagree."""
    iso_path = _iso_ablation_path(order, index, width, coset_dir)
    matmul_path = _gcr_matmul_path(order, index, width, gcr_matmul_dir)
    missing = [_relative_to_repo(path) for path in (iso_path, matmul_path) if not path.is_file()]
    if missing:
        return {"status": "pending", "missing": missing}

    iso_record = json.loads(iso_path.read_text())
    matmul_payload = json.loads(matmul_path.read_text())
    causal = causally_used_blocks(iso_record)
    product_carrying = product_carrying_blocks(matmul_payload["aggregate"])
    intersection = tuple(sorted(set(causal) & set(product_carrying)))
    return {
        "status": "measured",
        "iso_ablation_path": _relative_to_repo(iso_path),
        "gcr_matmul_path": _relative_to_repo(matmul_path),
        "used_blocks_causal": list(causal),
        "used_blocks_product_carrying": list(product_carrying),
        "used_blocks_intersection": list(intersection),
    }


# ---------------------------------------------------------------------------
# Per-run measurement.
# ---------------------------------------------------------------------------


def template_divergence_run(
    run_dir: Path,
    *,
    metric: str = "val/accuracy",
    threshold: float = 0.99,
    max_support_fraction: float = MAX_COSET_SUPPORT_FRACTION,
    datadriven: bool = True,
    coset_dir: Path = DEFAULT_COSET_DIR,
    gcr_matmul_dir: Path = DEFAULT_GCR_MATMUL_DIR,
) -> dict[str, Any]:
    """One run's template-divergence record: dip-aware checkpoint selection
    (recorded, substitutions included), then the screened GCR-vs-coset
    comparison for both Cayley-grid arguments. A run with no stable
    checkpoint comes back ``status: "skipped"``; a run whose group does not
    screen ``DEFINED`` comes back ``status: "screen_undefined"`` with no
    checkpoint loaded (the whole point of the screen is to make that check
    cheap and checkpoint-free).

    When ``datadriven`` (the default), each argument's occupancy record also
    gets a ``data_driven`` key (see :func:`data_driven_used_blocks`): either
    ``{"status": "pending", "missing": [...]}`` when I-15's or the GCR
    matrix-product fit's result file is not yet on disk under
    ``coset_dir``/``gcr_matmul_dir``, or, once both exist, the measured used
    sets plus :func:`instruments.template_divergence.data_driven_comparison`'s
    ``.to_record()`` (using the causally-used set, I-15, as the default
    measured-used set compared against each account's predicted set) nested
    under ``"comparison"``."""
    manifest = read_manifest(run_dir)
    config = validate_config(yaml.safe_load((run_dir / "resolved_config.yaml").read_text()))
    group = resolve_group(config.data.group)
    screen = degeneracy_screen(group, max_support_fraction=max_support_fraction)
    selection = select_checkpoint(run_dir, metric=metric, threshold=threshold)
    record: dict[str, Any] = {
        "instrument": "template-divergence-driver",
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
        "screen": screen.to_record(),
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
    if screen.verdict != DEFINED:
        record["status"] = "screen_undefined"
        return record
    checkpoint_path = selection.path
    if checkpoint_path is None:
        fallback = _curated_final_fallback(run_dir, selection, config)
        if fallback is None:
            record["status"] = "skipped"
            return record
        checkpoint_path, fallback_reason = fallback
        record["checkpoint_selection"]["curated_final_fallback"] = fallback_reason

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(config, group)
    model.load_state_dict(checkpoint["model_state_dict"])

    group_artifact = artifact_path(config.data.group.order, config.data.group.index)
    record["status"] = "measured"
    record["provenance"]["checkpoint_sha256"] = file_sha256(checkpoint_path)
    record["provenance"]["group_artifact_path"] = _relative_to_repo(group_artifact)
    record["provenance"]["group_artifact_sha256"] = file_sha256(group_artifact)

    dd_inputs: dict[str, Any] | None = None
    if datadriven:
        dd_inputs = data_driven_used_blocks(
            config.data.group.order,
            config.data.group.index,
            config.model.d_model,
            coset_dir=Path(coset_dir),
            gcr_matmul_dir=Path(gcr_matmul_dir),
        )

    activations = neuron_activations(model, group.order)
    occupancy: dict[str, Any] = {}
    for argument in ARGUMENTS:
        energies = isotypic_energies(activations, group, argument=argument)
        pop_occupancy = population_occupancy(energies)
        divergence = compare_to_templates(
            pop_occupancy, group, max_support_fraction=max_support_fraction
        )
        argument_record = divergence.to_record()
        if dd_inputs is not None:
            if dd_inputs["status"] == "pending":
                argument_record["data_driven"] = dd_inputs
            else:
                comparison = data_driven_comparison(
                    pop_occupancy,
                    group,
                    dd_inputs["used_blocks_causal"],
                    coset_template=screen.coset_template,
                    max_support_fraction=max_support_fraction,
                )
                argument_record["data_driven"] = {
                    **dd_inputs,
                    "comparison": comparison.to_record(),
                }
        occupancy[argument] = argument_record
    record["occupancy"] = occupancy
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
    """One ``DEFINED`` cell's aggregate over its measured seeds: per-argument
    mean/CI of ``tv_to_null``/``tv_to_gcr``/``tv_to_coset``, and a count of
    which template each measured seed landed closest to. Built only from
    ``status: "measured"`` records."""
    measured = [r for r in records if r.get("status") == "measured"]
    n_skipped = len(records) - len(measured)

    per_argument: dict[str, Any] = {}
    for argument in ARGUMENTS:
        tv_null = [float(r["occupancy"][argument]["tv_to_null"]) for r in measured]
        tv_gcr = [float(r["occupancy"][argument]["tv_to_gcr"]) for r in measured]
        tv_coset = [float(r["occupancy"][argument]["tv_to_coset"]) for r in measured]
        closest: dict[str, int] = {}
        for r in measured:
            name = r["occupancy"][argument]["closest_template"]
            closest[name] = closest.get(name, 0) + 1
        per_argument[argument] = {
            "tv_to_null": _summarise(tv_null),
            "tv_to_gcr": _summarise(tv_gcr),
            "tv_to_coset": _summarise(tv_coset),
            "closest_template_counts": closest,
        }

    return {
        "order": order,
        "index": index,
        "width": width,
        "n_seeds_attempted": len(records),
        "n_measured": len(measured),
        "n_skipped": n_skipped,
        "occupancy": per_argument,
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

    n_skipped_runs = 0
    n_written = 0
    for order, index, width in sorted(cells):
        screen = screen_cell(order, index, max_support_fraction=args.max_support_fraction)
        if screen["status"] != "screened" or screen["verdict"] != DEFINED:
            reason = screen.get("reason") or screen.get("screen", {}).get("reason")
            print(
                f"[template_divergence] ({order},{index}) w{width}: "
                f"screen={screen.get('verdict', screen['status'])} -- not attempting seeds "
                f"({reason})"
            )
            continue

        run_dirs = cells[(order, index, width)][: args.max_seeds_per_cell]
        records = [
            template_divergence_run(
                run_dir,
                metric=args.metric,
                threshold=args.threshold,
                max_support_fraction=args.max_support_fraction,
                datadriven=args.datadriven,
                coset_dir=Path(args.coset_dir),
                gcr_matmul_dir=Path(args.gcr_matmul_dir),
            )
            for run_dir in run_dirs
        ]
        n_skipped_runs += sum(1 for r in records if r["status"] == "skipped")
        summary = summarise_cell(order, index, width, records)
        payload = {
            "cell": {"order": order, "index": index, "width": width},
            "n_seeds_available": len(cells[(order, index, width)]),
            "max_seeds_per_cell": args.max_seeds_per_cell,
            "max_support_fraction": args.max_support_fraction,
            "summary": summary,
            "runs": records,
        }
        out_path = Path(args.out_dir) / f"template_divergence_{order}_{index}_w{width}.json"
        _write_json(out_path, payload)
        n_written += 1
        print(
            f"[template_divergence] ({order},{index}) w{width}: DEFINED, "
            f"n_measured={summary['n_measured']}/{summary['n_seeds_attempted']} -> {out_path}"
        )
    print(f"[template_divergence] wrote {n_written} DEFINED cell record(s) to {args.out_dir}")
    return 0 if n_skipped_runs == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser(
        "measure", help="run the screened GCR-vs-coset comparison across panel cells"
    )
    measure.add_argument(
        "--runs-dir", default="runs", help="root directory of trained run directories"
    )
    measure.add_argument(
        "--out-dir",
        default="results/template_divergence",
        help="directory to write one JSON per DEFINED cell",
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
        help="cap on seeds measured per DEFINED cell (default 50)",
    )
    measure.add_argument(
        "--max-support-fraction",
        type=float,
        default=MAX_COSET_SUPPORT_FRACTION,
        dest="max_support_fraction",
        help=f"degeneracy-screen threshold (default {MAX_COSET_SUPPORT_FRACTION})",
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
        "--datadriven",
        "--no-datadriven",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="datadriven",
        help=(
            "compare each occupancy against the MEASURED causally-used "
            "(I-15) and product-carrying (GCR matrix-product) block sets, "
            "reporting a pending status when either result file is not yet "
            "on disk (default: on)"
        ),
    )
    measure.add_argument(
        "--coset-dir",
        default=str(DEFAULT_COSET_DIR),
        dest="coset_dir",
        help=f"directory to read I-15's per-cell records from (default {DEFAULT_COSET_DIR})",
    )
    measure.add_argument(
        "--gcr-matmul-dir",
        default=str(DEFAULT_GCR_MATMUL_DIR),
        dest="gcr_matmul_dir",
        help=(
            "directory to read the GCR matrix-product fit's per-cell records from "
            f"(default {DEFAULT_GCR_MATMUL_DIR})"
        ),
    )
    measure.set_defaults(func=_cmd_measure)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
