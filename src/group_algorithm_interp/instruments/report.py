"""Record builders behind ``scripts/measure_occupancy.py``.

Three responsibilities, all producing JSON-serialisable measurement records --
a vector of signals with nulls and provenance attached, never a verdict label:

* :func:`measure_run` -- one run's occupancy measurement: dip-aware checkpoint
  selection (recorded, substitutions included), the I-10 occupancy vector for
  both arguments against the analytic null, the I-09 per-neuron concentration
  secondary, the Dirichlet noise floor at this run's neuron count, and the
  ``Ind_H^G 1`` template comparisons (``UNDEFINED`` where the coset account has
  no distinct prediction).
* :func:`pool_records` -- seed pooling within one experiment configuration
  (keyed on the manifest's ``config_group_hash``: same model/data/optim apart
  from the seed). Pooled occupancy is the energy-weighted sum over every
  pooled neuron, so the floor scales as ``1/sqrt(neurons x seeds)``.
* :func:`null_gate` -- the I-11 null-calibration gate: the identical pipeline
  on random-init (untrained) models of two groups, reporting each condition's
  null statistic and the per-seed paired difference with a bootstrap CI. The
  ship/no-ship decision this gate feeds is estimation-first and stays with the
  reader; the record carries the measurements.

Provenance per record: the run's manifest hashes (config, config-group,
dataset) and git commit, the analysis-time git commit, the selected
checkpoint's sha256, and the sha256 of every instrument module that computed
the numbers -- so any reported figure traces back to code and data.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .. import stats
from ..config import (
    DataConfig,
    GroupSpec,
    LoggingConfig,
    ModelConfig,
    ProjectConfig,
    validate_config,
)
from ..groups.catalog import resolve_group
from ..groups.group import FiniteGroup
from ..manifest import get_git_commit, read_manifest
from ..model import GroupModel
from ..seed import set_seed
from ..training.trainer import build_model
from .checkpoints import select_checkpoint
from .occupancy import (
    analytic_null,
    dirichlet_noise_floor,
    isotypic_energies,
    neuron_activations,
    per_neuron_concentration,
    population_occupancy,
    restrict_to_nontrivial,
    total_variation,
    trivial_block_index,
)
from .templates import TemplateLibrary, template_library

ARGUMENTS = ("left", "right")
FORMS = ("full", "nontrivial")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instrument_code_hashes() -> dict[str, str]:
    """sha256 of every instrument module, so a record pins the exact analysis
    code that produced it (frozen-and-hashed analysis scripts)."""
    package_dir = Path(__file__).resolve().parent
    return {path.name: file_sha256(path) for path in sorted(package_dir.glob("*.py"))}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tv_block(occupancy: np.ndarray, null: np.ndarray, n_units: int) -> dict[str, Any]:
    """TV against a null, always quoted with its Dirichlet floor at
    ``n_units``."""
    tv = total_variation(occupancy, null)
    floor = dirichlet_noise_floor(null, n_units)
    return {"tv_to_null": tv, "noise_floor": floor, "tv_over_floor": tv / floor}


def _occupancy_forms(
    occupancy: np.ndarray,
    pi0: np.ndarray,
    trivial_index: int,
    library: TemplateLibrary,
    n_units: int,
) -> dict[str, Any]:
    """Both statistic forms for one occupancy vector: the full-block spec form
    (trivial share reported as its own signal) and the nontrivial-renormalised
    form, the null-calibrated headline. Template comparisons are made
    like-for-like in each form; ``"UNDEFINED"`` (a string, deliberately not a
    number) where no nontrivial core-free subgroup exists."""
    occupancy_nt = restrict_to_nontrivial(occupancy, trivial_index)
    pi0_nt = restrict_to_nontrivial(pi0, trivial_index)
    comparisons: list[dict[str, Any]] | str
    if not library.coset_defined:
        comparisons = "UNDEFINED"
    else:
        comparisons = [
            {
                "subgroup_index": entry.subgroup_index,
                "coset_index": entry.coset_index,
                "tv_occupancy_to_template": total_variation(occupancy, entry.template),
                "tv_nontrivial_occupancy_to_template": total_variation(
                    occupancy_nt, restrict_to_nontrivial(entry.template, trivial_index)
                ),
            }
            for entry in library.entries
        ]
    return {
        "occupancy": occupancy.tolist(),
        "trivial_block_index": trivial_index,
        "trivial_block_share": float(occupancy[trivial_index]),
        "full": _tv_block(occupancy, pi0, n_units),
        "nontrivial": {
            "occupancy": occupancy_nt.tolist(),
            "null": pi0_nt.tolist(),
            **_tv_block(occupancy_nt, pi0_nt, n_units),
        },
        "template_comparisons": comparisons,
    }


def _occupancy_block(
    model: GroupModel,
    group: FiniteGroup,
    library: TemplateLibrary,
    n_units_for_floor: int,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    """Both arguments' occupancy measurements for one model, plus the raw
    per-block energy totals the pooling path accumulates. Per-neuron
    concentration (I-09) is computed over the nontrivial blocks -- the DC-heavy
    trivial block would otherwise be almost every random-init neuron's top
    block -- with each neuron's trivial share reported alongside."""
    pi0 = analytic_null(group)
    trivial = trivial_block_index(group)
    activations = neuron_activations(model, group.order)
    per_argument: dict[str, Any] = {}
    block_energy: dict[str, list[float]] = {}
    for argument in ARGUMENTS:
        energies = isotypic_energies(activations, group, argument=argument)
        occupancy = population_occupancy(energies)
        nontrivial_energies = np.delete(energies, trivial, axis=1)
        top_share, top_block = per_neuron_concentration(nontrivial_energies)
        # Remap argmax indices back into full-block-list numbering.
        top_block_full = np.where(top_block >= trivial, top_block + 1, top_block)
        top_block_full = np.where(top_block < 0, -1, top_block_full)
        with np.errstate(invalid="ignore", divide="ignore"):
            totals = energies.sum(axis=1)
            trivial_share = np.where(totals > 0.0, energies[:, trivial] / totals, np.nan)
        per_argument[argument] = {
            **_occupancy_forms(occupancy, pi0, trivial, library, n_units_for_floor),
            "n_zero_energy_units": int((energies.sum(axis=1) <= 0.0).sum()),
            "per_neuron_top_share": [None if np.isnan(v) else float(v) for v in top_share],
            "per_neuron_top_block": top_block_full.tolist(),
            "per_neuron_trivial_share": [None if np.isnan(v) else float(v) for v in trivial_share],
        }
        block_energy[argument] = energies.sum(axis=0).tolist()
    return per_argument, block_energy


def measure_run(
    run_dir: Path,
    *,
    metric: str = "val/accuracy",
    threshold: float = 0.99,
) -> dict[str, Any]:
    """One run's occupancy measurement record (module docstring). A run whose
    dip-aware selection finds no stable checkpoint is returned with
    ``status: "skipped"`` and the full selection record -- reported as data,
    never silently dropped."""
    manifest = read_manifest(run_dir)
    config = validate_config(yaml.safe_load((run_dir / "resolved_config.yaml").read_text()))
    selection = select_checkpoint(run_dir, metric=metric, threshold=threshold)
    record: dict[str, Any] = {
        "instrument": "occupancy",
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
    library = template_library(group)
    per_argument, block_energy = _occupancy_block(model, group, library, config.model.d_mlp)

    record["status"] = "measured"
    record["provenance"]["checkpoint_sha256"] = file_sha256(selection.path)
    record["n_units"] = config.model.d_mlp
    record["analytic_null"] = analytic_null(group).tolist()
    record["block_ranks"] = [block.block_rank for block in group.isotypic_blocks]
    record["block_irrep_degrees"] = [block.irrep_degree for block in group.isotypic_blocks]
    record["trivial_block_index"] = trivial_block_index(group)
    record["templates"] = library.to_record()
    record["occupancy"] = per_argument
    record["block_energy"] = block_energy
    return record


def pool_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pool measured records over seeds within one experiment configuration.

    The pooling key is the manifest's ``config_group_hash`` -- two runs share
    it iff they define the same experiment apart from the seed -- so seeds are
    never pooled across groups, widths, or optimiser settings. Pooled
    occupancy is the total energy per block over every pooled neuron,
    normalised; the noise floor is recomputed at ``N = neurons x seeds``."""
    pools: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("status") != "measured":
            continue
        key = record["provenance"].get("config_group_hash") or record["run_id"]
        pools.setdefault(key, []).append(record)

    pooled: list[dict[str, Any]] = []
    for key, members in sorted(pools.items()):
        first = members[0]
        pi0 = np.asarray(first["analytic_null"], dtype=np.float64)
        trivial = int(first["trivial_block_index"])
        n_units = sum(int(member["n_units"]) for member in members)
        per_argument: dict[str, Any] = {}
        for argument in ARGUMENTS:
            energy = np.zeros_like(pi0)
            for member in members:
                energy += np.asarray(member["block_energy"][argument], dtype=np.float64)
            occupancy = energy / float(energy.sum())
            occupancy_nt = restrict_to_nontrivial(occupancy, trivial)
            pi0_nt = restrict_to_nontrivial(pi0, trivial)
            comparisons: list[dict[str, Any]] | str
            template_record = first["templates"]
            if template_record["entries"] == "UNDEFINED":
                comparisons = "UNDEFINED"
            else:
                comparisons = [
                    {
                        "subgroup_index": entry["subgroup_index"],
                        "coset_index": entry["coset_index"],
                        "tv_occupancy_to_template": total_variation(
                            occupancy, np.asarray(entry["template"], dtype=np.float64)
                        ),
                        "tv_nontrivial_occupancy_to_template": total_variation(
                            occupancy_nt,
                            restrict_to_nontrivial(
                                np.asarray(entry["template"], dtype=np.float64), trivial
                            ),
                        ),
                    }
                    for entry in template_record["entries"]
                ]
            per_argument[argument] = {
                "occupancy": occupancy.tolist(),
                "trivial_block_index": trivial,
                "trivial_block_share": float(occupancy[trivial]),
                "full": _tv_block(occupancy, pi0, n_units),
                "nontrivial": {
                    "occupancy": occupancy_nt.tolist(),
                    "null": pi0_nt.tolist(),
                    **_tv_block(occupancy_nt, pi0_nt, n_units),
                },
                "template_comparisons": comparisons,
            }
        pooled.append(
            {
                "instrument": "occupancy-pooled",
                "config_group_hash": key,
                "group": first["group"],
                "model": first["model"],
                "n_runs": len(members),
                "run_ids": [member["run_id"] for member in members],
                "seeds": [member["seed"] for member in members],
                "n_units": n_units,
                "analytic_null": first["analytic_null"],
                "block_ranks": first["block_ranks"],
                "trivial_block_index": trivial,
                "templates": first["templates"],
                "occupancy": per_argument,
            }
        )
    return pooled


def _random_init_tvs(
    config: ProjectConfig, group: FiniteGroup, seeds: list[int]
) -> dict[str, dict[str, list[float]]]:
    """TV(occupancy, own analytic null) for a random-init (untrained) model of
    ``group`` at each seed -- both arguments, both statistic forms. The
    initialisation replays the training paths' seeding exactly (``set_seed``
    then ``build_model``), so the null models are the models training would
    have started from."""
    pi0 = analytic_null(group)
    trivial = trivial_block_index(group)
    pi0_nt = restrict_to_nontrivial(pi0, trivial)
    tvs: dict[str, dict[str, list[float]]] = {
        argument: {form: [] for form in FORMS} for argument in ARGUMENTS
    }
    for seed in seeds:
        set_seed(seed, deterministic=False)
        model = build_model(config, group)
        activations = neuron_activations(model, group.order)
        for argument in ARGUMENTS:
            energies = isotypic_energies(activations, group, argument=argument)
            occupancy = population_occupancy(energies)
            tvs[argument]["full"].append(total_variation(occupancy, pi0))
            tvs[argument]["nontrivial"].append(
                total_variation(restrict_to_nontrivial(occupancy, trivial), pi0_nt)
            )
    return tvs


def null_gate(
    group_a: tuple[int, int],
    group_b: tuple[int, int],
    seeds: list[int],
    *,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """I-11: the occupancy pipeline on random-init models of both groups.

    Reports each condition's null statistic (TV to its own analytic null) per
    seed, the per-seed paired difference (a - b) with a 95% bootstrap CI, and
    whether the two analytic nulls agree as multisets (the CT-equal free pass
    is asserted, not assumed). No verdict is emitted: a statistic whose null
    differs across conditions does not ship, and that reading is made from the
    interval."""
    conditions: dict[str, dict[str, Any]] = {}
    for label, (order, index) in (("a", group_a), ("b", group_b)):
        config = ProjectConfig(
            device="cpu",
            data=DataConfig(group=GroupSpec(order=order, index=index)),
            model=ModelConfig(**(model_config or {})),
            logging=LoggingConfig(mode="disabled"),
        )
        group = resolve_group(config.data.group)
        pi0 = analytic_null(group)
        trivial = trivial_block_index(group)
        tvs = _random_init_tvs(config, group, seeds)
        per_argument: dict[str, Any] = {}
        for argument in ARGUMENTS:
            per_argument[argument] = {}
            for form in FORMS:
                mean, std = stats.mean_std(tvs[argument][form])
                per_argument[argument][form] = {
                    "tv_per_seed": tvs[argument][form],
                    "mean": mean,
                    "std": std,
                }
        conditions[label] = {
            "group": {"order": order, "index": index, "name": group.canonical_name},
            "analytic_null": pi0.tolist(),
            "block_ranks": [block.block_rank for block in group.isotypic_blocks],
            "trivial_block_index": trivial,
            "noise_floor_per_model": {
                "full": dirichlet_noise_floor(pi0, config.model.d_mlp),
                "nontrivial": dirichlet_noise_floor(
                    restrict_to_nontrivial(pi0, trivial), config.model.d_mlp
                ),
            },
            "tv_to_own_null": per_argument,
            "n_units_per_model": config.model.d_mlp,
        }

    differences: dict[str, Any] = {}
    for argument in ARGUMENTS:
        differences[argument] = {}
        for form in FORMS:
            diffs = [
                a - b
                for a, b in zip(
                    conditions["a"]["tv_to_own_null"][argument][form]["tv_per_seed"],
                    conditions["b"]["tv_to_own_null"][argument][form]["tv_per_seed"],
                    strict=True,
                )
            ]
            mean, std = stats.mean_std(diffs)
            low, high = stats.bootstrap_ci(diffs)
            differences[argument][form] = {
                "per_seed": diffs,
                "mean": mean,
                "std": std,
                "bootstrap_ci_95": [low, high],
            }

    pi0_equal = sorted(conditions["a"]["block_ranks"]) == sorted(conditions["b"]["block_ranks"])
    return {
        "instrument": "occupancy-null-calibration",
        "statistic": "tv_to_own_analytic_null",
        "seeds": seeds,
        "conditions": conditions,
        "pi0_equal_as_multisets": pi0_equal,
        "paired_difference_a_minus_b": differences,
        "provenance": {
            "analysis_git_commit": get_git_commit(),
            "analysed_at": _utcnow(),
            "instrument_code_sha256": instrument_code_hashes(),
        },
        "note": (
            "Estimation-first record of the I-11 gate inputs: a statistic whose "
            "null differs across the two conditions is disqualified and does not "
            "ship. The reading is made from the paired-difference interval; no "
            "verdict label is emitted here."
        ),
    }


__all__ = [
    "ARGUMENTS",
    "FORMS",
    "file_sha256",
    "instrument_code_hashes",
    "measure_run",
    "null_gate",
    "pool_records",
]
