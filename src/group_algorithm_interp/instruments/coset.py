"""The coset arm (I-17/I-18/I-19) and the isotypic-block usage check (I-15).

This module turns the coset (T5) account into causal measurements on a single
trained model. It is the mechanism half of the C2 D32/QD32/Q32 case study
(plan §2, §5, §8): occupancy (I-10) says which isotypic blocks are *occupied*;
nothing here or there licenses "the model *uses* block ``rho``" until an
intervention shows a behavioural cost. The four instruments built here:

* **I-15** :func:`isotypic_block_ablation` -- rung 3, Necessary. Ablate an
  isotypic block from the shared input embedding and measure the behavioural
  drop on the transpose-unleaked held-out set, against a random subspace matched
  on norm and ``block_rank``. This is the instrument that converts "block j is
  occupied" (I-10, rung 0/2) into "the model uses block j" (rung 3).
* **I-17** :func:`coset_collapse_probe` -- rung 1, ceiling 1 (decodable != used).
  Whether the model's internal representation of the left input ``a`` is a
  function of ``a``'s coset of a core-free ``H`` (within- vs between-coset
  representation distance), against a random-partition null matched on block
  sizes.
* **I-18** :func:`coset_subspace_ablation` -- rung 3, Necessary. Ablate the whole
  ``Ind_H^G 1`` isotypic subspace (the coset circuit's support), against a random
  subspace matched on norm and total rank.
* **I-19** :func:`coset_patching` -- rung 4, Causal/pathway. Within-coset
  (``a -> a*h``, ``h in H``) vs across-coset (``a -> a*g``, ``g not in H``)
  perturbations: a coset circuit's intermediate is invariant under the first.
  Baseline representation invariance plus exact path patching (MLP path vs the
  direct/attention path) to confirm the coset information reaches the logits
  through the MLP.

**D32/QD32 only (the ``UNDEFINED`` design, plan §2).** The entire coset arm is
gated on the existence of a nontrivial core-free subgroup (I-12). On Q32 (and
Q64, Q128, every abelian group) the only core-free subgroup is trivial,
``Ind_1^G 1`` is the regular representation, and the coset template equals the
analytic null by theorem -- so I-17/I-18/I-19 return the string ``"UNDEFINED"``,
never a number. That undefinedness is the result, not a gap: the coset,
extension-coordinate and power-map accounts have no distinct prediction where no
core-free ``H`` exists. This gate is read straight off
:func:`templates.template_library` (``coset_defined``), so it is exact and
shares the occupancy arm's convention.

**Ablation as an embedding edit.** A "block/subspace ablation" here removes a
group-representation component from the *shared* input embedding ``W_E``. Each
column of the element-embedding block ``E = W_E[:|G|]`` is a function on ``G``;
projecting it with an isotypic (or the induced-subspace) projector ``P`` and
subtracting removes exactly that representation's content of the model's inputs,
which is precisely the object occupancy measures. The intervention needs no
forward hook -- it swaps ``W_E`` for the edited copy and restores it -- and its
matched random-subspace control isolates the block-specific behavioural cost
from the cost of deleting *any* rank-``r`` subspace of the same norm.

**DC-dominance convention.** As in ``occupancy.py``, the read position is the
``=`` token, whose constant component dominates a random-init model (~96% of the
energy sits in the trivial block). Representation-geometry instruments here mean
-centre the per-element representation across elements before measuring
distances, so coset structure is read from the nontrivial content rather than
the architectural DC offset; block ablation targets the nontrivial blocks by
default for the same reason.

Estimation-first throughout: every effect is an effect size with a null and/or a
bootstrap CI, never a thresholded verdict. Random draws use a private NumPy
generator so the global RNG streams a run depends on are never perturbed.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from .. import stats
from ..groups.group import FiniteGroup
from ..model import GroupModel, OneLayerTransformer
from .occupancy import (
    neuron_activations,
    trivial_block_index,
)
from .templates import induction_template, template_library

_TOL = 1e-9
UNDEFINED = "UNDEFINED"
ABLATION_MODES = ("zero", "mean", "resample")


# ---------------------------------------------------------------------------
# Behavioural substrate: the transpose-unleaked held-out set and accuracy
# ---------------------------------------------------------------------------


def _identity_index(table: np.ndarray) -> int:
    n = table.shape[0]
    idx = np.arange(n)
    candidates = np.flatnonzero(np.all(table == idx[None, :], axis=1))
    if candidates.size == 0:
        raise ValueError("Cayley table has no identity element")
    return int(candidates[0])


def unleaked_heldout(
    group: FiniteGroup, *, train_frac: float, split_seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The transpose-unleaked held-out subset for one group and split, as
    ``(tokens, targets)``. Tokens are ``(a, b, '=')`` triples (``'='`` is the
    ``order`` token); targets are ``a*b``. This is the single accuracy surface
    every instrument here scores on -- ``task.py``'s realised split with the
    transpose-leaked pairs removed (leaked test answers are readable off a
    training example and do not test generalisation).

    Imported lazily from ``task`` so this instrument module carries no import-
    time dependency on the task/data layer.
    """
    from ..task import build_group_task, train_test_split, transpose_leaked_mask

    task = build_group_task(group)
    split = train_test_split(task, train_frac, split_seed)
    leaked = transpose_leaked_mask(split, group.cayley_table)
    keep = ~leaked
    inputs = split.test_inputs[keep]
    targets = split.test_targets[keep]
    order = group.order
    equals = np.full((inputs.shape[0], 1), order, dtype=np.int64)
    tokens = np.concatenate([inputs.astype(np.int64), equals], axis=1)
    return torch.from_numpy(tokens), torch.from_numpy(targets.astype(np.int64))


def correct_mask(model: GroupModel, tokens: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    """Per-example correctness at the read position, ``argmax logits[:, -1] ==
    target``. The behavioural unit every ablation is scored in (flips)."""
    model.eval()
    with torch.no_grad():
        logits = model(tokens)
    pred = logits[:, -1, :].argmax(dim=-1)
    return (pred == targets).cpu().numpy().astype(bool)


def _flip_stats(clean: np.ndarray, ablated: np.ndarray) -> dict[str, Any]:
    """Behavioural cost of an intervention in flip units: examples that were
    correct clean and are wrong after, as a count and a fraction (1/n_test), plus
    the two accuracies. Estimation-first: raw quantities, no threshold."""
    n = int(clean.size)
    flips = int(np.count_nonzero(clean & ~ablated))
    return {
        "n_test": n,
        "clean_accuracy": float(clean.mean()) if n else float("nan"),
        "ablated_accuracy": float(ablated.mean()) if n else float("nan"),
        "flips": flips,
        "flip_fraction": (flips / n) if n else float("nan"),
    }


# ---------------------------------------------------------------------------
# Ablation: edit the shared input embedding's representation subspace
# ---------------------------------------------------------------------------


def random_rank_projector(order: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    """A uniformly-random real orthogonal projector of the given rank on the
    ``order``-dimensional element-function space (``Q Q^T`` for an orthonormal
    ``Q`` from the QR of a Gaussian). The matched control for a block/subspace
    ablation: same rank, different subspace."""
    if not 0 <= rank <= order:
        raise ValueError(f"rank {rank} out of range for order {order}")
    if rank == 0:
        return np.zeros((order, order), dtype=np.float64)
    gaussian = rng.standard_normal((order, rank))
    q, _ = np.linalg.qr(gaussian)
    return q @ q.T


def _apply_mode(
    embed: np.ndarray, component: np.ndarray, mode: str, rng: np.random.Generator
) -> np.ndarray:
    """Return the edited element embedding after removing ``component`` (the
    projected subspace) under one of the three ablation modes, compared side by
    side per the suite convention:

    * ``zero`` -- delete the component outright (``E - P E``);
    * ``mean`` -- replace it with its across-element mean (a constant offset);
    * ``resample`` -- replace it with the component of a random element
      permutation, preserving its marginal but destroying its alignment with
      element identity.
    """
    if mode == "zero":
        return embed - component
    if mode == "mean":
        return embed - component + component.mean(axis=0, keepdims=True)
    if mode == "resample":
        perm = rng.permutation(embed.shape[0])
        return embed - component + component[perm]
    raise ValueError(f"unknown ablation mode {mode!r}; expected one of {ABLATION_MODES}")


@contextmanager
def _swapped_embedding(model: GroupModel, new_element_embed: np.ndarray) -> Iterator[None]:
    """Temporarily replace the ``|G|`` element rows of ``W_E`` with
    ``new_element_embed`` (the ``'='`` row is left untouched), restoring the
    original parameter on exit. Exact and hook-free."""
    order = new_element_embed.shape[0]
    original = model.W_E
    full = original.detach().clone()
    full[:order] = torch.from_numpy(np.asarray(new_element_embed, dtype=np.float64)).to(full.dtype)
    model.W_E = torch.nn.Parameter(full, requires_grad=False)
    try:
        yield
    finally:
        model.W_E = original


def _ablation_flip_record(
    model: GroupModel,
    projector: np.ndarray,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    clean: np.ndarray,
    *,
    rank: int,
    n_random: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Behavioural cost of ablating the subspace ``projector`` picks out, for
    each mode, with a rank- and norm-matched random-subspace control. The random
    control is a distribution (``n_random`` draws): its flip fractions get a
    mean and a bootstrap CI, and the effect size is ``block flip fraction - mean
    random flip fraction`` -- the damage attributable to *this* subspace above
    deleting any subspace of the same size and norm."""
    element_embed = model.W_E.detach().cpu().numpy()[: projector.shape[0]].astype(np.float64)
    component = projector @ element_embed
    component_norm = float(np.linalg.norm(component))

    modes: dict[str, Any] = {}
    for mode in ABLATION_MODES:
        edited = _apply_mode(element_embed, component, mode, np.random.default_rng(0))
        with _swapped_embedding(model, edited):
            ablated = correct_mask(model, tokens, targets)
        target_stats = _flip_stats(clean, ablated)

        random_fracs: list[float] = []
        for _ in range(n_random):
            rand_proj = random_rank_projector(projector.shape[0], rank, rng)
            rand_component = rand_proj @ element_embed
            rand_norm = float(np.linalg.norm(rand_component))
            if rand_norm > _TOL:
                rand_component = rand_component * (component_norm / rand_norm)
            rand_edited = _apply_mode(element_embed, rand_component, mode, rng)
            with _swapped_embedding(model, rand_edited):
                rand_ablated = correct_mask(model, tokens, targets)
            random_fracs.append(_flip_stats(clean, rand_ablated)["flip_fraction"])

        random_summary = _distribution_summary(random_fracs)
        modes[mode] = {
            **target_stats,
            "random_subspace": random_summary,
            "flip_fraction_over_random": target_stats["flip_fraction"] - random_summary["mean"],
        }
    return {
        "rank": rank,
        "component_norm": component_norm,
        "n_random_subspaces": n_random,
        "modes": modes,
    }


def _distribution_summary(values: Sequence[float]) -> dict[str, Any]:
    """Mean and, when there are at least two draws, a 95% bootstrap CI of a
    control distribution."""
    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0, "mean": float("nan"), "bootstrap_ci_95": None, "values": []}
    mean = float(np.mean(vals))
    ci: list[float] | None = None
    if len(vals) >= 2 and len(set(vals)) > 1:
        low, high = stats.bootstrap_ci(vals)
        ci = [low, high]
    elif len(vals) >= 2:
        ci = [vals[0], vals[0]]
    return {"n": len(vals), "mean": mean, "bootstrap_ci_95": ci, "values": vals}


# ---------------------------------------------------------------------------
# I-15: isotypic-block ablation (rung 3) -- "occupied" -> "used"
# ---------------------------------------------------------------------------


def isotypic_block_ablation(
    model: GroupModel,
    group: FiniteGroup,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    block_indices: Sequence[int] | None = None,
    n_random: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """I-15 (rung 3, Necessary): ablate each isotypic block from the shared input
    embedding and measure the behavioural drop on ``(tokens, targets)`` against a
    random subspace matched on norm and ``block_rank``.

    Blocks default to every *nontrivial* block (the trivial block is the DC
    offset). Zero/mean/resample are compared side by side; the random control is
    a distribution with a bootstrap CI. The reported ``flip_fraction_over_random``
    per (block, mode) is the effect that licenses "the model uses block j": a
    block whose ablation costs no more than a matched random subspace is occupied
    but not shown to be used. No verdict is emitted -- the reading is the effect
    size and its CI.
    """
    clean = correct_mask(model, tokens, targets)
    trivial = trivial_block_index(group)
    if block_indices is None:
        block_indices = [j for j in range(len(group.isotypic_blocks)) if j != trivial]
    rng = np.random.default_rng(seed)

    blocks: list[dict[str, Any]] = []
    for j in block_indices:
        block = group.isotypic_blocks[j]
        projector = np.asarray(block.projector, dtype=np.float64)
        record = _ablation_flip_record(
            model,
            projector,
            tokens,
            targets,
            clean,
            rank=block.block_rank,
            n_random=n_random,
            rng=rng,
        )
        blocks.append(
            {
                "block_index": j,
                "irrep_degree": block.irrep_degree,
                "block_rank": block.block_rank,
                "is_trivial": j == trivial,
                **record,
            }
        )
    return {
        "instrument": "isotypic-block-ablation",
        "rung": 3,
        "trivial_block_index": trivial,
        "clean_accuracy": float(clean.mean()) if clean.size else float("nan"),
        "n_test": int(clean.size),
        "blocks": blocks,
    }


# ---------------------------------------------------------------------------
# Coset-arm gate and shared representation extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CosetTarget:
    """The minimal-index core-free subgroup the coset arm runs on, plus the
    left-coset partition and the ``Ind_H^G 1`` support the arm needs.
    ``count_at_min_index`` records how many core-free subgroups tie at the
    minimal index. Those ties need NOT be conjugate to one another -- e.g. D8's
    four index-4 reflection subgroups fall into two conjugacy classes, not one --
    so they can give genuinely different coset actions; the arm fixes one
    deterministically (the lowest ``subgroup_index``) rather than assuming they
    agree."""

    subgroup_index: int
    subgroup: np.ndarray
    coset_index: int
    cosets: tuple[np.ndarray, ...]
    element_to_coset: np.ndarray  # [|G|] -> coset id
    occupied_blocks: tuple[int, ...]
    subspace_projector: np.ndarray
    subspace_rank: int
    count_at_min_index: int


def coset_target(group: FiniteGroup) -> CosetTarget | None:
    """Select the coset arm's subgroup, or ``None`` when there is none to run on.
    ``None`` covers two distinct causes -- a structural UNDEFINED (no nontrivial
    core-free subgroup exists, Q32 and its family) and an artifact-incomplete
    group (no subgroups were exported at all) -- which this function does not
    itself separate: consult :func:`templates.template_library`'s
    ``coset_defined`` / ``artifact_incomplete`` (as :func:`coset_arm` does) to
    tell them apart. When a subgroup is returned the selection is deterministic:
    the lowest-``subgroup_index`` core-free subgroup at the minimal core-free
    index (I-12), which fixes a concrete left-coset partition and the
    ``Ind_H^G 1`` isotypic support (I-13)."""
    library = template_library(group)
    if not library.coset_defined:
        return None
    min_index = library.min_corefree_index
    at_min = [entry for entry in library.entries if entry.coset_index == min_index]
    entry = min(at_min, key=lambda e: e.subgroup_index)
    subgroup = np.asarray(group.subgroups[entry.subgroup_index], dtype=np.int64)
    cosets = group.cosets_for_subgroup(subgroup)

    element_to_coset = np.full(group.order, -1, dtype=np.int64)
    for coset_id, coset in enumerate(cosets):
        element_to_coset[np.asarray(coset, dtype=np.int64)] = coset_id
    if int(element_to_coset.min()) < 0:
        raise ValueError("left cosets do not partition the group")

    template = induction_template(group, subgroup)
    occupied = tuple(int(j) for j in np.flatnonzero(template > _TOL))
    projector = np.zeros((group.order, group.order), dtype=np.float64)
    rank = 0
    for j in occupied:
        projector = projector + np.asarray(group.isotypic_blocks[j].projector, dtype=np.float64)
        rank += group.isotypic_blocks[j].block_rank
    return CosetTarget(
        subgroup_index=entry.subgroup_index,
        subgroup=subgroup,
        coset_index=entry.coset_index,
        cosets=cosets,
        element_to_coset=element_to_coset,
        occupied_blocks=occupied,
        subspace_projector=projector,
        subspace_rank=rank,
        count_at_min_index=len(at_min),
    )


def left_representations(model: GroupModel, group: FiniteGroup, *, source: str) -> np.ndarray:
    """A per-element representation of the left input ``a``, mean-centred across
    elements (the DC-dominance convention). ``source="embed"`` is the shared
    embedding row ``W_E[a]``; ``source="mlp_pre"`` is the read-position
    pre-activation averaged over the right argument, ``mean_b A[a, b]`` (I-08).
    Returns ``[|G|, feature]``."""
    if source == "embed":
        rep = model.W_E.detach().cpu().numpy()[: group.order].astype(np.float64)
    elif source == "mlp_pre":
        activations = neuron_activations(model, group.order)  # [d_mlp, |G|, |G|]
        rep = activations.mean(axis=2).T  # [|G|, d_mlp]
    else:
        raise ValueError(f"unknown representation source {source!r}")
    return rep - rep.mean(axis=0, keepdims=True)


def _partition_distance_ratio(reps: np.ndarray, labels: np.ndarray, n_labels: int) -> float:
    """Mean within-group over mean between-group pairwise Euclidean distance for
    a labelling of the rows of ``reps``. A representation that collapses within
    each group drives the ratio below 1."""
    n = reps.shape[0]
    diff = reps[:, None, :] - reps[None, :, :]
    dist = np.sqrt(np.square(diff).sum(axis=2))
    same = labels[:, None] == labels[None, :]
    eye = np.eye(n, dtype=bool)
    within = same & ~eye
    between = ~same
    within_mean = float(dist[within].mean()) if within.any() else float("nan")
    between_mean = float(dist[between].mean()) if between.any() else float("nan")
    if not np.isfinite(between_mean) or between_mean <= _TOL:
        return float("nan")
    return within_mean / between_mean


def _random_partition_labels(sizes: Sequence[int], n: int, rng: np.random.Generator) -> np.ndarray:
    """A random labelling of ``n`` items into groups of the given sizes (the
    coset partition's block sizes) -- the I-17/I-19 null matched on block
    sizes. The sizes must sum to ``n`` or the labelling would leave items
    unassigned (a silent bug with ``np.empty``); this is guarded."""
    total = int(sum(int(s) for s in sizes))
    if total != n:
        raise ValueError(f"partition sizes sum to {total}, expected {n}")
    labels = np.full(n, -1, dtype=np.int64)
    perm = rng.permutation(n)
    start = 0
    for label, size in enumerate(sizes):
        labels[perm[start : start + size]] = label
        start += size
    if int(labels.min()) < 0:
        raise ValueError("partition left some items unlabelled")
    return labels


# ---------------------------------------------------------------------------
# I-17: coset-collapse probe (rung 1, ceiling 1 -- decodable != used)
# ---------------------------------------------------------------------------


def coset_collapse_probe(
    model: GroupModel,
    group: FiniteGroup,
    target: CosetTarget,
    *,
    sources: Sequence[str] = ("embed", "mlp_pre"),
    n_null: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """I-17 (rung 1, ceiling 1): is the model's representation of the left input a
    function of its coset of ``H``? Reports the within/between-coset distance
    ratio against a random-partition null matched on the coset block sizes, for
    each representation source. A ratio well below the null (z < 0) is coset
    collapse; the ceiling is 1 -- decodable is not used, so no "uses" claim is
    made here (that is I-18/I-19).
    """
    labels = target.element_to_coset
    sizes = [int(len(coset)) for coset in target.cosets]
    rng = np.random.default_rng(seed)
    per_source: dict[str, Any] = {}
    for source in sources:
        reps = left_representations(model, group, source=source)
        observed = _partition_distance_ratio(reps, labels, len(target.cosets))
        null_ratios = [
            _partition_distance_ratio(
                reps, _random_partition_labels(sizes, group.order, rng), len(sizes)
            )
            for _ in range(n_null)
        ]
        finite = [r for r in null_ratios if np.isfinite(r)]
        null_mean = float(np.mean(finite)) if finite else float("nan")
        null_std = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
        z = (
            (observed - null_mean) / null_std
            if np.isfinite(null_std) and null_std > _TOL
            else float("nan")
        )
        frac_le = float(np.mean([r <= observed for r in finite])) if finite else float("nan")
        per_source[source] = {
            "within_over_between_ratio": observed,
            "null_mean": null_mean,
            "null_std": null_std,
            "z_score": z,
            "collapse_effect": (null_mean - observed) if np.isfinite(null_mean) else float("nan"),
            "null_fraction_at_or_below_observed": frac_le,
            "n_null": len(finite),
        }
    return {
        "instrument": "coset-collapse-probe",
        "rung": 1,
        "ceiling": 1,
        "subgroup_index": target.subgroup_index,
        "coset_index": target.coset_index,
        "sources": per_source,
    }


# ---------------------------------------------------------------------------
# I-18: coset-subspace ablation (rung 3, Necessary)
# ---------------------------------------------------------------------------


def coset_subspace_ablation(
    model: GroupModel,
    group: FiniteGroup,
    target: CosetTarget,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    n_random: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """I-18 (rung 3, Necessary): ablate the whole ``Ind_H^G 1`` isotypic subspace
    (the union of the occupied blocks, ``rank`` = sum of their ``block_rank``s --
    30 on D32 for the minimal core-free ``H`` of index 16, not a single degree-2
    block's 4; the ranks sum to |G|=32, so a value like 48 is impossible. See
    ``tests/test_coset.py`` for the computed 30) from the shared embedding,
    against a random subspace matched on norm and that total rank. Reports the
    behavioural drop in flip units per mode with the matched-random control,
    exactly as I-15 but for the coset support rather than one block."""
    clean = correct_mask(model, tokens, targets)
    rng = np.random.default_rng(seed)
    record = _ablation_flip_record(
        model,
        target.subspace_projector,
        tokens,
        targets,
        clean,
        rank=target.subspace_rank,
        n_random=n_random,
        rng=rng,
    )
    return {
        "instrument": "coset-subspace-ablation",
        "rung": 3,
        "subgroup_index": target.subgroup_index,
        "coset_index": target.coset_index,
        "occupied_blocks": list(target.occupied_blocks),
        "clean_accuracy": float(clean.mean()) if clean.size else float("nan"),
        "n_test": int(clean.size),
        **record,
    }


# ---------------------------------------------------------------------------
# I-19: within-coset / across-coset patching (rung 4, Causal/pathway)
# ---------------------------------------------------------------------------


def _path_patch_recovery(
    model: OneLayerTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
) -> dict[str, float]:
    """Exact path patching for the 1-layer model, via the additive residual
    decomposition ``resid_final = embed + attn_out + mlp_out`` (no hooks needed).
    Patches the clean MLP output into the corrupted run (MLP path) and,
    separately, the clean attention+embedding (the direct path), and reports the
    fraction of the corrupt->clean read-logit gap each path recovers. A coset
    circuit routes the coset dependence through the MLP path."""
    model.eval()
    with torch.no_grad():
        clean = model(clean_tokens, return_cache=True)
        corrupt = model(corrupt_tokens, return_cache=True)
        w_u = model.W_U
        clean_logits = clean["logits"][:, -1, :]
        corrupt_logits = corrupt["logits"][:, -1, :]

        clean_embed = clean["embed"][:, -1, :]
        clean_attn = clean["attn_out"][:, -1, :]
        clean_resid = clean["resid_final"][:, -1, :]
        clean_mlp = clean_resid - clean_embed - clean_attn

        corrupt_embed = corrupt["embed"][:, -1, :]
        corrupt_attn = corrupt["attn_out"][:, -1, :]
        corrupt_resid = corrupt["resid_final"][:, -1, :]
        corrupt_mlp = corrupt_resid - corrupt_embed - corrupt_attn

        mlp_patched = (corrupt_embed + corrupt_attn + clean_mlp) @ w_u
        direct_patched = (clean_embed + clean_attn + corrupt_mlp) @ w_u

        gap = torch.linalg.vector_norm(corrupt_logits - clean_logits, dim=-1)
        mlp_residual = torch.linalg.vector_norm(mlp_patched - clean_logits, dim=-1)
        direct_residual = torch.linalg.vector_norm(direct_patched - clean_logits, dim=-1)
        safe = gap > _TOL
        mlp_recovery = torch.where(safe, 1.0 - mlp_residual / gap, torch.zeros_like(gap))
        direct_recovery = torch.where(safe, 1.0 - direct_residual / gap, torch.zeros_like(gap))
    return {
        "mean_logit_gap": float(gap.mean()),
        "mlp_path_recovery": float(mlp_recovery[safe].mean()) if bool(safe.any()) else float("nan"),
        "direct_path_recovery": (
            float(direct_recovery[safe].mean()) if bool(safe.any()) else float("nan")
        ),
    }


def coset_patching(
    model: GroupModel,
    group: FiniteGroup,
    target: CosetTarget,
    *,
    source: str = "mlp_pre",
    n_across: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """I-19 (rung 4, Causal/pathway): within-coset (``a -> a*h``, ``h in H``) vs
    across-coset (``a -> a*g``, ``g not in H``) perturbations of the left input.

    Two readouts. (1) Representation invariance: the mean read-position
    representation change under within- vs across-coset moves -- a coset
    circuit's intermediate is invariant under the first (within/across ratio near
    0), reported against the same random-partition scale as I-17. (2) Path
    patching (transformer only): the fraction of the corrupt->clean logit gap
    recovered through the MLP path vs the direct path, for within and across
    corruptions, confirming the coset information reaches the logits through the
    MLP. FC models report the invariance readout with patching marked
    ``UNDEFINED`` (no attention/MLP path split)."""
    table = group.cayley_table
    identity = _identity_index(table)
    subgroup = set(int(x) for x in target.subgroup.tolist())
    within_gen = [h for h in subgroup if h != identity]
    outside = [g for g in range(group.order) if g not in subgroup]
    rng = np.random.default_rng(seed)

    reps = left_representations(model, group, source=source)
    within_dists: list[float] = []
    across_dists: list[float] = []
    for a in range(group.order):
        for h in within_gen:
            a_h = int(table[a, h])
            within_dists.append(float(np.linalg.norm(reps[a] - reps[a_h])))
        chosen = rng.choice(len(outside), size=min(n_across, len(outside)), replace=False)
        for gi in chosen:
            a_g = int(table[a, outside[gi]])
            across_dists.append(float(np.linalg.norm(reps[a] - reps[a_g])))
    within_mean = float(np.mean(within_dists)) if within_dists else float("nan")
    across_mean = float(np.mean(across_dists)) if across_dists else float("nan")
    invariance = {
        "within_coset_mean_shift": within_mean,
        "across_coset_mean_shift": across_mean,
        "within_over_across_ratio": (
            within_mean / across_mean if across_mean > _TOL else float("nan")
        ),
        "source": source,
    }

    patching: dict[str, Any] | str
    if isinstance(model, OneLayerTransformer):
        b_choices = rng.choice(group.order, size=min(group.order, 8), replace=False)
        patching = {}
        for label, movers in (("within", within_gen), ("across", outside)):
            clean_rows: list[list[int]] = []
            corrupt_rows: list[list[int]] = []
            for a in range(group.order):
                mover = movers[int(rng.integers(len(movers)))]
                a_moved = int(table[a, mover])
                for b in b_choices:
                    clean_rows.append([a, int(b), group.order])
                    corrupt_rows.append([a_moved, int(b), group.order])
            clean_tokens = torch.tensor(clean_rows, dtype=torch.long)
            corrupt_tokens = torch.tensor(corrupt_rows, dtype=torch.long)
            patching[label] = _path_patch_recovery(model, clean_tokens, corrupt_tokens)
    else:
        patching = UNDEFINED

    return {
        "instrument": "coset-patching",
        "rung": 4,
        "subgroup_index": target.subgroup_index,
        "coset_index": target.coset_index,
        "representation_invariance": invariance,
        "path_patching": patching,
    }


# ---------------------------------------------------------------------------
# Record builder: the coset arm + I-15 for one model
# ---------------------------------------------------------------------------


def coset_arm(
    model: GroupModel,
    group: FiniteGroup,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    n_random: int = 16,
    n_null: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """The full I-17/I-18/I-19 coset arm for one model, or one of two non-measured
    records when there is no nontrivial core-free subgroup to run on:

    * *structural UNDEFINED* -- subgroups were examined and none beyond the
      trivial one is core-free (Q32 and its family -- the D32/QD32-only scoping,
      plan §2). ``Ind_1^G 1`` is the regular representation, so the coset account
      has no distinct prediction: a theorem. The record carries
      ``undefined_reason`` and the minimal core-free index.
    * *artifact-incomplete* -- the group's artifact was exported without any
      subgroup data, so whether a core-free subgroup exists is simply unknown. The
      record is marked ``status: "skipped"`` with a ``skip_reason``; it makes NO
      theorem claim, so a subgroup-less artifact can never masquerade as a
      structural UNDEFINED verdict.

    ``n_subgroups_examined`` is carried in every case so a reviewer can tell an
    examined-and-empty verdict from an unexamined one."""
    library = template_library(group)
    target = coset_target(group)
    if target is None:
        record: dict[str, Any] = {
            "instrument": "coset-arm",
            "coset_defined": False,
            "n_subgroups_examined": library.n_subgroups_examined,
            "min_corefree_index": library.min_corefree_index,
            "i17_coset_collapse": UNDEFINED,
            "i18_coset_subspace_ablation": UNDEFINED,
            "i19_coset_patching": UNDEFINED,
        }
        if library.artifact_incomplete:
            record["status"] = "skipped"
            record["skip_reason"] = (
                "artifact incomplete: this group's artifact was exported without "
                "subgroup data, so the coset account cannot be evaluated. This is "
                "NOT the structural UNDEFINED theorem; re-export the artifact with "
                "--include-subgroups to measure the coset arm"
            )
        else:
            record["undefined_reason"] = (
                "no nontrivial core-free subgroup: Ind_1^G 1 is the regular "
                "representation, so the coset account has no distinct prediction "
                "(the D32/QD32-only scoping; Q32 is UNDEFINED by design)"
            )
        return record
    return {
        "instrument": "coset-arm",
        "coset_defined": True,
        "n_subgroups_examined": library.n_subgroups_examined,
        "subgroup_index": target.subgroup_index,
        "subgroup_order": int(target.subgroup.size),
        "coset_index": target.coset_index,
        "corefree_subgroups_at_min_index": target.count_at_min_index,
        "occupied_blocks": list(target.occupied_blocks),
        "i17_coset_collapse": coset_collapse_probe(model, group, target, n_null=n_null, seed=seed),
        "i18_coset_subspace_ablation": coset_subspace_ablation(
            model, group, target, tokens, targets, n_random=n_random, seed=seed
        ),
        "i19_coset_patching": coset_patching(model, group, target, seed=seed),
    }


# ---------------------------------------------------------------------------
# Run-directory record builder (offline, dip-aware checkpoint selection)
# ---------------------------------------------------------------------------


def measure_coset_run(
    run_dir: Path,
    *,
    metric: str = "val/accuracy",
    threshold: float = 0.99,
    n_random: int = 16,
    n_null: int = 500,
) -> dict[str, Any]:
    """The C2 mechanism measurement for one run: the I-17/I-18/I-19 coset arm plus
    the I-15 isotypic-block usage check, all scored on the transpose-unleaked
    held-out set, behind the dip-aware checkpoint rule.

    Offline and reproducible: no network, no W&B, no Sage/GAP. A run whose
    dip-aware selection finds no stable checkpoint (a censored/never-grokked
    seed) returns ``status: "skipped"`` with the selection record, never a
    silently-analysed unstable model. Provenance pins the manifest hashes, the
    checkpoint sha256, this module's sha256, and the group-artifact file the run
    is analysed against (its relative path and sha256) -- the one input that
    flips the coset-arm verdict between measured, structural UNDEFINED and
    artifact-incomplete.
    """
    import yaml

    from ..config import validate_config
    from ..groups.catalog import resolve_group
    from ..groups.data import artifact_path
    from ..manifest import get_git_commit, read_manifest
    from ..training.trainer import build_model
    from .checkpoints import select_checkpoint
    from .report import file_sha256, instrument_code_hashes

    manifest = read_manifest(run_dir)
    config = validate_config(yaml.safe_load((run_dir / "resolved_config.yaml").read_text()))
    selection = select_checkpoint(run_dir, metric=metric, threshold=threshold)
    # The group artifact is resolved from the same (order, index) and
    # GROUP_ARTIFACTS_DIR that ``resolve_group`` below reads, so hashing it here
    # pins exactly the file the verdict depends on.
    art_path = artifact_path(config.data.group.order, config.data.group.index)
    repo_root = Path(__file__).resolve().parents[3]
    try:
        art_rel = str(art_path.relative_to(repo_root))
    except ValueError:
        art_rel = art_path.name
    record: dict[str, Any] = {
        "instrument": "coset",
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

    record["status"] = "measured"
    record["provenance"]["checkpoint_sha256"] = file_sha256(selection.path)
    record["n_test"] = int(tokens.shape[0])
    record["isotypic_block_ablation"] = isotypic_block_ablation(
        model, group, tokens, targets, n_random=n_random, seed=config.seed
    )
    record["coset_arm"] = coset_arm(
        model, group, tokens, targets, n_random=n_random, n_null=n_null, seed=config.seed
    )
    return record


__all__ = [
    "ABLATION_MODES",
    "UNDEFINED",
    "CosetTarget",
    "coset_arm",
    "coset_collapse_probe",
    "coset_patching",
    "coset_subspace_ablation",
    "coset_target",
    "correct_mask",
    "isotypic_block_ablation",
    "left_representations",
    "measure_coset_run",
    "random_rank_projector",
    "unleaked_heldout",
]
