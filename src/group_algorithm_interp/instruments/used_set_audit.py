"""The used-set causal-sufficiency audit: completeness + minimality of a
model's "used" isotypic-block set as its computational circuit.

The thesis this study is built on -- the network uses a near-minimal
*faithful* subset of irreducible representations -- currently rests on three
separate legs: occupancy (:mod:`occupancy`, the set is occupied),
:func:`coset.isotypic_block_ablation` / I-15 (the set is causally used, each
block's ablation costs more than a matched random subspace), and
:func:`template_divergence.used_set_minimality` (the set is near-minimal
faithful, compared against the cost-minimum faithful set). None of the three
asks the one question that turns "near-minimal faithful subset" into a
circuit claim: restricted to *only* the used blocks, does the model still
compute the group operation -- and does removing any one of them break it?
That is COMPLETENESS (the used set is causally sufficient) and MINIMALITY
(each block in it is causally necessary), the same two criteria
:func:`audit.circuit_audit` (I-34) already reports for a neuron-level
circuit, applied here to an isotypic-block subspace of the shared input
embedding instead of an ``mlp_post`` neuron set.

**Machinery reused, not reinvented.** Every ablation here is the existing
isotypic-block embedding edit from ``coset.py`` (I-15/I-18): a block/subspace
projector applied to the element-embedding rows of ``W_E``, compared against
a rank- and norm-matched random subspace
(:func:`coset._ablation_flip_record`). Completeness ablates the projector of
every block *outside* the used set (the union's projector, exactly as
:func:`coset.coset_subspace_ablation` ablates the coset target's subspace);
minimality is :func:`coset.isotypic_block_ablation` called with
``block_indices`` restricted to the used set, unchanged. Faithfulness reuses
:func:`audit.kl_divergence` on the model's read-position logits before and
after the completeness ablation -- the same KL-based faithfulness criterion
I-34 reports for its neuron circuit, wired here to the used-set subspace as
the circuit definition.

**Degeneracy guard.** A used set spanning almost every isotypic dimension
makes completeness trivially hold: there is nothing left in the complement to
ablate, so "restricting to the used set preserves accuracy" is vacuous, not
evidence of sufficiency. :data:`NEAR_FULL_RANK_FRACTION` (default ``0.9``)
gates a descriptive ``degenerate`` flag exactly as
``template_divergence.MAX_COSET_SUPPORT_FRACTION`` flags the mirror-image
hazard for the coset target -- named, overridable, and reported alongside the
raw rank numbers rather than silently rewriting the record.

**Estimation-first.** Every number here is an effect size (an accuracy, a
KL, a flip fraction), most with the matched-random-subspace control I-15
already provides. The one categorical field, ``verdict``, is a descriptive
classification exactly like ``ReplicationRecord.architecture_conditional`` or
``DegeneracyScreen.verdict`` elsewhere in this package -- a convenience label
over thresholds that are named constants, never a hidden significance gate --
and it is never reported without the raw numbers it summarises sitting
beside it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .. import stats
from ..groups.group import FiniteGroup
from ..model import GroupModel
from .audit import kl_divergence
from .coset import _ablation_flip_record, _apply_mode, _swapped_embedding, correct_mask
from .coset import isotypic_block_ablation as _isotypic_block_ablation
from .occupancy import trivial_block_index

_TOL = 1e-9

#: The used set's rank must be at least this fraction of ``|G|`` for the
#: completeness ablation to be flagged :data:`degenerate` -- almost nothing
#: left in the complement to ablate, so a high retained accuracy is vacuous
#: rather than evidence of sufficiency. Mirrors
#: ``template_divergence.MAX_COSET_SUPPORT_FRACTION``'s role for the coset
#: target, in the opposite direction (there: too MUCH of the group in the
#: ablated target; here: too little left OUTSIDE the used set). Named and
#: overridable, never a bare literal.
NEAR_FULL_RANK_FRACTION = 0.9

#: The zero-mode ``flip_fraction_over_random`` a used-set block must exceed to
#: be classified "necessary" in the descriptive ``verdict`` field. Same value
#: and rationale as ``template_divergence.MIN_FLIP_OVER_RANDOM`` (I-15's own
#: matched-random control already screens out "any subspace this size does
#: this much damage", so a small positive margin above it is informative) --
#: defined locally rather than imported so this module's classification bar
#: does not silently move if the aggregation-context constant is retuned.
MIN_FLIP_OVER_RANDOM = 0.05

#: The completeness accuracy-retention fraction (ablated / clean) a
#: non-degenerate used set must reach to be classified "sufficient" in the
#: descriptive ``verdict`` field. A convenience cut for the label only; the
#: raw retention fraction is always reported alongside it.
SUFFICIENCY_RETENTION_FRACTION = 0.9


# ---------------------------------------------------------------------------
# Complement projector: the subspace ablated by "restrict to the used set"
# ---------------------------------------------------------------------------


def complement_blocks_and_rank(
    group: FiniteGroup, used_blocks: Sequence[int]
) -> tuple[tuple[int, ...], np.ndarray, int, int]:
    """The blocks OUTSIDE ``used_blocks``, their union projector and rank, and
    the used set's own rank. Returns ``(complement_blocks, complement_projector,
    complement_rank, used_rank)``.

    ``used_blocks`` may be empty (complement is everything, ``used_rank ==
    0``) or may cover every block (complement is empty, ``complement_rank ==
    0`` and ``complement_projector`` the zero matrix) -- both are valid,
    measured inputs, not errors: an empty used set is the "nothing measured
    as used" degenerate case, and a full used set is the near-full-rank
    degenerate case :data:`NEAR_FULL_RANK_FRACTION` flags."""
    n_blocks = len(group.isotypic_blocks)
    used = {int(b) for b in used_blocks}
    if used and (min(used) < 0 or max(used) >= n_blocks):
        raise ValueError(f"used_blocks out of range for {n_blocks} isotypic blocks: {used}")
    order = group.order
    complement = tuple(j for j in range(n_blocks) if j not in used)
    projector = np.zeros((order, order), dtype=np.float64)
    complement_rank = 0
    for j in complement:
        block = group.isotypic_blocks[j]
        projector = projector + np.asarray(block.projector, dtype=np.float64)
        complement_rank += block.block_rank
    used_rank = order - complement_rank
    return complement, projector, complement_rank, used_rank


# ---------------------------------------------------------------------------
# Completeness: ablate everything OUTSIDE the used set
# ---------------------------------------------------------------------------


def completeness_ablation(
    model: GroupModel,
    group: FiniteGroup,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    used_blocks: Sequence[int],
    *,
    n_random: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Ablate the union of every isotypic block OUTSIDE ``used_blocks`` from
    the shared input embedding and report the retained behavioural accuracy
    per mode (zero/mean/resample), against a matched random subspace of the
    same complement rank -- exactly :func:`coset.coset_subspace_ablation`'s
    machinery (:func:`coset._ablation_flip_record`), pointed at the used
    set's complement rather than a coset target.

    A high ``ablated_accuracy`` (close to ``clean_accuracy``) is the
    completeness reading: the used set is causally SUFFICIENT to reproduce
    the model's behaviour. See :func:`used_set_audit` for the degeneracy
    guard on a used set whose complement is nearly empty."""
    complement, projector, complement_rank, used_rank = complement_blocks_and_rank(
        group, used_blocks
    )
    clean = correct_mask(model, tokens, targets)
    rng = np.random.default_rng(seed)
    record = _ablation_flip_record(
        model,
        projector,
        tokens,
        targets,
        clean,
        rank=complement_rank,
        n_random=n_random,
        rng=rng,
    )
    clean_accuracy = float(clean.mean()) if clean.size else float("nan")
    retention_by_mode = {
        mode: (
            record["modes"][mode]["ablated_accuracy"] / clean_accuracy
            if clean_accuracy > _TOL
            else float("nan")
        )
        for mode in record["modes"]
    }
    return {
        "complement_blocks": list(complement),
        "complement_rank": complement_rank,
        "used_rank": used_rank,
        "clean_accuracy": clean_accuracy,
        "n_test": int(clean.size),
        "accuracy_retention_fraction": retention_by_mode,
        **record,
    }


# ---------------------------------------------------------------------------
# Minimality: ablate each block INSIDE the used set in turn
# ---------------------------------------------------------------------------


def minimality_ablation(
    model: GroupModel,
    group: FiniteGroup,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    used_blocks: Sequence[int],
    *,
    n_random: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Ablate each block in ``used_blocks`` individually and report the
    behavioural drop, against a matched random subspace of the same block
    rank -- :func:`coset.isotypic_block_ablation` (I-15) restricted to the
    used set exactly as given, unchanged. A block whose ablation costs no
    more than its matched random control is occupied-in-the-used-set but not
    shown to be necessary to it."""
    used = tuple(sorted({int(b) for b in used_blocks}))
    return _isotypic_block_ablation(
        model, group, tokens, targets, block_indices=used, n_random=n_random, seed=seed
    )


# ---------------------------------------------------------------------------
# Faithfulness: KL(model, restricted circuit) -- audit.py's I-34 criterion,
# wired to the used-set subspace as the circuit definition
# ---------------------------------------------------------------------------


def _read_logits(model: GroupModel, tokens: torch.Tensor) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(tokens.to(device))
    return logits[:, -1, :].detach().cpu().numpy()


def _mean_ci(values: Sequence[float]) -> dict[str, float]:
    """Mean and, at 2+ values, a 95% bootstrap CI -- the same convention
    ``audit._mean_ci`` uses, kept local so this module does not depend on
    ``audit.py``'s private helper."""
    arr = [float(v) for v in values]
    if not arr:
        raise ValueError("_mean_ci requires at least one value, got an empty sequence")
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": float(len(arr))}
    low, high = stats.bootstrap_ci(arr)
    return {"mean": mean, "ci_low": low, "ci_high": high, "n": float(len(arr))}


def restricted_circuit_faithfulness(
    model: GroupModel,
    group: FiniteGroup,
    tokens: torch.Tensor,
    used_blocks: Sequence[int],
) -> dict[str, Any]:
    """KL from the full model's read-position output to the "restricted
    circuit" -- the model with everything outside ``used_blocks`` zero-ablated
    from the shared embedding -- on ``tokens``, plus argmax agreement.
    :func:`audit.kl_divergence`, wired here to the used-set subspace as the
    circuit definition exactly as :func:`audit.circuit_audit`'s own
    faithfulness criterion is wired to a neuron set."""
    _, projector, _, _ = complement_blocks_and_rank(group, used_blocks)
    order = group.order
    baseline_logits = _read_logits(model, tokens)
    element_embed = model.W_E.detach().cpu().numpy()[:order].astype(np.float64)
    component = projector @ element_embed
    restricted_embed = _apply_mode(element_embed, component, "zero", np.random.default_rng(0))
    with _swapped_embedding(model, restricted_embed):
        restricted_logits = _read_logits(model, tokens)
    kl = kl_divergence(baseline_logits, restricted_logits)
    agreement = (baseline_logits.argmax(axis=1) == restricted_logits.argmax(axis=1)).astype(
        np.float64
    )
    return {
        "kl_model_to_restricted_circuit": _mean_ci(kl.tolist()),
        "argmax_agreement": _mean_ci(agreement.tolist()),
        "note": (
            "KL from the full model to the used-set-restricted (outside "
            "zero-ablated) output at the read position; the used-set analogue "
            "of audit.circuit_audit's faithfulness criterion"
        ),
    }


# ---------------------------------------------------------------------------
# The combined record + descriptive verdict
# ---------------------------------------------------------------------------


def _verdict(
    *,
    degenerate: bool,
    retention_fraction: float,
    n_used: int,
    n_necessary: int,
    sufficiency_retention_fraction: float,
) -> str:
    if n_used == 0:
        return "empty_used_set"
    if degenerate:
        return "degenerate_uninformative"
    sufficient = np.isfinite(retention_fraction) and retention_fraction >= (
        sufficiency_retention_fraction
    )
    minimal = n_necessary == n_used
    if sufficient and minimal:
        return "sufficient_and_minimal_circuit"
    if sufficient and not minimal:
        return "sufficient_but_not_fully_minimal"
    if not sufficient and minimal:
        return "minimal_but_not_sufficient"
    return "neither_sufficient_nor_minimal"


def used_set_audit(
    model: GroupModel,
    group: FiniteGroup,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    used_blocks: Sequence[int],
    *,
    n_random: int = 16,
    seed: int = 0,
    near_full_rank_fraction: float = NEAR_FULL_RANK_FRACTION,
    min_flip_over_random: float = MIN_FLIP_OVER_RANDOM,
    sufficiency_retention_fraction: float = SUFFICIENCY_RETENTION_FRACTION,
    minimality_mode: str = "zero",
) -> dict[str, Any]:
    """The causal-sufficiency audit of a model's "used" isotypic-block set
    (rung 5, the missing piece completing the near-minimal-faithful-subset
    thesis): completeness (restrict to the used set, does behaviour survive),
    minimality (ablate each block inside it, does behaviour break), and
    faithfulness (KL from the model to the restricted circuit).

    ``used_blocks`` is supplied by the caller -- typically
    :func:`template_divergence.causally_used_blocks` applied to an I-15
    record, or an occupancy-derived set -- and is never re-derived here.

    Returns a record with ``used_blocks``, ``used_rank``/``used_rank_fraction``,
    ``complement_blocks``/``complement_rank``, the ``degenerate`` flag (see
    :data:`NEAR_FULL_RANK_FRACTION`), ``clean_accuracy``, ``completeness``
    (:func:`completeness_ablation`'s record), ``minimality``
    (:func:`minimality_ablation`'s record), ``faithfulness``
    (:func:`restricted_circuit_faithfulness`'s record), and a descriptive
    ``verdict`` -- one of ``"empty_used_set"``, ``"degenerate_uninformative"``,
    ``"sufficient_and_minimal_circuit"``, ``"sufficient_but_not_fully_minimal"``,
    ``"minimal_but_not_sufficient"``, or ``"neither_sufficient_nor_minimal"``.
    ``verdict`` is a convenience classification over the named threshold
    constants above, always reported beside the raw numbers it summarises,
    never in place of them (estimation-first: no hidden significance gate).
    """
    used = tuple(sorted({int(b) for b in used_blocks}))
    n_blocks = len(group.isotypic_blocks)
    order = group.order
    clean = correct_mask(model, tokens, targets)
    clean_accuracy = float(clean.mean()) if clean.size else float("nan")

    _, _, complement_rank, used_rank = complement_blocks_and_rank(group, used)
    used_rank_fraction = used_rank / order if order else float("nan")
    degenerate = used_rank_fraction >= near_full_rank_fraction

    completeness = completeness_ablation(
        model, group, tokens, targets, used, n_random=n_random, seed=seed
    )
    minimality = minimality_ablation(
        model, group, tokens, targets, used, n_random=n_random, seed=seed
    )
    faithfulness = restricted_circuit_faithfulness(model, group, tokens, used)

    retention_fraction = completeness["accuracy_retention_fraction"][minimality_mode]
    n_necessary = 0
    for block in minimality["blocks"]:
        # A single-model measurement (not aggregated across seeds), so this is
        # the raw scalar `coset._ablation_flip_record` reports -- not the
        # aggregated `{"mean": ...}` dict `template_divergence.causally_used_
        # blocks` reads off `scripts/measure_isotypic_ablation.py`'s per-cell
        # aggregation.
        flip = block["modes"][minimality_mode]["flip_fraction_over_random"]
        if flip is not None and np.isfinite(flip) and flip > min_flip_over_random:
            n_necessary += 1

    verdict = _verdict(
        degenerate=degenerate,
        retention_fraction=retention_fraction,
        n_used=len(used),
        n_necessary=n_necessary,
        sufficiency_retention_fraction=sufficiency_retention_fraction,
    )

    return {
        "instrument": "used-set-causal-sufficiency-audit",
        "rung": 5,
        "trivial_block_index": trivial_block_index(group),
        "n_blocks_total": n_blocks,
        "used_blocks": list(used),
        "used_rank": used_rank,
        "used_rank_fraction": used_rank_fraction,
        "complement_rank": complement_rank,
        "degenerate": degenerate,
        "near_full_rank_fraction": near_full_rank_fraction,
        "clean_accuracy": clean_accuracy,
        "n_test": int(clean.size),
        "completeness": completeness,
        "minimality": minimality,
        "faithfulness": faithfulness,
        "minimality_mode": minimality_mode,
        "n_necessary_blocks": n_necessary,
        "n_used_blocks": len(used),
        "verdict": verdict,
        "verdict_note": (
            "descriptive classification over named threshold constants "
            "(NEAR_FULL_RANK_FRACTION, MIN_FLIP_OVER_RANDOM, "
            "SUFFICIENCY_RETENTION_FRACTION), never a substitute for the raw "
            "accuracy/flip numbers reported alongside it"
        ),
    }


__all__ = [
    "MIN_FLIP_OVER_RANDOM",
    "NEAR_FULL_RANK_FRACTION",
    "SUFFICIENCY_RETENTION_FRACTION",
    "complement_blocks_and_rank",
    "completeness_ablation",
    "minimality_ablation",
    "restricted_circuit_faithfulness",
    "used_set_audit",
]
