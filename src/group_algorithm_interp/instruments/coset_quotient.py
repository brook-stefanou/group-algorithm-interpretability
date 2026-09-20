"""I-20: the coset-quotient route decider (non-degenerate coset signature).

This is the instrument that *decides* the coset question on the D32/QD32 case
study where I-18 (:func:`coset.coset_subspace_ablation`) is undecidable. The
obstacle I-18 hits is degeneracy of *size*: it ablates the ``Ind_H^G 1`` isotypic
subspace of the minimal-index core-free ``H``, which on D32 has rank 30 of 32
(every core-free subgroup of D32 is an order-2 reflection subgroup of index 16,
and all of them give the same rank-30 support -- see ``tests/test_coset.py`` and
the exploration behind this module). Ablating rank 30 removes almost the whole
representation, so a rank-matched random subspace does the same damage and the
control has no resolving power. There is no non-degenerate signature *inside the
core-free ``Ind_H^G 1`` framing* on D32.

Two moves make this instrument non-degenerate.

1. **Use a NORMAL subgroup, not a core-free one.** "The model tracks which coset
   of ``H`` each input lies in and combines cosets" is only a coherent algorithm
   when the cosets themselves form a group -- i.e. when ``H = N`` is *normal* and
   ``Q = G/N`` is the quotient. For a non-normal core-free ``H`` the left cosets
   carry no group structure to "combine", so I-18's target is the wrong object
   for the sentence it is meant to test. The quotient map ``q: G -> Q`` is the
   real locus of a coset route.

2. **Work in the rank-``[G:N]`` quotient subspace and read a coset-level target.**
   The functions on ``G`` that are constant on each coset of ``N`` -- the
   pullback ``q^* C[Q]`` of the regular representation of the quotient -- form a
   subspace of the ``|G|``-dimensional element-function space of rank exactly
   ``[G:N] = |Q|``. For D32's index-2 rotation subgroup that rank is **2**, not
   30. The coset-route claim becomes two non-degenerate, low-rank measurements
   on this subspace, each with a rank- and norm-matched random-subspace control:

   * **sufficiency (rung 1->3)** -- *restrict* the shared element embedding to the
     rank-``[G:N]`` quotient subspace (replace each element's embedding row by its
     coset mean, ``E -> P_quot E``) and measure the model's output-*coset*
     accuracy ``q(argmax) == q(a b)``. A coset route computes ``q(a b) =
     q(a) q(b)`` from the coset coordinate alone, so its coset accuracy survives
     the restriction; the readout target is the ``|Q|``-way coset, matched to the
     subspace's information content, not the full ``|G|``-way answer.
   * **necessity (rung 3)** -- *ablate* the same rank-``[G:N]`` subspace
     (``E -> E - P_quot E``) and measure the drop in output-coset accuracy. If the
     coset coordinate lives there, ablation drops coset accuracy to the
     random-control floor.

   Neither edit touches a near-full-rank subspace, so the matched random control
   retains full resolving power -- that is the whole point.

**What this decides, and what it deliberately does not.** A large sufficiency
effect with a matching necessity effect decides, non-degenerately, that the model
*organises its computation around the cosets of* ``N`` -- it carries and uses a
low-rank quotient coordinate to compute the output's coset. That is the coset
question the rank-30 test could not answer. It does **not**, on D32, separate the
coset account from the representation (GCR) account: D32's only clean quotient
coordinate is the index-2 rotation parity, whose subspace is exactly the
trivial + sign linear-character blocks (verified: the rank-2 projector overlaps
blocks 0 and 3 with weight 1), and those linear irreps are ones the
representation account uses too. The carriers coincide, which is precisely why
:func:`template_divergence.degeneracy_screen` returns ``UNDEFINED`` for the
occupancy-side coset-vs-GCR comparison on D32. This instrument therefore reports
a *positive coset-organisation* verdict with an explicit
:data:`DISCRIMINATION_NOTE`; it is a decider for "does the model use the coset
route", not for "coset rather than GCR".

Estimation-first throughout: effect sizes against a matched random-subspace null
with bootstrap CIs, never a thresholded verdict. Random draws use a private
NumPy generator so the global RNG stream is never perturbed. Hook-free: every
intervention swaps ``W_E`` for an edited copy and restores it (:func:`coset._swapped_embedding`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from ..groups.group import FiniteGroup
from ..model import GroupModel
from .coset import (
    _distribution_summary,
    _swapped_embedding,
    random_rank_projector,
)

_TOL = 1e-9

DISCRIMINATION_NOTE = (
    "Decides coset-route ORGANISATION (does the model compute the output coset "
    "from a low-rank quotient coordinate), NOT coset-vs-GCR. On D32 the only "
    "clean quotient coordinate is the index-2 rotation parity, whose rank-2 "
    "subspace is exactly the trivial+sign linear-character blocks -- irreps the "
    "representation account also uses -- so a positive result here cannot be "
    "attributed against GCR (the carriers coincide; cf. the template-divergence "
    "degeneracy screen)."
)


# ---------------------------------------------------------------------------
# Normal-subgroup quotient structure (read from the artifact's own subgroups)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalQuotient:
    """One proper, nontrivial quotient ``Q = G/N`` for a normal subgroup ``N``
    exported in the artifact, plus the rank-``[G:N]`` coset-mean projector on the
    element-function space and the element->coset labelling the coset-level
    readout scores against."""

    subgroup_index: int
    subgroup_order: int
    coset_index: int  # [G:N] = |Q| = rank of the quotient subspace
    cosets: tuple[np.ndarray, ...]
    element_to_coset: np.ndarray  # [|G|] -> coset id
    quotient_projector: np.ndarray  # [|G|, |G|], rank [G:N]


def _identity_index(table: np.ndarray) -> int:
    n = table.shape[0]
    idx = np.arange(n)
    candidates = np.flatnonzero(np.all(table == idx[None, :], axis=1))
    if candidates.size == 0:
        raise ValueError("Cayley table has no identity element")
    return int(candidates[0])


def _inverses(table: np.ndarray, identity: int) -> np.ndarray:
    n = table.shape[0]
    inv = np.empty(n, dtype=np.int64)
    for g in range(n):
        inv[g] = int(np.flatnonzero(table[g] == identity)[0])
    return inv


def _is_normal(table: np.ndarray, subgroup: np.ndarray, inv: np.ndarray) -> bool:
    """``x N x^-1 == N`` for every ``x`` -- pure Cayley-table index arithmetic."""
    members = {int(x) for x in subgroup.tolist()}
    for x in range(table.shape[0]):
        conjugate = {int(table[table[x, h], inv[x]]) for h in members}
        if conjugate != members:
            return False
    return True


def _coset_mean_projector(cosets: Sequence[np.ndarray], order: int) -> np.ndarray:
    """The orthogonal projector onto functions constant on each coset -- the
    pullback ``q^* C[Q]``. Block-constant averaging: ``P[a, a'] = 1/|coset|`` when
    ``a`` and ``a'`` share a coset, else 0. Symmetric idempotent of rank equal to
    the number of cosets ``[G:N]``."""
    projector = np.zeros((order, order), dtype=np.float64)
    for coset in cosets:
        members = np.asarray(coset, dtype=np.int64)
        projector[np.ix_(members, members)] = 1.0 / members.size
    return projector


def normal_quotients(group: FiniteGroup) -> tuple[NormalQuotient, ...]:
    """Every proper, nontrivial ``Q = G/N`` over the artifact's exported
    subgroups, ordered by increasing ``[G:N]`` (coarsest quotient first). Empty
    when no subgroup data was exported or the group is simple with no proper
    normal subgroup of the right size (e.g. a group of prime order)."""
    table = group.cayley_table
    identity = _identity_index(table)
    inv = _inverses(table, identity)
    quotients: list[NormalQuotient] = []
    for position, subgroup in enumerate(group.subgroups):
        members = np.asarray(subgroup, dtype=np.int64)
        size = int(members.size)
        if size <= 1 or size == group.order:
            continue  # trivial / whole group: no proper nontrivial quotient
        if not _is_normal(table, members, inv):
            continue
        cosets = group.cosets_for_subgroup(members)
        element_to_coset = np.full(group.order, -1, dtype=np.int64)
        for coset_id, coset in enumerate(cosets):
            element_to_coset[np.asarray(coset, dtype=np.int64)] = coset_id
        if int(element_to_coset.min()) < 0:
            raise ValueError("cosets do not partition the group")
        quotients.append(
            NormalQuotient(
                subgroup_index=position,
                subgroup_order=size,
                coset_index=group.order // size,
                cosets=tuple(np.asarray(c, dtype=np.int64) for c in cosets),
                element_to_coset=element_to_coset,
                quotient_projector=_coset_mean_projector(cosets, group.order),
            )
        )
    quotients.sort(key=lambda q: (q.coset_index, q.subgroup_index))
    return tuple(quotients)


# ---------------------------------------------------------------------------
# Behavioural readout: full-answer and output-coset accuracy under an edit
# ---------------------------------------------------------------------------


def _predictions(model: GroupModel, tokens: torch.Tensor) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(tokens.to(device))
    return logits[:, -1, :].argmax(dim=-1).cpu().numpy()


def _accuracies(
    pred: np.ndarray, targets: np.ndarray, element_to_coset: np.ndarray
) -> tuple[float, float]:
    """``(full_accuracy, coset_accuracy)`` -- ``argmax == a*b`` and
    ``q(argmax) == q(a*b)``."""
    if pred.size == 0:
        return float("nan"), float("nan")
    full = float((pred == targets).mean())
    coset = float((element_to_coset[pred] == element_to_coset[targets]).mean())
    return full, coset


def _edit_and_score(
    model: GroupModel,
    element_embed_edit: np.ndarray,
    tokens: torch.Tensor,
    targets_np: np.ndarray,
    element_to_coset: np.ndarray,
) -> tuple[float, float]:
    with _swapped_embedding(model, element_embed_edit):
        pred = _predictions(model, tokens)
    return _accuracies(pred, targets_np, element_to_coset)


def _matched_random_component(
    element_embed: np.ndarray,
    rank: int,
    target_norm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """A random rank-``rank`` projection of the embedding, rescaled to
    ``target_norm`` -- the rank- and norm-matched control component."""
    projector = random_rank_projector(element_embed.shape[0], rank, rng)
    component = projector @ element_embed
    norm = float(np.linalg.norm(component))
    if norm > _TOL:
        component = component * (target_norm / norm)
    return component


# ---------------------------------------------------------------------------
# I-20: the per-quotient probe (sufficiency + necessity) and the arm
# ---------------------------------------------------------------------------


def coset_quotient_probe(
    model: GroupModel,
    group: FiniteGroup,
    quotient: NormalQuotient,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    n_random: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """The two non-degenerate readouts for one quotient ``Q = G/N``.

    **Sufficiency** restricts the element embedding to the rank-``[G:N]`` coset
    subspace (``E -> P_quot E``, each element replaced by its coset mean) and
    reports the output-coset accuracy against a rank- and norm-matched
    random-subspace control. **Necessity** ablates the same subspace
    (``E -> E - P_quot E``) and reports the output-coset accuracy against the same
    style of control. Both effects are estimation-first: the control is a
    distribution with a bootstrap CI, and the effect is the signed gap to its
    mean. No verdict string is emitted."""
    element_embed = model.W_E.detach().cpu().numpy()[: group.order].astype(np.float64)
    targets_np = targets.cpu().numpy()
    labels = quotient.element_to_coset
    rank = quotient.coset_index
    projector = quotient.quotient_projector
    kept = projector @ element_embed
    kept_norm = float(np.linalg.norm(kept))
    rng = np.random.default_rng(seed)

    # Sufficiency: keep only the coset subspace.
    suff_full, suff_coset = _edit_and_score(model, kept, tokens, targets_np, labels)
    suff_random: list[float] = []
    for _ in range(n_random):
        component = _matched_random_component(element_embed, rank, kept_norm, rng)
        _, coset_acc = _edit_and_score(model, component, tokens, targets_np, labels)
        suff_random.append(coset_acc)
    suff_summary = _distribution_summary(suff_random)

    # Necessity: remove the coset subspace.
    abl_full, abl_coset = _edit_and_score(model, element_embed - kept, tokens, targets_np, labels)
    abl_random: list[float] = []
    for _ in range(n_random):
        component = _matched_random_component(element_embed, rank, kept_norm, rng)
        _, coset_acc = _edit_and_score(model, element_embed - component, tokens, targets_np, labels)
        abl_random.append(coset_acc)
    abl_summary = _distribution_summary(abl_random)

    return {
        "subgroup_index": quotient.subgroup_index,
        "subgroup_order": quotient.subgroup_order,
        "coset_index": quotient.coset_index,
        "quotient_rank": rank,
        "chance_coset_accuracy": 1.0 / quotient.coset_index,
        "sufficiency": {
            "restricted_coset_accuracy": suff_coset,
            "restricted_full_accuracy": suff_full,
            "random_subspace_coset_accuracy": suff_summary,
            "coset_accuracy_over_random": suff_coset - suff_summary["mean"],
        },
        "necessity": {
            "ablated_coset_accuracy": abl_coset,
            "ablated_full_accuracy": abl_full,
            "random_subspace_coset_accuracy": abl_summary,
            # positive => removing THIS subspace costs more coset accuracy than
            # removing a matched random one, i.e. the coset coordinate lives here.
            "coset_accuracy_drop_over_random": abl_summary["mean"] - abl_coset,
        },
    }


def coset_quotient_route(
    model: GroupModel,
    group: FiniteGroup,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    n_random: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """I-20: the coset-quotient route decider over every proper nontrivial
    ``Q = G/N``. Runs :func:`coset_quotient_probe` for each and marks the
    *decisive* quotient -- the one with the largest sufficiency effect
    (``coset_accuracy_over_random``) -- as the headline. Returns a ``defined:
    False`` record (never a number) when the artifact exports no proper normal
    subgroup to run on."""
    quotients = normal_quotients(group)
    if not quotients:
        return {
            "instrument": "coset-quotient-route",
            "rung": 3,
            "defined": False,
            "reason": (
                "no proper nontrivial normal subgroup exported for this group, so "
                "there is no quotient Q = G/N for a coset route to combine cosets in"
            ),
            "discrimination_note": DISCRIMINATION_NOTE,
        }
    baseline = _predictions(model, tokens)
    baseline_full = float((baseline == targets.cpu().numpy()).mean())
    probes = [
        coset_quotient_probe(model, group, q, tokens, targets, n_random=n_random, seed=seed)
        for q in quotients
    ]
    decisive = max(probes, key=lambda p: p["sufficiency"]["coset_accuracy_over_random"])
    return {
        "instrument": "coset-quotient-route",
        "rung": 3,
        "defined": True,
        "n_test": int(tokens.shape[0]),
        "clean_full_accuracy": baseline_full,
        "n_quotients": len(quotients),
        "decisive_subgroup_index": decisive["subgroup_index"],
        "quotients": probes,
        "discrimination_note": DISCRIMINATION_NOTE,
    }


__all__ = [
    "DISCRIMINATION_NOTE",
    "NormalQuotient",
    "coset_quotient_probe",
    "coset_quotient_route",
    "normal_quotients",
]
