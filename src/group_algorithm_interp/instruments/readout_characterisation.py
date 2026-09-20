"""The positive readout characterisation: does the read-position logit need
more than the scalar character?

Two existing readout results are each negative or class-function-caveated on
their own:

* ``gcr_matmul.py`` (the matrix-product test) shows the post-MLP *neurons* do
  **not** carry the full shared-index matrix product ``rho(a) rho(b)`` -- the
  generic bilinear alternative is not beaten by the constrained matrix-product
  form on every occupied block, so GCR's "the network forms the intermediate
  matrix ``rho(ab)``" claim is not credited at the neuron level.
* ``probes.gcr_character_readout_instrument`` shows the read-position
  *logits* fit the character form ``Phi_rho(a, b, c) = Re tr(rho(a) rho(b)
  rho(c^-1))``, but that design's own caveat is that ``Phi_rho`` is a class
  function of the single product ``a*b`` (for fixed ``c``), so a high raw
  held-out FVE is consistent with *any* correct algorithm and is not evidence
  for the character specifically.

This module supplies the positive characterisation that unifies the two: the
output stage reads off only the *scalar character* ``Re tr(rho(a b)
rho(c^-1))`` per occupied irrep, and never needs the full ``d x d`` matrix
product ``rho(a b) rho(c^-1))`` entrywise. It runs the same nested,
held-out-scored comparison the rest of the mechanism arm uses (I-20's
:func:`probes.functional_form_fit`), between two designs over the read-
position logits:

* **character-only** -- one column per candidate irrep, exactly
  :func:`probes.gcr_character_design`'s ``Phi_rho`` (the trace).
* **full-matrix-entry** -- ``2 * d**2`` columns per candidate irrep (Re and
  Im of every entry, not summed): the *un-contracted* matrix product
  ``M(a, b, c) = rho(a b) @ rho(c^-1)`` (:func:`gcr_matrix_entry_design`).

Because ``Re tr(M) = sum_i Re(M_ii)`` is exactly the sum of the full design's
``(Re, i == j)`` columns, the character-only design's single column lies in
the full-matrix-entry design's column span for every candidate irrep: the two
forms are a genuinely nested :class:`probes.FunctionalForm` pair (the full
form can never fit worse in sample), so the only question the held-out
comparison can honestly answer is whether the extra ``2 * d**2 - 1``
degrees of freedom per irrep buy anything held out.

CAVEAT, carried through to the record (read before citing raw FVE): ``M(a, b,
c)`` -- and therefore every entry of it, including the trace -- is a
function of the single product ``a*b`` once ``c`` is fixed, so a high raw
held-out FVE from *either* design is consistent with any algorithm that
computes the group product correctly, not only GCR, and proves nothing about
GCR specifically on its own (the same caveat ``gcr_character_design`` already
carries). The honest, load-bearing statistics are:

* ``held_out_fve_gain_full_minus_character`` -- does the strictly larger
  full-matrix-entry span buy anything held out over the character-only span
  it contains; near zero (within ``tie_tol``) is the positive "the readout
  needs only the scalar character" result, a clear positive gap means the
  readout uses more than the character;
* ``minimal_irrep_set`` -- the sparse subset of irreps the character-only
  readout needs to reach ``target_fve`` (reusing
  :func:`probes.gcr_character_readout_instrument`'s own greedy search
  unchanged, so the two instruments report the same minimal set).

Raw ``character_only_held_out_fve`` / ``full_matrix_entry_held_out_fve`` are
reported for context but are never, on their own, evidence either way.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from ..groups.group import FiniteGroup
from ..model import GroupModel
from .probes import (
    UNDEFINED,
    FunctionalForm,
    functional_form_fit,
    gcr_candidate_irreps,
    gcr_character_form,
    gcr_character_readout_instrument,
    identity_index,
    inverses,
)

__all__ = [
    "UNDEFINED",
    "gcr_matrix_entry_design",
    "gcr_matrix_entry_form",
    "readout_characterisation_instrument",
]


def gcr_matrix_entry_design(group: FiniteGroup, irrep_indices: Sequence[int]) -> np.ndarray:
    """The full-matrix-entry readout's design matrix: shape ``[order, order,
    order, sum_k 2 * d_k**2]`` where ``d_k`` is the degree of
    ``group.irreps[irrep_indices[k]]``.

    For each candidate irrep, the ``2 * d**2`` columns are the real and
    imaginary parts of every entry of the *un-contracted* matrix product
    ``M(a, b, c) = rho(a b) @ rho(c^-1)`` -- the same product whose trace is
    :func:`probes.gcr_character_design`'s single scalar column, but here kept
    as ``d x d`` separate entries rather than summed along the diagonal.
    ``rho(a b)`` is computed the same way ``gcr_character_design`` does (via
    the homomorphism property ``rho(a) rho(b) = rho(a b)``, straight from the
    Sage/GAP irrep matrices, never from the activations); ``rho(c^-1)`` is
    read off by indexing with the Cayley-table inverse map.

    Because ``Re tr(M) = sum_i Re(M_ii)``, ``gcr_character_design``'s column
    for the same irrep is exactly the sum of this design's ``(Re, i == j)``
    columns -- a genuine linear combination inside this design's column
    span, which is what makes (:func:`gcr_character_form`,
    :func:`gcr_matrix_entry_form`) a properly nested
    :class:`probes.FunctionalForm` pair.
    """
    table = group.cayley_table
    order = int(table.shape[0])
    e = identity_index(table)
    inv = inverses(table, e)
    irrep_indices = list(irrep_indices)
    blocks: list[np.ndarray] = []
    for idx in irrep_indices:
        matrices = group.irreps[idx].matrices  # [order, d, d] complex128: rho(g)
        matrices_inv = matrices[inv]  # rho(c^-1), indexed directly by c
        # ab[a, b]_{ik} = sum_j rho(a)_{ij} rho(b)_{jk} = rho(a b)_{ik}.
        ab = np.einsum("aij,bjk->abik", matrices, matrices)
        # m[a, b, c]_{ij} = sum_k ab[a, b]_{ik} rho(c^-1)_{kj} = [rho(a b) rho(c^-1)]_{ij}.
        m = np.einsum("abik,ckj->abcij", ab, matrices_inv)
        d = matrices.shape[1]
        blocks.append(m.real.reshape(order, order, order, d * d))
        blocks.append(m.imag.reshape(order, order, order, d * d))
    if not blocks:
        return np.zeros((order, order, order, 0), dtype=np.float64)
    return np.concatenate(blocks, axis=3)


def gcr_matrix_entry_form(
    group: FiniteGroup,
    irrep_indices: Sequence[int] | None = None,
    *,
    name: str | None = None,
) -> FunctionalForm:
    """A :class:`probes.FunctionalForm` wrapping :func:`gcr_matrix_entry_design`
    over ``irrep_indices`` (every nontrivial irrep by default, matching
    :func:`probes.gcr_candidate_irreps`)."""
    resolved = gcr_candidate_irreps(group) if irrep_indices is None else list(irrep_indices)
    design = gcr_matrix_entry_design(group, resolved)
    if name is None:
        dims = ",".join(str(group.irreps[i].dimension) for i in resolved)
        name = f"gcr_matrix_entries[{dims}]"
    return FunctionalForm(name=name, design=design)


def readout_characterisation_instrument(
    model: GroupModel,
    group: FiniteGroup,
    *,
    train_frac: float = 0.7,
    seed: int = 0,
    null_model: GroupModel | None = None,
    target_fve: float = 0.95,
    improvement_tol: float = 0.01,
    tie_tol: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    """The positive readout characterisation: character-only vs full-matrix-
    entry, held out.

    Fits the two nested designs described in the module docstring
    (:func:`gcr_character_form`, :func:`gcr_matrix_entry_form`, both over
    every nontrivial candidate irrep) to the model's read-position logits with
    I-20's harness (:func:`probes.functional_form_fit`, reused unchanged), and
    separately reuses :func:`probes.gcr_character_readout_instrument`'s own
    greedy minimal-irrep-set search so the reported minimal set matches that
    instrument's exactly.

    ``status: "UNDEFINED"`` when the group has no nontrivial irrep (the
    comparison is vacuous -- not expected on any fixture in this corpus, but
    guarded rather than assumed away).

    ``character_only_sufficient`` is the verdict: True when the full-matrix-
    entry design's extra columns buy no meaningful held-out FVE over the
    character-only design it contains (``held_out_fve_gain_full_minus_character
    <= tie_tol``) -- the positive "the readout needs only the scalar
    character" result. False means the readout uses more than the character
    (the full-matrix-entry design wins held out).

    CAVEAT (also carried in the record's ``caveat``): read before citing raw
    FVE. ``M(a, b, c) = rho(a b) rho(c^-1)`` is a function of the single
    product ``a*b`` once ``c`` is fixed, so a high raw held-out FVE for either
    design is consistent with any correct algorithm, not only GCR -- the
    load-bearing statistics are the held-out gap and the minimal irrep set,
    never the raw FVE numbers alone.
    """
    order = group.order
    full_irreps = gcr_candidate_irreps(group)
    if not full_irreps:
        return {
            "instrument": "readout-characterisation",
            "target_theory": "GCR",
            "status": UNDEFINED,
            "reason": (
                "group has no nontrivial irrep: the character-only vs "
                "full-matrix-entry comparison is vacuous"
            ),
        }

    character_form = gcr_character_form(group, full_irreps, name="character_only")
    matrix_form = gcr_matrix_entry_form(group, full_irreps, name="full_matrix_entry")

    fit = functional_form_fit(
        model,
        order,
        [character_form, matrix_form],
        train_frac=train_frac,
        seed=seed,
        null_model=null_model,
        device=device,
    )
    fve_by_name = {entry["name"]: entry["held_out_fve"] for entry in fit["forms"]}
    character_fve = fve_by_name["character_only"]
    matrix_fve = fve_by_name["full_matrix_entry"]
    gain = matrix_fve - character_fve
    character_only_sufficient = bool(gain <= tie_tol)

    # Reuse the character-readout instrument's own greedy minimal-irrep-set
    # search unchanged, so this instrument's reported minimal set is exactly
    # what gcr_character_readout_instrument reports -- never a second,
    # independently-tuned search that could quietly disagree.
    character_readout = gcr_character_readout_instrument(
        model,
        group,
        train_frac=train_frac,
        seed=seed,
        null_model=null_model,
        target_fve=target_fve,
        improvement_tol=improvement_tol,
        device=device,
    )
    minimal_irrep_set = character_readout["primary"]["minimal_irrep_set"]

    record: dict[str, Any] = {
        "instrument": "readout-characterisation",
        "target_theory": "GCR",
        "rung": 5,
        "status": "measured",
        "n_candidate_irreps": len(full_irreps),
        "candidate_irrep_dimensions": [int(group.irreps[i].dimension) for i in full_irreps],
        "character_only_held_out_fve": character_fve,
        "full_matrix_entry_held_out_fve": matrix_fve,
        "held_out_fve_gain_full_minus_character": gain,
        "tie_tol": float(tie_tol),
        "character_only_sufficient": character_only_sufficient,
        "minimal_irrep_set": minimal_irrep_set,
        "functional_form_fit": fit,
        "caveat": (
            "M(a, b, c) = rho(a b) rho(c^-1) -- and every entry of it, "
            "including the trace -- is a function of the single product a*b "
            "once c is fixed, so a high raw held-out FVE for either design "
            "(character_only_held_out_fve or full_matrix_entry_held_out_fve) "
            "is consistent with any algorithm that computes the group "
            "product correctly, not only GCR, and is not by itself evidence "
            "for a character-only readout. The load-bearing statistics are "
            "held_out_fve_gain_full_minus_character (does the strictly "
            "larger full-matrix-entry span, which contains the character-"
            "only span, buy anything held out) and minimal_irrep_set (the "
            "sparse subset of irreps the character alone needs), never the "
            "raw FVE numbers on their own."
        ),
        "note": (
            "character_only and full_matrix_entry are a genuinely nested "
            "FunctionalForm pair: Re tr(M) = sum_i Re(M_ii) is exactly a "
            "linear combination of full_matrix_entry's (Re, i == j) columns, "
            "so the full form can never fit worse in sample. "
            "character_only_sufficient is True when the extra columns buy no "
            "meaningful held-out FVE (gain <= tie_tol) -- the positive "
            "readout characterisation: the output stage reads off only the "
            "scalar character per occupied irrep and never reconstructs the "
            "full matrix. False means the readout uses more than the "
            "character."
        ),
    }
    if null_model is not None:
        null_by_name = {entry["name"]: entry.get("null_held_out_fve") for entry in fit["forms"]}
        record["null"] = {
            "character_only_null_held_out_fve": null_by_name.get("character_only"),
            "full_matrix_entry_null_held_out_fve": null_by_name.get("full_matrix_entry"),
        }
    return record
