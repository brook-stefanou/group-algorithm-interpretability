"""The spectral core of the occupancy instrument subset (I-08/I-09/I-10/I-14).

The object of study is activation-space, never weight-space: a "neuron" is an
MLP unit ``m`` and its function on the group is
``A_m[a, b] = mlp_pre[(a, b), -1, m]``, an ``|G| x |G|`` matrix over the whole
Cayley grid (attention mixes the argument positions, so the effective
per-argument weight is input-dependent and ``W_E @ W_in[:, m]`` is not a valid
Fourier-mass definition).

Isotypic energy for the left argument is ``E_m[j] = ||P_j A_m||_F^2`` (right
argument: ``||A_m P_j||_F^2``) over the real isotypic blocks of the regular
representation (complex-conjugate irrep pairs merged, as exported in
``groups/data.py::IsotypicBlock``). The blocks partition ``C[G]``, so the
energies are non-negative and sum exactly to ``||A_m||_F^2`` -- both asserted,
never assumed.

The only admissible occupancy null is each group's own analytic
``pi0_j = block_rank_j / |G|`` -- never the uniform distribution, never a
normaliser whose achievable range depends on the irrep degrees (the audited
condition-dependent-normalisation failure). Total-variation distances are
always quoted with the group's own noise floor,
``floor = 1/2 * sum_j sqrt(2/pi) * sqrt(pi0_j * (1 - pi0_j) / N)`` with
``N = neurons x seeds`` -- the Dirichlet/multinomial-derived planning floor,
which self-averaging of the energy-weighted statistic can beat but never
fabricate against.

Population occupancy (I-10, energy-weighted) is the primary readout; per-neuron
concentration (I-09) is a reportable secondary whose structured failure is a
coset-circuit signature. Neither licenses "the model uses irrep rho" -- that is
ablation's job (I-15, not in this subset).

The trivial block needs its own handling, measured rather than assumed: the
read position is the '=' token, whose embedding is identical for every input
pair, so at random initialisation the constant (and, for the left argument,
every ``b``-only) component dominates -- ~96% of a random-init model's energy
lands in the trivial block. That component is a DC offset of the architecture,
not group structure, and a statistic that counts it fails the suite's mandatory
untrained-model regression test. The record therefore carries both forms: the
full-block vector with the trivial share reported as its own signal, and the
nontrivial-renormalised form (null ``block_rank_j / (|G| - 1)`` over nontrivial
blocks), which is the null-calibrated headline statistic.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..groups.group import FiniteGroup
from ..model import GroupModel

_REL_TOL = 1e-6


def cayley_grid_tokens(order: int) -> torch.Tensor:
    """Token triples ``(a, b, '=')`` for every ordered pair of the group, in the
    row-major order ``row k = (k // n, k % n)`` -- the same enumeration
    ``task.build_group_task`` uses, so row ``k`` of the activations aligns with
    ``cayley_table.reshape(-1)[k]``. The '=' token is ``order`` (the vocabulary
    is the ``order`` elements plus '=')."""
    grid = torch.arange(order, dtype=torch.long)
    a = grid.repeat_interleave(order)
    b = grid.repeat(order)
    equals = torch.full_like(a, order)
    return torch.stack([a, b, equals], dim=1)


def neuron_activations(
    model: GroupModel,
    order: int,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    """I-08: the per-neuron function on ``G x G``, ``A[m, a, b]`` with shape
    ``[d_mlp, |G|, |G|]``, read from ``mlp_pre`` at the read position (-1) over
    the full Cayley grid. Requires an MLP (``use_mlp=false`` models have no
    neurons to measure)."""
    tokens = cayley_grid_tokens(order)
    chunks: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, tokens.shape[0], batch_size):
            cache = model(tokens[start : start + batch_size], return_cache=True)
            mlp_pre = cache["mlp_pre"]
            if mlp_pre is None:
                raise ValueError(
                    "model has no MLP (use_mlp=false); neuron activations are undefined"
                )
            chunks.append(mlp_pre[:, -1, :].detach().to(torch.float64).cpu())
    flat = torch.cat(chunks, dim=0)  # [|G|^2, d_mlp]
    d_mlp = flat.shape[1]
    return flat.numpy().reshape(order, order, d_mlp).transpose(2, 0, 1)


def analytic_null(group: FiniteGroup) -> np.ndarray:
    """The group's own analytic occupancy null ``pi0_j = block_rank_j / |G|``.
    Exact and per-group; sums to 1 because the block ranks sum to ``|G|``
    (validated at artifact load)."""
    pi0 = np.array(
        [block.block_rank / group.order for block in group.isotypic_blocks], dtype=np.float64
    )
    if abs(float(pi0.sum()) - 1.0) > _REL_TOL:
        raise ValueError(f"analytic null sums to {pi0.sum()}, expected 1")
    return pi0


def isotypic_energies(
    activations: np.ndarray,
    group: FiniteGroup,
    *,
    argument: str = "left",
) -> np.ndarray:
    """Per-neuron isotypic energies ``E[m, j]``, shape ``[d_mlp, n_blocks]``.

    ``argument="left"`` projects the first (``a``) index: ``||P_j A_m||_F^2``;
    ``argument="right"`` projects the second: ``||A_m P_j||_F^2``. Asserts the
    completeness identity ``sum_j E[m, j] == ||A_m||_F^2`` (the block
    projectors resolve the identity) and non-negativity."""
    if argument not in ("left", "right"):
        raise ValueError(f"argument must be 'left' or 'right', got {argument!r}")
    a = np.asarray(activations, dtype=np.float64)
    if a.ndim != 3 or a.shape[1] != group.order or a.shape[2] != group.order:
        raise ValueError(f"activations must have shape [d_mlp, {group.order}, {group.order}]")
    energies = np.empty((a.shape[0], len(group.isotypic_blocks)), dtype=np.float64)
    for j, block in enumerate(group.isotypic_blocks):
        projector = block.projector
        if argument == "left":
            projected = np.matmul(projector[None, :, :], a)
        else:
            projected = np.matmul(a, projector[None, :, :])
        energies[:, j] = np.square(projected).sum(axis=(1, 2))
    totals = np.square(a).sum(axis=(1, 2))
    if not np.allclose(energies.sum(axis=1), totals, rtol=_REL_TOL, atol=_REL_TOL):
        raise ValueError(
            "isotypic energies do not sum to the total activation energy; the "
            "block projectors of this artifact do not resolve the identity"
        )
    if float(energies.min(initial=0.0)) < -_REL_TOL:
        raise ValueError("negative isotypic energy; projector data is corrupt")
    return np.maximum(energies, 0.0)


def population_occupancy(energies: np.ndarray) -> np.ndarray:
    """I-10, energy-weighted: ``occ_j = sum_m E[m, j] / sum_{m, j} E[m, j]``.
    Sums to 1 by the completeness identity asserted in
    :func:`isotypic_energies`."""
    per_block = np.asarray(energies, dtype=np.float64).sum(axis=0)
    total = float(per_block.sum())
    if total <= 0.0:
        raise ValueError("total activation energy is zero; occupancy is undefined")
    return per_block / total


def trivial_block_index(group: FiniteGroup) -> int:
    """The index of the isotypic block containing the trivial irrep (the block
    whose projector averages over the group: constant functions). Identified by
    the character being identically 1 -- every group has exactly one."""
    candidates = [
        j
        for j, block in enumerate(group.isotypic_blocks)
        if block.irrep_degree == 1
        and all(np.allclose(group.irreps[i].character, 1.0) for i in block.irrep_indices)
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one trivial block, found {candidates}")
    return candidates[0]


def restrict_to_nontrivial(vector: np.ndarray, trivial_index: int) -> np.ndarray:
    """Drop the trivial block's entry and renormalise to a distribution over
    the nontrivial blocks. Applied identically to occupancy vectors, the
    analytic null (giving ``block_rank_j / (|G| - 1)``), and templates, so
    every comparison stays like-for-like."""
    v = np.asarray(vector, dtype=np.float64)
    remaining = np.delete(v, trivial_index)
    total = float(remaining.sum())
    if total <= 0.0:
        raise ValueError("no mass outside the trivial block; nontrivial form is undefined")
    return remaining / total


def per_neuron_concentration(energies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """I-09: per neuron, the fraction of its energy in its top isotypic block
    and that block's index. Zero-energy (dead) neurons get ``nan`` / ``-1``
    rather than a fabricated concentration."""
    e = np.asarray(energies, dtype=np.float64)
    totals = e.sum(axis=1)
    top_block = np.where(totals > 0.0, e.argmax(axis=1), -1).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        top_share = np.where(totals > 0.0, e.max(axis=1) / totals, np.nan)
    return top_share, top_block


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """``TV(p, q) = 1/2 * sum_j |p_j - q_j|`` over aligned block vectors."""
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    if p_arr.shape != q_arr.shape:
        raise ValueError(f"shape mismatch: {p_arr.shape} vs {q_arr.shape}")
    return 0.5 * float(np.abs(p_arr - q_arr).sum())


def dirichlet_noise_floor(pi0: np.ndarray, n_units: int) -> float:
    """The Dirichlet/multinomial-derived expected TV of a null sample of
    ``n_units`` units from ``pi0``:
    ``1/2 * sum_j sqrt(2/pi) * sqrt(pi0_j * (1 - pi0_j) / n_units)``.

    ``n_units`` is neurons x seeds, so pooling ``S`` seeds scales the floor by
    ``1/sqrt(S)``. A raw TV quoted without this floor is meaningless; the
    floor is a conservative planning number (count-form variance applied to
    the energy-weighted statistic, which self-averages further), so it can
    understate power but cannot fabricate decidability."""
    if n_units <= 0:
        raise ValueError(f"n_units must be positive, got {n_units}")
    p = np.asarray(pi0, dtype=np.float64)
    return 0.5 * float(np.sqrt(2.0 / math.pi) * np.sqrt(p * (1.0 - p) / n_units).sum())


__all__ = [
    "analytic_null",
    "cayley_grid_tokens",
    "dirichlet_noise_floor",
    "isotypic_energies",
    "neuron_activations",
    "per_neuron_concentration",
    "population_occupancy",
    "restrict_to_nontrivial",
    "total_variation",
    "trivial_block_index",
]
