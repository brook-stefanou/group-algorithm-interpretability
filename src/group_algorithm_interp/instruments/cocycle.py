"""The C5 cocycle instrument: extension precompute (I-21), the cocycle
probe/ablation/fit (I-22/I-22b/I-22c), and the intervention-based predicted-error
test (I-22d), T8.

This is the one instrument built for a single claim -- C5, the
``(48,29)`` GL(2,3) [split] vs ``(48,28)`` SL(2,3).C2 [non-split] contrast --
and it carries C5's scoped risk (a null probe/ablation ships as a measurement).

**Why I-22d exists (the ``|Q| = 2`` degeneracy of I-22/I-22b/I-22c).** At the
registered contrast ``Q = C2`` (``|Q| = 2``) the normalised cocycle ``f`` is
nontrivial on exactly one of the four ``(q1, q2)`` cells, cell ``(1, 1)``. Two
things follow, and the recent structural-degeneracy guards on I-22/I-22b/I-22c
record both: (a) leave-one-cell-out can never train on the sole nontrivial
class, so the decode probe is capped at chance regardless of the model
(``STRUCTURALLY_DEGENERATE``); and (b) more fundamentally, ``f`` as a per-pair
label is informationally identical to the cell-``(1, 1)`` indicator, so *any*
decode-``f`` or ablate-the-``f``-direction approach is confounded with ordinary
coset-membership sensitivity, which every accurate model must have. Decoding
cannot separate "computes via the twisted rule" from "is sensitive to cell
membership" here.

**I-22d's lever (intervention, not decoding).** The twisted and untwisted rules
make different *element-level* predictions about which wrong answer a perturbed
model produces. On a twist-active cell the true (twisted) answer and the
untwisted answer are related by a fixed right-translation ``true = untwisted . w``
(``w = s(q1 q2)^{-1} f(q1, q2) s(q1 q2)``), so the untwisted answer is a specific,
per-``(a, b)``-row target that *varies within the cell*. A model that computes
the untwisted product and then applies the cocycle correction, when its winning
prediction is knocked out, falls back to the untwisted answer specifically; a
lookup/cell-membership model falls back to arbitrary wrong answers. Scoring the
fraction of these induced errors that land on the row-specific untwisted target
-- against a shuffled-correspondence null and (where ``|N| >= 3``) counterfactual
cocycle-value targets -- is therefore *not* a function of cell membership: cell
membership is constant on the cell, whereas the untwisted target is not. I-22d is
additive to and does not alter I-21/I-22/I-22b/I-22c.

The object is an extension ``1 -> N -> G -> Q -> 1`` for a pre-registered normal
subgroup ``N`` (``Q = G/N``). Fix a transversal ``s: Q -> G`` with ``s(e) = e``.
Every ``g`` in ``G`` then has coordinates ``(n, q)`` with ``g = n . s(q)``,
``n in N``, ``q = coset(g)``, and the product law is the *twisted rule*

    (n1, q1) . (n2, q2) = (n1 . phi_{q1}(n2) . f(q1, q2),  q1 q2),

where ``phi_q(n) = s(q) n s(q)^{-1}`` is the action of ``Q`` on ``N`` and
``f(q1, q2) = s(q1) s(q2) s(q1 q2)^{-1} in N`` is the 2-cocycle (factor set).
Dropping ``f`` gives the *untwisted* (semidirect) rule. Two self-verifying
identities are asserted, never assumed:

* the (possibly non-abelian) cocycle identity
  ``f(a,b) f(ab,c) == phi_a(f(b,c)) f(a,bc)`` for all ``a,b,c in Q`` -- this is
  associativity of ``G`` re-expressed in coordinates;
* ``f == e`` is achievable by *some* transversal iff the extension *splits*
  (a complement to ``N`` exists). This is the crux of the split/non-split
  contrast: the untwisted rule reproduces the full Cayley table exactly iff a
  trivialising transversal exists, so the ``f == 1`` fit SUCCEEDS on a split
  group and FAILS on a non-split one, for *every* transversal.

On a split member the cocycle probe is degenerate by construction: with the
complement transversal ``f`` is constant, so the decode target has no variation
(the split member's control is a fit residual, not a probe score).

The model-facing probe (I-22) decodes ``f(q1, q2)`` held out over ``(q1, q2)``
cells -- never over ``(a, b)`` pairs, because many pairs share one cell and an
``(a, b)`` split would be pseudoreplication (banned practice 6). The ablation
(I-22b) removes the ``f``-carrying direction and is what licenses "the model
uses the cocycle"; a norm- and dimension-matched random subspace is the control
(I-03).

The twisted-rule FVE fit (I-22c) regresses the model's logits onto the answer
the twisted coordinate rule predicts and, on the same data, onto the untwisted
(``f == e``) rule; the held-out FVE gap is the extra logit variance the cocycle
term explains. Like the probe it is held out over ``(q1, q2)`` cells -- the
cocycle label is constant within a cell, so an ``(a, b)``-pair split would leak
it (banned practice 6). On a split member with the complement transversal the
two rules coincide (``f == e``), so the gap is ~0 by construction; a non-split
member needs the twist and the gap is positive.

Pure group theory here uses only the exported Cayley table and never Sage/GAP.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .. import stats
from ..groups.group import FiniteGroup
from ..model import GroupModel
from .occupancy import cayley_grid_tokens

_TOL = 1e-9


def _require_finite(array: np.ndarray, name: str) -> None:
    """Raise if ``array`` carries a non-finite value (NaN or inf).

    Checkpoint-derived features and activations can turn non-finite (a diverged
    run, a corrupt snapshot). Without this guard a NaN flows silently through
    ``np.linalg.solve``/``np.linalg.lstsq`` and out into a plausible-looking
    record; fail loudly at the boundary instead.
    """
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values (NaN or inf); refusing to fit")


# ---------------------------------------------------------------------------
# Cayley-table primitives (index arithmetic only; no group discovery)
# ---------------------------------------------------------------------------


def _identity_index(table: np.ndarray) -> int:
    n = table.shape[0]
    idx = np.arange(n)
    candidates = np.flatnonzero(np.all(table == idx[None, :], axis=1))
    if candidates.size == 0:
        raise ValueError("Cayley table has no identity element")
    return int(candidates[0])


def _inverses(table: np.ndarray, identity: int) -> np.ndarray:
    n = table.shape[0]
    inverses = np.empty(n, dtype=np.int64)
    for g in range(n):
        inverses[g] = int(np.flatnonzero(table[g] == identity)[0])
    return inverses


def is_subgroup(table: np.ndarray, members: np.ndarray) -> bool:
    """``members`` is closed under the product and inversion and contains the
    identity -- a subgroup. Pure index arithmetic on the Cayley table."""
    elems = sorted(int(x) for x in np.asarray(members).tolist())
    if not elems:
        return False
    subset = set(elems)
    identity = _identity_index(table)
    if identity not in subset:
        return False
    for a in elems:
        for b in elems:
            if int(table[a, b]) not in subset:
                return False
    return True


def is_normal(table: np.ndarray, members: np.ndarray) -> bool:
    """``members`` is a normal subgroup: a subgroup with ``g N g^{-1} = N`` for
    every ``g`` in ``G``."""
    if not is_subgroup(table, members):
        return False
    identity = _identity_index(table)
    inverses = _inverses(table, identity)
    subset = {int(x) for x in np.asarray(members).tolist()}
    n = table.shape[0]
    for g in range(n):
        for h in subset:
            if int(table[table[g, h], inverses[g]]) not in subset:
                return False
    return True


def subgroup_closure(table: np.ndarray, generators: set[int]) -> frozenset[int]:
    """The subgroup generated by ``generators`` (product closure including the
    identity), by breadth-first multiplication over the Cayley table."""
    identity = _identity_index(table)
    members = {identity, *(int(g) for g in generators)}
    frontier = list(members)
    while frontier:
        a = frontier.pop()
        for b in list(members):
            for prod in (int(table[a, b]), int(table[b, a])):
                if prod not in members:
                    members.add(prod)
                    frontier.append(prod)
    return frozenset(members)


# ---------------------------------------------------------------------------
# I-21: quotient, transversal, action, cocycle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Extension:
    """The extension ``1 -> N -> G -> Q -> 1`` coordinatised by one transversal.

    ``normal`` are the sorted element indices of ``N``; ``n_local`` maps a global
    element index in ``N`` to its position ``0..|N|-1``. ``coset_label`` maps every
    element of ``G`` to its ``Q``-label (the ``N``-coset it lies in), with the
    identity coset labelled ``0``. ``quotient_table`` is ``Q``'s multiplication.
    ``transversal[q]`` is ``s(q)`` (a global index; ``s(0)`` is the identity).
    ``action[q]`` is ``phi_q`` as a permutation of ``N`` in local indices.
    ``cocycle[q1, q2]`` is ``f(q1, q2)`` as a local ``N`` index.
    """

    order: int
    normal: tuple[int, ...]
    quotient_order: int
    coset_label: np.ndarray  # [|G|] -> Q-label
    quotient_table: np.ndarray  # [|Q|, |Q|]
    transversal: np.ndarray  # [|Q|] -> global element index
    action: np.ndarray  # [|Q|, |N|] local permutation
    cocycle: np.ndarray  # [|Q|, |Q|] local N index
    n_local: dict[int, int]

    @property
    def normal_order(self) -> int:
        return len(self.normal)

    def to_record(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "normal_order": self.normal_order,
            # The normal subgroup's membership pins which N was coordinatised,
            # not just its order -- a group can have several normal subgroups of
            # the same order.
            "normal": list(self.normal),
            "quotient_order": self.quotient_order,
            "cocycle_trivial": bool(cocycle_is_trivial(self)),
            "transversal": self.transversal.tolist(),
            "cocycle": self.cocycle.tolist(),
        }


def _quotient(table: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, list[list[int]], int]:
    """Left cosets ``gN`` as ``Q``-labels, the identity coset labelled ``0``.

    Returns ``(coset_label, coset_members, quotient_order)``.
    """
    n = table.shape[0]
    members = sorted(int(x) for x in np.asarray(normal).tolist())
    identity = _identity_index(table)
    coset_label = -np.ones(n, dtype=np.int64)
    coset_members: list[list[int]] = []
    # Process the identity first so its coset (== N) is label 0.
    order = [identity] + [g for g in range(n) if g != identity]
    for g in order:
        if coset_label[g] >= 0:
            continue
        coset = sorted({int(table[g, h]) for h in members})
        label = len(coset_members)
        for x in coset:
            coset_label[x] = label
        coset_members.append(coset)
    return coset_label, coset_members, len(coset_members)


def build_extension(
    group: FiniteGroup,
    normal: np.ndarray,
    transversal: np.ndarray | None = None,
) -> Extension:
    """I-21: coordinatise ``G`` over a normal subgroup ``N`` with a transversal.

    ``transversal`` maps each ``Q``-label to a representative of that coset;
    ``s(0)`` must be the identity. When ``None`` a canonical transversal is used
    (the smallest element index of each coset, with the identity for coset 0).

    Asserts the extension identities: ``N`` is normal; the transversal picks one
    element per coset with ``s(0) = e``; the action lands in ``N``; the cocycle
    lands in ``N`` and is normalised (``f(0, q) = f(q, 0) = e``); and the
    non-abelian cocycle identity holds.
    """
    table = group.cayley_table
    n_elements = table.shape[0]
    members = np.asarray(normal, dtype=np.int64)
    if not is_normal(table, members):
        raise ValueError("the given subgroup is not normal in G; no extension is defined")
    identity = _identity_index(table)
    inverses = _inverses(table, identity)
    normal_sorted = tuple(sorted(int(x) for x in members.tolist()))
    n_local = {g: i for i, g in enumerate(normal_sorted)}
    normal_array = np.array(normal_sorted, dtype=np.int64)

    coset_label, coset_members, quotient_order = _quotient(table, members)

    if transversal is None:
        chosen = np.empty(quotient_order, dtype=np.int64)
        for label, coset in enumerate(coset_members):
            chosen[label] = identity if label == 0 else min(coset)
    else:
        chosen = np.asarray(transversal, dtype=np.int64).copy()
        if chosen.shape != (quotient_order,):
            raise ValueError(
                f"transversal must have one representative per coset ({quotient_order})"
            )
        # Reject negative / out-of-range indices up front: NumPy would otherwise
        # index-wrap a negative representative silently and only trip much later
        # with a misleading "cocycle value fell outside N" error.
        if np.any(chosen < 0) or np.any(chosen >= n_elements):
            raise ValueError(
                "transversal entries must be element indices in [0, |G|); "
                "got a negative or out-of-range index"
            )
        if int(chosen[0]) != identity:
            raise ValueError("transversal must satisfy s(0) = e (the identity coset)")
        # ``s(q)`` must lie in coset ``q`` itself -- covering every coset once is
        # not enough, since a permutation of a valid transversal (right reps,
        # wrong slots) covers all cosets yet mislabels the quotient. Check
        # membership per slot so a permuted transversal is rejected here with a
        # precise error, not downstream as "cocycle value fell outside N".
        for q in range(quotient_order):
            if int(coset_label[int(chosen[q])]) != q:
                raise ValueError(f"transversal[{q}] = {int(chosen[q])} does not lie in coset {q}")

    # Quotient multiplication (well-defined by normality; read off the reps).
    quotient_table = np.empty((quotient_order, quotient_order), dtype=np.int64)
    for qi in range(quotient_order):
        for qj in range(quotient_order):
            quotient_table[qi, qj] = int(coset_label[table[chosen[qi], chosen[qj]]])

    # Action phi_q(n) = s(q) n s(q)^{-1}, as a local permutation of N.
    action = np.empty((quotient_order, len(normal_sorted)), dtype=np.int64)
    for q in range(quotient_order):
        s_q, s_q_inv = int(chosen[q]), int(inverses[chosen[q]])
        for i, g in enumerate(normal_sorted):
            image = int(table[table[s_q, g], s_q_inv])
            if image not in n_local:
                raise ValueError("action does not preserve N; the subgroup is not normal")
            action[q, i] = n_local[image]

    # Cocycle f(qi, qj) = s(qi) s(qj) s(qi qj)^{-1} in N (local index).
    cocycle = np.empty((quotient_order, quotient_order), dtype=np.int64)
    for qi in range(quotient_order):
        for qj in range(quotient_order):
            qk = int(quotient_table[qi, qj])
            value = int(table[table[chosen[qi], chosen[qj]], inverses[chosen[qk]]])
            if value not in n_local:
                raise ValueError("cocycle value fell outside N; transversal is inconsistent")
            cocycle[qi, qj] = n_local[value]

    extension = Extension(
        order=group.order,
        normal=normal_sorted,
        quotient_order=quotient_order,
        coset_label=coset_label,
        quotient_table=quotient_table,
        transversal=chosen,
        action=action,
        cocycle=cocycle,
        n_local=n_local,
    )
    _assert_normalised(extension)
    _assert_cocycle_identity(extension, table, normal_array)
    return extension


def _n_mul_local(table: np.ndarray, normal_array: np.ndarray, i: int, j: int) -> int:
    """Product of two ``N`` elements (given by local index), as a local index."""
    prod = int(table[normal_array[i], normal_array[j]])
    return int(np.flatnonzero(normal_array == prod)[0])


def _assert_normalised(extension: Extension) -> None:
    zero = extension.n_local[_identity_from_normal(extension)]
    q = extension.quotient_order
    if not all(int(extension.cocycle[0, j]) == zero for j in range(q)):
        raise ValueError("cocycle is not normalised: f(0, q) != e")
    if not all(int(extension.cocycle[i, 0]) == zero for i in range(q)):
        raise ValueError("cocycle is not normalised: f(q, 0) != e")


def _identity_from_normal(extension: Extension) -> int:
    """The identity's global index. ``s(0) = e`` by construction and ``e in N``."""
    return int(extension.transversal[0])


def _assert_cocycle_identity(
    extension: Extension, table: np.ndarray, normal_array: np.ndarray
) -> None:
    """``f(a,b) f(ab,c) == phi_a(f(b,c)) f(a,bc)`` for all ``a, b, c`` in ``Q``."""
    q = extension.quotient_order
    qt = extension.quotient_table
    f = extension.cocycle
    action = extension.action
    for a in range(q):
        for b in range(q):
            ab = int(qt[a, b])
            for c in range(q):
                bc = int(qt[b, c])
                lhs = _n_mul_local(table, normal_array, int(f[a, b]), int(f[ab, c]))
                phi_a_fbc = int(action[a, int(f[b, c])])
                rhs = _n_mul_local(table, normal_array, phi_a_fbc, int(f[a, bc]))
                if lhs != rhs:
                    raise ValueError(
                        f"cocycle identity fails at (a,b,c)=({a},{b},{c}): {lhs} != {rhs}"
                    )


def cocycle_is_trivial(extension: Extension) -> bool:
    """Whether ``f == e`` for this transversal (the untwisted rule is exact)."""
    zero = extension.n_local[int(extension.transversal[0])]
    return bool(np.all(extension.cocycle == zero))


# ---------------------------------------------------------------------------
# Coordinate rule and the f == 1 fit
# ---------------------------------------------------------------------------


def coordinates(extension: Extension, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each element's coordinates ``(n, q)`` with ``g = n . s(q)``.

    Returns ``(n_local_of, q_of)`` arrays over ``0..|G|-1``. Asserts the map
    ``G -> N x Q`` is a bijection.
    """
    identity = _identity_index(table)
    inverses = _inverses(table, identity)
    normal_array = np.array(extension.normal, dtype=np.int64)
    q_of = extension.coset_label.copy()
    n_of = np.empty(extension.order, dtype=np.int64)
    for g in range(extension.order):
        q = int(q_of[g])
        n_elem = int(table[g, inverses[int(extension.transversal[q])]])
        slot = np.flatnonzero(normal_array == n_elem)
        if slot.size == 0:
            raise ValueError("coordinate decomposition left N; transversal is inconsistent")
        n_of[g] = int(slot[0])
    pairs = {(int(n_of[g]), int(q_of[g])) for g in range(extension.order)}
    if len(pairs) != extension.order:
        raise ValueError("coordinate map G -> N x Q is not a bijection")
    return n_of, q_of


def coordinate_product_accuracy(
    extension: Extension, table: np.ndarray, *, include_cocycle: bool
) -> float:
    """Fraction of ordered pairs ``(g1, g2)`` the coordinate rule reproduces.

    With ``include_cocycle=True`` this is the twisted rule and is exact (1.0) by
    construction. With ``include_cocycle=False`` this is the untwisted
    (semidirect) rule; it equals 1.0 iff ``f == e`` for this transversal.
    """
    normal_array = np.array(extension.normal, dtype=np.int64)
    n_of, q_of = coordinates(extension, table)
    transversal = extension.transversal
    action = extension.action
    cocycle = extension.cocycle
    qt = extension.quotient_table
    correct = 0
    total = extension.order * extension.order
    for g1 in range(extension.order):
        n1, q1 = int(n_of[g1]), int(q_of[g1])
        for g2 in range(extension.order):
            n2, q2 = int(n_of[g2]), int(q_of[g2])
            phi = int(action[q1, n2])
            n_res = _n_mul_local(table, normal_array, n1, phi)
            if include_cocycle:
                n_res = _n_mul_local(table, normal_array, n_res, int(cocycle[q1, q2]))
            qk = int(qt[q1, q2])
            predicted = int(table[normal_array[n_res], int(transversal[qk])])
            if predicted == int(table[g1, g2]):
                correct += 1
    return correct / total


@dataclass(frozen=True)
class SplitResult:
    """The split/non-split determination and, when split, a complement."""

    splits: bool
    complement: tuple[int, ...] | None
    quotient_order: int
    method: str

    def to_record(self) -> dict[str, Any]:
        return {
            "splits": self.splits,
            "complement": None if self.complement is None else list(self.complement),
            "quotient_order": self.quotient_order,
            "method": self.method,
        }


def _find_complement(table: np.ndarray, normal: np.ndarray) -> tuple[int, ...] | None:
    """A complement ``C`` to ``N``: a subgroup with ``|C| = [G:N]`` and
    ``C intersect N = {e}``, found by backtracking closure. ``None`` when the
    extension does not split.

    A complement is exactly a set of coset representatives closed under the
    product, so growth is pruned hard: any candidate whose closure revisits a
    coset or re-enters ``N`` nontrivially is rejected immediately.
    """
    identity = _identity_index(table)
    coset_label, coset_members, quotient_order = _quotient(table, normal)
    normal_set = {int(x) for x in np.asarray(normal).tolist()}
    target = quotient_order

    def covered(members: frozenset[int]) -> set[int]:
        return {int(coset_label[g]) for g in members}

    def valid(members: frozenset[int]) -> bool:
        labels = [int(coset_label[g]) for g in members]
        if len(set(labels)) != len(labels):
            return False  # two elements in one coset
        if any(g in normal_set and g != identity for g in members):
            return False  # re-entered N off the identity
        return True

    start = frozenset({identity})

    def search(current: frozenset[int], seen: set[frozenset[int]]) -> tuple[int, ...] | None:
        if len(current) == target:
            return tuple(sorted(current))
        have = covered(current)
        for label in range(quotient_order):
            if label in have:
                continue
            for x in coset_members[label]:
                grown = subgroup_closure(table, set(current) | {x})
                if not valid(grown) or grown in seen:
                    continue
                seen.add(grown)
                result = search(grown, seen)
                if result is not None:
                    return result
            # Committing to cover this (smallest uncovered) coset and failing
            # means no complement extends ``current``.
            break
        return None

    return search(start, {start})


def split_status(group: FiniteGroup, normal: np.ndarray) -> SplitResult:
    """Decide whether ``G`` splits over ``N`` -- equivalently, whether ``f == e``
    is achievable by some transversal (I-21's ``f == 1 iff split`` invariant)."""
    if not is_normal(group.cayley_table, normal):
        raise ValueError("the given subgroup is not normal in G")
    complement = _find_complement(group.cayley_table, np.asarray(normal, dtype=np.int64))
    _, _, quotient_order = _quotient(group.cayley_table, np.asarray(normal, dtype=np.int64))
    return SplitResult(
        splits=complement is not None,
        complement=complement,
        quotient_order=quotient_order,
        method="backtracking-complement",
    )


def trivialising_transversal(group: FiniteGroup, normal: np.ndarray) -> np.ndarray | None:
    """A transversal ``s`` with ``f == e`` (the complement, as a ``Q -> G`` map),
    or ``None`` when the extension does not split."""
    result = split_status(group, normal)
    if result.complement is None:
        return None
    coset_label, _, quotient_order = _quotient(
        group.cayley_table, np.asarray(normal, dtype=np.int64)
    )
    transversal = np.empty(quotient_order, dtype=np.int64)
    for g in result.complement:
        transversal[int(coset_label[g])] = g
    return transversal


@dataclass(frozen=True)
class CocycleFit:
    """The ``f == 1`` fit (I-22 twisted-rule fit at the ground-truth level).

    ``untwisted_accuracy`` is the fraction of products the untwisted
    (semidirect) rule reproduces under the best available transversal; it is
    1.0 iff the extension splits. ``twisted_accuracy`` is always 1.0 (the
    twisted rule is exact). ``splits`` records the group-theoretic determination
    (``f == 1`` achievable iff split), which is decisive independently of the
    fit residual.
    """

    splits: bool
    untwisted_accuracy: float
    twisted_accuracy: float
    quotient_order: int
    normal_order: int
    used_trivialising_transversal: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "splits": self.splits,
            "untwisted_accuracy": self.untwisted_accuracy,
            "twisted_accuracy": self.twisted_accuracy,
            "quotient_order": self.quotient_order,
            "normal_order": self.normal_order,
            "used_trivialising_transversal": self.used_trivialising_transversal,
        }


def f_equiv_one_fit(group: FiniteGroup, normal: np.ndarray) -> CocycleFit:
    """The decisive split-vs-non-split discriminator: does the untwisted
    (``f == 1``) rule reproduce the whole Cayley table?

    On a *split* extension the complement transversal makes the untwisted rule
    exact -- the fit SUCCEEDS (accuracy 1.0). On a *non-split* extension no
    transversal trivialises ``f``, so the untwisted rule fails for every
    transversal -- the fit FAILS (accuracy < 1.0). This is the GL(2,3) [split]
    vs SL(2,3).C2 [non-split] contrast.
    """
    table = group.cayley_table
    normal_array = np.asarray(normal, dtype=np.int64)
    transversal = trivialising_transversal(group, normal_array)
    splits = transversal is not None
    extension = build_extension(group, normal_array, transversal)
    untwisted = coordinate_product_accuracy(extension, table, include_cocycle=False)
    twisted = coordinate_product_accuracy(extension, table, include_cocycle=True)
    if splits and untwisted < 1.0 - _TOL:
        raise ValueError("split extension but the trivialising transversal did not trivialise f")
    if not splits and untwisted > 1.0 - _TOL:
        raise ValueError("non-split extension but the untwisted rule was exact")
    return CocycleFit(
        splits=splits,
        untwisted_accuracy=untwisted,
        twisted_accuracy=twisted,
        quotient_order=extension.quotient_order,
        normal_order=extension.normal_order,
        used_trivialising_transversal=splits,
    )


# ---------------------------------------------------------------------------
# I-22 / I-22b: model-facing cocycle probe and ablation
# ---------------------------------------------------------------------------


def pair_features(model: GroupModel, order: int, *, site: str = "resid_final") -> np.ndarray:
    """Per-pair activation vectors over the whole Cayley grid, read at the '='
    position (-1): ``[|G|^2, d]`` for ``site in {"resid_final", "mlp_post"}``.

    Rows follow ``cayley_grid_tokens`` order (row ``k = (k // n, k % n)``).
    """
    if site not in ("resid_final", "mlp_post"):
        raise ValueError(f"site must be 'resid_final' or 'mlp_post', got {site!r}")
    tokens = cayley_grid_tokens(order)
    model.eval()
    rows: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, tokens.shape[0], 8192):
            cache = model(tokens[start : start + 8192], return_cache=True)
            value = cache[site]
            if value is None:
                raise ValueError(f"model has no {site} (use_mlp=false?)")
            rows.append(value[:, -1, :].detach().to(torch.float64).cpu())
    return torch.cat(rows, dim=0).numpy()


def cell_labels(extension: Extension, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Per grid-row ``(q1, q2)`` cell id and the cocycle value ``f(q1, q2)``.

    ``cell`` is ``q1 * |Q| + q2`` (the pseudoreplication unit -- many ``(a, b)``
    rows share one cell); ``f_label`` is the local ``N`` index of ``f(q1, q2)``.
    """
    grid = np.arange(order)
    a = np.repeat(grid, order)
    b = np.tile(grid, order)
    q1 = extension.coset_label[a]
    q2 = extension.coset_label[b]
    cell = q1 * extension.quotient_order + q2
    f_label = extension.cocycle[q1, q2]
    return cell.astype(np.int64), f_label.astype(np.int64)


def _degenerate_folds(cell: np.ndarray, f_label: np.ndarray) -> list[tuple[int, tuple[int, ...]]]:
    """Leave-one-cell-out folds whose held-out cell carries an ``f`` class that
    appears in no other cell.

    When the only cell realising an ``f`` class is held out, that class is absent
    from the training rows, so the decode cannot possibly predict it and the fold
    is structurally capped at chance -- independent of what the model encodes. For
    a normalised cocycle with ``Q = C2`` (``|Q| = 2``) the sole nontrivial cell is
    ``(1, 1)``, whose class exists nowhere else, so that fold is always degenerate.
    Returns ``(held_cell, missing_classes)`` for every degenerate fold.
    """
    degenerate: list[tuple[int, tuple[int, ...]]] = []
    for held in np.unique(cell):
        test_mask = cell == held
        train_classes = set(f_label[~test_mask].tolist())
        missing = sorted(set(f_label[test_mask].tolist()) - train_classes)
        if missing:
            degenerate.append((int(held), tuple(int(m) for m in missing)))
    return degenerate


def _f_partition_single_cell(extension: Extension, order: int) -> int | None:
    """The cell whose indicator the ``f``-label partition coincides with, or
    ``None``.

    When the ``f`` partition splits the rows into exactly one cell versus all the
    others -- the ``|Q| = 2`` normalised shape, where only cell ``(1, 1)`` is
    nontrivial -- the ``f``-carrying direction is that one cell's mean offset,
    which is confounded with ordinary quotient-pair sensitivity: any accurate
    model moves that mean and so shows an ablation effect regardless of whether it
    represents the cocycle. Returns the cell id so the record can flag the
    confound.
    """
    cell, f_label = cell_labels(extension, order)
    classes = np.unique(f_label)
    if classes.size != 2:
        return None
    for cls in classes:
        rows = np.flatnonzero(f_label == cls)
        cells_here = np.unique(cell[rows])
        if cells_here.size == 1:
            single = int(cells_here[0])
            if np.array_equal(np.sort(rows), np.flatnonzero(cell == single)):
                return single
    return None


def _ridge_multiclass(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, classes: np.ndarray, lam: float
) -> np.ndarray:
    """Closed-form ridge one-vs-all decode; returns predicted class per test row.

    A bias column is appended; the penalty ``lam`` is not applied to it.
    """
    n_features = x_train.shape[1]
    xb = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
    onehot = (y_train[:, None] == classes[None, :]).astype(np.float64)
    penalty = lam * np.eye(n_features + 1)
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(xb.T @ xb + penalty, xb.T @ onehot)
    xtb = np.hstack([x_test, np.ones((x_test.shape[0], 1))])
    scores = xtb @ weights
    return classes[scores.argmax(axis=1)]


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for cls in np.unique(y_true):
        mask = y_true == cls
        recalls.append(float((y_pred[mask] == cls).mean()))
    return float(np.mean(recalls))


@dataclass(frozen=True)
class ProbeResult:
    """I-22 cocycle-value probe, held out over ``(q1, q2)`` cells.

    ``defined`` is False on a split member with the complement transversal:
    ``f`` is constant, the decode target is degenerate, and the control is a fit
    residual, not a probe score (the record says ``UNDEFINED``). ``accuracy`` is
    over held-out cell rows; ``balanced_accuracy`` is the chance-corrected form;
    ``chance`` is the majority-class rate on the held-out rows.

    ``fold_degenerate`` is True when at least one leave-one-cell-out fold holds
    out the only cell that carries some ``f`` class -- so that class is absent
    from training and the fold is capped at chance whatever the model encodes.
    This is the ``Q = C2`` shape: the normalised cocycle is nontrivial in exactly
    one cell ``(1, 1)``, so holding that cell out always removes its class. A
    degenerate result must never ship as a plain ``measured`` chance-level number:
    ``accuracy``/``balanced_accuracy``/``chance`` are left ``None`` and
    ``to_record`` reports ``status == "STRUCTURALLY_DEGENERATE"``, naming the
    affected cells and classes. What C5 should measure instead for ``|Q| = 2`` is
    a pre-registration-level redesign, not something this instrument decides.
    """

    defined: bool
    site: str
    n_cells: int
    n_classes: int
    accuracy: float | None
    balanced_accuracy: float | None
    chance: float | None
    lam: float | None = None
    fold_degenerate: bool = False
    degenerate_cells: tuple[int, ...] = ()
    degenerate_classes: tuple[int, ...] = ()
    reason: str | None = None

    def to_record(self) -> dict[str, Any]:
        if not self.defined:
            return {"instrument": "cocycle_probe", "status": "UNDEFINED", "reason": self.reason}
        if self.fold_degenerate:
            return {
                "instrument": "cocycle_probe",
                "status": "STRUCTURALLY_DEGENERATE",
                "reason": self.reason,
                "site": self.site,
                "n_cells": self.n_cells,
                "n_classes": self.n_classes,
                "degenerate_cells": list(self.degenerate_cells),
                "degenerate_classes": list(self.degenerate_classes),
            }
        return {
            "instrument": "cocycle_probe",
            "status": "measured",
            "site": self.site,
            "n_cells": self.n_cells,
            "n_classes": self.n_classes,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "chance": self.chance,
            "lam": self.lam,
        }


def probe_cocycle(
    features: np.ndarray,
    extension: Extension,
    order: int,
    *,
    site: str = "resid_final",
    lam: float = 1.0,
) -> ProbeResult:
    """Decode ``f(q1, q2)`` from per-pair ``features``, held out over cells.

    Leave-one-cell-out: for each ``(q1, q2)`` cell, train the ridge decode on the
    rows of every *other* cell and predict the held-out cell's rows, so a cell's
    label is never seen in its own training fold. Degenerate (constant ``f``)
    targets return ``UNDEFINED`` -- the split member's control. When some fold
    holds out the only cell carrying an ``f`` class (the ``Q = C2`` shape, where
    the sole nontrivial cell is ``(1, 1)``), the probe is structurally capped at
    chance and the record is marked ``STRUCTURALLY_DEGENERATE`` rather than shipped
    as a measured chance-level number.
    """
    _require_finite(features, "probe features")
    cell, f_label = cell_labels(extension, order)
    distinct_f = np.unique(f_label)
    if distinct_f.size < 2:
        return ProbeResult(
            defined=False,
            site=site,
            n_cells=int(np.unique(cell).size),
            n_classes=int(distinct_f.size),
            accuracy=None,
            balanced_accuracy=None,
            chance=None,
            lam=lam,
            reason="cocycle is constant (split member with the complement transversal): "
            "the decode target is degenerate, so the control is a fit residual not a probe score",
        )
    degenerate = _degenerate_folds(cell, f_label)
    if degenerate:
        degenerate_cells = tuple(held for held, _ in degenerate)
        degenerate_classes = tuple(sorted({m for _, missing in degenerate for m in missing}))
        return ProbeResult(
            defined=True,
            site=site,
            n_cells=int(np.unique(cell).size),
            n_classes=int(distinct_f.size),
            accuracy=None,
            balanced_accuracy=None,
            chance=None,
            lam=lam,
            fold_degenerate=True,
            degenerate_cells=degenerate_cells,
            degenerate_classes=degenerate_classes,
            reason="leave-one-cell-out is structurally degenerate: cell(s) "
            f"{list(degenerate_cells)} carry f class(es) {list(degenerate_classes)} that "
            "appear in no other cell, so the class is absent from training whenever its only "
            "cell is held out and the probe is capped at chance regardless of the model "
            "(the Q = C2 shape: the sole nontrivial cocycle cell is (1, 1)). What C5 should "
            "measure for |Q| = 2 is a pre-registration-level redesign.",
        )
    cells = np.unique(cell)
    preds = np.empty(features.shape[0], dtype=np.int64)
    preds.fill(-1)
    for held in cells:
        test_mask = cell == held
        train_mask = ~test_mask
        train_classes = np.unique(f_label[train_mask])
        preds[test_mask] = _ridge_multiclass(
            features[train_mask], f_label[train_mask], features[test_mask], train_classes, lam
        )
    accuracy = float((preds == f_label).mean())
    balanced = _balanced_accuracy(f_label, preds)
    _, counts = np.unique(f_label, return_counts=True)
    chance = float(counts.max() / counts.sum())
    return ProbeResult(
        defined=True,
        site=site,
        n_cells=int(cells.size),
        n_classes=int(distinct_f.size),
        accuracy=accuracy,
        balanced_accuracy=balanced,
        chance=chance,
        lam=lam,
    )


def _unembed(model: GroupModel) -> np.ndarray:
    return model.W_U.detach().to(torch.float64).cpu().numpy()


def _grid_targets(group: FiniteGroup) -> np.ndarray:
    return group.cayley_table.reshape(-1)


@dataclass(frozen=True)
class AblationResult:
    """I-22b: ablation of the ``f``-carrying direction vs a matched random one.

    ``defined`` is False on a split member (no ``f`` direction to remove).
    ``delta_accuracy`` is the drop in full-grid argmax accuracy caused by
    removing the decoded ``f``-subspace at ``site``; ``random_delta_accuracy`` is
    the same for a norm/dimension-matched random subspace (I-03). A larger
    ``f``-subspace drop than the random control is what licenses "the model uses
    the cocycle"; the record is estimation-first (effect sizes, no threshold).

    Confound at ``|Q| = 2``: with a two-cell ``f`` partition the ``f``-carrying
    subspace is just the single nontrivial cell's mean-offset direction, so the
    ablation is confounded with ordinary quotient-pair (cell) sensitivity -- any
    model accurate on that cell shows an effect, whether or not it represents the
    cocycle as such. ``f_partition_single_cell`` records the cell whose indicator
    the ``f`` partition coincides with (``None`` when it does not), so a reader can
    see when this confound applies; the measurement itself is unchanged.
    """

    defined: bool
    site: str
    subspace_dim: int
    baseline_accuracy: float | None
    delta_accuracy: float | None
    random_delta_accuracy: float | None
    seed: int = 0
    f_partition_single_cell: int | None = None
    reason: str | None = None

    def to_record(self) -> dict[str, Any]:
        if not self.defined:
            return {"instrument": "cocycle_ablation", "status": "UNDEFINED", "reason": self.reason}
        return {
            "instrument": "cocycle_ablation",
            "status": "measured",
            "site": self.site,
            "subspace_dim": self.subspace_dim,
            "baseline_accuracy": self.baseline_accuracy,
            "delta_accuracy": self.delta_accuracy,
            "random_delta_accuracy": self.random_delta_accuracy,
            "seed": self.seed,
            "f_partition_single_cell": self.f_partition_single_cell,
            "f_partition_is_single_cell_indicator": self.f_partition_single_cell is not None,
        }


def _accuracy_from_features(
    features: np.ndarray, unembed: np.ndarray, targets: np.ndarray
) -> float:
    logits = features @ unembed
    return float((logits.argmax(axis=1) == targets).mean())


def cocycle_ablation(
    model: GroupModel,
    group: FiniteGroup,
    extension: Extension,
    *,
    subspace_dim: int | None = None,
    seed: int = 0,
) -> AblationResult:
    """I-22b: remove the decoded ``f``-direction at ``resid_final`` and measure
    the accuracy drop, against a norm/dimension-matched random subspace.

    The ``f``-subspace is the leading right-singular directions of the
    class-mean matrix of ``f`` over cells (the directions along which the mean
    ``resid_final`` moves with ``f``). Both architectures compute
    ``logits = resid_final @ W_U``, so ablation is exact by projecting the
    subspace out of ``resid_final`` and re-reading the logits.

    ``subspace_dim`` defaults to ``|distinct f| - 1`` (the rank of the centred
    class-mean matrix); an explicit value must be a positive integer.

    At ``|Q| = 2`` the ``f`` partition is a single-cell indicator and the
    ablation is confounded with quotient-pair sensitivity; the returned record's
    ``f_partition_single_cell`` field flags this (the measurement is unchanged).
    """
    if subspace_dim is not None and subspace_dim <= 0:
        raise ValueError("subspace_dim must be a positive integer or None (0 is rejected)")
    single_cell = _f_partition_single_cell(extension, group.order)
    features = pair_features(model, group.order, site="resid_final")
    _require_finite(features, "resid_final features")
    _, f_label = cell_labels(extension, group.order)
    distinct = np.unique(f_label)
    if distinct.size < 2:
        return AblationResult(
            defined=False,
            site="resid_final",
            subspace_dim=0,
            baseline_accuracy=None,
            delta_accuracy=None,
            random_delta_accuracy=None,
            seed=seed,
            f_partition_single_cell=single_cell,
            reason="cocycle is constant (split member): no f-direction exists to ablate",
        )
    unembed = _unembed(model)
    _require_finite(unembed, "unembedding W_U")
    targets = _grid_targets(group)
    baseline = _accuracy_from_features(features, unembed, targets)

    # Class means of resid_final by f value, centred: their span is the
    # f-carrying subspace.
    grand = features.mean(axis=0, keepdims=True)
    means = np.stack([features[f_label == cls].mean(axis=0) for cls in distinct]) - grand
    _, _, vh = np.linalg.svd(means, full_matrices=False)
    requested = subspace_dim if subspace_dim is not None else distinct.size - 1
    dim = min(requested, vh.shape[0])
    basis = vh[:dim]  # [dim, d]
    projected = features - (features @ basis.T) @ basis
    delta = baseline - _accuracy_from_features(projected, unembed, targets)

    # Norm/dimension-matched random subspace control (I-03).
    rng = np.random.default_rng(seed)
    random_basis = rng.standard_normal((dim, features.shape[1]))
    q_basis, _ = np.linalg.qr(random_basis.T)
    q_basis = q_basis[:, :dim].T
    projected_rand = features - (features @ q_basis.T) @ q_basis
    random_delta = baseline - _accuracy_from_features(projected_rand, unembed, targets)

    return AblationResult(
        defined=True,
        site="resid_final",
        subspace_dim=dim,
        baseline_accuracy=baseline,
        delta_accuracy=delta,
        random_delta_accuracy=random_delta,
        seed=seed,
        f_partition_single_cell=single_cell,
    )


# ---------------------------------------------------------------------------
# I-22c: twisted-rule fraction-of-variance-explained (FVE) fit
# ---------------------------------------------------------------------------


def _predicted_products(
    extension: Extension, table: np.ndarray, *, include_cocycle: bool
) -> np.ndarray:
    """Predicted product element for every grid row under the coordinate rule.

    Row ``k = (g1, g2)`` in ``cayley_grid_tokens`` order (``g1 = k // |G|``,
    ``g2 = k % |G|``), so the returned ``[|G|^2]`` array aligns with
    :func:`cell_labels` and :func:`pair_features` row-for-row. With
    ``include_cocycle=True`` this is the twisted rule and reproduces the whole
    table; with ``include_cocycle=False`` it is the untwisted (``f == e``) rule,
    which errs on every cell whose cocycle value is nontrivial. Applies the same
    rule as :func:`coordinate_product_accuracy` without altering that function.
    """
    normal_array = np.array(extension.normal, dtype=np.int64)
    n_of, q_of = coordinates(extension, table)
    transversal = extension.transversal
    action = extension.action
    cocycle = extension.cocycle
    qt = extension.quotient_table
    order = extension.order
    predicted = np.empty(order * order, dtype=np.int64)
    for g1 in range(order):
        n1, q1 = int(n_of[g1]), int(q_of[g1])
        for g2 in range(order):
            n2, q2 = int(n_of[g2]), int(q_of[g2])
            phi = int(action[q1, n2])
            n_res = _n_mul_local(table, normal_array, n1, phi)
            if include_cocycle:
                n_res = _n_mul_local(table, normal_array, n_res, int(cocycle[q1, q2]))
            qk = int(qt[q1, q2])
            predicted[g1 * order + g2] = int(table[normal_array[n_res], int(transversal[qk])])
    return predicted


def _rule_indicator_design(predicted: np.ndarray, n_classes: int) -> np.ndarray:
    """One-hot design ``[N, n_classes, 1]`` -- the answer a coordinate rule
    predicts per pair, the single closed-form feature (mirrors I-27's indicator
    design in ``probes._signed_cyclic_forms``)."""
    design = np.zeros((predicted.shape[0], n_classes, 1), dtype=np.float64)
    design[np.arange(predicted.shape[0]), predicted, 0] = 1.0
    return design


def _leave_one_cell_out(cell: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_mask, test_mask)`` holding out one whole ``(q1, q2)`` cell
    at a time.

    A cell is the pseudoreplication unit -- many ``(a, b)`` rows share it and the
    cocycle value is constant within it -- so it is never split across the
    fit/held-out boundary. This is the same discipline :func:`probe_cocycle`
    uses (banned practice 6).
    """
    for held in np.unique(cell):
        test_mask = cell == held
        yield ~test_mask, test_mask


def _cell_held_out_fve(logits: np.ndarray, design: np.ndarray, cell: np.ndarray) -> float:
    """Held-out FVE of a closed-form rule against the model's logits, held out
    over ``(q1, q2)`` cells.

    ``logits`` is ``[N, n_classes]`` and ``design`` is ``[N, n_classes,
    n_features]``. Softmax is shift-invariant, so both are mean-centred across the
    class axis first (a constant per-pair shift carries no information). For each
    held-out cell the regression coefficients are fitted on every *other* cell's
    rows and used to predict the held-out cell; the FVE pools all held-out
    predictions. Mirrors ``probes._held_out_fve`` but with a leave-one-cell-out
    split, never an ``(a, b)``-pair split.
    """
    _require_finite(logits, "FVE target logits")
    y = logits - logits.mean(axis=1, keepdims=True)
    x = design - design.mean(axis=1, keepdims=True)
    preds = np.empty_like(y)
    for train_mask, test_mask in _leave_one_cell_out(cell):
        x_fit = x[train_mask].reshape(-1, x.shape[2])
        y_fit = y[train_mask].reshape(-1)
        coef, _, _, _ = np.linalg.lstsq(x_fit, y_fit, rcond=None)
        preds[test_mask] = x[test_mask] @ coef
    residual = (y - preds).reshape(-1)
    y_flat = y.reshape(-1)
    ss_res = float(residual @ residual)
    centred = y_flat - y_flat.mean()
    ss_tot = float(centred @ centred)
    if ss_tot <= 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


@dataclass(frozen=True)
class TwistedFveResult:
    """I-22c: the twisted-rule fraction-of-variance-explained (FVE) fit.

    ``twisted_fve`` and ``untwisted_fve`` are the held-out FVE of the model's
    logits regressed onto the answer the twisted and the untwisted (``f == e``)
    coordinate rules predict; ``twist_fve_gain`` (twisted minus untwisted) is the
    extra logit variance the cocycle term explains. Both are held out over
    ``(q1, q2)`` cells, never over ``(a, b)`` pairs (the cocycle label is cell-
    constant, so a pair split would leak it -- banned practice 6).

    ``twist_is_trivial`` records whether ``f == e`` for this transversal: on a
    split member with the complement transversal the two rules coincide and the
    gain is ~0 by construction (the split control); a non-split member needs the
    twist and the gain is positive.

    Interpretation caveat at ``|Q| = 2`` (``fold_degenerate``): when the only cell
    carrying the nontrivial cocycle is a single cell ``(1, 1)``, the twisted and
    untwisted designs coincide on every *training* cell of that fold and differ
    only on the held-out cell itself. The gain then measures accuracy on the
    twisted cell -- whether the model gets that cell right -- not that the model
    computes multiplication *via* the cocycle mechanism; the two are not
    distinguishable here. ``degenerate_cells`` names the affected cell(s). The
    number is a genuine held-out variance-explained; only the mechanistic reading
    is limited, so the record stays ``measured`` with the flag set.
    """

    site: str
    n_cells: int
    n_features: int
    twist_is_trivial: bool
    twisted_fve: float
    untwisted_fve: float
    twist_fve_gain: float
    fold_degenerate: bool = False
    degenerate_cells: tuple[int, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "instrument": "cocycle_twisted_fve",
            "status": "measured",
            "site": self.site,
            "n_cells": self.n_cells,
            "n_features": self.n_features,
            "twist_is_trivial": self.twist_is_trivial,
            "twisted_fve": self.twisted_fve,
            "untwisted_fve": self.untwisted_fve,
            "twist_fve_gain": self.twist_fve_gain,
            "fold_degenerate": self.fold_degenerate,
            "degenerate_cells": list(self.degenerate_cells),
        }


def twisted_rule_fve(
    model: GroupModel,
    group: FiniteGroup,
    extension: Extension,
) -> TwistedFveResult:
    """I-22c: fit the twisted multiplication rule to the model's logits and report
    how much extra variance the cocycle term explains.

    The FVE target is the model's read-position logits ``resid_final @ W_U``, the
    network's own output -- both architectures compute the logits this way (the
    same read :func:`cocycle_ablation` uses, so ``mlp_post`` is not a logit
    source here). The twisted rule's one-hot answer indicator reproduces the
    Cayley table exactly; the untwisted (``f == e``) indicator errs wherever the
    cocycle is nontrivial, so the held-out FVE gap (``twist_fve_gain``) is the
    twist's contribution. This is the GL(2,3) [split] vs SL(2,3).C2 [non-split]
    contrast: on the split member the trivialising transversal makes ``f == e``
    and the gain is ~0; on the non-split member no transversal trivialises ``f``
    and the gain is positive.

    Held out over ``(q1, q2)`` cells (the cocycle label is cell-constant, so an
    ``(a, b)``-pair split would be pseudoreplication -- banned practice 6), the
    same discipline :func:`probe_cocycle` uses. Deliberately *not*
    ``probes.functional_form_fit``, which holds out over ``(a, b)`` pairs and
    would reintroduce the banned practice.

    At ``|Q| = 2`` the gain is mechanism-independent: the twisted and untwisted
    designs differ only on the single nontrivial cell, so a positive gain merely
    means the model is accurate on that cell, not that it uses the cocycle. The
    returned record's ``fold_degenerate`` flag records this (see
    :class:`TwistedFveResult`); the FVE numbers themselves stay valid.
    """
    features = pair_features(model, group.order, site="resid_final")
    _require_finite(features, "resid_final features")
    unembed = _unembed(model)
    _require_finite(unembed, "unembedding W_U")
    logits = features @ unembed
    n_classes = logits.shape[1]
    cell, f_label = cell_labels(extension, group.order)
    table = group.cayley_table
    twisted_design = _rule_indicator_design(
        _predicted_products(extension, table, include_cocycle=True), n_classes
    )
    untwisted_design = _rule_indicator_design(
        _predicted_products(extension, table, include_cocycle=False), n_classes
    )
    twisted_fve = _cell_held_out_fve(logits, twisted_design, cell)
    untwisted_fve = _cell_held_out_fve(logits, untwisted_design, cell)
    degenerate = _degenerate_folds(cell, f_label)
    return TwistedFveResult(
        site="resid_final",
        n_cells=int(np.unique(cell).size),
        n_features=int(twisted_design.shape[2]),
        twist_is_trivial=bool(cocycle_is_trivial(extension)),
        twisted_fve=float(twisted_fve),
        untwisted_fve=float(untwisted_fve),
        twist_fve_gain=float(twisted_fve - untwisted_fve),
        fold_degenerate=bool(degenerate),
        degenerate_cells=tuple(held for held, _ in degenerate),
    )


# ---------------------------------------------------------------------------
# I-22d: intervention-based predicted-error test (the |Q| = 2 replacement)
# ---------------------------------------------------------------------------


def _row_coordinate_products(
    extension: Extension, table: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per grid-row ``(n_res, qk, untwisted)`` in ``cayley_grid_tokens`` order.

    ``n_res[k]`` is the local ``N`` index of ``n1 . phi_{q1}(n2)`` (the twisted
    product's ``N``-part *before* the cocycle factor); ``qk[k]`` is the quotient
    product ``q1 q2``; ``untwisted[k]`` is the untwisted (``f == e``) product
    element ``compose(n_res, s(qk))``. Row ``k = (g1, g2)`` with ``g1 = k //
    |G|``, ``g2 = k % |G|`` -- aligned row-for-row with :func:`cell_labels`,
    :func:`pair_features`, and :func:`_predicted_products`. Computed in one pass
    rather than by reusing :func:`_predicted_products` because I-22d needs
    ``n_res``/``qk`` to build counterfactual cocycle-value targets, which that
    function does not expose; :func:`_predicted_products` is left untouched.
    """
    order = extension.order
    normal_array = np.array(extension.normal, dtype=np.int64)
    n_of, q_of = coordinates(extension, table)
    action = extension.action
    qt = extension.quotient_table
    transversal = extension.transversal
    n_res = np.empty(order * order, dtype=np.int64)
    qk = np.empty(order * order, dtype=np.int64)
    untwisted = np.empty(order * order, dtype=np.int64)
    for g1 in range(order):
        n1, q1 = int(n_of[g1]), int(q_of[g1])
        for g2 in range(order):
            n2, q2 = int(n_of[g2]), int(q_of[g2])
            phi = int(action[q1, n2])
            nr = _n_mul_local(table, normal_array, n1, phi)
            k = int(qt[q1, q2])
            idx = g1 * order + g2
            n_res[idx] = nr
            qk[idx] = k
            untwisted[idx] = int(table[normal_array[nr], int(transversal[k])])
    return n_res, qk, untwisted


def _cocycle_value_target(
    extension: Extension, table: np.ndarray, n_res: np.ndarray, qk: np.ndarray, c_local: int
) -> np.ndarray:
    """Per-row product element the coordinate rule predicts when the cocycle
    factor is the fixed ``N``-element ``c_local`` (a local index) on *every* row:
    ``compose(n_res . c_local, s(qk))``. ``c_local = e`` reproduces the untwisted
    answer; ``c_local = f(q1, q2)`` reproduces the true answer on that cell. Other
    values give structurally identical wrong targets in the same coset -- the
    counterfactual-cocycle matched controls."""
    order = extension.order
    normal_array = np.array(extension.normal, dtype=np.int64)
    out = np.empty(order * order, dtype=np.int64)
    for r in range(order * order):
        nf = _n_mul_local(table, normal_array, int(n_res[r]), c_local)
        out[r] = int(table[normal_array[nf], int(extension.transversal[int(qk[r])])])
    return out


def _hit_ci(indicators: np.ndarray, *, seed: int = 0) -> list[float] | None:
    """95% bootstrap CI of a 0/1 hit-rate over rows, or ``None`` when it is
    undefined (fewer than two rows) or degenerate (every row identical, so the
    rate has no sampling spread)."""
    values = [float(v) for v in indicators.tolist()]
    if len(values) < 2 or len(set(values)) < 2:
        return None
    low, high = stats.bootstrap_ci(values, seed=seed)
    return [low, high]


@dataclass(frozen=True)
class PredictedErrorResult:
    """I-22d: the intervention-based predicted-error (untwisted-target) test.

    The discriminator, stated as an argument that it is **not** a function of
    cell membership: on a twist-active cell the model's winning (correct)
    prediction is knocked out -- its own argmax logit is set to ``-inf`` and the
    readout re-argmaxed -- and we score the fraction of these induced errors that
    land on the *row-specific* untwisted answer ``untwisted(a, b)``. Because
    ``true = untwisted . w`` for a fixed ``w`` on the cell, ``untwisted(a, b)``
    varies from row to row *within* the cell, while cell membership is by
    definition constant on the cell. So a model whose cell-(1,1) behaviour is a
    pure function of cell membership (a lookup table) cannot preferentially hit
    the row-specific target, whereas a model that computes the untwisted product
    and applies the cocycle correction falls back onto it exactly. Two matched
    controls turn the hit rate into an effect: a *shuffled-correspondence* null
    (permute which untwisted target each induced error is scored against, so only
    the row-specific correspondence -- not the marginal, not cell membership --
    can raise the score) and, where ``|N| >= 3``, *counterfactual cocycle-value*
    targets (structurally identical wrong answers for the wrong cocycle value).

    ``status``: ``"measured"`` when there are twist-active cells and at least one
    correctly-answered twist-active row; ``"twist_inactive"`` (a UNDEFINED-shaped
    verdict) when the coordinatisation has no twist-active cell -- the split
    member with the trivialising transversal, the built-in negative control;
    ``"no_correct_active_rows"`` when twist-active cells exist but the model is
    wrong on all of them (a censored/ungrokked seed), so the knockout has no
    correct prediction to displace. Estimation-first: effect sizes with CIs, no
    threshold. ``untwist_hit_rate`` and its ``untwist_hit_ci`` are the headline;
    ``effect_vs_shuffle``/``effect_vs_counterfactual`` are the hit rate minus each
    matched null's mean.
    """

    status: str
    n_active_cells: int
    n_active_rows: int
    n_correct_active_rows: int
    normal_order: int
    n_shuffle: int
    seed: int
    untwist_hit_rate: float | None = None
    untwist_hit_ci: list[float] | None = None
    shuffle_null_mean: float | None = None
    shuffle_null_ci: list[float] | None = None
    effect_vs_shuffle: float | None = None
    counterfactual_null_mean: float | None = None
    effect_vs_counterfactual: float | None = None
    active_cells: tuple[int, ...] = ()
    reason: str | None = None

    def to_record(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "instrument": "cocycle_predicted_error",
            "status": self.status,
            "n_active_cells": self.n_active_cells,
            "n_active_rows": self.n_active_rows,
            "n_correct_active_rows": self.n_correct_active_rows,
            "normal_order": self.normal_order,
            "active_cells": list(self.active_cells),
            "seed": self.seed,
        }
        if self.status != "measured":
            base["reason"] = self.reason
            return base
        base.update(
            {
                "n_shuffle": self.n_shuffle,
                "untwist_hit_rate": self.untwist_hit_rate,
                "untwist_hit_ci": self.untwist_hit_ci,
                "shuffle_null": {"mean": self.shuffle_null_mean, "ci_95": self.shuffle_null_ci},
                "effect_vs_shuffle": self.effect_vs_shuffle,
                "counterfactual_null": (
                    None
                    if self.counterfactual_null_mean is None
                    else {"mean": self.counterfactual_null_mean}
                ),
                "effect_vs_counterfactual": self.effect_vs_counterfactual,
            }
        )
        return base


def predicted_error_intervention(
    logits: np.ndarray,
    group: FiniteGroup,
    extension: Extension,
    *,
    seed: int = 0,
    n_shuffle: int = 500,
) -> PredictedErrorResult:
    """I-22d: the winner-knockout predicted-error test on the model's read-position
    ``logits`` (``[|G|^2, |G|]`` in :func:`cayley_grid_tokens` order -- typically
    ``pair_features(...) @ W_U``).

    For every grid row where the untwisted product differs from the true product
    (the twist-active rows) *and* the model's argmax equals the true product, the
    model's own winning logit is knocked out (set to ``-inf``) and the readout is
    re-argmaxed. The instrument scores the fraction of these knocked-out rows
    whose fallback answer is the row-specific untwisted product, against a
    shuffled-correspondence null (``n_shuffle`` seeded permutations of the
    target-to-row assignment) and, where ``|N| >= 3``, counterfactual
    cocycle-value targets. All randomness is seeded; the record is estimation-
    first (see :class:`PredictedErrorResult`).

    ``twist_inactive`` (the split control) when the coordinatisation has no
    twist-active cell; ``no_correct_active_rows`` when the model is wrong on every
    twist-active row. Use the trivialising transversal for a split member (so it
    reports ``twist_inactive``) and any transversal for a non-split member.
    """
    _require_finite(logits, "predicted-error logits")
    table = group.cayley_table
    order = group.order
    if logits.shape != (order * order, order):
        raise ValueError(
            f"logits must be [|G|^2, |G|] = [{order * order}, {order}]; got {logits.shape}"
        )
    true = table.reshape(-1)
    n_res, qk, untwisted = _row_coordinate_products(extension, table)
    active = untwisted != true
    cell, _ = cell_labels(extension, order)
    active_cells = tuple(int(c) for c in np.unique(cell[active]))
    n_active_rows = int(active.sum())

    if n_active_rows == 0:
        return PredictedErrorResult(
            status="twist_inactive",
            n_active_cells=0,
            n_active_rows=0,
            n_correct_active_rows=0,
            normal_order=extension.normal_order,
            n_shuffle=n_shuffle,
            seed=seed,
            reason="no twist-active cell for this coordinatisation: the untwisted rule "
            "already reproduces every product (a split member with the trivialising "
            "transversal, or a trivial cocycle) -- the built-in split negative control",
        )

    pred = logits.argmax(axis=1)
    active_idx = np.flatnonzero(active)
    correct_active = active_idx[pred[active_idx] == true[active_idx]]
    if correct_active.size == 0:
        return PredictedErrorResult(
            status="no_correct_active_rows",
            n_active_cells=len(active_cells),
            n_active_rows=n_active_rows,
            n_correct_active_rows=0,
            normal_order=extension.normal_order,
            n_shuffle=n_shuffle,
            seed=seed,
            active_cells=active_cells,
            reason="the model answers no twist-active row correctly (a censored/ungrokked "
            "seed): the winner-knockout has no correct prediction to displace",
        )

    # Winner-knockout: remove each row's own argmax logit, re-argmax for the fallback.
    knocked = logits.copy()
    knocked[np.arange(order * order), pred] = -np.inf
    fallback = knocked.argmax(axis=1)

    untwist_targets = untwisted[correct_active]
    fallback_active = fallback[correct_active]
    hit_indicator = (fallback_active == untwist_targets).astype(np.float64)
    untwist_hit_rate = float(hit_indicator.mean())

    # Shuffled-correspondence null: permute which untwisted target each fallback
    # is scored against, so only the true row-specific correspondence -- not the
    # marginal frequency of untwisted-shaped answers, and not cell membership --
    # can raise the score above this null.
    rng = np.random.default_rng(seed)
    shuffle_rates = np.empty(n_shuffle, dtype=np.float64)
    for i in range(n_shuffle):
        permuted = untwist_targets[rng.permutation(correct_active.size)]
        shuffle_rates[i] = float((fallback_active == permuted).mean())
    shuffle_mean = float(shuffle_rates.mean())
    shuffle_ci = [
        float(np.quantile(shuffle_rates, 0.025)),
        float(np.quantile(shuffle_rates, 0.975)),
    ]

    # Counterfactual cocycle-value null (|N| >= 3): the fraction of fallbacks
    # landing on target(c) for cocycle values c that are neither the identity
    # (the untwisted target) nor the row's own true cocycle value.
    counterfactual_mean: float | None = None
    if extension.normal_order >= 3:
        zero_local = extension.n_local[_identity_from_normal(extension)]
        f_local = extension.cocycle[
            extension.coset_label[correct_active // order],
            extension.coset_label[correct_active % order],
        ]
        per_c = []
        for c_local in range(extension.normal_order):
            if c_local == zero_local:
                continue
            target_c = _cocycle_value_target(extension, table, n_res, qk, c_local)[correct_active]
            usable = f_local != c_local  # exclude the true cocycle value per row
            if bool(usable.any()):
                per_c.append(float((fallback_active[usable] == target_c[usable]).mean()))
        if per_c:
            counterfactual_mean = float(np.mean(per_c))

    return PredictedErrorResult(
        status="measured",
        n_active_cells=len(active_cells),
        n_active_rows=n_active_rows,
        n_correct_active_rows=int(correct_active.size),
        normal_order=extension.normal_order,
        n_shuffle=n_shuffle,
        seed=seed,
        untwist_hit_rate=untwist_hit_rate,
        untwist_hit_ci=_hit_ci(hit_indicator, seed=seed),
        shuffle_null_mean=shuffle_mean,
        shuffle_null_ci=shuffle_ci,
        effect_vs_shuffle=untwist_hit_rate - shuffle_mean,
        counterfactual_null_mean=counterfactual_mean,
        effect_vs_counterfactual=(
            None if counterfactual_mean is None else untwist_hit_rate - counterfactual_mean
        ),
        active_cells=active_cells,
    )


def cocycle_predicted_error(
    model: GroupModel,
    group: FiniteGroup,
    extension: Extension,
    *,
    seed: int = 0,
    n_shuffle: int = 500,
) -> PredictedErrorResult:
    """I-22d on a trained ``model``: read the read-position logits
    (``resid_final @ W_U``, both architectures' output path) and run
    :func:`predicted_error_intervention`. Within-model only (two-track rule)."""
    features = pair_features(model, group.order, site="resid_final")
    _require_finite(features, "resid_final features")
    unembed = _unembed(model)
    _require_finite(unembed, "unembedding W_U")
    logits = features @ unembed
    return predicted_error_intervention(logits, group, extension, seed=seed, n_shuffle=n_shuffle)


# ---------------------------------------------------------------------------
# Run-directory record builder: the full C5 arm (I-21/I-22/I-22b/I-22c/I-22d)
# ---------------------------------------------------------------------------


def select_normal_subgroup(group: FiniteGroup, normal_order: int) -> np.ndarray | None:
    """The deterministic normal subgroup of order ``normal_order`` C5 coordinatises
    over, or ``None`` when the group's artifact exports no such subgroup.

    Picks the lowest-``subgroups``-index normal subgroup of the requested order
    (the same rule the tests' ``_first_normal`` uses), so the choice is fixed and
    the record pins its exact membership -- a group can carry several normal
    subgroups of one order, and the cocycle depends on *which* ``N`` is chosen."""
    if not getattr(group, "subgroups", None):
        return None
    table = group.cayley_table
    for subgroup in group.subgroups:
        members = np.asarray(subgroup, dtype=np.int64)
        if members.size == normal_order and is_normal(table, members):
            return np.array(sorted(int(x) for x in members.tolist()), dtype=np.int64)
    return None


def measure_cocycle_run(
    run_dir: Path,
    *,
    metric: str = "val/accuracy",
    threshold: float = 0.99,
    normal_order: int | None = None,
    n_shuffle: int = 500,
) -> dict[str, Any]:
    """The C5 mechanism measurement for one run: I-21 precompute, the I-22/I-22b/
    I-22c decode/ablation/fit (carrying their ``|Q| = 2`` structural-degeneracy
    guards), and the I-22d intervention-based predicted-error test, all behind the
    dip-aware checkpoint rule.

    Offline and reproducible: no network, no W&B, no Sage/GAP. A run whose
    dip-aware selection finds no stable checkpoint (a censored/never-grokked seed)
    returns ``status: "skipped"`` with the selection record, never a silently-
    analysed unstable model. When the group's artifact exports no normal subgroup
    of the requested order the record is ``status: "skipped"`` with a
    ``skip_reason`` (never a wrong number). ``normal_order`` defaults to
    ``order // 2`` (the index-2 ``N`` of the ``|Q| = C2`` C5 pair); the split
    member is coordinatised with its trivialising transversal so I-22/I-22b/I-22c
    report their split controls and I-22d reports ``twist_inactive``. Provenance
    pins the manifest hashes, the analysis git commit and dirty flag, the
    checkpoint sha256, this package's module sha256s, and the group-artifact file
    (relative path and sha256) the extension is built from."""
    import yaml

    from ..config import validate_config
    from ..groups.catalog import resolve_group
    from ..groups.data import artifact_path
    from ..manifest import get_git_commit, get_git_dirty, read_manifest
    from ..training.trainer import build_model
    from .checkpoints import select_checkpoint
    from .report import file_sha256, instrument_code_hashes

    manifest = read_manifest(run_dir)
    config = validate_config(yaml.safe_load((run_dir / "resolved_config.yaml").read_text()))
    selection = select_checkpoint(run_dir, metric=metric, threshold=threshold)
    art_path = artifact_path(config.data.group.order, config.data.group.index)
    repo_root = Path(__file__).resolve().parents[3]
    try:
        art_rel = str(art_path.relative_to(repo_root))
    except ValueError:
        art_rel = art_path.name
    record: dict[str, Any] = {
        "instrument": "cocycle",
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
            "analysis_git_dirty": get_git_dirty(),
            "instrument_code_sha256": instrument_code_hashes(),
            "group_artifact": art_rel,
            "group_artifact_sha256": (file_sha256(art_path) if art_path.is_file() else None),
        },
    }
    if selection.path is None:
        record["status"] = "skipped"
        return record

    group = resolve_group(config.data.group)
    chosen_order = normal_order if normal_order is not None else group.order // 2
    record["normal_order"] = chosen_order
    normal = select_normal_subgroup(group, chosen_order)
    if normal is None:
        record["status"] = "skipped"
        record["skip_reason"] = (
            f"no normal subgroup of order {chosen_order} in this group's exported artifact "
            "(re-export with --include-subgroups); the C5 extension cannot be coordinatised"
        )
        return record

    checkpoint = torch.load(selection.path, map_location="cpu", weights_only=False)
    model = build_model(config, group)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Split member: coordinatise with the trivialising transversal so I-22/I-22b/
    # I-22c report their split controls (probe UNDEFINED, no f-direction, gain ~0)
    # and I-22d reports twist_inactive; non-split: any (canonical) transversal.
    transversal = trivialising_transversal(group, normal)
    extension = build_extension(group, normal, transversal)
    features = pair_features(model, group.order, site="resid_final")

    record["status"] = "measured"
    record["provenance"]["checkpoint_sha256"] = file_sha256(selection.path)
    record["normal_membership"] = [int(x) for x in normal.tolist()]
    record["extension"] = extension.to_record()
    record["f_equiv_one_fit"] = f_equiv_one_fit(group, normal).to_record()
    record["i22_cocycle_probe"] = probe_cocycle(features, extension, group.order).to_record()
    record["i22b_cocycle_ablation"] = cocycle_ablation(
        model, group, extension, seed=config.seed
    ).to_record()
    record["i22c_twisted_fve"] = twisted_rule_fve(model, group, extension).to_record()
    record["i22d_predicted_error"] = cocycle_predicted_error(
        model, group, extension, seed=config.seed, n_shuffle=n_shuffle
    ).to_record()
    return record


__all__ = [
    "AblationResult",
    "CocycleFit",
    "Extension",
    "PredictedErrorResult",
    "ProbeResult",
    "SplitResult",
    "TwistedFveResult",
    "build_extension",
    "cocycle_predicted_error",
    "measure_cocycle_run",
    "predicted_error_intervention",
    "select_normal_subgroup",
    "cell_labels",
    "cocycle_ablation",
    "cocycle_is_trivial",
    "coordinate_product_accuracy",
    "coordinates",
    "f_equiv_one_fit",
    "is_normal",
    "is_subgroup",
    "pair_features",
    "probe_cocycle",
    "split_status",
    "subgroup_closure",
    "trivialising_transversal",
    "twisted_rule_fve",
]
