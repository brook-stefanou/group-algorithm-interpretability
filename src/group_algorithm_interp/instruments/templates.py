"""Core-free subgroup enumeration (I-12) and ``Ind_H^G 1`` templates (I-13).

The coset (T5) account predicts occupancy concentrated on the irreps of
``Ind_H^G 1`` for a small-index core-free subgroup ``H``. This module computes,
from exported artifact data alone (Cayley table, per-irrep characters, isotypic
blocks -- no Sage/GAP at analysis time):

* which exported subgroups are core-free (the core ``core(H) = intersection of
  all conjugates x H x^-1`` is the largest normal subgroup inside ``H``;
  core-free means it is trivial), and the minimal core-free index over them;
* for each nontrivial core-free ``H``, the multiplicity of each irrep in
  ``Ind_H^G 1`` by Frobenius reciprocity, ``m_chi = (1/|H|) sum_{h in H}
  chi(h)``, and the energy template over real isotypic blocks
  ``t_j = sum_{i in block j} m_i * d_i / [G:H]`` -- with the self-verifying
  identity ``sum_i m_i d_i = [G:H]`` asserted, never assumed.

``UNDEFINED`` is a designed outcome, not a failure: when the only core-free
subgroup is trivial (``min_corefree_index == |G|``, e.g. Q32 and every abelian
group), ``Ind_1^G 1`` is the regular representation, the template equals the
analytic null exactly (a theorem -- asserted here on the trivial subgroup), and
the coset account has no distinct prediction. Downstream consumers must report
``UNDEFINED`` there, never a number.

The enumeration covers the subgroups the artifact exports. The standard
exporter ships every subgroup; a selectively exported artifact (the large-group
escape hatch) constrains what this module can see, so the number of subgroups
examined is recorded alongside the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..groups.group import FiniteGroup
from .occupancy import analytic_null, total_variation

_TOL = 1e-8


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


def subgroup_core(table: np.ndarray, subgroup: np.ndarray) -> np.ndarray:
    """The core of ``H``: the intersection of all conjugates ``x H x^-1``,
    i.e. the largest normal subgroup of ``G`` contained in ``H``. Pure index
    arithmetic on the Cayley table; sorted element indices returned."""
    identity = _identity_index(table)
    inverses = _inverses(table, identity)
    members = np.asarray(subgroup, dtype=np.int64)
    core = set(members.tolist())
    for x in range(table.shape[0]):
        conjugate = {int(table[table[x, h], inverses[x]]) for h in members}
        core &= conjugate
        if core == {identity}:
            break
    return np.array(sorted(core), dtype=np.int64)


def is_core_free(table: np.ndarray, subgroup: np.ndarray) -> bool:
    """``core(H)`` is trivial. The trivial subgroup is core-free by
    convention."""
    return subgroup_core(table, subgroup).size == 1


def induction_multiplicities(group: FiniteGroup, subgroup: np.ndarray) -> np.ndarray:
    """Frobenius reciprocity: the multiplicity of each irrep in ``Ind_H^G 1``,
    ``m_chi = (1/|H|) sum_{h in H} chi(h)``, one entry per exported irrep.

    Self-verifying: each multiplicity must be (numerically) a non-negative
    integer, and ``sum_i m_i d_i`` must equal ``[G:H]`` exactly -- the
    dimension of the induced module."""
    members = np.asarray(subgroup, dtype=np.int64)
    size = int(members.size)
    if size == 0 or group.order % size != 0:
        raise ValueError(f"subgroup of size {size} does not divide |G|={group.order}")
    index = group.order // size
    multiplicities = np.empty(len(group.irreps), dtype=np.float64)
    for i, irrep in enumerate(group.irreps):
        value = complex(irrep.character[members].sum()) / size
        if abs(value.imag) > _TOL:
            raise ValueError(f"multiplicity of irrep {i} is not real: {value}")
        rounded = round(value.real)
        if rounded < 0 or abs(value.real - rounded) > 1e-6:
            raise ValueError(f"multiplicity of irrep {i} is not a non-negative integer: {value}")
        multiplicities[i] = float(rounded)
    dimension = sum(multiplicities[i] * irrep.dimension for i, irrep in enumerate(group.irreps))
    if abs(dimension - index) > _TOL:
        raise ValueError(
            f"Ind_H^G 1 dimension check failed: sum m_i d_i = {dimension}, expected [G:H] = {index}"
        )
    return multiplicities


def induction_template(group: FiniteGroup, subgroup: np.ndarray) -> np.ndarray:
    """I-13: the energy template of ``Ind_H^G 1`` over real isotypic blocks,
    ``t_j = sum_{i in block j} m_i * d_i / [G:H]``. Sums to 1 by the asserted
    dimension identity."""
    members = np.asarray(subgroup, dtype=np.int64)
    index = group.order // int(members.size)
    multiplicities = induction_multiplicities(group, subgroup)
    template = np.zeros(len(group.isotypic_blocks), dtype=np.float64)
    for j, block in enumerate(group.isotypic_blocks):
        template[j] = (
            sum(multiplicities[i] * group.irreps[i].dimension for i in block.irrep_indices) / index
        )
    if abs(float(template.sum()) - 1.0) > _TOL:
        raise ValueError(f"template sums to {template.sum()}, expected 1")
    return template


@dataclass(frozen=True)
class TemplateEntry:
    """One nontrivial core-free subgroup's ``Ind_H^G 1`` template."""

    subgroup_index: int  # position in the artifact's exported subgroup list
    subgroup_order: int
    coset_index: int  # [G:H]
    template: np.ndarray
    tv_to_null: float

    def to_record(self) -> dict[str, Any]:
        return {
            "subgroup_index": self.subgroup_index,
            "subgroup_order": self.subgroup_order,
            "coset_index": self.coset_index,
            "template": self.template.tolist(),
            "tv_to_null": self.tv_to_null,
        }


@dataclass(frozen=True)
class TemplateLibrary:
    """Every nontrivial core-free subgroup's template for one group, plus the
    structural facts the coset arm gates on. ``coset_defined`` is False exactly
    when ``min_corefree_index == |G|`` (only the trivial subgroup is core-free)
    -- the ``UNDEFINED`` condition."""

    order: int
    n_subgroups_examined: int
    min_corefree_index: int
    coset_defined: bool
    entries: tuple[TemplateEntry, ...]

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "n_subgroups_examined": self.n_subgroups_examined,
            "min_corefree_index": self.min_corefree_index,
            "coset_defined": self.coset_defined,
        }
        if self.coset_defined:
            record["entries"] = [entry.to_record() for entry in self.entries]
        else:
            # The theorem, stated as data: Ind_1^G 1 is the regular
            # representation, the template equals pi0, TV = 0 structurally.
            record["entries"] = "UNDEFINED"
            record["undefined_reason"] = (
                "no nontrivial core-free subgroup: Ind_1^G 1 is the regular "
                "representation, so the coset template equals the analytic null "
                "by theorem and the coset account has no distinct prediction"
            )
        return record


def template_library(group: FiniteGroup) -> TemplateLibrary:
    """Build the ``Ind_H^G 1`` template library over the artifact's exported
    subgroups (I-12 + I-13). Also asserts the trivial-subgroup theorem when the
    trivial subgroup is exported: its template must equal the analytic null
    exactly."""
    table = group.cayley_table
    pi0 = analytic_null(group)
    entries: list[TemplateEntry] = []
    corefree_indices: list[int] = [group.order]  # the trivial subgroup, always core-free
    for position, subgroup in enumerate(group.subgroups):
        members = np.asarray(subgroup, dtype=np.int64)
        if int(members.size) == group.order:
            continue  # H = G is normal in itself, never core-free (unless trivial group)
        if not is_core_free(table, members):
            continue
        coset_index = group.order // int(members.size)
        if int(members.size) == 1:
            trivial_template = induction_template(group, members)
            if total_variation(trivial_template, pi0) > _TOL:
                raise ValueError(
                    "trivial-subgroup template differs from the analytic null; "
                    "the artifact's characters and blocks are inconsistent"
                )
            continue
        corefree_indices.append(coset_index)
        template = induction_template(group, members)
        entries.append(
            TemplateEntry(
                subgroup_index=position,
                subgroup_order=int(members.size),
                coset_index=coset_index,
                template=template,
                tv_to_null=total_variation(template, pi0),
            )
        )
    entries.sort(key=lambda entry: (entry.coset_index, entry.subgroup_index))
    min_index = min(corefree_indices)
    return TemplateLibrary(
        order=group.order,
        n_subgroups_examined=len(group.subgroups),
        min_corefree_index=min_index,
        coset_defined=min_index < group.order,
        entries=tuple(entries),
    )


__all__ = [
    "TemplateEntry",
    "TemplateLibrary",
    "induction_multiplicities",
    "induction_template",
    "is_core_free",
    "subgroup_core",
    "template_library",
]
