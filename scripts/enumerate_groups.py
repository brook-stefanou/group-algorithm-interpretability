#!/usr/bin/env python3
"""Enumerate SmallGroup invariants via Sage/libgap and emit an enriched JSONL dataset.

Two-stage pipeline
-------------------
Stage 1 (GAP, regenerable): ``scripts/enumerate_groups.g`` enumerates SmallGroup
invariants and writes the immutable source catalogue, ``data/group_properties.jsonl``.

Stage 2 (this script): reads that stage-1 catalogue via
``--source-group-properties`` (defaults to ``data/group_properties.jsonl``),
enriches it, and writes the canonical dataset via ``--output`` (defaults to
``data/group_properties_full.jsonl``). ``--output`` must not resolve to the same
file as the source -- the script refuses to run (non-zero exit) if it does, so it
can never silently clobber the stage-1 input.

Usage::
    gap -b -q -T scripts/enumerate_groups.g > data/group_properties.jsonl
    sage -python scripts/enumerate_groups.py --output data/group_properties_full.jsonl
    python scripts/enumerate_groups.py --output data/group_properties_full.jsonl  # via passagemath

The ordinary Python package never imports Sage or GAP at module load.

Column semantics that are easy to misread
-----------------------------------------
``nilpotency_class``            null means NOT NILPOTENT, not "value missing".
``pc_rank``                     null means NOT SOLVABLE (no polycyclic generating sequence).
``derived_length``              null means NOT SOLVABLE.  GAP's ``DerivedLength`` happily
                                returns a finite value on non-solvable input (0 for perfect
                                groups, 1 for S5), which is meaningless; it is suppressed
                                here rather than emitted as a plausible-looking integer.
``min_faithful_irrep_degree``   null means NO faithful irreducible complex representation
                                EXISTS -- i.e. the socle of Z(G) is non-cyclic -- not "value
                                missing".  ``has_faithful_irrep`` records this explicitly so
                                downstream code cannot mistake absence for a gap in the data.
``irrR_degree``                 null under exactly the same condition.
``frobenius_kernel_order``      null means the group is NOT a Frobenius group.
``frobenius_complement_order``  null means the group is NOT a Frobenius group.
``aut_derived_length``          null means Aut(G) is NOT SOLVABLE, for the same reason as
                                ``derived_length``: GAP's ``DerivedLength`` returns a
                                meaningless finite value on non-solvable input, suppressed here.
``aut_nilpotency_class``        null means Aut(G) is not nilpotent.
``fourier_block_cost``          Sum of d**3 (CUBE, not square) over irreducible degrees.
``fourier_block_cost_normalized``   fourier_block_cost / |G|; always >= 1.
``nilpotency_class_at_most_2``  G' <= Z(G), hence TRUE FOR EVERY ABELIAN GROUP.  Replaces the
                                old ``class2``, which was routinely misread as "class == 2".
``nilpotency_class_exactly_2``  strict: nilpotency class == 2 (abelian groups are False).
``pc_rank``, ``composition_length``, ``log_order``
                                CONSTANT ACROSS EVERY GROUP OF A FIXED ORDER (they are all
                                functions of |G| alone -- pc_rank == Omega(|G|)).  Useless as
                                a within-order distance feature; see CONSTANT_AT_FIXED_ORDER.
                                The genuine "coordinate digit count" is ngens = d(G).
``is_semidirect``               THE canonical extension axis: does there exist a normal
                                1 < N < G with a complement?  Series-free, a genuine group
                                invariant, and safe to select or compute distances on.
``chief_factor_split``          DIAGNOSTIC ONLY -- NOT A GROUP INVARIANT.  Per-step flags
                                along GAP's ONE chosen chief series; a group has many chief
                                series and they can disagree.  MUST NOT be used as a distance
                                feature.  Use ``is_semidirect`` for the real axis.
``central_product``             INCLUSIVE definition, true for ~2600 of 6958 groups and
                                dominated by trivial cases.  Prefer
                                ``is_essential_central_product`` (central AND directly
                                indecomposable), which is the discriminating column.
``isoclinism_family_id``        A PROXY, NOT a true isoclinism class.  It is a hash of
                                (|G/Z|, |G'|, commutator fibre histogram, exponent of G/Z).
                                Groups sharing it are candidates, not certified isoclinic.
``character_table_fingerprint`` Canonical, permutation-invariant hash of the ordinary
                                character table.  Equal fingerprint <=> equal character table
                                up to simultaneous row/column permutation.  It deliberately
                                excludes power maps and Frobenius-Schur indicators, because
                                nu_2(chi) = (1/|G|) sum_g chi(g^2) is computed THROUGH the
                                squaring power map and is therefore power-map output, not a
                                character-table invariant.  (Real-VALUEDNESS is readable off
                                the table alone, so fs_complex_count IS a CT invariant while
                                the real/quaternionic split is not.)  D8 and Q8 share a
                                fingerprint, as they must.
``character_table_fingerprint_exact``
                                False iff the canonical-labelling search hit its node budget
                                and fell back to a (still permutation-invariant) refinement
                                hash.  Equal tables ALWAYS collide either way, so a bucket is
                                never split; only false merges are possible when this is False.

Every field is assigned through :class:`FieldAudit`.  A column that raises for every group,
or that comes out entirely null without being declared in ``EXPECTED_SPARSE_FIELDS``, makes
the run exit non-zero.  An all-null column must never be produced silently again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ORDER_MIN = 21
ORDER_MAX = 255
PROGRESS_INTERVAL = 50  # print a progress marker every N groups
# Stage-1 (GAP) source catalogue -- immutable, regenerable via enumerate_groups.g.
SOURCE_GROUP_PROPERTIES_PATH = "data/group_properties.jsonl"
# Stage-2 (this script) canonical, enriched output -- every claim in
# data/selected_contrast_candidates.md is sourced to this file.
CANONICAL_OUTPUT_PATH = "data/group_properties_full.jsonl"

# Fields this script computes and writes over the source catalogue.  Everything else in a
# source row is copied through verbatim.
ALGORITHMIC_CORE_FIELDS = (
    "direct_factor_orders",
    "direct_factor_count",
    "directly_indecomposable",
    "chief_factor_orders",
    "chief_factor_central",
    "chief_factor_split",
    "normal_quotient_index_spectrum",
    "minimal_faithful_permutation_degree",
    "min_corefree_index",
    "corefree_index_spectrum",
    "near_min_corefree_index_count",
    "fourier_block_cost",
    "fourier_block_cost_normalized",
    "min_faithful_irrep_degree",
    "has_faithful_irrep",
    "element_order_histogram",
    # Power maps for EVERY prime dividing |G|, keyed by prime.  The old schema hardcoded
    # p in {2, 3, 5}, which silently omitted p = 7, 11, 13, 17, 19, 23 -- all of which divide
    # orders in range (21 = 3*7, 253 = 11*23).  That was a hole, not a simplification.
    "power_map_image_fraction",
    "power_map_fibre_histogram",
    "commutator_image_size",
    "commutator_fibre_histogram",
    "nilpotency_class_at_most_2",
    "nilpotency_class_exactly_2",
    # Columns a previous run emitted as 100% null and that are now computed.  The GAP names
    # originally used for most of these do not exist in this GAP; see each call site.
    "is_frobenius",
    "frobenius_kernel_order",
    "frobenius_complement_order",
    "num_rational_characters",
    "num_real_conjugacy_classes",
    "hypercenter_order",
    "num_conjugacy_classes_maximal",
    "metabelian",
    "metacyclic",
    "rational",
    "dihedral",
    "is_semidirect",
    "central_product",
    "pc_rank",
    "aut_nilpotency_class",
    "linR_count",
    "irrR_degree",
    # Recomputed because the previous value was wrong on every non-solvable group.
    "derived_length",
    # Recomputed to suppress it on every non-solvable-Aut group, exactly as derived_length.
    "aut_derived_length",
    # --- Derived scalars, stored rather than recomputed downstream ---
    "log_order",
    "centre_fraction",
    "abelianisation_fraction",
    "fitting_fraction",
    "involution_fraction",
    "permutation_degree_ratio",
    "fs_complex_fraction",
    "fs_quaternionic_fraction",
    "self_dual_fraction",
    "rational_character_fraction",
    "commutator_surjectivity_ratio",
    "plancherel_entropy",
    "plancherel_max",
    "element_order_entropy",
    # --- Representation route ---
    "low_dim_irrep_kernel_profile",
    "min_faithful_rep_degree_sum",
    "min_faithful_rep_block_count",
    "character_field_degrees",
    "max_character_field_degree",
    # --- Fusion / tensor route ---
    "max_fusion_multiplicity",
    "multiplicity_free_tensor_fraction",
    "mean_tensor_support",
    # --- Canonical character-table fingerprint ---
    "character_table_fingerprint",
    "character_table_fingerprint_exact",
    # --- Isoclinism (PROXY -- see the module docstring) ---
    "isoclinism_family_id",
    # --- Extension / affine / dihedral family ---
    "is_essential_central_product",
    "wreath_product",
    "frobenius_complement_is_cyclic",
    "has_cyclic_subgroup_index_2",
    "dihedral_family_type",
    "schur_multiplier",
    "schur_multiplier_order",
    "stabilizer_chain_depth",
    # --- Class-2 commutator form ---
    "commutator_form_rank",
    "commutator_form_radical_order",
)

# Source columns deliberately removed from the emitted schema.  Every one of these was 100%
# null, and is either an exact restatement of a column that already works -- emitting both
# would double-weight one invariant in any Gower distance -- or has no canonical definition.
DROPPED_FIELDS: dict[str, str] = {
    "permutation_degree": "duplicate of minimal_faithful_permutation_degree",
    "transitive_degree": "duplicate of min_corefree_index (minimal faithful TRANSITIVE degree)",
    "irrC_degree": "duplicate of min_faithful_irrep_degree",
    "direct_product": "duplicate of (not directly_indecomposable)",
    "almost_simple": "duplicate of is_almost_simple",
    "quasisimple": "duplicate of is_quasisimple",
    "Agroup": "duplicate of all_sylow_abelian",
    "Zgroup": "duplicate of all_sylow_cyclic",
    "rank": "ambiguous; the minimal generator count is ngens, the p-rank is rank_p_group",
    "number_divisions": "duplicate of num_rational_conjugacy_classes",
    "semidirect_product": "renamed is_semidirect, to separate it from the series-dependent "
    "chief_factor_split (see DIAGNOSTIC_ONLY_FIELDS)",
}

# Columns that are NOT group invariants and MUST NOT be used as distance or selection
# features.  They are emitted for diagnosis only.
#   chief_factor_split is defined per step of a CHOSEN chief series, and a group has many
#   chief series -- GAP returns one of them.  Two different chief series of the same group can
#   disagree about which steps split, so the flag list is a property of (group, series), not
#   of the group.  The canonical, series-free question -- "does a complement exist at all?" --
#   is is_semidirect, which IS a group invariant and IS safe to select on.
DIAGNOSTIC_ONLY_FIELDS = frozenset({"chief_factor_split"})

# Columns that are CONSTANT across every group of a fixed order, and so carry no signal for
# the within-order contrasts this study runs on.  They are emitted because they are free and
# occasionally useful across orders, but they must never be used as a distance feature.
#   pc_rank == composition_length == Omega(|G|) for solvable groups, so it is a function of
#   the order alone.  The real "number of coordinate digits" is d(G) = ngens, which varies at
#   fixed order and is already populated.
CONSTANT_AT_FIXED_ORDER = frozenset({"pc_rank", "composition_length", "log_order"})

# Columns whose null is a mathematical answer rather than a failure.  These are exempt from
# the all-null abort; every other column must be populated for at least one group.
EXPECTED_SPARSE_FIELDS = frozenset(
    {
        "nilpotency_class",  # not nilpotent
        "prime_p_group",  # not a p-group
        "rank_p_group",  # not a p-group
        "p_class_p_group",  # not a p-group
        "pc_rank",  # not solvable
        "derived_length",  # not solvable
        "elementary_abelian_series_length",  # not solvable
        "min_faithful_irrep_degree",  # no faithful irreducible complex representation exists
        "irrR_degree",  # same condition
        "frobenius_kernel_order",  # not a Frobenius group
        "frobenius_complement_order",  # not a Frobenius group
        "frobenius_complement_is_cyclic",  # not a Frobenius group
        "aut_nilpotency_class",  # Aut(G) is not nilpotent
        "stddev_character_degree",  # a single irreducible character
        "commutator_form_rank",  # not a class-2 p-group
        "commutator_form_radical_order",  # not a class-2 p-group
    }
)


class FieldAudit:
    """Per-column attempt/failure/null accounting for a run.

    A previous enumeration lost 29 columns because every field was assigned inside a bare
    ``except Exception: return None``.  A GAP name that does not exist raises on every single
    group, so those columns came out entirely null and nothing about the run said so.  This
    records what actually happened to each column and makes the run fail loudly when a column
    is dead.
    """

    def __init__(self) -> None:
        self.groups = 0
        self.attempts: Counter[str] = Counter()
        self.failures: Counter[str] = Counter()
        self.nulls: Counter[str] = Counter()
        self.first_error: dict[str, str] = {}

    def succeeded(self, field: str, value: Any) -> None:
        self.attempts[field] += 1
        if value is None:
            self.nulls[field] += 1

    def failed(self, field: str, exc: BaseException) -> None:
        self.attempts[field] += 1
        self.nulls[field] += 1
        self.failures[field] += 1
        message = str(exc).splitlines()[0] if str(exc).strip() else repr(exc)
        self.first_error.setdefault(field, f"{type(exc).__name__}: {message[:160]}")

    def dead_columns(self) -> list[str]:
        """Columns that raised for every group, or are entirely null without being sparse."""
        dead = []
        for field, attempts in sorted(self.attempts.items()):
            if attempts == 0:
                continue
            raised_always = self.failures[field] == attempts
            null_always = self.nulls[field] == attempts and field not in EXPECTED_SPARSE_FIELDS
            if raised_always or null_always:
                dead.append(field)
        return dead

    def report(self, stream: Any) -> None:
        """Write a per-column summary, then a prominent block for anything dead."""
        troubled = sorted(
            field
            for field, attempts in self.attempts.items()
            if self.failures[field] or (self.nulls[field] and field not in EXPECTED_SPARSE_FIELDS)
        )
        print(f"# field audit over {self.groups} group(s)", file=stream)
        if not troubled:
            print("#   every column populated, no exceptions", file=stream)
        for field in troubled:
            attempts = self.attempts[field]
            detail = f"  [{self.first_error[field]}]" if field in self.first_error else ""
            print(
                f"#   {field}: {self.failures[field]}/{attempts} raised, "
                f"{self.nulls[field]}/{attempts} null{detail}",
                file=stream,
            )

        dead = self.dead_columns()
        if dead:
            print("#", file=stream)
            print("# " + "!" * 74, file=stream)
            print(
                f"# !!! {len(dead)} DEAD COLUMN(S) -- every value is null. DO NOT SHIP.",
                file=stream,
            )
            for field in dead:
                reason = self.first_error.get(field, "no exception; the field is never assigned")
                print(f"# !!!   {field}: {reason}", file=stream)
            print("# " + "!" * 74, file=stream)


class _Stage:
    """A shared intermediate GAP result.

    Several columns are derived from one expensive object (the element list, the character
    table, the chief series).  If that object cannot be built, every dependent column must
    record the ORIGINAL error rather than a downstream ``NoneType`` symptom, so ``get``
    re-raises the captured exception once per dependent column.
    """

    __slots__ = ("value", "error")

    def __init__(self, producer: Callable[[], Any]) -> None:
        try:
            self.value: Any = producer()
            self.error: BaseException | None = None
        except Exception as exc:  # noqa: BLE001 - re-raised for each dependent column
            self.value = None
            self.error = exc

    def get(self) -> Any:
        if self.error is not None:
            raise self.error
        return self.value


def _require_sage() -> Any:
    """Return ``libgap`` from the SageMath / passagemath namespace."""
    try:
        from importlib.util import find_spec

        if find_spec("sage") is None:
            raise ImportError("sage not found")
        from sage.all import libgap  # type: ignore[import-untyped]

        return libgap
    except ImportError:
        pass

    raise SystemExit(
        "SageMath is required for local group enumeration.  Install it via one of:\n"
        "  uv add passagemath-standard           (pip-installable)\n"
        "  sage -python scripts/enumerate_groups.py …\n"
        "Or use the standalone GAP script: gap -b -q -T scripts/enumerate_groups.g\n"
        "See https://doc.sagemath.org/html/en/installation/index.html"
    )


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def _to_python(item: Any) -> Any:
    """Convert a GAP scalar to the closest Python scalar, falling back to its printed form."""
    try:
        return int(item)
    except Exception:  # noqa: BLE001
        try:
            return float(item)
        except Exception:  # noqa: BLE001
            return str(item)


def _fibre_histogram(images: list[Any]) -> dict[str, int]:
    """Return the distribution of fibre cardinalities for a finite map.

    GAP elements are not guaranteed to be hashable Python objects, whereas their GAP external
    representations are stable within a group calculation.  The result maps a fibre size to
    the number of image elements with that size, which is substantially smaller than
    retaining the map itself.
    """
    fibre_sizes = Counter(str(image) for image in images)
    return {str(size): count for size, count in sorted(Counter(fibre_sizes.values()).items())}


def _power_map_summary(elements: list[Any], power: int) -> tuple[float, dict[str, int]]:
    """Summarise ``x -> x**power`` by its image fraction and fibre profile."""
    images = [elt**power for elt in elements]
    image_size = len({str(image) for image in images})
    return image_size / len(elements), _fibre_histogram(images)


def _frobenius_kernel(libgap: Any, G: Any, order: int) -> Any | None:
    """Return the Frobenius kernel of ``G``, or None if ``G`` is not a Frobenius group.

    This GAP has no ``IsFrobeniusGroup`` / ``FrobeniusKernel`` / ``FrobeniusComplement`` --
    all three raise AttributeError -- so the standard characterisation is used directly: G is
    Frobenius with kernel K exactly when K is normal, 1 < K < G, and C_G(k) <= K for every
    non-identity k in K.  That centraliser condition is a class function, so one
    representative per G-class meeting K suffices.  Coprimality of |K| and |G:K|, and hence
    the existence of a complement by Schur-Zassenhaus, follows from the condition rather than
    needing a separate test.
    """
    for kernel in list(libgap.NormalSubgroups(G)):
        kernel_order = int(libgap.Size(kernel))
        if kernel_order == 1 or kernel_order == order:
            continue
        semiregular = True
        for conjugacy_class in list(libgap.ConjugacyClasses(G)):
            representative = libgap.Representative(conjugacy_class)
            if int(libgap.Order(representative)) == 1:
                continue
            if not bool(libgap.IN(representative, kernel)):
                continue
            if not bool(libgap.IsSubgroup(kernel, libgap.Centralizer(G, representative))):
                semiregular = False
                break
        if semiregular:
            return kernel
    return None


def _is_semidirect_product(libgap: Any, G: Any, order: int) -> bool:
    """True iff G = N : H for some normal 1 < N < G admitting a complement H.

    Direct products satisfy this, as they should: a direct product is a semidirect product
    with trivial action.
    """
    for normal in list(libgap.NormalSubgroups(G)):
        normal_order = int(libgap.Size(normal))
        if normal_order == 1 or normal_order == order:
            continue
        if len(list(libgap.ComplementClassesRepresentatives(G, normal))) > 0:
            return True
    return False


def _is_metacyclic(libgap: Any, G: Any) -> bool:
    """True iff G has a cyclic normal subgroup with cyclic quotient.

    This GAP has no ``IsMetacyclicGroup``; the definition is cheap to test directly.
    """
    for normal in list(libgap.NormalSubgroups(G)):
        if bool(libgap.IsCyclic(normal)) and bool(libgap.IsCyclic(libgap.FactorGroup(G, normal))):
            return True
    return False


def _chief_factor_split(libgap: Any, G: Any, chief_series: list[Any]) -> list[bool]:
    """Per-step split flags along GAP's chosen chief series.  DIAGNOSTIC ONLY.

    !!! THIS IS NOT A GROUP INVARIANT.  Splitness is defined per step of a CHOSEN chief
    series, and a group has many chief series; ``ChiefSeries`` returns one of them.  Two chief
    series of the same group can disagree about which steps split, so this list is a property
    of (group, series), NOT of the group.  It MUST NOT be used as a distance or selection
    feature.  The canonical, series-free question -- "does a complement exist at all?" -- is
    ``is_semidirect``, which IS a group invariant.

    Why the column exists at all: when a step splits, elements factor uniquely as (n, q) and
    multiplication is the twisted rule (n1, q1)(n2, q2) = (n1 * q1(n2), q1 q2).  When it does
    not, there is no complement and the rule acquires a cocycle f in H^2(Q, N):
    (n1, q1)(n2, q2) = (n1 * q1(n2) * f(q1, q2), q1 q2).  The cocycle is NOT derivable from
    the action -- it is an irreducible extra term a network has to learn.  Archetype: D8
    splits (C4 has a complement), Q8 does not (its only involution is central and therefore
    lies in every non-trivial subgroup).

    ChiefSeries is descending (G = N_0 > ... > N_r = 1).  The factor N_i/N_{i+1} splits when
    1 -> N_i/N_{i+1} -> G/N_{i+1} -> G/N_i -> 1 splits, i.e. when N_i/N_{i+1} has a complement
    in G/N_{i+1}.  ``ComplementClassesRepresentatives`` decides this exactly.
    """
    flags = []
    for upper, lower in zip(chief_series, chief_series[1:]):
        homomorphism = libgap.NaturalHomomorphismByNormalSubgroup(G, lower)
        quotient = libgap.ImagesSource(homomorphism)
        factor = libgap.Image(homomorphism, upper)
        flags.append(len(list(libgap.ComplementClassesRepresentatives(quotient, factor))) > 0)
    return flags


_GAP_CACHE: dict[str, Any] = {}


def _gap_function(libgap: Any, key: str, code: str) -> Any:
    """Compile a GAP lambda once and reuse it for the whole run."""
    if key not in _GAP_CACHE:
        _GAP_CACHE[key] = libgap.EvalString(code)
    return _GAP_CACHE[key]


def _wreath_idgroups(libgap: Any) -> set[tuple[int, int]]:
    """Every (order, index) in range that is isomorphic to a wreath product B wr T.

    Built by FORWARD enumeration: construct B wr T for each transitive T of degree n and each
    small group B with |B|^n * |T| in range, and record its IdGroup.  This is far cheaper than
    a per-group structural test, and " wr " never appears in a GAP StructureDescription, so
    there is no name-parsing shortcut.  The set is small (26 groups) and is built once.
    """
    if "wreath" in _GAP_CACHE:
        return _GAP_CACHE["wreath"]  # type: ignore[no-any-return]
    found: set[tuple[int, int]] = set()
    moved = libgap.EvalString("NrMovedPoints")
    for degree in range(2, 8):
        try:
            transitive = libgap.AllTransitiveGroups(moved, degree)
        except Exception:  # noqa: BLE001 - transgrp unavailable for this degree
            continue
        for top in transitive:
            top_order = int(top.Size())
            for base_order in range(2, 64):
                if not ORDER_MIN <= base_order**degree * top_order <= ORDER_MAX:
                    continue
                for base_index in range(1, int(libgap.NumberSmallGroups(base_order)) + 1):
                    product = libgap.WreathProduct(libgap.SmallGroup(base_order, base_index), top)
                    identifier = libgap.IdGroup(product)
                    found.add((int(identifier[0]), int(identifier[1])))
    _GAP_CACHE["wreath"] = found
    return found


def _table_matrix(libgap: Any, character_table: Any) -> tuple[Any, list[str]]:
    """The character table as an integer code matrix plus its sorted distinct entries.

    Entries are GAP's canonical printed forms of cyclotomics, so equal algebraic numbers have
    equal strings.  Coding by SORTED distinct string keeps the matrix permutation-invariant.
    """
    stringify = _gap_function(
        libgap, "str_table", "tbl -> List(Irr(tbl), chi -> List(chi, String))"
    )
    rows = [[str(value) for value in row] for row in stringify(character_table)]
    distinct = sorted({value for row in rows for value in row})
    code = {value: position for position, value in enumerate(distinct)}
    matrix = np.array([[code[value] for value in row] for row in rows], dtype=np.int64)
    return matrix, distinct


def _complex_values(libgap: Any, distinct: list[str]) -> Any:
    """Numeric value of each DISTINCT table entry.

    Converting all k^2 entries costs 18s at k=255; the distinct entries number at most a few
    hundred, so this is the difference between a feasible run and an infeasible one.
    """
    values = np.empty(len(distinct), dtype=complex)
    for position, text in enumerate(distinct):
        entry = libgap.EvalString(text)
        conductor = int(libgap.Conductor(entry))
        coefficients = np.array([float(c) for c in libgap.CoeffsCyc(entry, conductor)])
        values[position] = coefficients @ np.exp(2j * np.pi * np.arange(conductor) / conductor)
    return values


def _colour_rank(signature: Any) -> Any:
    """Rank rows of ``signature`` by lexicographic order, returning a colour per row.

    ``np.unique(..., axis=0)`` does this too, but it builds a structured-array view and is
    catastrophically slow here -- it was 93% of the whole fingerprint cost.  lexsort gives the
    same canonical (lexicographic) ranking on plain int64.
    """
    if signature.ndim == 1:
        signature = signature[:, None]
    order = np.lexsort(signature.T[::-1])
    ordered = signature[order]
    boundary = np.empty(len(ordered), dtype=bool)
    boundary[0] = True
    if len(ordered) > 1:
        np.any(ordered[1:] != ordered[:-1], axis=1, out=boundary[1:])
    colours = np.cumsum(boundary) - 1
    result = np.empty(len(ordered), dtype=np.int64)
    result[order] = colours
    return result


def _refine(matrix: Any, row_colours: Any, col_colours: Any) -> tuple[Any, Any]:
    """1-WL refinement of the row/column colouring induced by the table entries."""
    size = matrix.shape[0]
    span = int(matrix.max()) + 1
    for _ in range(2 * size + 2):
        row_key = np.sort(col_colours[None, :] * span + matrix, axis=1)
        col_key = np.sort(row_colours[:, None] * span + matrix, axis=0).T
        new_rows = _colour_rank(np.concatenate([row_colours[:, None], row_key], axis=1))
        new_cols = _colour_rank(np.concatenate([col_colours[:, None], col_key], axis=1))
        if np.array_equal(new_rows, row_colours) and np.array_equal(new_cols, col_colours):
            break
        row_colours, col_colours = new_rows, new_cols
    return row_colours, col_colours


def _canonical_form(matrix: Any, budget: int = 4000) -> tuple[bytes, bool]:
    """Canonical form of ``matrix`` under INDEPENDENT row and column permutation.

    Individualisation-refinement: refine, and if the colouring is not discrete, branch on the
    smallest tied colour class and keep the lexicographically least leaf.  Equivalent tables
    explore isomorphic search trees, so the result is a genuine invariant even when the node
    budget truncates the search -- truncation can merge distinct tables, never split equal ones.
    """
    size = matrix.shape[0]
    best: list[bytes | None] = [None]
    nodes = [0]
    exact = [True]

    def emit(row_colours: Any, col_colours: Any) -> bytes:
        rows = np.argsort(row_colours, kind="stable")
        cols = np.argsort(col_colours, kind="stable")
        return matrix[rows][:, cols].tobytes()

    def search(row_colours: Any, col_colours: Any) -> None:
        if nodes[0] > budget:
            exact[0] = False
            return
        nodes[0] += 1
        row_colours, col_colours = _refine(matrix, row_colours, col_colours)
        if len(np.unique(row_colours)) == size and len(np.unique(col_colours)) == size:
            leaf = emit(row_colours, col_colours)
            if best[0] is None or leaf < best[0]:
                best[0] = leaf
            return
        for colours, is_row in ((row_colours, True), (col_colours, False)):
            values, counts = np.unique(colours, return_counts=True)
            tied = values[counts > 1]
            if tied.size:
                for member in np.flatnonzero(colours == tied[0]):
                    branch = colours.copy()
                    branch[member] = -1  # individualise: strictly the smallest colour
                    branch = _colour_rank(branch[:, None])
                    if is_row:
                        search(branch, col_colours.copy())
                    else:
                        search(row_colours.copy(), branch)
                    if nodes[0] > budget:
                        return
                return
        leaf = emit(row_colours, col_colours)
        if best[0] is None or leaf < best[0]:
            best[0] = leaf

    zeros = np.zeros(size, dtype=np.int64)
    search(zeros, zeros.copy())
    if best[0] is None:  # budget exhausted before any leaf: fall back to the refinement hash
        row_colours, col_colours = _refine(matrix, zeros, zeros.copy())
        exact[0] = False
        best[0] = repr(
            (
                np.bincount(row_colours).tolist(),
                np.bincount(col_colours).tolist(),
                np.sort(matrix.ravel()).tolist(),
            )
        ).encode()
    return best[0], exact[0]  # type: ignore[return-value]


def _character_table_fingerprint(
    libgap: Any, G: Any, abelian: bool | None, invariants: list[Any] | None
) -> tuple[str, bool]:
    """Canonical permutation-invariant hash of the ordinary character table."""
    if abelian and invariants is not None:
        # The character table of an abelian group is its Fourier matrix, which determines the
        # group.  So the isomorphism type IS the canonical form -- exact, and instant.  This
        # also removes the most symmetric tables (C2^7, C255) from the search entirely.
        payload = repr(("abelian", tuple(sorted(int(v) for v in invariants)))).encode()
        return hashlib.sha256(payload).hexdigest()[:24], True
    matrix, distinct = _table_matrix(libgap, libgap.CharacterTable(G))
    form, exact = _canonical_form(matrix)
    payload = repr(distinct).encode() + b"|" + form
    return hashlib.sha256(payload).hexdigest()[:24], exact


def _fusion_statistics(
    libgap: Any, G: Any, character_table: Any, order: int, abelian: bool | None
) -> tuple[int, float, float]:
    """Summarise the tensor fusion rules N_ij^k = <chi_i chi_j, chi_k>.

    Returns (max multiplicity, multiplicity-free fraction of pairs, mean tensor support).
    """
    if abelian:
        # A tensor product of two linear characters is a single linear character: the fusion
        # rules are the multiplication table of the dual group.  Exact, and avoids the k=255
        # abelian tables entirely.
        return 1, 1.0, 1.0
    matrix, distinct = _table_matrix(libgap, character_table)
    table = _complex_values(libgap, distinct)[matrix]  # (irrep, class)
    size = table.shape[0]
    weights = (
        np.array([int(s) for s in libgap.SizesConjugacyClasses(character_table)], dtype=float)
        / order
    )
    dual = table.conj().T  # (class, irrep)
    max_multiplicity = 0
    multiplicity_free = 0
    support = 0.0
    for i in range(size):
        products = (table[i] * weights)[None, :] * table  # (j, class)
        coefficients = np.rint((products @ dual).real)  # (j, k)
        max_multiplicity = max(max_multiplicity, int(coefficients.max()))
        multiplicity_free += int((coefficients.max(axis=1) <= 1).sum())
        support += float((coefficients > 0.5).sum(axis=1).mean())
    return max_multiplicity, multiplicity_free / (size * size), support / size


def _entropy(counts: Any) -> float:
    """Shannon entropy (nats) of a count vector."""
    weights = np.asarray(counts, dtype=float)
    weights = weights[weights > 0]
    weights = weights / weights.sum()
    return float(-(weights * np.log(weights)).sum())


def _is_central_product(libgap: Any, G: Any, order: int, centre_order: int) -> bool:
    """True iff G = A o C_G(A) for a proper normal A with 1 < A n C_G(A).

    Using the centraliser makes this linear in the number of normal subgroups rather than
    quadratic.  The non-trivial intersection is what separates a central product from a
    direct product.  NOTE this is the INCLUSIVE definition and holds for ~2600 of the 6958
    groups; ``is_essential_central_product`` is the discriminating one.
    """
    if centre_order <= 1:
        return False
    for first in libgap.NormalSubgroups(G):
        first_order = int(first.Size())
        if first_order <= 1 or first_order >= order:
            continue
        second = libgap.Centralizer(G, first)
        second_order = int(second.Size())
        if second_order <= 1 or second_order >= order:
            continue
        if int(libgap.ClosureGroup(first, second).Size()) != order:
            continue
        if int(libgap.Intersection(first, second).Size()) <= 1:
            continue  # a direct product, not a central product
        return True
    return False


def _dihedral_family(libgap: Any, G: Any, order: int) -> tuple[bool, str]:
    """Classify the D / Q / SD / M family: a cyclic subgroup of index 2 plus its twist rule.

    These four families share an order and often an entire character table, differing only in
    how the outer involution acts on the cyclic subgroup -- the tightest minimal contrast in
    finite group theory.  Returns (has a cyclic index-2 subgroup, family name).
    """
    if order % 2 != 0:
        return False, "none"
    cyclic = None
    for normal in libgap.NormalSubgroups(G):
        if int(normal.Size()) * 2 == order and bool(libgap.IsCyclic(normal)):
            cyclic = normal
            break
    if cyclic is None:
        return False, "none"
    if bool(libgap.IsAbelian(G)):
        return True, "none"  # C_2m or C_m x C_2, not a twisted family
    if bool(libgap.IsDihedralGroup(G)):
        return True, "dihedral"
    if bool(libgap.IsGeneralisedQuaternionGroup(G)):
        return True, "dicyclic_or_generalized_quaternion"

    half = order // 2
    generator = libgap.MinimalGeneratingSet(cyclic)[0]
    outside = next(g for g in libgap.Elements(G) if not bool(libgap.IN(g, cyclic)))
    conjugate = outside * generator * outside**-1
    twist = next(r for r in range(half) if math.gcd(r, half) == 1 and conjugate == generator**r)
    if twist % half == half - 1:
        return True, "dicyclic_or_generalized_quaternion"  # inversion, non-split
    if half % 2 == 0 and twist % half == half // 2 - 1:
        return True, "semidihedral"
    if half % 2 == 0 and twist % half == half // 2 + 1:
        return True, "modular"
    return True, "other_cyclic_index_2"


def _minimal_faithful_representation(
    libgap: Any, G: Any, kernels: list[tuple[int, Any]], minimal_normals: list[Any]
) -> tuple[int, int]:
    """Minimal total dimension of a faithful, possibly REDUCIBLE, representation.

    A set of irreducibles is faithful iff the intersection of their kernels is trivial, and a
    normal subgroup is trivial iff it contains no minimal normal subgroup.  So faithfulness is
    exactly a set cover of the minimal normal subgroups by "kernels that miss them", solved
    here exactly by subset DP.  This is the quantity that matters mechanistically: "no
    faithful irrep" means at least two blocks are needed, NOT that the group cannot be
    represented.
    """
    count = len(minimal_normals)
    full = (1 << count) - 1
    if full == 0:  # only the trivial group has no minimal normal subgroup
        return 0, 0
    options = []
    for degree, kernel in kernels:
        mask = 0
        for position, minimal in enumerate(minimal_normals):
            if not bool(libgap.IsSubgroup(kernel, minimal)):
                mask |= 1 << position
        if mask:
            options.append((degree, mask))
    best: dict[int, tuple[int, int]] = {0: (0, 0)}
    for _ in range(count):  # a cover never needs more blocks than there are minimal normals
        for covered, (total, blocks) in list(best.items()):
            for degree, mask in options:
                reached = covered | mask
                candidate = (total + degree, blocks + 1)
                if reached not in best or candidate < best[reached]:
                    best[reached] = candidate
    if full not in best:
        raise RuntimeError("no faithful representation found: kernels do not separate")
    return best[full]


def _load_gap_package(libgap: Any, name: str) -> bool:
    """Load a GAP package once per run, remembering whether it is actually available."""
    key = f"package:{name}"
    if key not in _GAP_CACHE:
        try:
            _GAP_CACHE[key] = bool(libgap.LoadPackage(name))
        except Exception:  # noqa: BLE001 - an absent package is a missing route, not a crash
            _GAP_CACHE[key] = False
    return bool(_GAP_CACHE[key])


def _schur_multiplier(libgap: Any, G: Any, solvable: bool | None) -> list[int]:
    """Abelian invariants of the Schur multiplier H_2(G, Z), in GAP's canonical ordering.

    ``AbelianInvariantsMultiplier`` is the obvious call, but it builds an fp-presentation and
    then runs coset enumeration, which on SmallGroup(192,198) blows GAP's 4,096,000-coset limit
    and left the column null.  That is a tooling limit, not a mathematical one: the group is
    solvable and GAP is already holding it as a pc group, so the enumeration is entirely
    avoidable.  The polycyclic package reads the multiplier straight off a pcp presentation
    (~3 ms on 192.198, vs an explosion), and HAP's ``GroupHomology(G, 2)`` is an independent
    enumeration-free route; on 192.198 both return [2].

    Routes are tried in order and the first success wins:
      1. polycyclic ``SchurMultPcpGroup`` -- solvable groups only, but that is 6944/6958 here
      2. ``AbelianInvariantsMultiplier`` -- the general route, and the only one for the
         non-solvable groups, which are small enough that the enumeration is fine
      3. HAP ``GroupHomology(G, 2)`` -- last-resort cross-check route

    If EVERY route fails the errors are re-raised together, so the column records a failure.
    A multiplier that cannot be obtained must never be silently downgraded to the trivial
    group: the previous code did exactly that on 192.198 and reported order 1 when the true
    answer is 2.
    """
    errors: list[str] = []

    if solvable and _load_gap_package(libgap, "polycyclic"):
        try:
            pcp = libgap.Image(libgap.IsomorphismPcpGroup(G))
            return [int(v) for v in libgap.SchurMultPcpGroup(pcp)]
        except Exception as exc:  # noqa: BLE001 - fall through to the next route
            errors.append(f"polycyclic: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")

    try:
        return [int(v) for v in libgap.AbelianInvariantsMultiplier(G)]
    except Exception as exc:  # noqa: BLE001 - fall through to the next route
        errors.append(
            f"AbelianInvariantsMultiplier: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
        )

    if _load_gap_package(libgap, "HAP"):
        try:
            return [int(v) for v in libgap.GroupHomology(G, 2)]
        except Exception as exc:  # noqa: BLE001 - recorded below with the rest
            errors.append(f"HAP: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
    else:
        errors.append("HAP: package unavailable")

    raise RuntimeError("no Schur multiplier route succeeded -- " + "; ".join(errors))


def _check_output_not_source(output_path: Path, source_path: Path) -> None:
    """Refuse to run if the resolved output path is the resolved source path.

    Compares resolved (absolute, symlink-free) paths, not raw strings, so that
    e.g. ``--output data/group_properties.jsonl`` and
    ``--output ./data/group_properties.jsonl`` are both caught even though they
    are spelled differently.  Without this guard, running the enrichment pass
    with ``--output`` pointed at the stage-1 source clobbers the immutable,
    regenerable stage-1 catalogue while leaving the canonical enriched dataset
    untouched and silently divergent.
    """
    resolved_output = output_path.resolve()
    resolved_source = source_path.resolve()
    if resolved_output == resolved_source:
        raise SystemExit(
            f"refusing to run: --output ({resolved_output}) resolves to the same "
            f"file as --source-group-properties ({resolved_source}). This would "
            "overwrite the immutable stage-1 source catalogue. Pass a distinct "
            "--output path (the canonical enriched dataset lives at "
            f"{CANONICAL_OUTPUT_PATH})."
        )


def _read_source_rows(source_path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    """Read JSON records, tolerating the three historical GAP diagnostic lines."""
    try:
        source_text = source_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(
            f"stage-1 source catalogue not found at {source_path}. Regenerate it "
            "with: gap -b -q -T scripts/enumerate_groups.g > "
            f"{SOURCE_GROUP_PROPERTIES_PATH}"
        ) from exc
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for line_number, line in enumerate(source_text.splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
            key = (int(row["order"]), int(row["index"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid source JSONL at line {line_number}") from exc
        if key in rows:
            raise ValueError(f"duplicate source identifier {key}")
        rows[key] = row
    return rows


def _emitted_row(source: dict[str, Any], computed: dict[str, Any]) -> dict[str, Any]:
    """Source row minus the dropped columns, with the algorithmic-core columns written over."""
    row = {field: value for field, value in source.items() if field not in DROPPED_FIELDS}
    row.update({field: computed[field] for field in ALGORITHMIC_CORE_FIELDS})
    return row


def _merge_chunks(
    source_rows: dict[tuple[int, int], dict[str, Any]],
    chunk_paths: list[Path],
    output_path: Path,
) -> None:
    """Validate worker chunks and atomically write them in source-row order."""
    expected_fields = (set(next(iter(source_rows.values()))) | set(ALGORITHMIC_CORE_FIELDS)) - set(
        DROPPED_FIELDS
    )
    enriched_rows: dict[tuple[int, int], dict[str, Any]] = {}
    for chunk_path in chunk_paths:
        for line_number, line in enumerate(chunk_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line:
                continue
            try:
                row = json.loads(line)
                key = (int(row["order"]), int(row["index"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid worker JSONL at {chunk_path}:{line_number}") from exc
            source = source_rows.get(key)
            if source is None:
                raise ValueError(f"worker emitted unexpected identifier {key}")
            if key in enriched_rows:
                raise ValueError(f"duplicate worker identifier {key}")
            if set(row) != expected_fields:
                raise ValueError(f"worker field set differs from source-plus-core for {key}")
            # Source fields are carried through verbatim.  The algorithmic-core fields are
            # recomputed by design -- several of them repair columns the source left null --
            # so they are exempt from the no-change check.
            stale = [
                field
                for field, value in source.items()
                if field not in ALGORITHMIC_CORE_FIELDS
                and field not in DROPPED_FIELDS
                and row[field] != value
            ]
            if stale:
                raise ValueError(f"worker changed source field {stale[0]} for {key}")
            enriched_rows[key] = row
    if set(enriched_rows) != set(source_rows):
        missing = len(set(source_rows) - set(enriched_rows))
        unexpected = len(set(enriched_rows) - set(source_rows))
        raise ValueError(
            f"worker chunks do not cover source exactly: {missing} missing, {unexpected} unexpected"
        )

    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as output:
        for key in source_rows:
            output.write(json.dumps(enriched_rows[key], default=str))
            output.write("\n")
    temporary_output.replace(output_path)


def enumerate_group(
    libgap: Any,
    order: int,
    index: int,
    audit: FieldAudit | None = None,
) -> dict[str, Any]:
    """Extract all invariants for ``SmallGroup(order, index)``.

    Returns a flat dictionary suitable for JSONL serialisation.  Every field is assigned
    through ``compute``, so an exception is counted against its column rather than silently
    turning into a null.  A null therefore means either a recorded exception or a deliberate
    mathematical "not applicable" -- see the module docstring for which columns have the
    latter and what it means for each.
    """
    audit = FieldAudit() if audit is None else audit
    audit.groups += 1
    G = libgap.SmallGroup(order, index)
    props: dict[str, Any] = {}

    def compute(field: str, producer: Callable[[], Any]) -> Any:
        """Assign ``props[field]``, recording any exception against ``field``."""
        try:
            value = producer()
        except Exception as exc:  # noqa: BLE001 - recorded in the audit, never swallowed
            audit.failed(field, exc)
            props[field] = None
            return None
        audit.succeeded(field, value)
        props[field] = value
        return value

    # --- Identity ---
    props["order"] = order
    props["index"] = index
    props["label"] = f"{order}.{index}"
    compute("name", lambda: str(libgap.StructureDescription(G)))

    # --- Structural booleans ---
    abelian = compute("abelian", lambda: bool(libgap.IsAbelian(G)))
    compute("cyclic", lambda: bool(libgap.IsCyclic(G)))
    nilpotent = compute("nilpotent", lambda: bool(libgap.IsNilpotentGroup(G)))
    solvable = compute("solvable", lambda: bool(libgap.IsSolvableGroup(G)))
    compute("simple", lambda: bool(libgap.IsSimpleGroup(G)))
    compute("perfect", lambda: bool(libgap.IsPerfectGroup(G)))
    compute("supersolvable", lambda: bool(libgap.IsSupersolvableGroup(G)))
    compute("monomial", lambda: bool(libgap.IsMonomialGroup(G)))
    compute("is_elementary_abelian", lambda: bool(libgap.IsElementaryAbelian(G)))
    is_p_group = compute("is_p_group", lambda: bool(libgap.IsPGroup(G)))
    compute("is_almost_simple", lambda: bool(libgap.IsAlmostSimpleGroup(G)))
    compute("is_quasisimple", lambda: bool(libgap.IsQuasisimpleGroup(G)))
    compute("is_nonabelian_simple", lambda: bool(libgap.IsNonabelianSimpleGroup(G)))

    # Unlike most of its neighbours below, ``IsDihedralGroup`` does exist in this GAP.
    compute("dihedral", lambda: bool(libgap.IsDihedralGroup(G)))

    # G is metabelian iff G'' = 1.  Deliberately NOT derived from ``derived_length``: GAP
    # returns 0 for perfect groups and 1 for S5, so a ``derived_length <= 2`` test would call
    # every non-solvable group metabelian.
    compute(
        "metabelian",
        lambda: int(libgap.Size(libgap.DerivedSubgroup(libgap.DerivedSubgroup(G)))) == 1,
    )
    compute("metacyclic", lambda: _is_metacyclic(libgap, G))

    props["prime_p_group"] = None
    props["rank_p_group"] = None
    props["p_class_p_group"] = None
    if is_p_group:
        compute("prime_p_group", lambda: int(libgap.PrimePGroup(G)))
        compute("rank_p_group", lambda: int(libgap.RankPGroup(G)))
        compute("p_class_p_group", lambda: int(libgap.PClassPGroup(G)))

    compute(
        "frattini_factor_size",
        lambda: int(libgap.Size(G)) // int(libgap.Size(libgap.FrattiniSubgroup(G))),
    )
    compute("abelian_invariants", lambda: [_to_python(v) for v in libgap.AbelianInvariants(G)])

    # --- Integer invariants ---
    compute("exponent", lambda: int(libgap.Exponent(G)))
    # Derived length is only defined for solvable groups.  GAP returns a finite, meaningless
    # value on non-solvable input, which a previous run recorded as fact.
    compute("derived_length", lambda: int(libgap.DerivedLength(G)) if solvable else None)
    compute(
        "nilpotency_class",
        lambda: int(libgap.NilpotencyClassOfGroup(G)) if nilpotent else None,
    )
    number_conjugacy_classes = compute(
        "number_conjugacy_classes", lambda: int(libgap.NrConjugacyClasses(G))
    )
    compute("composition_length", lambda: len(list(libgap.CompositionSeries(G))) - 1)
    centre_order = compute("center_order", lambda: int(libgap.Size(libgap.Centre(G))))
    commutator_size = compute(
        "commutator_size", lambda: int(libgap.Size(libgap.DerivedSubgroup(G)))
    )
    # Pcgs returns the GAP boolean ``fail`` for non-solvable groups, which is not iterable.
    compute("pc_rank", lambda: len(list(libgap.Pcgs(G))) if solvable else None)

    # --- Automorphism group ---
    aut = _Stage(lambda: libgap.AutomorphismGroup(G))
    aut_order = compute("aut_order", lambda: int(libgap.Size(aut.get())))
    aut_solvable = compute("aut_solvable", lambda: bool(libgap.IsSolvableGroup(aut.get())))
    # Derived length is only defined for solvable groups.  GAP returns a finite, meaningless
    # value on non-solvable Aut(G), exactly as for ``derived_length`` above; suppress it.
    compute(
        "aut_derived_length",
        lambda: int(libgap.DerivedLength(aut.get())) if aut_solvable else None,
    )
    aut_nilpotent = compute("aut_nilpotent", lambda: bool(libgap.IsNilpotentGroup(aut.get())))
    compute("aut_nr_conjugacy_classes", lambda: int(libgap.NrConjugacyClasses(aut.get())))
    compute("aut_exponent", lambda: int(libgap.Exponent(aut.get())))
    compute(
        "aut_order_ratio",
        lambda: float(aut_order) / float(order) if aut_order is not None and order else None,
    )
    # Nearly free: Aut(G) is already built.  Null means Aut(G) is not nilpotent.
    compute(
        "aut_nilpotency_class",
        lambda: int(libgap.NilpotencyClassOfGroup(aut.get())) if aut_nilpotent else None,
    )

    compute("ngens", lambda: len(list(libgap.MinimalGeneratingSet(G))))

    # --- Direct decomposition ---
    # GAP decomposes into directly indecomposable factors; an indecomposable group is the
    # one-factor list ``[G]``.
    direct_factors = _Stage(
        lambda: sorted(int(libgap.Size(f)) for f in libgap.DirectFactorsOfGroup(G))
    )
    compute("direct_factor_orders", lambda: list(direct_factors.get()))
    compute("direct_factor_count", lambda: len(direct_factors.get()))
    compute("directly_indecomposable", lambda: len(direct_factors.get()) == 1)

    # --- Extension structure ---
    # THE canonical extension axis: does a coordinate system exist at all?  Series-free, and
    # so a genuine group invariant -- unlike chief_factor_split below.
    compute("is_semidirect", lambda: _is_semidirect_product(libgap, G, order))
    central = compute(
        "central_product",
        lambda: _is_central_product(libgap, G, order, centre_order or 0),
    )
    # The inclusive central-product flag is true for ~2600 of 6958 groups and is dominated by
    # trivial cases.  Requiring direct indecomposability leaves the ~650 that actually say
    # something structural, so THAT is the primary column.
    compute(
        "is_essential_central_product",
        lambda: bool(central) and bool(props["directly_indecomposable"]),
    )
    compute("wreath_product", lambda: (order, index) in _wreath_idgroups(libgap))

    # --- Subgroups ---
    all_subgroups = _Stage(lambda: list(libgap.AllSubgroups(G)))
    normal_subgroups = _Stage(lambda: list(libgap.NormalSubgroups(G)))
    compute("number_subgroups", lambda: len(all_subgroups.get()))
    compute("number_normal_subgroups", lambda: len(normal_subgroups.get()))

    # Only quotient indices are retained: quotient isomorphism types would be too
    # information-rich for the intended diversity selection.
    compute(
        "normal_quotient_index_spectrum",
        lambda: sorted({order // int(libgap.Size(n)) for n in normal_subgroups.get()}),
    )

    # --- Faithful action routes ---
    # The GAP routine is authoritative.  The explicit core-free enumeration is retained as an
    # interpretable companion: it is the minimal faithful TRANSITIVE degree, which can exceed
    # the minimal faithful degree.
    compute(
        "minimal_faithful_permutation_degree",
        lambda: int(libgap.MinimalFaithfulPermutationDegree(G)),
    )

    def corefree_spectrum() -> list[int]:
        return sorted(
            {
                order // int(libgap.Size(subgroup))
                for subgroup in all_subgroups.get()
                if int(libgap.Size(libgap.Core(G, subgroup))) == 1
            }
        )

    spectrum = compute("corefree_index_spectrum", corefree_spectrum)
    compute("min_corefree_index", lambda: spectrum[0] if spectrum else None)
    # "Near-minimal" is deliberately a small, scale-free window.
    compute(
        "near_min_corefree_index_count",
        lambda: sum(1 for value in spectrum if value <= 2 * spectrum[0]) if spectrum else None,
    )

    # --- Character theory ---
    character_table = _Stage(lambda: libgap.CharacterTable(G))
    irreducibles = _Stage(lambda: list(libgap.Irr(character_table.get())))
    indicators = _Stage(lambda: [int(v) for v in libgap.Indicator(character_table.get(), 2)])

    compute("indicator_vector", lambda: list(indicators.get()))
    compute("fs_real_count", lambda: sum(1 for v in indicators.get() if v == 1))
    compute("fs_complex_count", lambda: sum(1 for v in indicators.get() if v == 0))
    compute("fs_quaternionic_count", lambda: sum(1 for v in indicators.get() if v == -1))

    degrees = compute("character_degrees", lambda: [int(chi[0]) for chi in irreducibles.get()])
    compute("max_irrep_dim", lambda: max(degrees) if degrees else -1)
    compute("distinct_degree_count", lambda: len(set(degrees)) if degrees else 0)
    compute("linC_count", lambda: sum(1 for d in degrees if d == 1) if degrees else 0)
    compute("average_character_degree", lambda: statistics.mean(degrees) if degrees else None)
    # pstdev, not stdev: a group's irreducible character degrees are the complete
    # set, not a sample drawn from one, so the population estimator (/n) is the
    # correct one.  `statistics.stdev` is ddof=1 and disagreed with the GAP
    # enumerator, which divides by n -- e.g. SmallGroup(21,1), degrees [1,1,1,3,3]:
    # 1.0954 (ddof=1) vs 0.9798 (population).
    compute(
        "stddev_character_degree",
        lambda: statistics.pstdev(degrees) if degrees and len(degrees) > 1 else None,
    )
    compute("median_character_degree", lambda: statistics.median(degrees) if degrees else None)

    # NOTE: the exponent is a CUBE.  This is a proxy for the cost of working in the Fourier
    # blocks, not the sum of squares (which is just |G|).  Dividing by |G| makes it comparable
    # across orders, and the normalised value is always >= 1.
    fourier_block_cost = compute(
        "fourier_block_cost", lambda: sum(degree**3 for degree in degrees) if degrees else None
    )
    compute(
        "fourier_block_cost_normalized",
        lambda: fourier_block_cost / order if fourier_block_cost is not None and order else None,
    )

    # A character is rational iff all of its values are.  The previous code indexed a class
    # function by a conjugacy-class OBJECT (``chi[c]``), which raises "TypeError: an integer
    # is required" on every group, so this column was entirely null.
    compute(
        "num_rational_characters",
        lambda: sum(1 for chi in irreducibles.get() if all(bool(libgap.IsRat(v)) for v in chi)),
    )
    # Real-valued linear characters: chi = conj(chi) on every class.
    compute(
        "linR_count",
        lambda: sum(
            1
            for chi in irreducibles.get()
            if int(chi[0]) == 1 and all(libgap.ComplexConjugate(v) == v for v in chi)
        ),
    )

    # A faithful irreducible complex representation exists iff the socle of Z(G) is cyclic, so
    # a null degree means "no such representation EXISTS", not "not computed".
    # Kernels of every irreducible, computed ONCE: the faithful ones, the low-degree kernel
    # profile and the minimal faithful reducible representation all read from this.
    kernels = _Stage(
        lambda: [(int(chi[0]), libgap.KernelOfCharacter(chi)) for chi in irreducibles.get()]
    )
    faithful = _Stage(
        lambda: [
            (degree, indicators.get()[position])
            for position, (degree, kernel) in enumerate(kernels.get())
            if int(libgap.Size(kernel)) == 1
        ]
    )
    compute("has_faithful_irrep", lambda: bool(faithful.get()))
    compute(
        "min_faithful_irrep_degree",
        lambda: min(degree for degree, _ in faithful.get()) if faithful.get() else None,
    )
    # A faithful irreducible REAL representation: Frobenius-Schur indicator +1 keeps the
    # degree, indicator 0 or -1 doubles it.
    compute(
        "irrR_degree",
        lambda: (
            min((degree if indicator == 1 else 2 * degree) for degree, indicator in faithful.get())
            if faithful.get()
            else None
        ),
    )

    # Kernels of the lowest-degree irreducibles.  For the ~67% of groups with NO faithful
    # irrep this is what says which QUOTIENTS still have a cheap representation route.
    def kernel_profile() -> list[list[int]]:
        smallest = sorted({degree for degree, _ in kernels.get()})[:5]
        return [
            [
                degree,
                min(
                    int(libgap.Size(kernel))
                    for kernel_degree, kernel in kernels.get()
                    if kernel_degree == degree
                ),
            ]
            for degree in smallest
        ]

    compute("low_dim_irrep_kernel_profile", kernel_profile)

    # The minimal faithful representation may be REDUCIBLE.  "No faithful irrep" means at
    # least two blocks are needed, not that the group is unrepresentable.
    minimal_normals = _Stage(lambda: list(libgap.MinimalNormalSubgroups(G)))
    faithful_rep = _Stage(
        lambda: _minimal_faithful_representation(libgap, G, kernels.get(), minimal_normals.get())
    )
    compute("min_faithful_rep_degree_sum", lambda: faithful_rep.get()[0])
    compute("min_faithful_rep_block_count", lambda: faithful_rep.get()[1])

    # Degree of the character field Q(chi) over Q, per irreducible.
    field_degrees = compute(
        "character_field_degrees",
        lambda: [int(libgap.DegreeOverPrimeField(libgap.Field(chi))) for chi in irreducibles.get()],
    )
    compute(
        "max_character_field_degree",
        lambda: max(field_degrees) if field_degrees else None,
    )

    # --- Fusion / tensor route ---
    fusion = _Stage(lambda: _fusion_statistics(libgap, G, character_table.get(), order, abelian))
    compute("max_fusion_multiplicity", lambda: fusion.get()[0])
    compute("multiplicity_free_tensor_fraction", lambda: fusion.get()[1])
    compute("mean_tensor_support", lambda: fusion.get()[2])

    # --- Canonical character-table fingerprint ---
    # Equal fingerprint <=> equal character table up to simultaneous row/column permutation.
    # Deliberately excludes power maps and Frobenius-Schur indicators (nu_2 is computed
    # THROUGH the squaring power map, so it is power-map output, not a table invariant).
    table_hash = _Stage(
        lambda: _character_table_fingerprint(libgap, G, abelian, props.get("abelian_invariants"))
    )
    compute("character_table_fingerprint", lambda: table_hash.get()[0])
    compute("character_table_fingerprint_exact", lambda: table_hash.get()[1])

    # --- Element order statistics ---
    elements = _Stage(lambda: list(libgap.Elements(G)))
    element_orders = _Stage(lambda: [int(libgap.Order(e)) for e in elements.get()])

    compute("element_order_spectrum", lambda: sorted(set(element_orders.get())))
    compute(
        "element_order_histogram",
        lambda: {
            str(value): count for value, count in sorted(Counter(element_orders.get()).items())
        },
    )
    compute("max_element_order", lambda: max(element_orders.get()))
    compute("num_involutions", lambda: element_orders.get().count(2))
    compute(
        "fraction_prime_order",
        lambda: (
            sum(1 for value in element_orders.get() if _is_prime(value)) / len(element_orders.get())
        ),
    )

    # --- Power maps and commutator geometry ---
    # Finite-map summaries, not raw element-labelled maps: invariant under relabelling and
    # small enough for JSONL.  x -> x^p is computed for EVERY prime p dividing |G|, keyed by
    # prime, so the schema is uniform across groups with different prime divisors.  The old
    # schema hardcoded p in {2, 3, 5} and silently dropped p = 7, 11, 13, 17, 19, 23.
    prime_divisors = _Stage(lambda: [int(p) for p in libgap.PrimeDivisors(libgap.Size(G))])
    power_maps = _Stage(
        lambda: {str(p): _power_map_summary(elements.get(), p) for p in prime_divisors.get()}
    )
    compute(
        "power_map_image_fraction",
        lambda: {p: summary[0] for p, summary in power_maps.get().items()},
    )
    compute(
        "power_map_fibre_histogram",
        lambda: {p: summary[1] for p, summary in power_maps.get().items()},
    )

    commutators = _Stage(
        lambda: [libgap.Comm(left, right) for left in elements.get() for right in elements.get()]
    )
    compute("commutator_image_size", lambda: len({str(v) for v in commutators.get()}))
    compute("commutator_fibre_histogram", lambda: _fibre_histogram(commutators.get()))

    # G' <= Z(G) iff G is nilpotent of class at most 2, which is TRUE FOR EVERY ABELIAN GROUP.
    # The old name ``class2`` hid that, so both readings now have explicit names.
    at_most_2 = compute(
        "nilpotency_class_at_most_2",
        lambda: bool(libgap.IsSubgroup(libgap.Centre(G), libgap.DerivedSubgroup(G))),
    )
    compute(
        "nilpotency_class_exactly_2",
        lambda: bool(at_most_2) and not bool(abelian) if at_most_2 is not None else None,
    )

    # --- Conjugacy class structure ---
    conjugacy_classes = _Stage(lambda: list(libgap.ConjugacyClasses(G)))
    class_sizes = compute(
        "conjugacy_class_sizes",
        lambda: [int(libgap.Size(c)) for c in conjugacy_classes.get()],
    )
    compute("max_conjugacy_class_size", lambda: max(class_sizes) if class_sizes else None)
    rational_classes = compute(
        "num_rational_conjugacy_classes",
        lambda: int(libgap.Length(libgap.RationalClasses(G))),
    )
    # A class is real iff it is its own inverse class.  The previous code never assigned this
    # column at all -- there was no exception, the line simply did not exist.
    compute(
        "num_real_conjugacy_classes",
        lambda: sum(
            1
            for position, inverse in enumerate(libgap.InverseClasses(character_table.get()))
            if int(inverse) == position + 1
        ),
    )
    # A rational group: every character is rational-valued, equivalently every conjugacy class
    # coincides with its rational class.
    compute(
        "rational",
        lambda: (
            rational_classes == number_conjugacy_classes
            if rational_classes is not None and number_conjugacy_classes is not None
            else None
        ),
    )

    # --- Characteristic subgroups ---
    compute("frattini_subgroup_order", lambda: int(libgap.Size(libgap.FrattiniSubgroup(G))))
    compute("fitting_subgroup_order", lambda: int(libgap.Size(libgap.FittingSubgroup(G))))
    compute("solvable_radical_order", lambda: int(libgap.Size(libgap.SolvableRadical(G))))
    compute("socle_order", lambda: int(libgap.Size(libgap.Socle(G))))
    compute("perfect_residuum_order", lambda: int(libgap.Size(libgap.PerfectResiduum(G))))
    compute(
        "supersolvable_residuum_order",
        lambda: int(libgap.Size(libgap.SupersolvableResiduum(G))),
    )
    compute(
        "p_core_orders",
        lambda: [
            int(libgap.Size(libgap.PCore(G, int(p)))) for p in libgap.PrimeDivisors(libgap.Size(G))
        ],
    )
    # There is no ``Hypercentre`` in this GAP (AttributeError).  GAP's upper central series is
    # returned in DECREASING order, so the hypercentre is its first term.
    compute(
        "hypercenter_order",
        lambda: int(libgap.Size(list(libgap.UpperCentralSeriesOfGroup(G))[0])),
    )

    # --- Series lengths ---
    compute(
        "upper_central_series_length",
        lambda: len(list(libgap.UpperCentralSeriesOfGroup(G))) - 1,
    )
    chief_series = _Stage(lambda: list(libgap.ChiefSeries(G)))
    compute("chief_series_length", lambda: len(chief_series.get()) - 1)
    compute(
        "elementary_abelian_series_length",
        lambda: len(list(libgap.ElementaryAbelianSeries(G))) - 1 if solvable else None,
    )

    # --- Chief-factor / extension structure ---
    # ChiefSeries is descending (G = N_0 > ... > N_r = 1).  A factor N_i/N_{i+1} is central in
    # G/N_{i+1} precisely when [G, N_i] <= N_{i+1}.
    compute(
        "chief_factor_orders",
        lambda: [
            int(libgap.Size(upper)) // int(libgap.Size(lower))
            for upper, lower in zip(chief_series.get(), chief_series.get()[1:])
        ],
    )
    compute(
        "chief_factor_central",
        lambda: [
            bool(libgap.IsSubgroup(lower, libgap.CommutatorSubgroup(G, upper)))
            for upper, lower in zip(chief_series.get(), chief_series.get()[1:])
        ],
    )
    compute("chief_factor_split", lambda: _chief_factor_split(libgap, G, chief_series.get()))

    # --- Sylow structure ---
    sylows = _Stage(
        lambda: [libgap.SylowSubgroup(G, int(p)) for p in libgap.PrimeDivisors(libgap.Size(G))]
    )
    compute("sylow_subgroup_orders", lambda: [int(libgap.Size(s)) for s in sylows.get()])
    compute(
        "sylow_numbers",
        lambda: [
            int(libgap.Size(G)) // int(libgap.Size(libgap.Normalizer(G, s))) for s in sylows.get()
        ],
    )
    compute("all_sylow_cyclic", lambda: all(bool(libgap.IsCyclic(s)) for s in sylows.get()))
    compute("all_sylow_abelian", lambda: all(bool(libgap.IsAbelian(s)) for s in sylows.get()))

    # --- Normal subgroups ---
    normal_orders = compute(
        "normal_subgroup_orders",
        lambda: [int(libgap.Size(n)) for n in normal_subgroups.get()],
    )
    compute(
        "num_normal_subgroups_prime_order",
        lambda: (
            sum(1 for value in normal_orders if _is_prime(value))
            if normal_orders is not None
            else None
        ),
    )

    # --- Maximal subgroups ---
    # ``MaximalSubgroups`` lists SUBGROUPS, not classes: SmallGroup(21,1) has 8 maximal
    # subgroups falling into 2 conjugacy classes.  The class count is therefore genuinely not
    # derivable from ``total_maximal_subgroups`` and needs its own GAP call.
    maximal_orders = compute(
        "maximal_subgroup_orders",
        lambda: [int(libgap.Size(m)) for m in libgap.MaximalSubgroups(G)],
    )
    compute(
        "total_maximal_subgroups",
        lambda: len(maximal_orders) if maximal_orders is not None else None,
    )
    compute(
        "num_conjugacy_classes_maximal",
        lambda: int(libgap.Length(libgap.ConjugacyClassesMaximalSubgroups(G))),
    )

    # --- Minimal normal subgroups, commuting probability, Frobenius ---
    # (minimal_normals is already staged above, for the faithful-representation cover.)
    compute("num_minimal_normal_subgroups", lambda: len(minimal_normals.get()))
    compute(
        "minimal_normal_subgroup_orders",
        lambda: [int(libgap.Size(m)) for m in minimal_normals.get()],
    )
    compute(
        "commuting_probability",
        lambda: (
            float(number_conjugacy_classes) / float(order)
            if number_conjugacy_classes is not None and order
            else None
        ),
    )

    # This GAP has no IsFrobeniusGroup / FrobeniusKernel / FrobeniusComplement -- all three
    # raise AttributeError, which is exactly why the three Frobenius columns were 100% null.
    kernel = _Stage(lambda: _frobenius_kernel(libgap, G, order))
    compute("is_frobenius", lambda: kernel.get() is not None)
    kernel_order = compute(
        "frobenius_kernel_order",
        lambda: int(libgap.Size(kernel.get())) if kernel.get() is not None else None,
    )
    compute("frobenius_complement_order", lambda: order // kernel_order if kernel_order else None)
    # A cyclic complement is the pure "ax + b" affine route; a non-abelian complement is not.
    compute(
        "frobenius_complement_is_cyclic",
        lambda: (
            bool(libgap.IsCyclic(libgap.ComplementClassesRepresentatives(G, kernel.get())[0]))
            if kernel.get() is not None
            else None
        ),
    )

    compute(
        "subgroup_conjugacy_class_count",
        lambda: int(libgap.Length(libgap.ConjugacyClassesSubgroups(libgap.LatticeSubgroups(G)))),
    )

    # --- D / Q / SD / M family: the tightest minimal contrast in finite group theory ---
    family = _Stage(lambda: _dihedral_family(libgap, G, order))
    compute("has_cyclic_subgroup_index_2", lambda: family.get()[0])
    compute("dihedral_family_type", lambda: family.get()[1])

    # The Schur multiplier is the obstruction to splitting a central extension.  Recorded, but
    # it is NOT intended as a primary distance feature.
    # ``_schur_multiplier`` prefers the polycyclic route, which needs no coset enumeration and
    # so does not blow GAP's 4,096,000-coset limit on SmallGroup(192,198); it still RAISES,
    # rather than guessing, if no route can obtain the multiplier.
    schur = compute("schur_multiplier", lambda: _schur_multiplier(libgap, G, solvable))

    def schur_order() -> int:
        # An EMPTY multiplier means the trivial group, whose order is 1.  An UNKNOWN multiplier
        # must not be reported as 1 -- deriving the order from a failed column would fabricate
        # a value, which is exactly the class of bug this module exists to prevent.
        if schur is None:
            raise RuntimeError("schur_multiplier is unavailable, so its order is unknown")
        return int(np.prod(schur)) if schur else 1

    compute("schur_multiplier_order", schur_order)

    # Depth of a stabiliser chain: the iterative-sifting route, analogous to net depth.
    compute(
        "stabilizer_chain_depth",
        lambda: len(
            list(
                libgap.BaseStabChain(libgap.StabChain(libgap.Image(libgap.IsomorphismPermGroup(G))))
            )
        ),
    )

    # --- Class-2 commutator form ---
    # The alternating form induced by [.,.].  NOTE the form on G/Z has trivial radical BY
    # DEFINITION (if [x, G] = 1 then x is already in Z), so that radical carries no
    # information.  The informative space is the Frattini quotient G/Phi(G) -- the generator
    # space -- whose radical is (Z(G)Phi(G))/Phi(G): the generators that act centrally.
    def commutator_form() -> tuple[int, int] | None:
        if not props.get("nilpotency_class_exactly_2") or not props.get("is_p_group"):
            return None
        prime = int(libgap.PrimePGroup(G))
        frattini = libgap.FrattiniSubgroup(G)
        frattini_order = int(libgap.Size(frattini))
        space = int(round(math.log(order // frattini_order, prime)))
        central = libgap.ClosureGroup(frattini, libgap.Centre(G))
        radical = int(libgap.Size(central)) // frattini_order
        return space - int(round(math.log(radical, prime))), radical

    form = _Stage(commutator_form)
    compute("commutator_form_rank", lambda: form.get()[0] if form.get() else None)
    compute("commutator_form_radical_order", lambda: form.get()[1] if form.get() else None)

    # --- Isoclinism PROXY ---
    # Isoclinic groups share G/Z, G' and the commutator map.  GAP has no cheap canonical
    # isoclinism invariant, so this is a best-effort fingerprint of exactly those data.
    # Groups sharing it are CANDIDATES, not certified isoclinic.
    compute(
        "isoclinism_family_id",
        lambda: hashlib.sha256(
            repr(
                (
                    order // (centre_order or 1),
                    commutator_size,
                    props.get("commutator_fibre_histogram"),
                    int(libgap.Exponent(libgap.FactorGroup(G, libgap.Centre(G)))),
                )
            ).encode()
        ).hexdigest()[:24],
    )

    # --- Derived scalars, stored rather than recomputed downstream ---
    def ratio(numerator: Any, denominator: Any) -> float | None:
        if numerator is None or not denominator:
            return None
        return float(numerator) / float(denominator)

    classes = props.get("number_conjugacy_classes")
    compute("log_order", lambda: math.log(order))
    compute("centre_fraction", lambda: ratio(centre_order, order))
    # |G/G'| / |G| == 1 / |G'|.
    compute("abelianisation_fraction", lambda: ratio(1.0, commutator_size))
    compute("fitting_fraction", lambda: ratio(props.get("fitting_subgroup_order"), order))
    compute("involution_fraction", lambda: ratio(props.get("num_involutions"), order))
    compute(
        "permutation_degree_ratio",
        lambda: ratio(props.get("minimal_faithful_permutation_degree"), order),
    )
    compute("fs_complex_fraction", lambda: ratio(props.get("fs_complex_count"), classes))
    compute(
        "fs_quaternionic_fraction",
        lambda: ratio(props.get("fs_quaternionic_count"), classes),
    )
    compute(
        "self_dual_fraction",
        lambda: ratio(
            (props.get("fs_real_count") or 0) + (props.get("fs_quaternionic_count") or 0),
            classes,
        ),
    )
    compute(
        "rational_character_fraction",
        lambda: ratio(props.get("num_rational_characters"), classes),
    )

    # For a class-2 group the commutator map is bilinear, so its image is a subgroup and
    # therefore equals G': the ratio is exactly 1.  A ratio below 1 is a CERTIFICATE of class
    # >= 3.  Violating that is a bug, so raise and let the audit report it.
    def surjectivity() -> float | None:
        value = ratio(props.get("commutator_image_size"), commutator_size)
        if value is not None and props.get("nilpotency_class_at_most_2") and value != 1.0:
            raise RuntimeError(
                f"class-2 group with commutator_surjectivity_ratio={value} (must be 1.0)"
            )
        return value

    compute("commutator_surjectivity_ratio", surjectivity)

    # Plancherel measure P(rho) = d^2 / |G| over the irreducibles; it sums to 1.
    compute(
        "plancherel_entropy",
        lambda: _entropy([d * d for d in degrees]) if degrees else None,
    )
    compute(
        "plancherel_max",
        lambda: max(d * d for d in degrees) / order if degrees else None,
    )
    compute(
        "element_order_entropy",
        lambda: (
            _entropy(list(props["element_order_histogram"].values()))
            if props.get("element_order_histogram")
            else None
        ),
    )

    return props


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enumerate SmallGroup invariants without modifying the source catalogue."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(CANONICAL_OUTPUT_PATH),
        help=(
            "JSONL destination for the enriched canonical dataset. Must resolve to a "
            f"file distinct from --source-group-properties. Defaults to {CANONICAL_OUTPUT_PATH}."
        ),
    )
    parser.add_argument(
        "--source-group-properties",
        type=Path,
        default=Path(SOURCE_GROUP_PROPERTIES_PATH),
        help="Immutable source catalogue recorded in each enriched row.",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Inclusive canonical-source row offset for a deterministic worker chunk.",
    )
    parser.add_argument(
        "--stop-offset",
        type=int,
        help="Exclusive canonical-source row offset for a deterministic worker chunk.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        help=(
            "Round-robin shard the source rows across this many workers. Cost per group varies "
            "by two orders of magnitude and the source is ordered by group order, so contiguous "
            "chunks put every order-128 group on three workers and idle the rest."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        help="Which round-robin shard this worker takes, in [0, --shard-count).",
    )
    parser.add_argument(
        "--merge-chunks",
        type=Path,
        nargs="+",
        help="Validate worker chunks and atomically merge them into --output in source order.",
    )
    parser.add_argument(
        "--allow-dead-columns",
        action="store_true",
        help="Report dead columns but still exit 0. Only for deliberately tiny probe runs.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    source_path = args.source_group_properties
    _check_output_not_source(args.output, source_path)
    source_rows = _read_source_rows(source_path)
    if args.merge_chunks:
        if args.output is None:
            raise ValueError("--merge-chunks requires --output")
        _merge_chunks(source_rows, args.merge_chunks, args.output)
        return
    identifiers = list(source_rows)
    if (args.shard_count is None) != (args.shard_index is None):
        raise ValueError("--shard-count and --shard-index must be given together")
    if args.shard_count is not None:
        if not 0 <= args.shard_index < args.shard_count:
            raise ValueError(f"--shard-index {args.shard_index} outside [0, {args.shard_count})")
        identifiers = identifiers[args.shard_index :: args.shard_count]
    else:
        stop_offset = args.stop_offset if args.stop_offset is not None else len(identifiers)
        if not 0 <= args.start_offset <= stop_offset <= len(identifiers):
            raise ValueError(
                f"invalid source offsets [{args.start_offset}, {stop_offset}) "
                f"for {len(identifiers)} rows"
            )
        identifiers = identifiers[args.start_offset : stop_offset]
    libgap = _require_sage()

    print(f"# enumerate_groups.py: orders {ORDER_MIN}-{ORDER_MAX}", file=sys.stderr)
    for field, reason in sorted(DROPPED_FIELDS.items()):
        print(f"#   dropped column {field}: {reason}", file=sys.stderr)

    audit = FieldAudit()
    count = 0
    t_start = time.monotonic()

    output = args.output.open("w", encoding="utf-8") if args.output else sys.stdout
    try:
        for order, index in identifiers:
            count += 1
            computed = enumerate_group(libgap, order, index, audit)
            props = _emitted_row(source_rows[(order, index)], computed)

            output.write(json.dumps(props, default=str))
            output.write("\n")
            output.flush()

            if count % PROGRESS_INTERVAL == 0:
                elapsed = time.monotonic() - t_start
                rate = count / elapsed if elapsed > 0 else 0
                print(
                    f"#   ... {count}/{len(identifiers)} groups done "
                    f"({elapsed:.0f}s, {rate:.1f} groups/s)",
                    file=sys.stderr,
                )
    finally:
        if args.output:
            output.close()

    elapsed = time.monotonic() - t_start
    rate = count / elapsed if elapsed > 0 else 0
    print(
        f"# Total groups enumerated: {count} in {elapsed:.0f}s ({rate:.1f} groups/s)",
        file=sys.stderr,
    )
    audit.report(sys.stderr)

    dead = audit.dead_columns()
    if dead and not args.allow_dead_columns:
        raise SystemExit(
            f"refusing to report success: {len(dead)} column(s) are entirely null "
            f"({', '.join(dead)}); pass --allow-dead-columns only if that is intended."
        )


if __name__ == "__main__":
    main()
