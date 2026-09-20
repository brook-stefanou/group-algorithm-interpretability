"""Mechanism-arm probes: I-20, I-26, I-27, I-28 (+ I-28b, and the I-21 digit
precompute I-26 consumes).

These instruments read a *trained* model (selected through the dip-aware rule in
``checkpoints.py``) and ask what algebraic structure its computation carries.
They are the source of the paper's mechanistic claims, so every one obeys the
suite-wide discipline (plan §8): a probe is rung 1 and never licenses "uses"
(only the paired ablation does); every score is chance-corrected against the
group's own baseline and never compared raw across orders; every functional-form
fit is scored on held-out ``(a, b)`` pairs, never an in-sample gap; and a
degenerate (constant) label is ``UNDEFINED``, deliberately a string and not a
number, rather than a "failed" probe.

The four instruments and where they are consumed:

* **I-20** ``functional_form_fit`` -- the generic, nested-safe fit harness. It
  regresses a model's read-position logits onto the design matrix of a candidate
  closed-form rule, fitting on one set of ``(a, b)`` pairs and scoring the
  fraction of variance explained on a disjoint held-out set, with a nested-model
  comparison (held-out FVE per form, never a raw in-sample gain) and an
  untrained-model null. Consumed by C1 [D32], C3, C5 and the twisted-rule fit of
  I-27. Softmax is shift-invariant, so logits and features are mean-centred
  across the class axis before the fit (a constant shift must not move the
  answer).
* **I-26** ``carry_digit_instrument`` -- the polycyclic digit probe, coordinate-
  direction count and ripple-carry structure for T16 (C3: C128 vs C2^7). Builds
  on the I-21 digit precompute (``polycyclic_digits``). The direction count is a
  structure metric read against each group's own random-init ``W_E`` null and
  never across orders.
* **I-27** ``signed_cyclic_instrument`` -- the signed-cyclic (dihedral) probe and
  twisted-rule fit for T12 (C2: D32/QD32/Q32; C4: the 104 pair). The negative
  control *is* the instrument: the ``(r, s)`` coordinate system exists only for a
  split ``C_m : C_2`` with the inversion action, so it is structurally
  ``UNDEFINED`` on the quaternionic member (Q32, C13:Q8). A probe that "succeeds"
  there would be fitting an artefact and would disqualify the dihedral result, so
  the coordinate construction refuses to produce coordinates that do not
  reproduce the Cayley table.
* **I-28** ``power_map_probe`` (+ **I-28b** ``involution_direction_ablation``) --
  the power-map / element-order probe for T10 (C1, C2, C4, C5). On a CT-equivalent
  pair this is the only instrument that can carry a difference the character table
  cannot see (the Frobenius-Schur axis He et al. exclude). Class balance differs
  across every pair it is used on (involutions 17 vs 1, 53 vs 1, ...), so the
  metric is chance-corrected balanced accuracy (adjusted so random = 0), never
  raw accuracy -- a raw score would encode the condition, a condition-dependent
  normaliser in a second guise.
* **GCR character-readout** ``gcr_character_readout_instrument`` -- builds on
  I-20's harness to test the Group Composition via Representations (GCR)
  account's readout prediction: read-position logits as a sparse sum over
  occupied irreps of ``Phi_rho(a, b, c) = Re tr(rho(a) rho(b) rho(c^-1))``.
  ``Phi_rho`` is a class function of the single product ``a*b`` (for fixed
  ``c``), so a high raw held-out FVE is consistent with *any* correct
  algorithm, not only GCR, and proves nothing on its own -- the load-bearing
  statistics are the out-of-sample nested comparison against a Fourier-only
  (abelian, degree-1-irrep) rival and the minimal irrep subset a greedy search
  needs, both under the record's ``primary`` key; raw FVE (including a
  saturated one-hot lookup ceiling) is demoted to ``secondary_raw_fve``.

Every element-level probe uses a parameter-free nearest-class-mean classifier
under stratified cross-validation: it cannot overfit random high-dimensional
features the way an unregularised linear probe can, so an untrained model scores
at chance (adjusted 0) as the mandatory rule-1 regression requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from ..groups.group import FiniteGroup
from ..model import GroupModel
from .device import run_with_device_fallback
from .interventions import ablate_direction, model_correct_mask
from .occupancy import cayley_grid_tokens, neuron_activations

UNDEFINED = "UNDEFINED"

_MAX_SPLITS = 5


# ---------------------------------------------------------------------------
# Cayley-table group theory (pure, from the exported table -- no Sage at
# analysis time, the same convention templates.py already follows).
# ---------------------------------------------------------------------------


def identity_index(table: np.ndarray) -> int:
    """The identity element's index (the row equal to ``0..n-1``)."""
    n = table.shape[0]
    idx = np.arange(n)
    candidates = np.flatnonzero(np.all(table == idx[None, :], axis=1))
    if candidates.size == 0:
        raise ValueError("Cayley table has no identity element")
    return int(candidates[0])


def inverses(table: np.ndarray, identity: int) -> np.ndarray:
    """``inv[g]`` with ``g * inv[g] == identity``, one entry per element."""
    n = table.shape[0]
    out = np.empty(n, dtype=np.int64)
    for g in range(n):
        out[g] = int(np.flatnonzero(table[g] == identity)[0])
    return out


def element_orders(table: np.ndarray) -> np.ndarray:
    """The multiplicative order of every element: the least ``k > 0`` with
    ``g^k == e``. Pure Cayley-table arithmetic."""
    n = table.shape[0]
    e = identity_index(table)
    orders = np.empty(n, dtype=np.int64)
    for g in range(n):
        power = g
        k = 1
        while power != e:
            power = int(table[power, g])
            k += 1
        orders[g] = k
    return orders


def square_map(table: np.ndarray) -> np.ndarray:
    """The squaring power map ``g -> g^2`` as element indices (T10)."""
    return np.array([int(table[g, g]) for g in range(table.shape[0])], dtype=np.int64)


def cube_map(table: np.ndarray) -> np.ndarray:
    """The cubing power map ``g -> g^3`` as element indices. Constant on some
    groups ((27,3)); the caller treats a constant map as ``UNDEFINED``."""
    return np.array([int(table[table[g, g], g]) for g in range(table.shape[0])], dtype=np.int64)


def cyclic_subgroup(table: np.ndarray, generator: int, identity: int) -> list[int]:
    """``{e, g, g^2, ...}`` in power order -- the powers of one element."""
    members = [identity]
    power = generator
    while power != identity:
        members.append(power)
        power = int(table[power, generator])
    return members


# ---------------------------------------------------------------------------
# I-21 (signed-cyclic slice): the (r, s) coordinate system for a split
# ``C_m : C_2``. This is the negative control that makes I-27 decisive.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignedCyclicCoords:
    """A split ``C_m : C_2`` coordinate system, verified against the Cayley table.

    ``coords[g] == (r, s)`` with ``r in range(radix)``, ``s in {0, 1}`` and the
    normal form ``g = c^r t^s`` for the chosen cyclic generator ``c`` (order
    ``radix``) and involution ``t``. ``action_sign`` is ``-1`` for the inversion
    action ``t c t^{-1} = c^{-1}`` (the signed-cyclic / dihedral case) and ``+1``
    for the trivial action (``C_m x C_2``); I-27's twisted rule is the
    ``action_sign == -1`` case. The multiplication rule
    ``(r1, s1)(r2, s2) = (r1 + action^{s1} r2 mod m, (s1 + s2) mod 2)`` is
    asserted to reproduce the whole table before this object is returned, so a
    coordinate system that does not reproduce the group is never produced.
    """

    radix: int
    generator: int
    involution: int
    action_sign: int
    coords: np.ndarray  # [order, 2] int64: (r, s)
    element_of: np.ndarray  # [radix, 2] int64: (r, s) -> element index

    @property
    def is_signed_cyclic(self) -> bool:
        """The dihedral (inversion-action) case -- the structure I-27 fits.

        Undefined-as-a-question, not just undefined-in-code, for ``radix <= 2``:
        an order-``<= 2`` cyclic generator is its own inverse, so the inversion
        action ``c -> c^{-1}`` and the trivial action ``c -> c`` are the *same
        function* on ``N``, and :func:`signed_cyclic_coordinates` always
        resolves this case to ``action_sign = +1`` (trivial) rather than
        picking the dihedral label arbitrarily (Klein-four, ``radix == 2``, is
        the case that bit -- see the regression test)."""
        return self.action_sign == -1


def signed_cyclic_coordinates(group: FiniteGroup) -> SignedCyclicCoords | None:
    """Construct the ``(r, s)`` coordinates of a split ``C_m : C_2`` (T12), or
    return ``None`` when no such system exists.

    ``None`` is the designed outcome on the quaternionic member: a generalised
    quaternion group ``Q_{2m}`` is non-split, so although it has a cyclic
    subgroup ``C_m`` of index 2, every element outside it has order 4 and there
    is no involution ``t`` completing the split. The construction requires

    * a cyclic normal subgroup ``N = <c>`` of order ``m = |G| / 2`` (index 2, so
      automatically normal); and
    * an involution ``t`` outside ``N`` acting on ``c`` by inversion
      (``t c t^{-1} = c^{-1}``, the signed-cyclic case) -- or trivially, recorded
      as ``action_sign = +1``.

    For ``m <= 2`` inversion and the trivial action coincide (an order-``<= 2``
    element is its own inverse, so ``c^{-1} == c``): the classification always
    resolves to ``action_sign = +1`` in that case, never ``-1`` by an accident
    of which branch a search checks first. Without this, the Klein four-group
    (``m = 2``) -- an abelian group -- would be labelled dihedral, because its
    unique non-identity element of ``N`` equals its own "inverse".

    On failure at either step the group has no signed-cyclic coordinate system
    and ``None`` is returned. On success the constructed rule is asserted to
    reproduce the Cayley table exactly, so an artefactual coordinate assignment
    can never slip through.
    """
    table = group.cayley_table
    n = int(table.shape[0])
    if n % 2 != 0:
        return None
    m = n // 2
    e = identity_index(table)
    orders = element_orders(table)

    generator = next((g for g in range(n) if orders[g] == m), None)
    if generator is None:
        return None
    rotation = cyclic_subgroup(table, generator, e)
    if len(rotation) != m:
        return None
    in_rotation = np.zeros(n, dtype=bool)
    in_rotation[rotation] = True
    dlog = {element: r for r, element in enumerate(rotation)}
    inv = inverses(table, e)
    c_inverse = inv[generator]

    # An involution outside N; classify its action on c as inversion or trivial.
    # For m <= 2, c is its own inverse (c_inverse == generator), so the two
    # branches below are the same test; checking the inversion branch first
    # would then always "win" and mislabel a trivial action as dihedral (the
    # Klein four-group case). Only trust the inversion match when it is
    # actually distinguishable from the trivial one.
    involution = None
    action_sign = 0
    inversion_is_distinguishable = c_inverse != generator
    for t in range(n):
        if in_rotation[t] or orders[t] != 2:
            continue
        conjugate = int(table[table[t, generator], inv[t]])
        if inversion_is_distinguishable and conjugate == c_inverse:
            involution, action_sign = t, -1
            break
        if conjugate == generator:
            involution, action_sign = t, 1
            break
    if involution is None:
        return None

    coords = np.empty((n, 2), dtype=np.int64)
    element_of = np.empty((m, 2), dtype=np.int64)
    for g in range(n):
        if in_rotation[g]:
            r, s = dlog[g], 0
        else:
            r, s = dlog[int(table[g, involution])], 1
        coords[g] = (r, s)
        element_of[r, s] = g

    # Decisive self-check: the signed-cyclic rule must rebuild the whole table.
    expected = np.empty((n, n), dtype=np.int64)
    for a in range(n):
        ra, sa = int(coords[a, 0]), int(coords[a, 1])
        for b in range(n):
            rb, sb = int(coords[b, 0]), int(coords[b, 1])
            r = (ra + (action_sign**sa) * rb) % m
            s = (sa + sb) % 2
            expected[a, b] = int(element_of[r, s])
    if not np.array_equal(expected, table):
        return None

    return SignedCyclicCoords(
        radix=m,
        generator=generator,
        involution=involution,
        action_sign=action_sign,
        coords=coords,
        element_of=element_of,
    )


# ---------------------------------------------------------------------------
# I-21 (digit precompute): a polycyclic / mixed-radix coordinate system, and
# the ripple-carry structure I-26 reads. Built for the abelian C3 members
# (C128, C2^7, the C127 anchor) from the Cayley table.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolycyclicDigits:
    """A mixed-radix coordinate system over a pinned polycyclic series.

    ``digits[g]`` is the length-``length`` tuple of coordinates of element ``g``
    in the normal form ``g = gen_0^{d_0} * gen_1^{d_1} * ... * gen_{length-1}^{d_{length-1}}``
    (index 0 first) along a greedily built
    chain ``1 = G_0 < G_1 < ... < G_length = G`` with prime relative orders
    ``radices``. ``length`` is the composition length ``l(G)`` (7 for both C128
    and C2^7). The digit map is asserted to be a set bijection. ``carry`` is the
    boolean ``length x length`` matrix ``carry[i, j] = True`` iff perturbing the
    single input digit ``i`` (adding a generator) can change output digit ``j``:
    diagonal for a group whose digits add independently (C2^7, elementary
    abelian), lower/general-triangular where low digits ripple into high ones
    (C128, radix-2 carry).
    """

    length: int
    radices: tuple[int, ...]
    generators: tuple[int, ...]
    digits: np.ndarray  # [order, length] int64
    element_of: Callable[[tuple[int, ...]], int]
    carry: np.ndarray  # [length, length] bool

    @property
    def coordinate_directions(self) -> int:
        """The number of carry-connected digit blocks: the connected components
        of the carry graph. This is the analytic companion to the model-measured
        direction count (``effective_rank`` of ``W_E``); for the abelian C3
        members it equals ``d(G) = ngens`` (1 for C128 and the C127 anchor, 7 for
        the elementary-abelian C2^7), a single cyclic radix chain collapsing to
        one carry-linked block while independent digits stay separate. It is a
        structural readout, not asserted to equal ``ngens`` for every group."""
        # Connected components of the (undirected) carry adjacency.
        adjacency = self.carry | self.carry.T
        seen = np.zeros(self.length, dtype=bool)
        components = 0
        for start in range(self.length):
            if seen[start]:
                continue
            components += 1
            stack = [start]
            seen[start] = True
            while stack:
                node = stack.pop()
                for other in range(self.length):
                    if adjacency[node, other] and not seen[other]:
                        seen[other] = True
                        stack.append(other)
        return components


def _prime_factor(value: int) -> int:
    """The least prime factor of ``value > 1``."""
    d = 2
    while d * d <= value:
        if value % d == 0:
            return d
        d += 1
    return value


def polycyclic_digits(group: FiniteGroup) -> PolycyclicDigits | None:
    """I-21 digit precompute: a mixed-radix coordinate system for a polycyclic
    (here abelian) group, or ``None`` when the greedy chain does not stay
    polycyclic (a non-abelian group whose greedy step is not normal).

    The chain is built greedily from the Cayley table: at each step pick any
    element outside the current subgroup, extend by the least prime power that
    lands back inside, and record the relative order as a radix. Every element
    then has a unique digit tuple in the resulting normal form. The construction
    is exact for abelian groups (every subgroup is normal), which is all the C3
    claim needs; for a non-abelian group it verifies normality of each step and
    returns ``None`` if it fails rather than shipping a coordinate system that is
    not polycyclic.

    On abelian groups (the C3 scope: C128, C2^7, the C127 anchor) every subgroup
    is normal, so the outcome (a coordinate system is always built) does not
    depend on which "outside" element the greedy step happens to pick, and is
    robust to the Cayley table's element enumeration.

    On a non-abelian group the outcome can depend on the artifact's element
    enumeration: the guard at each step requires the *chosen* extension to stay
    normal in the whole of ``G`` (``x g x^{-1}`` inside the extended subgroup
    for every ``x``, not merely the weaker "normal in the next step's subgroup"
    that polycyclicity actually requires), and which candidate the greedy search
    picks first is enumeration order-dependent. D8 has been observed to return a
    coordinate system under one element enumeration and ``None`` under another
    -- both are real, correct outcomes of this specific (enumeration-sensitive)
    construction, not a bug in either. Do not read a non-abelian ``None``/measured
    outcome as a property of the group; it is a property of this greedy search
    over this artifact's enumeration.
    """
    table = group.cayley_table
    n = int(table.shape[0])
    e = identity_index(table)
    inv = inverses(table, e)

    # Greedy subnormal chain of cyclic prime-order extensions.
    members = [e]
    member_set = {e}
    generators: list[int] = []
    radices: list[int] = []
    while len(member_set) < n:
        outside = next(g for g in range(n) if g not in member_set)
        # Smallest power of `outside` that re-enters the current subgroup.
        rel = 1
        power = outside
        while power not in member_set:
            power = int(table[power, outside])
            rel += 1
        radix = _prime_factor(rel)
        # Use g^(rel/radix) so the added generator has prime relative order.
        step = rel // radix
        gen = outside
        for _ in range(step - 1):
            gen = int(table[gen, outside])
        # New coset representatives: gen^k * current members, k = 0..radix-1.
        new_members: list[int] = []
        coset_rep = e
        for _ in range(radix):
            for h in members:
                new_members.append(int(table[coset_rep, h]))
            coset_rep = int(table[coset_rep, gen])
        if len(set(new_members)) != radix * len(members):
            return None  # not a clean cyclic extension (step was not polycyclic)
        # Guard: the added generator's conjugates (in the whole of G, which is
        # stronger than polycyclicity actually requires -- normal-in-the-next-
        # subgroup would suffice) must stay inside the extended subgroup, or the
        # chain is rejected. Trivially satisfied on an abelian group (every
        # subgroup is normal -- the C3 scope, enumeration-robust); on a
        # non-abelian group whether a given greedy pick passes is enumeration-
        # order-dependent (see the docstring above).
        extended = set(new_members)
        if any(int(table[table[x, gen], inv[x]]) not in extended for x in range(n)):
            return None
        generators.append(gen)
        radices.append(radix)
        members = new_members
        member_set = set(new_members)

    length = len(generators)
    radix_tuple = tuple(radices)

    # Digit assignment by the normal form g = gen_0^{d_0} * gen_1^{d_1} * ...
    # * gen_{L-1}^{d_{L-1}} (index 0 first -- confirmed against _element_for's
    # actual left-multiply accumulation below, not merely asserted). Build
    # element -> digits by enumerating the mixed-radix grid in the same order
    # the chain was grown (generator i multiplies on the outside of G_i).
    digits = np.empty((n, length), dtype=np.int64)

    def _element_for(digit_tuple: tuple[int, ...]) -> int:
        # Accumulates right-to-left (i from length-1 down to 0), left-
        # multiplying each new factor onto the running product -- by
        # associativity this equals gen_0^{d_0} * gen_1^{d_1} * ...
        # * gen_{length-1}^{d_{length-1}} (index 0 first), the normal form
        # documented above. The digit-tuple bijection check below is what
        # actually keeps this correct regardless of convention; the order
        # only matters for reading the digits off correctly by hand.
        acc = e
        for i in range(length - 1, -1, -1):
            g = e
            for _ in range(digit_tuple[i]):
                g = int(table[g, generators[i]])
            acc = int(table[g, acc])
        return acc

    seen = set()
    # Enumerate every digit combination once.
    total = 1
    for radix in radices:
        total *= radix
    if total != n:
        return None
    for flat in range(n):
        rem = flat
        combo = [0] * length
        for i in range(length):
            combo[i] = rem % radices[i]
            rem //= radices[i]
        element = _element_for(tuple(combo))
        digits[element] = combo
        seen.add(element)
    if len(seen) != n:
        return None  # not a bijection

    digit_index = {tuple(int(v) for v in digits[g]): g for g in range(n)}

    def element_of(digit_tuple: tuple[int, ...]) -> int:
        return digit_index[tuple(int(v) for v in digit_tuple)]

    # Carry structure: does adding one generator (bumping input digit i) change
    # output digit j? Read off the group product of every element with each
    # generator and compare digit tuples.
    carry = np.zeros((length, length), dtype=bool)
    for i in range(length):
        gen = generators[i]
        for g in range(n):
            before = digits[g]
            after = digits[int(table[g, gen])]
            changed = np.flatnonzero(before != after)
            for j in changed.tolist():
                carry[i, int(j)] = True
    return PolycyclicDigits(
        length=length,
        radices=radix_tuple,
        generators=tuple(generators),
        digits=digits,
        element_of=element_of,
        carry=carry,
    )


# ---------------------------------------------------------------------------
# Element-level features and the chance-corrected cross-validated probe.
# ---------------------------------------------------------------------------


def _assert_finite(array: np.ndarray, what: str) -> None:
    """Raise a clear ``ValueError`` if ``array`` carries a NaN or Inf.

    Checkpoint-derived features and logits feed a nearest-centroid classifier
    (``argmin`` over NaN distances silently lands on class 0) and an
    unregularised least-squares fit (NaN propagates through and serialises as
    a bare, invalid ``NaN`` JSON token) -- neither fails loudly on its own, so
    every producer of checkpoint-derived numbers must guard here instead of
    downstream."""
    if not np.all(np.isfinite(array)):
        n_bad = int(np.size(array) - np.count_nonzero(np.isfinite(array)))
        raise ValueError(
            f"{what} contains {n_bad} non-finite value(s) (NaN/Inf); the "
            "checkpoint may be corrupted or the model diverged. Refusing to "
            "feed non-finite values into a probe or fit -- they would pass "
            "silently through nearest-centroid argmin or lstsq."
        )


def embedding_features(model: GroupModel, order: int) -> np.ndarray:
    """The model's learned embedding of each group element, ``W_E[g]``, mean-
    centred across elements (both architectures expose ``W_E``; the ``=`` read
    token, index ``order``, is dropped). Mean-centring removes the DC offset the
    read position injects, per the occupancy DC-dominance finding."""
    w_e = model.W_E.detach().to(torch.float64).cpu().numpy()[:order]
    _assert_finite(w_e, "embedding_features (model.W_E)")
    return w_e - w_e.mean(axis=0, keepdims=True)


def neuron_features(model: GroupModel, group: FiniteGroup, *, argument: str) -> np.ndarray:
    """A per-element feature from the read-position activations ``A[m, a, b]``:
    ``argument="left"`` averages over ``b`` to a function of ``a``, ``"right"``
    over ``a`` to a function of ``b``. Mean-centred across elements (the trivial
    DC block would otherwise dominate)."""
    activations = neuron_activations(model, group.order)  # [d_mlp, order, order]
    _assert_finite(activations, "neuron_features (neuron_activations)")
    if argument == "left":
        features = activations.mean(axis=2).T  # [order, d_mlp], indexed by a
    elif argument == "right":
        features = activations.mean(axis=1).T  # [order, d_mlp], indexed by b
    else:
        raise ValueError(f"argument must be 'left' or 'right', got {argument!r}")
    return features - features.mean(axis=0, keepdims=True)


def _nearest_centroid_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray
) -> np.ndarray:
    """Assign each test row to the nearest training class mean (Euclidean).

    A parameter-free nearest-class-mean classifier, written directly to avoid a
    division-by-zero warning ``sklearn.neighbors.NearestCentroid`` raises on
    single-sample folds. It has no fitted parameters, so it cannot overfit random
    high-dimensional features -- which is what makes the untrained-model null
    land at chance."""
    classes = np.unique(train_y)
    centroids = np.stack([train_x[train_y == c].mean(axis=0) for c in classes])
    distances = np.linalg.norm(test_x[:, None, :] - centroids[None, :, :], axis=2)
    return classes[distances.argmin(axis=1)]


def crossval_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, Any] | str:
    """A chance-corrected, cross-validated decodability score for ``labels`` from
    ``features`` (one row per element).

    Returns ``UNDEFINED`` (a string) for a degenerate label -- a constant label,
    or any class with fewer than two members, which cannot be
    stratified/held-out -- rather than a spurious "failed" number, per the
    degenerate-label rule. Otherwise returns the *adjusted* balanced accuracy
    (``sklearn`` ``adjusted=True``: chance maps to 0, perfect to 1), so the score
    is comparable across the differing class balances of different groups and
    never encodes the condition. The classifier is a parameter-free nearest-
    class-mean under stratified k-fold cross-validation: it does not overfit
    random high-dimensional features, so an untrained model scores ~0.
    """
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    if classes.size < 2 or counts.min() < 2:
        return UNDEFINED
    n_splits = min(_MAX_SPLITS, int(counts.min()))
    if n_splits < 2:
        return UNDEFINED
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predictions = np.empty_like(labels)
    for train_idx, test_idx in splitter.split(features, labels):
        predictions[test_idx] = _nearest_centroid_predict(
            features[train_idx], labels[train_idx], features[test_idx]
        )
    chance = 1.0 / classes.size
    return {
        "adjusted_balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions, adjusted=True)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "chance_level": chance,
        "n_classes": int(classes.size),
        "n_splits": n_splits,
        "class_counts": {int(c): int(n) for c, n in zip(classes, counts)},
    }


# ---------------------------------------------------------------------------
# I-28: power-map / element-order probe (T10), chance-corrected.
# ---------------------------------------------------------------------------


def _feature_matrix(model: GroupModel, group: FiniteGroup, source: str) -> np.ndarray:
    if source == "embed":
        return embedding_features(model, group.order)
    if source in ("left", "right"):
        return neuron_features(model, group, argument=source)
    raise ValueError(f"source must be 'embed', 'left' or 'right', got {source!r}")


def power_map_probe(
    model: GroupModel,
    group: FiniteGroup,
    *,
    source: str = "embed",
    seed: int = 0,
) -> dict[str, Any]:
    """I-28: decode element order, involution-ness and the square/cube power map
    from the model's per-element representation, each with a chance-corrected
    score (adjusted balanced accuracy). On a CT-equivalent pair this is the only
    instrument that can carry a difference the character table cannot see (the
    Frobenius-Schur axis). Constant targets (the cube map on some 3-groups) come
    back ``UNDEFINED`` -- a fit-residual degeneracy, not a failure.

    Rung 1 (decodable, not "used"); the involution-direction ablation (I-28b,
    :func:`involution_direction_ablation`) is what raises "used".
    """
    table = group.cayley_table
    features = _feature_matrix(model, group, source)
    orders = element_orders(table)
    identity = identity_index(table)
    non_identity = np.arange(group.order) != identity
    # The identity is the unique order-1 element (a singleton class that no
    # stratified fold can hold out) and carries no power-map structure, so the
    # element-order target is decoded over the non-identity elements only.
    targets: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "element_order": (features[non_identity], orders[non_identity]),
        "is_involution": (features, (orders == 2).astype(np.int64)),
        "square_map": (features, square_map(table)),
        "cube_map": (features, cube_map(table)),
    }
    results: dict[str, Any] = {}
    for name, (probe_features, labels) in targets.items():
        results[name] = crossval_probe(probe_features, labels, seed=seed)
    return {
        "instrument": "power-map-probe",
        "target_theory": "T10",
        "rung": 1,
        "source": source,
        "metric": "adjusted_balanced_accuracy",
        "n_involutions": int((orders == 2).sum()),
        "probes": results,
        "note": (
            "Chance-corrected (adjusted balanced accuracy): random init scores ~0, "
            "and the score is never compared raw across orders. Rung 1 -- decodable, "
            "not used; see involution_direction_ablation for the causal check."
        ),
    }


# ---------------------------------------------------------------------------
# I-28b: involution-direction ablation. The behavioural machinery -- zero-
# ablating a representational axis out of W_E and scoring the cost in flips
# against random-direction controls -- lives in the shared I-07 harness
# (interventions.ablate_direction); this probe only picks the direction.
# ``model_correct_mask`` is re-exported from that harness for callers that read
# the model's correctness directly.
# ---------------------------------------------------------------------------


def involution_direction_ablation(
    model: GroupModel,
    group: FiniteGroup,
    *,
    source: str = "embed",
    subset: np.ndarray | None = None,
    n_controls: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """I-28b: ablate the involution-carrying direction and measure the drop in
    behavioural accuracy (in flips: the change in the number of correctly
    answered pairs), against norm-matched random-direction controls.

    Only ``source="embed"`` is valid here, unlike the probe (I-28, which
    decodes structure in whatever feature space the caller asks for):
    :func:`interventions.ablate_direction` always projects the direction out of
    ``W_E`` rows, in ``d_model`` space, because zero-ablating an embedding axis
    is the one representational edit every architecture this harness serves
    shares. A direction built from ``source="left"``/``"right"``
    (:func:`neuron_features`, ``d_mlp`` space) has the wrong dimensionality to
    ablate against ``W_E`` -- it either crashes on a shape mismatch, or, on a
    config where ``d_mlp`` happens to equal ``d_model``, silently zero-ablates
    a meaningless axis and reports a spurious ``status: "measured"`` record.
    Both have been observed; this function refuses any ``source`` but
    ``"embed"`` rather than risk either. (:func:`power_map_probe` and the other
    element-level probes are unaffected -- they only ever classify, never
    ablate, so any feature space is safe for them.)

    The involution direction is the difference of class means (involutions minus
    non-involutions) in the embedding space -- the axis a nearest-mean probe
    would use. Zero-ablation projects it out of ``W_E``. ``UNDEFINED`` when the
    involution label is degenerate (no involutions, or all elements are
    involutions), matching the probe. ``subset`` restricts the accuracy count to
    an unleaked held-out mask when the caller supplies one; by default every pair
    is counted.

    Rung 3 (causal, Necessary) *only* alongside the probe; reported in flips, the
    behavioural unit, never a bare percentage.
    """
    if source != "embed":
        raise ValueError(
            "involution_direction_ablation only supports source='embed': "
            "ablate_direction always projects the direction out of W_E rows "
            f"in d_model space, but source={source!r} builds the direction in "
            "d_mlp space (neuron_features) -- a dimensional mismatch that "
            "either crashes (d_mlp != d_model) or silently ablates a "
            "meaningless axis and reports a spurious measured record "
            "(d_mlp == d_model). Valid sources for ablation: 'embed'."
        )
    table = group.cayley_table
    orders = element_orders(table)
    involution = orders == 2
    if involution.sum() == 0 or involution.sum() == group.order:
        return {
            "instrument": "involution-direction-ablation",
            "status": UNDEFINED,
            "reason": "involution label is degenerate (0 or all elements)",
        }
    features = _feature_matrix(model, group, source)
    direction = features[involution].mean(axis=0) - features[~involution].mean(axis=0)

    # The direction is instrument-specific; removing it and scoring the
    # behavioural cost (in flips, against norm-matched random directions) is the
    # shared I-07 harness's job.
    harness = ablate_direction(
        model, group, direction, subset=subset, n_controls=n_controls, seed=seed
    )

    return {
        "instrument": "involution-direction-ablation",
        "target_theory": "T10",
        "rung": 3,
        "source": source,
        "status": "measured",
        "n_scored_pairs": harness["n_scored_pairs"],
        "baseline_correct": harness["baseline_correct"],
        "ablated_correct": harness["ablated_correct"],
        "involution_direction_drop_flips": harness["direction_drop_flips"],
        "random_direction_drop_flips": harness["random_direction_drop_flips"],
        "note": (
            "Effect size in flips (change in correctly-answered pairs). The "
            "involution-direction drop is read against the norm-matched random-"
            "direction control, not an absolute threshold."
        ),
    }


# ---------------------------------------------------------------------------
# I-20: generic, nested-safe functional-form fit harness.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FunctionalForm:
    """A candidate closed-form rule as a design matrix over ``(a, b, c)``.

    ``design`` has shape ``[order, order, n_classes, n_features]``: the features
    the rule assigns to answer ``c`` for input pair ``(a, b)``. Regressing the
    model's read-position logits onto these features (per pair, over the class
    axis) tests how much of the logit structure the rule explains. Held-out FVE
    is well defined for any pair of forms; it is the *interpretation* of the
    comparison that depends on how the pair relates:

    * a genuinely **nested** pair -- the reduced form's feature columns a
      subset of the full form's -- isolates what the full form's extra columns
      buy over the reduced one, with neither rewarded for fitting noise
      (held-out scoring);
    * a **non-nested** pair of alternatives (e.g. two different one-hot
      answer-indicator rules, as in the signed-cyclic twisted-rule fit,
      :func:`_signed_cyclic_forms`) instead compares which of two candidate
      *rules* the model's logits agree with -- read the gap as a difference in
      whole-answer agreement between the two rules, not as isolated evidence
      for the extra structure the nested case would license. Callers that pass
      a non-nested pair to :func:`functional_form_fit` document the
      interpretation that actually applies at the call site.
    """

    name: str
    design: np.ndarray


def read_position_logits(
    model: GroupModel, order: int, *, device: torch.device = torch.device("cpu")
) -> np.ndarray:
    """The model's read-position logits over the whole Cayley grid, shaped
    ``[order, order, n_classes]`` (row ``(a, b)``).

    ``device`` runs the forward pass there in the model's native dtype
    (float32); the logits returned are cast to CPU float64 exactly as before,
    so the fits this feeds are unaffected in precision. The model is moved to
    ``device`` for the duration of this call and always restored to CPU on
    return. An MPS op gap falls back to CPU automatically, logged as a
    ``RuntimeWarning`` (see :func:`.device.run_with_device_fallback`)."""
    tokens = cayley_grid_tokens(order)
    model.eval()

    def _compute(dev: torch.device) -> np.ndarray:
        moved = dev.type != "cpu"
        if moved:
            model.to(dev)
        try:
            with torch.no_grad():
                logits = model(tokens.to(dev))[:, -1, :].detach().cpu().to(torch.float64).numpy()
            return logits
        finally:
            if moved:
                model.to(torch.device("cpu"))

    logits, _note = run_with_device_fallback(_compute, device)
    _assert_finite(logits, "read_position_logits")
    return logits.reshape(order, order, -1)


def pair_split(
    order: int, *, train_frac: float = 0.7, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """A reproducible split of the ``order^2`` input pairs into fit/held-out
    row indices (into the flattened ``(a, b)`` grid). The functional-form fit is
    fitted on the first and scored only on the second -- the held-out ``(a, b)``
    pairs a nested basis cannot be rewarded for on noise."""
    n = order * order
    permutation = np.random.default_rng(seed).permutation(n)
    n_fit = int(round(train_frac * n))
    n_fit = max(1, min(n - 1, n_fit))
    return np.sort(permutation[:n_fit]), np.sort(permutation[n_fit:])


def _held_out_fve(
    logits: np.ndarray,
    design: np.ndarray,
    fit_pairs: np.ndarray,
    test_pairs: np.ndarray,
) -> float:
    """Fraction of variance explained on held-out pairs when the (class-axis)
    mean-centred logits are regressed onto the mean-centred design. Softmax is
    shift-invariant, so both are centred across the class axis first: a constant
    per-pair shift carries no information and must not enter the fit."""
    order = logits.shape[0]
    n_classes = logits.shape[2]
    y = logits.reshape(order * order, n_classes)
    y = y - y.mean(axis=1, keepdims=True)
    x = design.reshape(order * order, n_classes, design.shape[3]).astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)

    def _stack(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_rows = y[rows].reshape(-1)
        x_rows = x[rows].reshape(-1, x.shape[2])
        return x_rows, y_rows

    x_fit, y_fit = _stack(fit_pairs)
    coef, _, _, _ = np.linalg.lstsq(x_fit, y_fit, rcond=None)
    x_test, y_test = _stack(test_pairs)
    residual = y_test - x_test @ coef
    ss_res = float(residual @ residual)
    ss_tot = float(((y_test - y_test.mean()) @ (y_test - y_test.mean())))
    if ss_tot <= 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def functional_form_fit(
    model: GroupModel,
    order: int,
    forms: list[FunctionalForm],
    *,
    train_frac: float = 0.7,
    seed: int = 0,
    null_model: GroupModel | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    """I-20: regress the model's logits onto each candidate closed-form rule and
    report held-out FVE, with a nested comparison and an untrained-model null.

    Every score is on held-out ``(a, b)`` pairs never used to fit, so a larger
    (nested) basis is *not* rewarded for its extra columns on noise: the reported
    ``held_out_fve`` per form and the ``held_out_fve_gain`` between the first two
    forms are held-out quantities, never an in-sample gap (banned practice 5).
    When ``null_model`` (a random-init model of the same shape) is supplied its
    held-out FVE is reported alongside and must be ~0 -- the rule-1 regression.

    ``device`` runs both models' read-position forward passes there (see
    :func:`read_position_logits`); the fit itself is unaffected in precision.

    Gates every rung-5 claim; consumed by C1 [D32], C3, C5 and I-27's twisted-
    rule fit. Rung 5 only when connected to circuit or causal evidence.
    """
    logits = read_position_logits(model, order, device=device)
    # Computed once, not per form: the null model's logits do not depend on
    # the form being scored, so recomputing them inside the loop was a
    # wasted forward pass per form for no different result.
    null_logits = (
        read_position_logits(null_model, order, device=device) if null_model is not None else None
    )
    fit_pairs, test_pairs = pair_split(order, train_frac=train_frac, seed=seed)
    per_form: list[dict[str, Any]] = []
    for form in forms:
        fve = _held_out_fve(logits, form.design, fit_pairs, test_pairs)
        entry: dict[str, Any] = {
            "name": form.name,
            "n_features": int(form.design.shape[3]),
            "held_out_fve": fve,
        }
        if null_logits is not None:
            entry["null_held_out_fve"] = _held_out_fve(
                null_logits, form.design, fit_pairs, test_pairs
            )
        per_form.append(entry)
    record: dict[str, Any] = {
        "instrument": "functional-form-fit",
        "rung": 5,
        "train_frac": float(train_frac),
        "split_seed": int(seed),
        "n_fit_pairs": int(fit_pairs.size),
        "n_held_out_pairs": int(test_pairs.size),
        "forms": per_form,
        "note": (
            "Held-out FVE on (a, b) pairs never used to fit; nested forms are "
            "compared on held-out FVE, never an in-sample gain. Logits and design "
            "are class-axis mean-centred (softmax shift-invariance)."
        ),
    }
    if len(per_form) >= 2:
        record["held_out_fve_gain"] = per_form[0]["held_out_fve"] - per_form[1]["held_out_fve"]
    return record


# ---------------------------------------------------------------------------
# I-27: signed-cyclic probe and twisted-rule fit (T12).
# ---------------------------------------------------------------------------


def _signed_cyclic_forms(coords: SignedCyclicCoords, order: int) -> list[FunctionalForm]:
    """The pair of closed-form rules for the twisted-rule fit: the full
    signed-cyclic rule (uses the ``(-1)^{s}`` inversion) and the reduced
    untwisted rule (adds the rotations directly). Each is a one-hot indicator of
    the predicted answer per ``(a, b)``.

    NOT a nested pair in :class:`FunctionalForm`'s sense (contrast the general
    nested case that class documents): the full rule predicts ``a*b`` exactly,
    by construction -- it *is* the ground-truth one-hot -- while the reduced
    rule predicts a different (wrong, wherever the inversion matters) element.
    Consequently the full form's held-out FVE is really a measurement of the
    model's accuracy on the held-out pairs, and the full-minus-reduced gap is
    accuracy-sensitive: it mostly reflects whether the model gets the
    inversion-sensitive pairs right at all, not whether it computes them via a
    signed-cyclic mechanism specifically. Any accurate (grokked) model would
    show the same positive gap here, on any group -- this fit is not
    independent mechanism evidence. I-27's mechanism content is the ``(r, s)``
    coordinate system existing at all (structurally ``UNDEFINED`` on the
    quaternionic member) and the ``rotation_r``/``reflection_s`` probes scored
    alongside it; see :func:`signed_cyclic_instrument`, whose record carries
    this caveat on the fit."""
    m = coords.radix
    r = coords.coords[:, 0]
    s = coords.coords[:, 1]

    def _indicator(action: int) -> np.ndarray:
        design = np.zeros((order, order, order, 1), dtype=np.float64)
        for a in range(order):
            for b in range(order):
                rr = (int(r[a]) + (action ** int(s[a])) * int(r[b])) % m
                ss = (int(s[a]) + int(s[b])) % 2
                design[a, b, int(coords.element_of[rr, ss]), 0] = 1.0
        return design

    full = FunctionalForm(name="signed_cyclic_twisted", design=_indicator(coords.action_sign))
    reduced = FunctionalForm(name="untwisted", design=_indicator(1))
    return [full, reduced]


def signed_cyclic_instrument(
    model: GroupModel,
    group: FiniteGroup,
    *,
    source: str = "embed",
    train_frac: float = 0.7,
    seed: int = 0,
    null_model: GroupModel | None = None,
) -> dict[str, Any]:
    """I-27: the signed-cyclic (dihedral) probe and twisted-rule fit (T12).

    Returns ``status: "UNDEFINED"`` on the quaternionic member -- the whole
    instrument, not a low score -- because :func:`signed_cyclic_coordinates`
    finds no ``(r, s)`` coordinate system there (Q32, C13:Q8 are non-split). This
    is the negative control that makes the dihedral result decisive: a probe that
    "succeeded" on the quaternionic member would be fitting an artefact and would
    disqualify the dihedral result, so no coordinates that fail to reproduce the
    table are ever produced.

    Where coordinates exist: the probe decodes the rotation coordinate ``r`` and
    the reflection bit ``s`` from the per-element representation (rung 1, chance-
    corrected), and the twisted-rule fit (via I-20) reports held-out FVE for the
    signed-cyclic rule against the untwisted reduced rule (rung 5 with I-15/I-28b
    as the causal check).

    CAVEAT (carried in the record's ``twisted_rule_fit["caveat"]``): the two
    forms are not a nested pair (see :func:`_signed_cyclic_forms`) -- the full
    form is exactly the ground-truth one-hot, so the fit's held-out FVE gap is
    accuracy-sensitive, not independent evidence that the model's mechanism is
    specifically signed-cyclic. That mechanism content is carried by the
    ``(r, s)`` probes above and by this instrument's structural ``UNDEFINED``
    on the quaternionic member, not by the fit's FVE gap.
    """
    coords = signed_cyclic_coordinates(group)
    if coords is None or not coords.is_signed_cyclic:
        return {
            "instrument": "signed-cyclic",
            "target_theory": "T12",
            "status": UNDEFINED,
            "reason": (
                "no split C_m : C_2 inversion structure: the (r, s) coordinate "
                "system does not exist on this group (the quaternionic / non-split "
                "member), so the signed-cyclic probe and fit are structurally "
                "undefined -- a success here would fit an artefact"
            ),
        }
    features = _feature_matrix(model, group, source)
    probe = {
        "rotation_r": crossval_probe(features, coords.coords[:, 0], seed=seed),
        "reflection_s": crossval_probe(features, coords.coords[:, 1], seed=seed),
    }
    forms = _signed_cyclic_forms(coords, group.order)
    fit = functional_form_fit(
        model,
        group.order,
        forms,
        train_frac=train_frac,
        seed=seed,
        null_model=null_model,
    )
    fit["caveat"] = (
        "The two forms are not a nested pair: the full form is exactly the "
        "ground-truth one-hot (it predicts a*b by construction), so this "
        "held-out FVE (and the full-minus-reduced gap) is accuracy-sensitive "
        "-- it mostly measures whether the model gets the inversion-sensitive "
        "pairs right, not whether it computes them via a signed-cyclic "
        "mechanism specifically. Any accurate (grokked) model would show the "
        "same positive gap here, signed-cyclic or not. This fit is not "
        "independent mechanism evidence; I-27's mechanism content is the "
        "rotation_r/reflection_s probe above and the structural UNDEFINED on "
        "the quaternionic member."
    )
    return {
        "instrument": "signed-cyclic",
        "target_theory": "T12",
        "status": "measured",
        "rung": 5,
        "radix": coords.radix,
        "action_sign": coords.action_sign,
        "source": source,
        "probe": probe,
        "twisted_rule_fit": fit,
        "note": (
            "Probe (rung 1, chance-corrected) decodes (r, s); the twisted-rule fit "
            "(rung 5, held-out FVE) compares the signed-cyclic rule against the "
            "untwisted reduced rule -- an accuracy-sensitive comparison, not "
            "independent mechanism evidence, see twisted_rule_fit['caveat']. "
            "Defined only on the split member."
        ),
    }


# ---------------------------------------------------------------------------
# I-26: polycyclic digit probe, direction count, carry structure (T16).
# ---------------------------------------------------------------------------


def carry_digit_instrument(
    model: GroupModel,
    group: FiniteGroup,
    *,
    source: str = "embed",
    seed: int = 0,
    null_model: GroupModel | None = None,
) -> dict[str, Any]:
    """I-26: the polycyclic digit probe, coordinate-direction count and ripple-
    carry structure (T16), for the C3 members (C128, C2^7, C127 anchor).

    Returns ``status: "UNDEFINED"`` when no polycyclic (mixed-radix) coordinate
    system can be built (:func:`polycyclic_digits` fails). Otherwise it reports,
    per digit, the chance-corrected decodability of that digit from the model's
    representation (rung 1); the carry matrix (diagonal for an elementary-abelian
    group whose digits add independently, triangular where low digits ripple into
    high ones); and the coordinate-direction count with its own random-init
    ``W_E`` null. Per the plan, the direction count is a structure metric read
    against each group's own null and never compared raw across orders.
    """
    precompute = polycyclic_digits(group)
    if precompute is None:
        return {
            "instrument": "carry-digit",
            "target_theory": "T16",
            "status": UNDEFINED,
            "reason": "no polycyclic (mixed-radix) coordinate system for this group",
        }
    features = _feature_matrix(model, group, source)
    digit_probes: list[dict[str, Any] | str] = []
    for i in range(precompute.length):
        digit_probes.append(crossval_probe(features, precompute.digits[:, i], seed=seed))

    directions = effective_rank(embedding_features(model, group.order))
    null_directions = (
        effective_rank(embedding_features(null_model, group.order))
        if null_model is not None
        else None
    )
    return {
        "instrument": "carry-digit",
        "target_theory": "T16",
        "status": "measured",
        "rung": 1,
        "source": source,
        "composition_length": precompute.length,
        "radices": list(precompute.radices),
        "coordinate_directions": precompute.coordinate_directions,
        "carry_matrix": precompute.carry.astype(int).tolist(),
        "carry_is_diagonal": bool(
            np.array_equal(precompute.carry, np.eye(precompute.length, dtype=bool))
        ),
        "digit_probes": digit_probes,
        "effective_rank_w_e": directions,
        "null_effective_rank_w_e": null_directions,
        "note": (
            "Per-digit decodability is chance-corrected (adjusted balanced "
            "accuracy). The coordinate-direction count and W_E effective rank are "
            "structure metrics read against this group's own random-init null and "
            "never compared raw across orders."
        ),
    }


# ---------------------------------------------------------------------------
# GCR character-readout functional form: the readout prediction of the Group
# Composition via Representations account, built on I-20's harness.
# ---------------------------------------------------------------------------


def _is_trivial_irrep(irrep: Any) -> bool:
    """The trivial irrep: degree 1, character identically 1."""
    return int(irrep.dimension) == 1 and bool(np.allclose(irrep.character, 1.0))


def gcr_candidate_irreps(group: FiniteGroup, *, include_trivial: bool = False) -> list[int]:
    """Indices into ``group.irreps`` of the GCR readout's candidate irreps:
    every nontrivial irrep by default. ``include_trivial=True`` adds the
    trivial irrep back (its ``Phi_rho`` column is the constant ``1``, which the
    fit's class-axis mean-centring removes anyway -- useful mainly to confirm
    it contributes nothing beyond that null column)."""
    return [
        i for i, irrep in enumerate(group.irreps) if include_trivial or not _is_trivial_irrep(irrep)
    ]


def gcr_fourier_only_irreps(group: FiniteGroup) -> list[int]:
    """The nontrivial one-dimensional (abelian-character) irreps: the
    "Fourier-only" rival to the full GCR irrep set. Every column this produces
    is also a column the full candidate set produces (a genuinely nested
    restriction, :class:`FunctionalForm`'s sense), so comparing the full set
    against it isolates what the matrix (degree >= 2) irreps buy over the
    group's abelianisation alone -- :func:`gcr_character_readout_instrument`'s
    primary, out-of-sample statistic."""
    return [i for i in gcr_candidate_irreps(group) if int(group.irreps[i].dimension) == 1]


def gcr_character_design(group: FiniteGroup, irrep_indices: Sequence[int]) -> np.ndarray:
    """The GCR readout's design matrix over a chosen irrep set: shape
    ``[order, order, order, len(irrep_indices)]``, column ``k`` at ``(a, b,
    c)`` equal to ``Phi_rho(a, b, c) = Re tr(rho(a) rho(b) rho(c^-1))`` for
    ``rho = group.irreps[irrep_indices[k]]``. Full irrep matrices come
    straight from the Sage/GAP artifact (``IrrepData.matrices``, ``[order,
    degree, degree]`` complex128, ``groups/data.py``); ``rho(c^-1)`` is read
    off by indexing the matrices with the Cayley-table inverse map.

    CAVEAT -- load-bearing, read before treating a fit against this design
    matrix as evidence (see :func:`gcr_character_readout_instrument` for the
    full statement): ``tr(rho(a) rho(b) rho(c^-1)) = tr(rho(a*b) rho(c^-1))``
    is fixed once the product ``a*b`` and ``c`` are fixed -- it does not
    depend on ``a`` and ``b`` separately. Any algorithm that computes the
    group product correctly therefore produces logits whose *information
    content* this design matrix can represent, so a high raw (held-out) FVE
    from it is consistent with every correct algorithm, not only GCR, and is
    not by itself evidence for GCR specifically.
    """
    table = group.cayley_table
    order = int(table.shape[0])
    e = identity_index(table)
    inv = inverses(table, e)
    irrep_indices = list(irrep_indices)
    design = np.empty((order, order, order, len(irrep_indices)), dtype=np.float64)
    for k, idx in enumerate(irrep_indices):
        matrices = group.irreps[idx].matrices  # [order, d, d] complex128: rho(g)
        matrices_inv = matrices[inv]  # rho(c^-1), indexed directly by c
        # ab[a, b]_{ik} = sum_j rho(a)_{ij} rho(b)_{jk} = (rho(a) @ rho(b))_{ik}.
        ab = np.einsum("aij,bjk->abik", matrices, matrices)
        # phi[a, b, c] = sum_{i,k} ab[a, b]_{ik} * rho(c^-1)_{ki} = trace(rho(a) rho(b) rho(c^-1)).
        phi = np.einsum("abik,cki->abc", ab, matrices_inv)
        design[:, :, :, k] = phi.real
    return design


def gcr_character_form(
    group: FiniteGroup,
    irrep_indices: Sequence[int] | None = None,
    *,
    name: str | None = None,
) -> FunctionalForm:
    """A :class:`FunctionalForm` whose design is the GCR character readout
    (:func:`gcr_character_design`) over ``irrep_indices`` (every nontrivial
    irrep by default, :func:`gcr_candidate_irreps`). Passing a specified
    sparse subset is what lets the minimal-key-irrep-set search in
    :func:`gcr_character_readout_instrument` score any candidate subset with
    the same nested-safe held-out machinery as the full fit."""
    resolved_indices = gcr_candidate_irreps(group) if irrep_indices is None else list(irrep_indices)
    design = gcr_character_design(group, resolved_indices)
    if name is None:
        dims = ",".join(str(group.irreps[i].dimension) for i in resolved_indices)
        name = f"gcr_character[{dims}]"
    return FunctionalForm(name=name, design=design)


def _saturated_lookup_form(group: FiniteGroup, name: str) -> FunctionalForm:
    """A saturated one-hot lookup of the true product ``a*b`` -- the accuracy
    ceiling any correct model's logits approach regardless of mechanism.
    Reported only as secondary context (:func:`gcr_character_readout_instrument`):
    this ceiling is not GCR-specific either, it is the accuracy ceiling itself,
    the same construction as :func:`_signed_cyclic_forms`'s full rule."""
    table = group.cayley_table
    order = int(table.shape[0])
    design = np.zeros((order, order, order, 1), dtype=np.float64)
    for a in range(order):
        for b in range(order):
            design[a, b, int(table[a, b]), 0] = 1.0
    return FunctionalForm(name=name, design=design)


def _greedy_minimal_irrep_search(
    group: FiniteGroup,
    logits: np.ndarray,
    fit_pairs: np.ndarray,
    test_pairs: np.ndarray,
    *,
    target_fve: float,
    improvement_tol: float,
) -> dict[str, Any]:
    """Greedy forward selection over the candidate nontrivial irreps -- the
    MINIMAL-irrep-set statistic :func:`gcr_character_readout_instrument` leads
    with, since raw full-set FVE cannot discriminate GCR from any other
    correct algorithm. At each step, add whichever remaining irrep raises the
    cumulative design's held-out FVE (:func:`_held_out_fve`, reused directly,
    not reimplemented) the most; stop once ``target_fve`` is reached, or the
    best available step improves held-out FVE by less than
    ``improvement_tol`` (once at least one irrep is already selected -- the
    first pick is always taken, so an empty selection is never reported for a
    nondegenerate search). A small selected set reaching high held-out FVE is
    a specific, falsifiable GCR prediction (sparse occupancy); the search
    trace records every step considered, selected or not.
    """
    remaining = gcr_candidate_irreps(group)
    selected: list[int] = []
    trace: list[dict[str, Any]] = []
    best_fve = 0.0
    while remaining:
        scored = [
            (
                _held_out_fve(
                    logits,
                    gcr_character_design(group, [*selected, idx]),
                    fit_pairs,
                    test_pairs,
                ),
                idx,
            )
            for idx in remaining
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        step_fve, step_idx = scored[0]
        improvement = step_fve - best_fve
        accept = not selected or improvement >= improvement_tol
        trace.append(
            {
                "irrep_index": int(step_idx),
                "irrep_dimension": int(group.irreps[step_idx].dimension),
                "cumulative_held_out_fve": float(step_fve),
                "improvement": float(improvement),
                "selected": bool(accept),
            }
        )
        if not accept:
            break
        selected.append(step_idx)
        remaining.remove(step_idx)
        best_fve = step_fve
        if best_fve >= target_fve:
            break
    return {
        "target_fve": float(target_fve),
        "improvement_tol": float(improvement_tol),
        "n_candidates": len(gcr_candidate_irreps(group)),
        "selected_irrep_indices": [int(i) for i in selected],
        "selected_irrep_dimensions": [int(group.irreps[i].dimension) for i in selected],
        "n_selected": len(selected),
        "held_out_fve_at_selection": float(best_fve),
        "reached_target": bool(best_fve >= target_fve),
        "search_trace": trace,
    }


def gcr_character_readout_instrument(
    model: GroupModel,
    group: FiniteGroup,
    *,
    train_frac: float = 0.7,
    seed: int = 0,
    null_model: GroupModel | None = None,
    target_fve: float = 0.95,
    improvement_tol: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    """The GCR character-readout functional form: tests the Group Composition
    via Representations account's readout prediction that a model's
    read-position logits are a sparse sum over occupied irreps of
    ``Phi_rho(a, b, c) = Re tr(rho(a) rho(b) rho(c^-1))``
    (:func:`gcr_character_design`).

    CAVEAT -- READ BEFORE CITING RAW FVE (the overclaim this instrument is
    built to avoid): ``Phi_rho`` is a class function of the single product
    ``a*b`` once ``c`` is fixed, so a high raw held-out FVE from the full
    irrep set is consistent with ANY correct algorithm, not only GCR -- it
    proves nothing about GCR specifically on its own. This instrument's
    load-bearing, reported statistics are therefore, under ``primary``:

    * ``nested_comparison`` -- the out-of-sample nested comparison (I-20) of
      the full candidate irrep set against the Fourier-only (abelian,
      degree-1-irrep) rival (:func:`gcr_fourier_only_irreps`, a genuine subset
      of the full set's columns): does the full set beat the abelian
      restriction out of sample, and by how much (``full_vs_fourier_held_out_fve_gain``).
    * ``minimal_irrep_set`` -- the minimal irrep subset a greedy forward
      search (:func:`_greedy_minimal_irrep_search`) needs to reach
      ``target_fve``: a small selected set is the specific, falsifiable GCR
      prediction (sparse occupancy) that raw full-set FVE cannot distinguish
      from any other correct algorithm.

    Raw held-out FVE -- including a saturated one-hot lookup ceiling (the
    accuracy ceiling any correct model's logits approach, GCR-specific or not)
    -- is reported only as secondary context, under ``secondary_raw_fve``,
    never as the headline number.

    Reuses I-20's nested-safe held-out machinery throughout:
    :func:`functional_form_fit` for the primary nested comparison (with the
    same untrained-model-null convention when ``null_model`` is supplied), and
    :func:`_held_out_fve` directly, not reimplemented, for the minimal-set
    search. When the group is perfect (no nontrivial one-dimensional irrep --
    no Fourier-only rival exists), ``primary.nested_comparison`` comes back
    ``UNDEFINED`` rather than compared against an empty design; the
    minimal-set search and secondary raw FVE are unaffected.
    """
    order = group.order
    full_irreps = gcr_candidate_irreps(group)
    fourier_irreps = gcr_fourier_only_irreps(group)

    full_form = gcr_character_form(group, full_irreps, name="gcr_character_full")
    ceiling_form = _saturated_lookup_form(group, name="saturated_lookup_ceiling")
    forms = [full_form]
    if fourier_irreps:
        forms.append(gcr_character_form(group, fourier_irreps, name="gcr_character_fourier_only"))
    forms.append(ceiling_form)

    fit = functional_form_fit(
        model, order, forms, train_frac=train_frac, seed=seed, null_model=null_model, device=device
    )
    fve_by_name = {entry["name"]: entry["held_out_fve"] for entry in fit["forms"]}
    null_fve_by_name = (
        {entry["name"]: entry.get("null_held_out_fve") for entry in fit["forms"]}
        if null_model is not None
        else None
    )

    if fourier_irreps:
        nested_comparison: dict[str, Any] = {
            "full_vs_fourier_held_out_fve_gain": fit["held_out_fve_gain"],
            "full_held_out_fve": fve_by_name["gcr_character_full"],
            "fourier_only_held_out_fve": fve_by_name["gcr_character_fourier_only"],
        }
    else:
        nested_comparison = {
            "status": UNDEFINED,
            "reason": (
                "group has no nontrivial one-dimensional irrep (perfect group): "
                "no Fourier-only rival exists to compare against"
            ),
            "full_held_out_fve": fve_by_name["gcr_character_full"],
        }

    logits = read_position_logits(model, order, device=device)
    fit_pairs, test_pairs = pair_split(order, train_frac=train_frac, seed=seed)
    minimal_set = _greedy_minimal_irrep_search(
        group,
        logits,
        fit_pairs,
        test_pairs,
        target_fve=target_fve,
        improvement_tol=improvement_tol,
    )

    record: dict[str, Any] = {
        "instrument": "gcr-character-readout",
        "target_theory": "GCR",
        "rung": 5,
        "status": "measured",
        "n_candidate_irreps": len(full_irreps),
        "n_fourier_irreps": len(fourier_irreps),
        "candidate_irrep_dimensions": [int(group.irreps[i].dimension) for i in full_irreps],
        "primary": {
            "nested_comparison": nested_comparison,
            "minimal_irrep_set": minimal_set,
        },
        "secondary_raw_fve": {
            "full_held_out_fve": fve_by_name["gcr_character_full"],
            "saturated_lookup_ceiling_held_out_fve": fve_by_name["saturated_lookup_ceiling"],
            "note": (
                "Raw FVE, including the saturated one-hot ceiling: informational "
                "only, never the reported evidence for GCR -- see the caveat."
            ),
        },
        "functional_form_fit": fit,
        "caveat": (
            "Phi_rho(a, b, c) is a class function of a*b (for fixed c), so a high "
            "raw held-out FVE is consistent with any correct algorithm, not only "
            "GCR. The load-bearing statistics are primary.nested_comparison (full "
            "irrep set vs the Fourier-only/abelian rival, out of sample) and "
            "primary.minimal_irrep_set (the sparse subset a greedy search needs "
            "to reach target_fve), never secondary_raw_fve."
        ),
    }
    if null_fve_by_name is not None:
        record["null"] = {
            "full_null_held_out_fve": null_fve_by_name.get("gcr_character_full"),
            "fourier_only_null_held_out_fve": null_fve_by_name.get("gcr_character_fourier_only"),
        }
    return record


def effective_rank(matrix: np.ndarray) -> float:
    """The participation-ratio effective rank of a matrix's singular spectrum,
    ``(sum sigma)^2 / sum sigma^2`` -- a smooth stand-in for the rank of the
    element embedding that a random-init null can be compared against within a
    pair (never across orders, where ``d_model`` and ``|G|`` set the null)."""
    singular_values = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    total = float((singular_values**2).sum())
    if total <= 0.0:
        return 0.0
    return float(singular_values.sum() ** 2 / total)


__all__ = [
    "FunctionalForm",
    "PolycyclicDigits",
    "SignedCyclicCoords",
    "UNDEFINED",
    "carry_digit_instrument",
    "crossval_probe",
    "cube_map",
    "cyclic_subgroup",
    "effective_rank",
    "element_orders",
    "embedding_features",
    "functional_form_fit",
    "gcr_candidate_irreps",
    "gcr_character_design",
    "gcr_character_form",
    "gcr_character_readout_instrument",
    "gcr_fourier_only_irreps",
    "identity_index",
    "inverses",
    "involution_direction_ablation",
    "model_correct_mask",
    "neuron_features",
    "pair_split",
    "polycyclic_digits",
    "power_map_probe",
    "read_position_logits",
    "signed_cyclic_coordinates",
    "signed_cyclic_instrument",
    "square_map",
]
