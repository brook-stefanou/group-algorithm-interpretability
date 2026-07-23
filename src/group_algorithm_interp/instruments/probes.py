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

Every element-level probe uses a parameter-free nearest-class-mean classifier
under stratified cross-validation: it cannot overfit random high-dimensional
features the way an unregularised linear probe can, so an untrained model scores
at chance (adjusted 0) as the mandatory rule-1 regression requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from ..groups.group import FiniteGroup
from ..model import GroupModel
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
        """The dihedral (inversion-action) case -- the structure I-27 fits."""
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
    involution = None
    action_sign = 0
    for t in range(n):
        if in_rotation[t] or orders[t] != 2:
            continue
        conjugate = int(table[table[t, generator], inv[t]])
        if conjugate == c_inverse:
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
    in the normal form ``g = prod_i gen_i^{digit_i}`` along a greedily built
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
        # Subnormality guard: the added generator's conjugates must stay inside
        # the extended subgroup, or the chain is not polycyclic (trivial for an
        # abelian group, where every subgroup is normal -- the C3 scope).
        extended = set(new_members)
        if any(int(table[table[x, gen], inv[x]]) not in extended for x in range(n)):
            return None
        generators.append(gen)
        radices.append(radix)
        members = new_members
        member_set = set(new_members)

    length = len(generators)
    radix_tuple = tuple(radices)

    # Digit assignment by the normal form g = gen_{L-1}^{d_{L-1}} ... gen_0^{d_0}.
    # Build element -> digits by enumerating the mixed-radix grid in the same
    # order the chain was grown (generator i multiplies on the outside of G_i).
    digits = np.empty((n, length), dtype=np.int64)

    def _element_for(digit_tuple: tuple[int, ...]) -> int:
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


def embedding_features(model: GroupModel, order: int) -> np.ndarray:
    """The model's learned embedding of each group element, ``W_E[g]``, mean-
    centred across elements (both architectures expose ``W_E``; the ``=`` read
    token, index ``order``, is dropped). Mean-centring removes the DC offset the
    read position injects, per the occupancy DC-dominance finding."""
    w_e = model.W_E.detach().to(torch.float64).cpu().numpy()[:order]
    return w_e - w_e.mean(axis=0, keepdims=True)


def neuron_features(model: GroupModel, group: FiniteGroup, *, argument: str) -> np.ndarray:
    """A per-element feature from the read-position activations ``A[m, a, b]``:
    ``argument="left"`` averages over ``b`` to a function of ``a``, ``"right"``
    over ``a`` to a function of ``b``. Mean-centred across elements (the trivial
    DC block would otherwise dominate)."""
    activations = neuron_activations(model, group.order)  # [d_mlp, order, order]
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

    The involution direction is the difference of class means (involutions minus
    non-involutions) in the chosen feature space -- the axis a nearest-mean probe
    would use. Zero-ablation projects it out of ``W_E``. ``UNDEFINED`` when the
    involution label is degenerate (no involutions, or all elements are
    involutions), matching the probe. ``subset`` restricts the accuracy count to
    an unleaked held-out mask when the caller supplies one; by default every pair
    is counted.

    Rung 3 (causal, Necessary) *only* alongside the probe; reported in flips, the
    behavioural unit, never a bare percentage.
    """
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
    axis) tests how much of the logit structure the rule explains. For a nested
    comparison the reduced form's feature columns must be a subset of the full
    form's.
    """

    name: str
    design: np.ndarray


def read_position_logits(model: GroupModel, order: int) -> np.ndarray:
    """The model's read-position logits over the whole Cayley grid, shaped
    ``[order, order, n_classes]`` (row ``(a, b)``)."""
    tokens = cayley_grid_tokens(order)
    model.eval()
    with torch.no_grad():
        logits = model(tokens)[:, -1, :].to(torch.float64).cpu().numpy()
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
) -> dict[str, Any]:
    """I-20: regress the model's logits onto each candidate closed-form rule and
    report held-out FVE, with a nested comparison and an untrained-model null.

    Every score is on held-out ``(a, b)`` pairs never used to fit, so a larger
    (nested) basis is *not* rewarded for its extra columns on noise: the reported
    ``held_out_fve`` per form and the ``held_out_fve_gain`` between the first two
    forms are held-out quantities, never an in-sample gap (banned practice 5).
    When ``null_model`` (a random-init model of the same shape) is supplied its
    held-out FVE is reported alongside and must be ~0 -- the rule-1 regression.

    Gates every rung-5 claim; consumed by C1 [D32], C3, C5 and I-27's twisted-
    rule fit. Rung 5 only when connected to circuit or causal evidence.
    """
    logits = read_position_logits(model, order)
    fit_pairs, test_pairs = pair_split(order, train_frac=train_frac, seed=seed)
    per_form: list[dict[str, Any]] = []
    for form in forms:
        fve = _held_out_fve(logits, form.design, fit_pairs, test_pairs)
        entry: dict[str, Any] = {
            "name": form.name,
            "n_features": int(form.design.shape[3]),
            "held_out_fve": fve,
        }
        if null_model is not None:
            null_logits = read_position_logits(null_model, order)
            entry["null_held_out_fve"] = _held_out_fve(
                null_logits, form.design, fit_pairs, test_pairs
            )
        per_form.append(entry)
    record: dict[str, Any] = {
        "instrument": "functional-form-fit",
        "rung": 5,
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
    """The nested pair of closed-form rules for the twisted-rule fit: the full
    signed-cyclic rule (uses the ``(-1)^{s}`` inversion) and the reduced
    untwisted rule (adds the rotations directly). Each is a one-hot indicator of
    the predicted answer per ``(a, b)``; the full rule predicts ``a*b`` exactly
    (by construction), the reduced rule predicts the wrong element wherever the
    inversion matters, so the held-out FVE gap is the twist's contribution."""
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
            "untwisted reduced rule. Defined only on the split member."
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
