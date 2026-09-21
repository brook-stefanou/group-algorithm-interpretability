"""The power-map attribution instrument: does the model's per-element
representation encode power-map structure beyond conjugacy-class membership.

## Why this instrument exists

Within a character-table-equivalent matched pair (the C1 pair family), the
Fourier, tensor-rank and coset accounts all read off *class functions* of the
group -- quantities constant on a conjugacy class, which the character table
already fixes identically across the pair by construction. The only axis a
matched pair is free to differ on is the group's power-map / element-order
structure: which specific element is the square, cube, ... k-th power of
which, and hence the fine Cayley-table structure the character table cannot
see (the classical "Brauer pair" phenomenon -- non-isomorphic groups sharing
a character table but differing in their power maps).

I-28 (:func:`probes.power_map_probe`) already asks "is element order /
involution-ness / the square or cube map decodable from the model's
embedding" with a chance-corrected score. What it does not do is attribute
that decodability: a positive I-28 score is also consistent with the model
simply encoding conjugacy class, since element order *is* a class function
(constant on every conjugacy class -- conjugation preserves order). This
module adds that attribution with a narrower, nested question: how much of a
target is decodable *beyond* conjugacy-class membership.

## Class functions vs the true residual

Conjugation commutes with powering: if ``g' = h g h^-1`` then
``g'^k = h g^k h^-1``, so ``g^k`` and ``g'^k`` are always conjugate whenever
``g`` and ``g'`` are. Consequently ``class(g^k)`` is a fixed function of
``class(g)`` alone -- a genuine class function, already fully determined by
conjugacy-class membership and therefore *not* a residual axis. (The same
argument makes element order and involution-ness class functions: this is
provable, not measured, and is used below as a built-in sanity check --
``element_order``/``is_involution``'s nested increment should read ~0.)

What is *not* a class function is the SPECIFIC element ``g^k`` (as opposed to
merely its class): two conjugate ``g, g'`` generally have ``g^k != g'^k`` as
literal elements, even though those images always share a class. That
specific identity is exactly the axis a character-table-only account cannot
see, and is what this module's headline nested test targets.

## Design

Per-element feature: the model's per-element embedding, reusing
:func:`probes.embedding_features` unchanged (``model.W_E`` rows for the
``order`` element tokens, dropping the ``=`` read token, mean-centred across
elements -- the same feature space I-20/I-27/I-28 already probe).

Two views, both reported by :func:`power_map_attribution_instrument`:

* **Direct decodability** (rung 1, context) -- reuses I-28's
  ``power_map_probe`` wholesale (element order, involution-ness, square map,
  cube map, chance-corrected nearest-centroid balanced accuracy under
  stratified CV) and adds a ``conjugacy_class`` target the same way,
  restricted to non-singleton classes (a singleton class -- e.g. the
  identity, or any other central element -- cannot be stratified/held-out,
  mirroring I-28's own exclusion of the identity from ``element_order``).

* **Nested beyond-class increment** (the headline) -- a genuinely nested
  linear-probe pair mirroring ``readout_characterisation``'s full-vs-
  restricted design, in the *decode* direction (predict a group-theoretic
  target from model-derived features) rather than that instrument's *fit*
  direction (predict model logits from group-theoretic designs) -- a
  deliberate, documented fork from the existing pattern (see "Two design
  forks" below):

  - ``restricted`` predictors: the ``[order, n_classes]`` one-hot conjugacy-
    class design (:func:`class_design`) -- literally the only per-element
    information a character-table-only observer has, since every element of
    a class gets the identical feature row by construction.
  - ``full`` predictors: ``restricted`` concatenated with the model's raw
    embedding features -- a genuine nested pair (``restricted``'s columns are
    a literal subset of ``full``'s), so ``full`` can never fit worse
    in-sample.
  - response: a one-hot encoding of the target (``power_map(table, k)`` for
    the headline, plus the ``element_order``/``is_involution`` sanity
    targets), compressed to the target's own observed cardinality (see
    "Response compaction" below), regressed via ordinary least squares and
    scored by held-out fraction of variance explained, cross-validated over
    the ``|G|`` elements (:class:`sklearn.model_selection.KFold`,
    unstratified -- unlike I-28's classifier, a regression has no per-class
    minimum-count constraint, so every element including the identity is
    used).
  - headline statistic:
    ``held_out_fve_gain_full_minus_restricted = full_fve - restricted_fve``.
    Materially positive (above ``tie_tol``) means the embedding encodes the
    specific power-map image beyond what conjugacy class alone predicts --
    the attribution this instrument exists to make. Near zero on the
    ``element_order``/``is_involution`` sanity targets is the expected,
    provable null (both are class functions, so ``restricted`` alone should
    already explain them).

## Two design forks from the existing pattern (flagged for review)

1. **Decode direction, not fit direction.** ``readout_characterisation``
   regresses the MODEL's read-position logits onto group-theoretic design
   matrices (fit direction: does the model's output match a candidate rule).
   This module instead regresses a group-theoretic TARGET onto model-derived
   features (decode direction: does the model's representation predict a
   candidate structural fact). The nesting property that makes the
   comparison honest -- ``restricted``'s columns a literal subset of
   ``full``'s -- is preserved either way, but the direction is a genuine
   fork, chosen because the task this module answers ("is X decodable") is
   naturally a decode question, not a fit question.
2. **``k=2`` only by default.** See "Response compaction" below for why
   ``k=3`` is dropped from the default ``power_ks`` (still available on
   request).

## Response compaction

The response's one-hot width is compressed to the target's own observed
cardinality (:func:`_compact_labels`), not a fixed ``order``-wide encoding.
This matters because a power map can be a bijection: cubing is bijective
whenever ``gcd(3, |G|) == 1``, which holds for every power-of-two order in
this project's panel (64, 128, 192's 2-part, 216 is the exception at
``3 | 216``). A bijective power map's "specific image identity" carries
exactly as much information as the element's own identity, so an
``order``-wide one-hot response there is really an ``order``-way
memorisation task -- indistinguishable, at the panel's element counts
(64-255), from massively overfitting a saturated regression (far more
response columns than training rows per fold). Compaction narrows the
response to only the images the map actually takes, which is a genuine
improvement whenever the map is non-injective (the common case for squaring
on the even-order panel groups) but cannot rescue a genuinely bijective map.
That is why ``k=3`` -- bijective across most of the panel -- is dropped from
the default; ``k=2`` (squaring) is typically non-injective on an even-order
group (every involution squares to the identity, collapsing many elements
onto one image), so compaction buys it real headroom. Pass ``power_ks``
explicitly to probe another power regardless.

CAVEAT, read before citing a raw FVE: even after compaction, ``full``'s
extra ``d_model`` columns can still exceed the per-fold training-row count
for the smaller panel groups, putting both designs in a saturated-regression
regime. This affects ``restricted`` and ``full`` identically (same response,
same row split), so the *gain* stays the load-bearing comparison; a raw
``full_held_out_fve`` alone is not evidence of anything on its own. "Fit
freely" is the deliberate choice here: no basis search, no regularisation
sweep, so the reported gain does not depend on a tuning decision.

The full-scale run (``results/power_map_attribution/``) bears out a sharper
limit than the design argument above: on the order-64 cells (``d_model`` 128 or
256 against ``|G|`` = 64) the gains are small, positive and interpretable, but
on the smaller-order cells (orders 27, 54, 104, 32 at their current widths) the
``full`` design collapses so severely, and asymmetrically between paired
members, that even the within-pair gain can ride on that differential collapse
rather than on a clean signal. Treat the headline gain as interpretable only
where ``d_model`` does not dominate the per-fold training-row count; a rank
reduction of ``full``'s embedding columns before concatenation is the fix if
the smaller-order cells are ever needed.

## Rung and what is not here

Rung 1 throughout (decodable, not "used") -- this module makes no causal
claim. A causal ablation arm (remove the beyond-class subspace and measure
the hit to held-out multiplication accuracy, mirroring I-28b) is a possible
future extension but is not implemented here: unlike I-28b's involution
direction (a single well-defined class-mean difference), the beyond-class
subspace this module's ``full`` design fits is a ``d_model``-dimensional
object with no single obviously-correct rank to ablate, and it stays out of
the public surface until that is worked out.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import KFold

from ..groups.group import FiniteGroup
from ..model import GroupModel
from .probes import (
    UNDEFINED,
    crossval_probe,
    element_orders,
    embedding_features,
    identity_index,
    power_map_probe,
)

_MAX_FOLDS = 5

__all__ = [
    "UNDEFINED",
    "class_design",
    "conjugacy_class_labels",
    "conjugacy_class_probe_target",
    "nested_beyond_class_fve",
    "power_map",
    "power_map_attribution_instrument",
]


# ---------------------------------------------------------------------------
# Pure Cayley-table group theory: the general power map and conjugacy-class
# design, both computed from the exported table only (no Sage at analysis
# time, matching probes.py's own convention).
# ---------------------------------------------------------------------------


def power_map(table: np.ndarray, k: int) -> np.ndarray:
    """The general power map ``g -> g^k`` as element indices.

    ``k=0`` is the constant identity map (degenerate; flagged ``UNDEFINED``
    downstream, never asserted away here), ``k=1`` is the identity function,
    and ``k=2``/``k=3`` agree exactly with :func:`probes.square_map`/
    :func:`probes.cube_map` (checked in the test suite). Pure Cayley-table
    arithmetic, no Sage/GAP dependency at analysis time.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    n = table.shape[0]
    e = identity_index(table)
    out = np.empty(n, dtype=np.int64)
    for g in range(n):
        power = e
        for _ in range(k):
            power = int(table[power, g])
        out[g] = power
    return out


def conjugacy_class_labels(group: FiniteGroup) -> np.ndarray:
    """``labels[g]`` = index of the conjugacy class containing ``g``, from
    the artifact's own ``conjugacy_classes`` (Sage/GAP-exported, never
    rediscovered from the Cayley table)."""
    labels = np.empty(group.order, dtype=np.int64)
    for idx, members in enumerate(group.conjugacy_classes):
        labels[members] = idx
    return labels


def class_design(group: FiniteGroup) -> np.ndarray:
    """``[order, n_classes]`` one-hot conjugacy-class design: row ``g`` is the
    indicator of ``g``'s class. This is literally the only per-element
    information available to an observer restricted to the character table --
    every element of a class receives an identical feature row -- which is
    what makes it the correct ``restricted`` baseline for the nested test."""
    labels = conjugacy_class_labels(group)
    n_classes = len(group.conjugacy_classes)
    design = np.zeros((group.order, n_classes), dtype=np.float64)
    design[np.arange(group.order), labels] = 1.0
    return design


def conjugacy_class_probe_target(group: FiniteGroup) -> tuple[np.ndarray, np.ndarray]:
    """Conjugacy-class labels for the direct-decode probe, restricted to
    elements in non-singleton classes.

    A singleton class (the identity is always one; any other central element
    is too) has only one member and so cannot be stratified/held-out by
    ``crossval_probe``'s own rule -- worse, a single leftover singleton class
    makes the *whole* label vector degenerate under that rule (its global
    per-class minimum-count check), silently reporting ``UNDEFINED`` even
    when the non-singleton classes carry real decodable structure. This
    mirrors I-28's own exclusion of the identity from the ``element_order``
    target, generalised to every singleton class. Returns ``(mask,
    labels[mask])``.
    """
    labels = conjugacy_class_labels(group)
    sizes = np.array([len(members) for members in group.conjugacy_classes], dtype=np.int64)
    mask = sizes[labels] >= 2
    return mask, labels[mask]


# ---------------------------------------------------------------------------
# The nested held-out-FVE test: restricted (class-only) vs full (class +
# embedding) designs predicting a one-hot response, cross-validated over the
# |G| elements.
# ---------------------------------------------------------------------------


def _compact_labels(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Re-index ``labels`` to a dense ``0..n_unique-1`` range and return the
    compact width -- the response-compaction that keeps the nested FVE
    test's response as narrow as the target's actual cardinality, rather
    than a fixed ``order``-wide one-hot regardless of how many distinct
    values the target actually takes (module docstring, "Response
    compaction"). ``labels`` must be non-empty."""
    _, inverse = np.unique(labels, return_inverse=True)
    inverse = inverse.astype(np.int64)
    return inverse, int(inverse.max()) + 1


def _one_hot(labels: np.ndarray, width: int) -> np.ndarray:
    n = labels.shape[0]
    out = np.zeros((n, width), dtype=np.float64)
    out[np.arange(n), labels] = 1.0
    return out


def _held_out_fve(
    y_train: np.ndarray, y_test: np.ndarray, x_train: np.ndarray, x_test: np.ndarray
) -> float:
    """Fraction of variance explained on the held-out rows: fit an
    unregularised least-squares map ``x -> y`` on the train rows, score
    residual sum of squares on the test rows against the test set's own
    total variance (the same convention ``probes._held_out_fve`` uses)."""
    coef, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    residual = y_test - x_test @ coef
    ss_res = float(np.square(residual).sum())
    centred = y_test - y_test.mean(axis=0, keepdims=True)
    ss_tot = float(np.square(centred).sum())
    if ss_tot <= 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def nested_beyond_class_fve(
    group: FiniteGroup,
    features: np.ndarray,
    response_labels: np.ndarray,
    *,
    n_folds: int = _MAX_FOLDS,
    seed: int = 0,
) -> dict[str, Any] | str:
    """The nested held-out-FVE test: does ``restricted`` (conjugacy-class
    one-hot alone) explain ``response_labels`` as well as ``full``
    (``restricted`` concatenated with ``features``), cross-validated over the
    ``|G|`` elements.

    Returns ``UNDEFINED`` (a string) for a degenerate (constant) response,
    mirroring ``crossval_probe``'s degenerate-label rule -- a constant target
    has zero variance and the comparison is vacuous, not a "failed" fit.
    Otherwise a dict with both designs' mean held-out FVE (averaged over
    ``min(n_folds, order)`` unstratified K-folds -- a regression has no
    per-class minimum-count constraint, unlike the classifier probes), the
    gain, and the per-fold values for inspection. The response is compacted
    to its own observed cardinality first (:func:`_compact_labels`, module
    docstring "Response compaction").
    """
    order = group.order
    unique = np.unique(response_labels)
    if unique.size < 2:
        return UNDEFINED
    compact, width = _compact_labels(response_labels)
    restricted = class_design(group)
    full = np.concatenate([restricted, features], axis=1)
    y = _one_hot(compact, width)

    n_splits = min(n_folds, order)
    if n_splits < 2:
        return UNDEFINED
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    restricted_fves: list[float] = []
    full_fves: list[float] = []
    for train_idx, test_idx in splitter.split(np.arange(order)):
        restricted_fves.append(
            _held_out_fve(y[train_idx], y[test_idx], restricted[train_idx], restricted[test_idx])
        )
        full_fves.append(_held_out_fve(y[train_idx], y[test_idx], full[train_idx], full[test_idx]))
    restricted_fve = float(np.mean(restricted_fves))
    full_fve = float(np.mean(full_fves))
    return {
        "restricted_held_out_fve": restricted_fve,
        "full_held_out_fve": full_fve,
        "held_out_fve_gain_full_minus_restricted": full_fve - restricted_fve,
        "n_folds": n_splits,
        "response_width": int(width),
        "response_width_uncompacted": int(order),
        "restricted_fve_per_fold": [float(v) for v in restricted_fves],
        "full_fve_per_fold": [float(v) for v in full_fves],
    }


# ---------------------------------------------------------------------------
# The instrument proper.
# ---------------------------------------------------------------------------


def power_map_attribution_instrument(
    model: GroupModel,
    group: FiniteGroup,
    *,
    power_ks: Sequence[int] = (2,),
    n_folds: int = _MAX_FOLDS,
    seed: int = 0,
    tie_tol: float = 0.01,
    null_model: GroupModel | None = None,
) -> dict[str, Any]:
    """The power-map attribution instrument on one trained model: direct
    decodability (rung 1, context, reusing I-28 unchanged) plus the nested
    beyond-conjugacy-class increment (the headline).

    ``power_ks`` are the powers probed as the headline residual axis
    (default squaring only -- see the module docstring for why cubing is
    dropped from the default); the first entry is the reported
    ``headline_target``. ``element_order``/``is_involution`` are always
    probed too, as a built-in sanity check -- both are provably class
    functions (module docstring), so their nested increment should read ~0;
    a materially positive increment there would indicate a bug in this
    instrument, not a finding about the model.

    ``null_model`` (a random-init model of the same shape) is optional; when
    supplied, the identical nested test runs against its embedding too and
    the result is reported under ``null`` -- the rule-1 regression this
    project's mechanism-arm instruments carry: an untrained model's gain
    should sit near zero, since held-out scoring cannot reward memorising
    noise.
    """
    order = group.order
    table = group.cayley_table
    features = embedding_features(model, order)

    direct = power_map_probe(model, group, source="embed", seed=seed)
    direct_probes: dict[str, Any] = dict(direct["probes"])
    class_mask, class_labels_restricted = conjugacy_class_probe_target(group)
    direct_probes["conjugacy_class"] = crossval_probe(
        features[class_mask], class_labels_restricted, seed=seed
    )

    def _nested_targets(feats: np.ndarray) -> dict[str, Any]:
        orders = element_orders(table)
        targets: dict[str, Any] = {
            "element_order": nested_beyond_class_fve(
                group, feats, orders, n_folds=n_folds, seed=seed
            ),
            "is_involution": nested_beyond_class_fve(
                group, feats, (orders == 2).astype(np.int64), n_folds=n_folds, seed=seed
            ),
        }
        for k in power_ks:
            targets[f"power_map_k{k}"] = nested_beyond_class_fve(
                group, feats, power_map(table, k), n_folds=n_folds, seed=seed
            )
        return targets

    nested = _nested_targets(features)

    headline_key = f"power_map_k{power_ks[0]}"
    headline = nested.get(headline_key)
    headline_gain: float | None
    power_map_beyond_class: bool | None
    if isinstance(headline, dict):
        headline_gain = headline["held_out_fve_gain_full_minus_restricted"]
        power_map_beyond_class = bool(headline_gain > tie_tol)
    else:
        headline_gain = None
        power_map_beyond_class = None

    record: dict[str, Any] = {
        "instrument": "power-map-attribution",
        "target_theory": "T10 (power-map / element-order axis; attributes a C1 within-pair "
        "difference specifically to the power-map axis, since it is the one axis a "
        "character-table-equivalent pair is free to differ on)",
        "status": "measured",
        "rung": 1,
        "group": group.canonical_name,
        "order": order,
        "n_conjugacy_classes": len(group.conjugacy_classes),
        "feature": "probes.embedding_features(model, order): mean-centred model.W_E rows "
        "for the order element tokens (the '=' read token dropped), unchanged from I-20/I-27/I-28",
        "power_ks": list(power_ks),
        "headline_target": headline_key,
        "tie_tol": float(tie_tol),
        "held_out_fve_gain_headline": headline_gain,
        "power_map_beyond_class": power_map_beyond_class,
        "direct_decode": direct_probes,
        "nested_beyond_class": nested,
        "note": (
            "direct_decode is rung-1 context (I-28 unchanged, plus a conjugacy_class target "
            "restricted to non-singleton classes): decodable does not mean beyond conjugacy "
            "class -- element_order and is_involution are provably class functions and can "
            "score well here for that reason alone. nested_beyond_class is the attribution: "
            "restricted uses only the conjugacy-class one-hot (the character-table-visible "
            "information), full adds the raw embedding; held_out_fve_gain_full_minus_restricted "
            "is the headline statistic, near zero on element_order/is_involution (the built-in "
            "sanity check) and materially positive on power_map_k<k> only if the embedding "
            "encodes the specific power-map image beyond what conjugacy class alone predicts."
        ),
        "caveat": (
            "Even after response compaction, the full design's extra d_model columns can "
            "exceed the per-fold training-row count for the smaller panel groups, putting "
            "both restricted and full in a saturated-regression regime; this affects both "
            "designs identically (same response, same row split), so the gain stays the "
            "load-bearing comparison, but a raw full_held_out_fve alone is not evidence of "
            "anything on its own. No basis search or regularisation sweep was run ('fit "
            "freely'), so the reported gain does not depend on a tuning choice."
        ),
    }
    if null_model is not None:
        null_features = embedding_features(null_model, order)
        record["null"] = {"nested_beyond_class": _nested_targets(null_features)}
    return record
