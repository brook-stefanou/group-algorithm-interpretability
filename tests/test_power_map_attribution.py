"""The power-map attribution instrument (instruments/power_map_attribution.py):
does the model's embedding encode power-map structure beyond conjugacy-class
membership.

Covers, in order:

* the pure group-theory building blocks (``power_map`` agreeing with
  ``probes.square_map``/``cube_map``, ``class_design``, and the
  conjugation-commutes-with-powering fact the whole module leans on, checked
  directly against a real Cayley table);
* the abelian edge case (every class a singleton -- ``conjugacy_class``
  direct decode must come back ``UNDEFINED`` rather than crash);
* two synthetic controls for ``nested_beyond_class_fve`` at a scale (order
  60, 5 classes) large enough for a stable signal -- mirroring
  ``test_readout_characterisation.py``'s "ties" / "wins" pair -- checking the
  nested design finds no benefit when the feature carries only class
  information and a clear benefit when it also carries the exact
  beyond-class target;
* response compaction (a non-injective power map's response width is its own
  observed cardinality, not a fixed ``order``);
* end-to-end shape and the null-model convention on a real (untrained)
  fixture model.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from group_algorithm_interp.config import ProjectConfig
from group_algorithm_interp.groups.catalog import resolve_group
from group_algorithm_interp.instruments import power_map_attribution as PM
from group_algorithm_interp.instruments import probes as P
from group_algorithm_interp.seed import set_seed
from group_algorithm_interp.training.trainer import build_model

D_MODEL = 32
D_MLP = 64


def _config(order: int, index: int) -> ProjectConfig:
    return ProjectConfig(
        device="cpu",
        data={"group": {"order": order, "index": index}},
        model={"arch": "transformer", "d_model": D_MODEL, "d_mlp": D_MLP, "n_heads": 4},
        logging={"mode": "disabled"},
    )


def _random_model(name: str, seed: int = 0):
    group = resolve_group(name)
    set_seed(seed, deterministic=False)
    return build_model(_config(group.order, group.index), group), group


class _FakeModel:
    """A model stub carrying only a hand-set ``W_E`` -- everything this
    module reads from a model (``probes.embedding_features``) only touches
    ``.W_E``, so no forward pass is needed."""

    def __init__(self, w_e: np.ndarray):
        self.W_E = torch.tensor(w_e, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Group-theory building blocks.
# ---------------------------------------------------------------------------


def test_power_map_agrees_with_probes_square_and_cube_map():
    group = resolve_group("Q8")
    table = group.cayley_table
    assert np.array_equal(PM.power_map(table, 2), P.square_map(table))
    assert np.array_equal(PM.power_map(table, 3), P.cube_map(table))


def test_power_map_k1_is_the_identity_function():
    group = resolve_group("S3")
    table = group.cayley_table
    assert np.array_equal(PM.power_map(table, 1), np.arange(group.order))


def test_power_map_rejects_negative_k():
    group = resolve_group("S3")
    with pytest.raises(ValueError):
        PM.power_map(group.cayley_table, -1)


def test_class_of_power_map_is_determined_by_class_of_g():
    """Conjugation commutes with powering: class(g^k) is a fixed function of
    class(g) alone, for every g in the same class -- the module docstring's
    central claim, checked directly against the Cayley table on a
    non-abelian fixture (S3)."""
    group = resolve_group("S3")
    table = group.cayley_table
    class_labels = PM.conjugacy_class_labels(group)
    for k in (2, 3):
        images = PM.power_map(table, k)
        image_classes = class_labels[images]
        for members in group.conjugacy_classes:
            if len(members) < 2:
                continue
            assert len(set(image_classes[members].tolist())) == 1, (
                f"class(g^{k}) varies within a single conjugacy class -- "
                "violates the conjugation-commutes-with-powering fact"
            )


def test_class_design_is_one_hot_per_element():
    group = resolve_group("S3")
    design = PM.class_design(group)
    assert design.shape == (group.order, len(group.conjugacy_classes))
    assert np.allclose(design.sum(axis=1), 1.0)
    labels = PM.conjugacy_class_labels(group)
    assert np.array_equal(design.argmax(axis=1), labels)


# ---------------------------------------------------------------------------
# The abelian edge case: every class is a singleton.
# ---------------------------------------------------------------------------


def test_conjugacy_class_probe_target_empty_on_abelian_group():
    group = resolve_group("C8")
    assert all(len(members) == 1 for members in group.conjugacy_classes)
    mask, labels = PM.conjugacy_class_probe_target(group)
    assert mask.sum() == 0
    assert labels.shape == (0,)


def test_instrument_handles_abelian_group_without_crashing():
    model, group = _random_model("C8", seed=1)
    record = PM.power_map_attribution_instrument(model, group, seed=1)
    assert record["status"] == "measured"
    assert record["direct_decode"]["conjugacy_class"] == PM.UNDEFINED
    for target, result in record["nested_beyond_class"].items():
        assert result == PM.UNDEFINED or "held_out_fve_gain_full_minus_restricted" in result, target


# ---------------------------------------------------------------------------
# Synthetic controls for the nested test itself, at a scale (order 60, 5
# classes of 12) large enough for a stable signal -- mirrors
# test_readout_characterisation.py's ties/wins pair.
# ---------------------------------------------------------------------------


def _synthetic_class_group(n_classes: int, class_size: int):
    order = n_classes * class_size
    classes = tuple(np.arange(i * class_size, (i + 1) * class_size) for i in range(n_classes))
    return SimpleNamespace(order=order, conjugacy_classes=classes)


def test_nested_fve_ties_when_features_carry_only_class_information():
    """A feature that is (a noisy copy of) the class one-hot itself, plus a
    few unrelated noise dimensions, buys nothing over the class-only
    restricted design when the response varies WITHIN class (position within
    class, uncorrelated with the noise) -- the nested test's honest null."""
    n_classes, class_size = 5, 12
    group = _synthetic_class_group(n_classes, class_size)
    class_design = PM.class_design(group)
    response = np.concatenate([np.arange(class_size) for _ in range(n_classes)])

    rng = np.random.default_rng(0)
    features = class_design + rng.normal(scale=0.01, size=class_design.shape)
    features = np.concatenate([features, rng.normal(scale=0.01, size=(group.order, 10))], axis=1)
    result = PM.nested_beyond_class_fve(group, features, response, seed=0)
    assert result != PM.UNDEFINED
    assert result["held_out_fve_gain_full_minus_restricted"] <= 0.01


def test_nested_fve_full_wins_when_features_carry_the_beyond_class_target():
    """A feature that additionally carries a one-hot of the exact
    within-class response lets ``full`` reconstruct it far better than
    ``restricted`` (which only ever sees the class, identical for every
    element of a class) can -- the nested test's positive result."""
    n_classes, class_size = 5, 12
    group = _synthetic_class_group(n_classes, class_size)
    class_design = PM.class_design(group)
    response = np.concatenate([np.arange(class_size) for _ in range(n_classes)])

    response_one_hot = np.zeros((group.order, class_size))
    response_one_hot[np.arange(group.order), response] = 1.0
    rng = np.random.default_rng(0)
    features = np.concatenate([class_design, response_one_hot], axis=1) + rng.normal(
        scale=0.01, size=(group.order, n_classes + class_size)
    )
    result = PM.nested_beyond_class_fve(group, features, response, seed=0)
    assert result != PM.UNDEFINED
    assert result["held_out_fve_gain_full_minus_restricted"] > 0.5
    assert result["full_held_out_fve"] > result["restricted_held_out_fve"]


def test_nested_fve_undefined_for_a_constant_response():
    group = _synthetic_class_group(3, 4)
    features = np.zeros((group.order, 5))
    constant = np.zeros(group.order, dtype=np.int64)
    assert PM.nested_beyond_class_fve(group, features, constant) == PM.UNDEFINED


def test_nested_fve_undefined_when_fewer_than_two_folds_fit():
    """A non-degenerate response with n_folds forced to 1 cannot be
    cross-validated (a single fold has no held-out rows) -- UNDEFINED, not a
    crash."""
    group = _synthetic_class_group(3, 4)
    features = np.zeros((group.order, 5))
    varying = np.arange(group.order) % 2
    assert PM.nested_beyond_class_fve(group, features, varying, n_folds=1) == PM.UNDEFINED


def test_headline_gain_and_verdict_are_none_when_the_headline_target_is_undefined():
    """k=0 is the constant identity power map (every element maps to e) --
    degenerate, so the headline nested test is UNDEFINED and the top-level
    gain/verdict fields must degrade to None rather than crash on a missing
    dict key."""
    model, group = _random_model("S3", seed=6)
    record = PM.power_map_attribution_instrument(model, group, power_ks=(0,), seed=6)
    assert record["nested_beyond_class"]["power_map_k0"] == PM.UNDEFINED
    assert record["held_out_fve_gain_headline"] is None
    assert record["power_map_beyond_class"] is None


# ---------------------------------------------------------------------------
# Response compaction: a non-injective power map's response width is its own
# observed cardinality, not the group order.
# ---------------------------------------------------------------------------


def test_response_width_is_compacted_to_observed_cardinality():
    group = resolve_group("S3")
    table = group.cayley_table
    square = PM.power_map(table, 2)
    assert len(set(square.tolist())) < group.order  # squaring on S3 is not injective
    features = np.zeros((group.order, 3))
    result = PM.nested_beyond_class_fve(group, features, square, seed=0)
    assert result != PM.UNDEFINED
    assert result["response_width"] == len(set(square.tolist()))
    assert result["response_width_uncompacted"] == group.order


# ---------------------------------------------------------------------------
# End-to-end shape, headline default, and the null-model convention.
# ---------------------------------------------------------------------------


def test_instrument_runs_end_to_end_on_s3_and_reports_expected_shape():
    model, group = _random_model("S3", seed=2)
    record = PM.power_map_attribution_instrument(model, group, seed=2)
    assert record["status"] == "measured"
    assert record["order"] == group.order
    assert set(record["direct_decode"]) >= {
        "element_order",
        "is_involution",
        "square_map",
        "cube_map",
        "conjugacy_class",
    }
    # k=2 only by default (module docstring: cubing is bijective on most of
    # the trained panel and is dropped from the default for that reason).
    assert record["power_ks"] == [2]
    assert set(record["nested_beyond_class"]) == {"element_order", "is_involution", "power_map_k2"}
    assert record["headline_target"] == "power_map_k2"
    assert record["power_map_beyond_class"] in (True, False)


def test_power_ks_can_be_overridden_to_include_cubing():
    model, group = _random_model("S3", seed=3)
    record = PM.power_map_attribution_instrument(model, group, power_ks=(2, 3), seed=3)
    assert set(record["nested_beyond_class"]) == {
        "element_order",
        "is_involution",
        "power_map_k2",
        "power_map_k3",
    }


def test_element_order_and_involution_are_near_zero_class_function_sanity_check():
    """Both targets are provably class functions (module docstring): a large
    nested increment on either would indicate a bug in this instrument, not
    a finding about the model."""
    model, group = _random_model("D8", seed=4)
    record = PM.power_map_attribution_instrument(model, group, seed=4)
    for target in ("element_order", "is_involution"):
        result = record["nested_beyond_class"][target]
        if result == PM.UNDEFINED:
            continue
        assert abs(result["held_out_fve_gain_full_minus_restricted"]) < 0.6, target


def test_null_model_reported_when_supplied():
    model, group = _random_model("S3", seed=5)
    set_seed(99, deterministic=False)
    null_model, _ = _random_model("S3", seed=99)
    record = PM.power_map_attribution_instrument(model, group, seed=5, null_model=null_model)
    assert "null" in record
    null_headline = record["null"]["nested_beyond_class"]["power_map_k2"]
    assert (
        null_headline == PM.UNDEFINED or "held_out_fve_gain_full_minus_restricted" in null_headline
    )


def test_direct_decode_conjugacy_class_matches_crossval_probe_on_nonsingleton_classes():
    """Sanity: the direct-decode conjugacy_class target really is
    ``crossval_probe`` restricted to non-singleton classes, not a
    reimplementation that could quietly diverge."""
    _, group = _random_model("S3", seed=0)
    raw = np.eye(group.order)  # perfectly separable stand-in features
    fake = _FakeModel(np.concatenate([raw, np.zeros((1, group.order))], axis=0))
    features = P.embedding_features(fake, group.order)
    mask, labels = PM.conjugacy_class_probe_target(group)
    expected = P.crossval_probe(features[mask], labels, seed=0)
    record = PM.power_map_attribution_instrument(fake, group, seed=0)
    assert record["direct_decode"]["conjugacy_class"] == expected
