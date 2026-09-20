"""The coset-quotient route decider (instruments/coset_quotient.py, I-20).

The decider that resolves the coset question where I-18 is undecidable: it works
in the rank-``[G:N]`` quotient subspace of a NORMAL subgroup (rank 2 for an
index-2 quotient) and reads an output-*coset* target, so it never ablates a
near-full-rank subspace. Tested on the fixture analogues: D8 (8,3) has an
index-2 normal rotation subgroup (a rank-2 parity quotient, the exact analogue
of D32's decisive quotient), and C7 (7,1) -- prime order -- has no proper normal
subgroup, so the arm is ``defined: False``. Structure is checked without
training: the projector algebra and the edit semantics are exact, and the
mandatory rule-1 regression (an untrained model shows no coset organisation) is
checked directly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from group_algorithm_interp.config import ProjectConfig
from group_algorithm_interp.groups.catalog import resolve_group
from group_algorithm_interp.instruments.coset_quotient import (
    DISCRIMINATION_NOTE,
    coset_quotient_probe,
    coset_quotient_route,
    normal_quotients,
)
from group_algorithm_interp.instruments.occupancy import cayley_grid_tokens
from group_algorithm_interp.seed import set_seed
from group_algorithm_interp.training.trainer import build_model


def _config(order: int, index: int) -> ProjectConfig:
    return ProjectConfig(
        device="cpu",
        data={"group": {"order": order, "index": index}, "train_frac": 0.8},
        model={"arch": "transformer", "d_model": 32, "d_mlp": 64, "n_heads": 4},
        logging={"mode": "disabled"},
    )


def _random_model(order: int, index: int, seed: int = 0):
    group = resolve_group((order, index))
    set_seed(seed, deterministic=False)
    return build_model(_config(order, index), group), group


def _grid(group):
    tokens = cayley_grid_tokens(group.order)
    targets = torch.from_numpy(group.cayley_table.reshape(-1).astype(np.int64))
    return tokens, targets


def _full_accuracy(model, tokens, targets) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(tokens)[:, -1, :].argmax(dim=-1)
    return float((pred == targets).float().mean())


# ---------------------------------------------------------------------------
# Normal-quotient enumeration and the coset-mean projector
# ---------------------------------------------------------------------------


def test_normal_quotients_on_d8_include_the_rank_2_index_2_quotient():
    group = resolve_group((8, 3))
    quotients = normal_quotients(group)
    assert quotients, "D8 must expose proper nontrivial normal quotients"
    # ordered coarsest (smallest index) first
    indices = [q.coset_index for q in quotients]
    assert indices == sorted(indices)
    # the index-2 rotation quotient: rank == index == 2, a proper partition
    index_2 = [q for q in quotients if q.coset_index == 2]
    assert index_2, "D8 has an index-2 (order-4) normal rotation subgroup"
    q2 = index_2[0]
    assert q2.subgroup_order == 4
    assert q2.quotient_projector.shape == (8, 8)
    assert set(q2.element_to_coset.tolist()) == {0, 1}
    assert q2.element_to_coset.min() >= 0  # every element is labelled


def test_every_returned_subgroup_is_normal_and_none_are_core_free_reflections():
    # D8's core-free reflection subgroups (order 2, index 4, non-normal) are the
    # I-18 targets; this arm must exclude them and keep only normal subgroups.
    group = resolve_group((8, 3))
    table = group.cayley_table
    idx = np.arange(group.order)
    e = int(np.flatnonzero(np.all(table == idx[None, :], axis=1))[0])
    inv = {g: int(np.flatnonzero(table[g] == e)[0]) for g in range(group.order)}
    for q in normal_quotients(group):
        members = {int(x) for x in group.subgroups[q.subgroup_index].tolist()}
        for x in range(group.order):
            conjugate = {int(table[table[x, h], inv[x]]) for h in members}
            assert conjugate == members, "a returned subgroup is not normal"


def test_quotient_projector_is_a_coset_constant_orthogonal_projector():
    group = resolve_group((8, 3))
    for q in normal_quotients(group):
        p = q.quotient_projector
        assert np.allclose(p, p.T)  # symmetric
        assert np.allclose(p @ p, p)  # idempotent
        assert abs(np.trace(p) - q.coset_index) < 1e-9  # rank == [G:N]
        # projecting any function gives one constant per coset
        v = np.arange(group.order, dtype=np.float64)
        projected = p @ v
        for coset in q.cosets:
            members = np.asarray(coset)
            assert np.allclose(projected[members], projected[members][0])


# ---------------------------------------------------------------------------
# The arm: schema, the defined/undefined gate, and edit semantics
# ---------------------------------------------------------------------------


def test_route_schema_and_decisive_quotient_on_d8():
    model, group = _random_model(8, 3)
    tokens, targets = _grid(group)
    record = coset_quotient_route(model, group, tokens, targets, n_random=6, seed=0)
    assert record["defined"] is True
    assert record["instrument"] == "coset-quotient-route"
    assert record["discrimination_note"] == DISCRIMINATION_NOTE
    assert record["n_test"] == group.order * group.order
    subgroup_indices = {q["subgroup_index"] for q in record["quotients"]}
    assert record["decisive_subgroup_index"] in subgroup_indices
    for probe in record["quotients"]:
        assert abs(probe["chance_coset_accuracy"] - 1.0 / probe["coset_index"]) < 1e-12
        suff = probe["sufficiency"]
        nec = probe["necessity"]
        assert 0.0 <= suff["restricted_coset_accuracy"] <= 1.0
        assert suff["random_subspace_coset_accuracy"]["n"] == 6
        assert "coset_accuracy_over_random" in suff
        assert "coset_accuracy_drop_over_random" in nec


def test_route_is_undefined_without_a_proper_normal_subgroup():
    # C7 is of prime order: its only subgroups are trivial and the whole group,
    # so there is no quotient Q = G/N for a coset route to combine cosets in.
    model, group = _random_model(7, 1)
    tokens, targets = _grid(group)
    record = coset_quotient_route(model, group, tokens, targets, n_random=4, seed=0)
    assert record["defined"] is False
    assert "no proper nontrivial normal subgroup" in record["reason"]
    assert record["discrimination_note"] == DISCRIMINATION_NOTE
    assert normal_quotients(group) == ()


def test_restriction_is_a_no_op_and_ablation_removes_all_when_embedding_is_coset_collapsed():
    # Plant a coset-collapsed embedding (each element row == its coset mean).
    # Then restricting to the quotient subspace is the identity (P is idempotent
    # and P E == E), so the restricted behaviour equals the model's own; ablating
    # it (E - P E) zeroes the element embedding entirely.
    model, group = _random_model(8, 3)
    q = next(iter(q for q in normal_quotients(group) if q.coset_index == 2))
    element = model.W_E.detach().clone()
    projected = torch.from_numpy(q.quotient_projector).to(element.dtype) @ element[: group.order]
    element[: group.order] = projected
    model.W_E = torch.nn.Parameter(element, requires_grad=False)

    tokens, targets = _grid(group)
    collapsed_accuracy = _full_accuracy(model, tokens, targets)
    probe = coset_quotient_probe(model, group, q, tokens, targets, n_random=4, seed=0)

    # restriction is a no-op on an already-collapsed embedding
    assert probe["sufficiency"]["restricted_full_accuracy"] == pytest.approx(
        collapsed_accuracy, abs=0.05
    )
    # ablation zeroes the element embedding -> at most a single constant answer
    assert probe["necessity"]["ablated_full_accuracy"] <= collapsed_accuracy + 1e-9


def test_untrained_model_shows_no_coset_organisation():
    # Rule-1 regression: an untrained model has no coset route, so restricting to
    # the quotient subspace should not recover the coset far above a matched
    # random subspace on any quotient.
    model, group = _random_model(8, 3, seed=1)
    tokens, targets = _grid(group)
    record = coset_quotient_route(model, group, tokens, targets, n_random=8, seed=0)
    for probe in record["quotients"]:
        assert probe["sufficiency"]["coset_accuracy_over_random"] < 0.25
