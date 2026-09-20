"""The used-set causal-sufficiency audit (instruments/used_set_audit.py): the
completeness/minimality/faithfulness triad that turns "the model uses a
near-minimal faithful subset of isotypic blocks" into a rung-5 circuit claim.

Structure without a grokking run, following ``test_audit.py``'s recipe: a
small S3 model (3 isotypic blocks -- trivial rank 1, sign rank 1, the
degree-2 "standard" block rank 4) is trained on the full Cayley table with
its element embedding's rows PROJECTED, after every optimiser step, onto a
chosen union of isotypic blocks. That projection is what makes the
completeness assertions below exact rather than approximate: the embedding's
complement is bit-exact zero throughout training, regardless of how well
training converges, so "restrict to the used set" is a true no-op whenever
the used set matches (or covers) the planted subspace.

The minimality/non-faithful-subset assertions instead rely on a
training-independent combinatorial bound: when an ablation collapses two or
more distinct group elements to an identical embedding row, the model's
prediction for them is necessarily identical too (same forward computation),
while their true products differ (group cancellation) -- so accuracy on the
affected rows is bounded above by a number derivable from the collision
sizes alone, never an empirical-only observation. On S3, ablating the
degree-2 "standard" block from an embedding that lives entirely in it
collapses every element to one row, bounding accuracy at ``1/|G|``; ablating
it from an embedding spanning both nontrivial blocks collapses the two
cosets of the sign kernel (``A3`` and its complement, 3 elements each),
bounding accuracy at ``2/6 = 1/3``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from group_algorithm_interp.config import ProjectConfig
from group_algorithm_interp.groups.catalog import resolve_group
from group_algorithm_interp.instruments import used_set_audit as U
from group_algorithm_interp.instruments.coset import unleaked_heldout
from group_algorithm_interp.instruments.occupancy import cayley_grid_tokens, trivial_block_index
from group_algorithm_interp.instruments.template_divergence import (
    is_faithful,
    minimum_faithful_sets,
)
from group_algorithm_interp.seed import set_seed
from group_algorithm_interp.training.trainer import build_model

ORDER, INDEX = 6, 1  # S3: 3 isotypic blocks (trivial, sign, degree-2 standard)


def _config(order: int, index: int, arch: str = "transformer") -> ProjectConfig:
    return ProjectConfig(
        device="cpu",
        data={"group": {"order": order, "index": index}},
        model={"arch": arch, "d_model": 32, "d_mlp": 64, "n_heads": 4},
        logging={"mode": "disabled"},
    )


def _block_union_projector(group, block_indices) -> np.ndarray:
    projector = np.zeros((group.order, group.order), dtype=np.float64)
    for j in block_indices:
        projector += np.asarray(group.isotypic_blocks[j].projector, dtype=np.float64)
    return projector


def _project_embedding(model, projector: np.ndarray, order: int) -> None:
    with torch.no_grad():
        element = model.W_E[:order].detach().cpu().numpy().astype(np.float64)
        restricted = projector @ element
        model.W_E[:order] = torch.from_numpy(restricted).to(model.W_E.dtype)


def _train_with_restricted_embedding(block_indices, *, epochs: int = 400, seed: int = 0):
    """A model whose element embedding is confined to exactly ``block_indices``'
    isotypic subspace throughout training: projected gradient descent
    re-projects ``W_E``'s element rows onto that subspace after every
    optimiser step, so the complement is bit-exact zero at the end regardless
    of how well training converges. Trained on the full Cayley grid,
    deterministic and fast for S3 -- 400 epochs reliably reaches accuracy
    1.0 for both the single- and two-block subspaces used below."""
    group = resolve_group("S3")
    set_seed(seed, deterministic=False)
    model = build_model(_config(ORDER, INDEX), group)
    projector = _block_union_projector(group, block_indices)
    _project_embedding(model, projector, ORDER)
    tokens = cayley_grid_tokens(ORDER)
    targets = torch.as_tensor(group.cayley_table.reshape(-1), dtype=torch.long)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1.0)
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = F.cross_entropy(model(tokens)[:, -1, :], targets)
        loss.backward()
        optimiser.step()
        _project_embedding(model, projector, ORDER)
    return model, group


@pytest.fixture(scope="module")
def s3_group():
    return resolve_group("S3")


@pytest.fixture(scope="module")
def s3_nontrivial_blocks(s3_group):
    trivial = trivial_block_index(s3_group)
    return tuple(j for j in range(len(s3_group.isotypic_blocks)) if j != trivial)


@pytest.fixture(scope="module")
def s3_minimal_faithful_block(s3_group):
    """The cardinality-minimum faithful set on S3 is the singleton degree-2
    "standard" block (its kernel is just the identity, so it is faithful
    alone); the sign block alone is not (kernel = A3). Derived
    programmatically rather than hardcoded so the test does not assume S3's
    block ordering."""
    minimum = minimum_faithful_sets(s3_group, cost="cardinality")
    assert minimum["min_cardinality"] == 1
    (block,) = minimum["example_min_set"]
    assert is_faithful(s3_group, (block,))
    return block


@pytest.fixture(scope="module")
def s3_single_block_model(s3_minimal_faithful_block):
    """The model's ENTIRE element embedding lives in the minimal faithful
    block's subspace -- the sole nonzero block, so ablating it removes every
    element's information."""
    return _train_with_restricted_embedding((s3_minimal_faithful_block,))


@pytest.fixture(scope="module")
def s3_two_block_model(s3_nontrivial_blocks):
    """The model's element embedding is confined to BOTH nontrivial blocks
    (sign + standard on S3): a faithful used set that is not
    cardinality-minimal, since the standard block alone already suffices."""
    return _train_with_restricted_embedding(s3_nontrivial_blocks)


# ---------------------------------------------------------------------------
# complement_blocks_and_rank: pure rank/projector arithmetic, no model needed
# ---------------------------------------------------------------------------


def test_complement_of_empty_used_set_is_everything(s3_group):
    complement, projector, complement_rank, used_rank = U.complement_blocks_and_rank(s3_group, ())
    assert complement == tuple(range(len(s3_group.isotypic_blocks)))
    assert complement_rank == s3_group.order
    assert used_rank == 0
    np.testing.assert_allclose(projector, np.eye(s3_group.order), atol=1e-9)


def test_complement_of_full_used_set_is_empty(s3_group):
    all_blocks = tuple(range(len(s3_group.isotypic_blocks)))
    complement, projector, complement_rank, used_rank = U.complement_blocks_and_rank(
        s3_group, all_blocks
    )
    assert complement == ()
    assert complement_rank == 0
    assert used_rank == s3_group.order
    np.testing.assert_allclose(projector, 0.0, atol=1e-9)


def test_complement_and_used_ranks_sum_to_group_order(s3_group, s3_nontrivial_blocks):
    _, _, complement_rank, used_rank = U.complement_blocks_and_rank(s3_group, s3_nontrivial_blocks)
    assert complement_rank + used_rank == s3_group.order
    assert used_rank == 5  # sign (1) + standard (4)


def test_complement_blocks_and_rank_rejects_out_of_range_block(s3_group):
    with pytest.raises(ValueError, match="out of range"):
        U.complement_blocks_and_rank(s3_group, (999,))


# ---------------------------------------------------------------------------
# Completeness + faithfulness: restricting to the planted (faithful) used set
# is an exact no-op
# ---------------------------------------------------------------------------


def test_completeness_is_exact_when_used_set_matches_the_planted_subspace(
    s3_single_block_model, s3_minimal_faithful_block
):
    model, group = s3_single_block_model
    tokens, targets = unleaked_heldout(group, train_frac=0.8, split_seed=0)
    record = U.used_set_audit(
        model, group, tokens, targets, (s3_minimal_faithful_block,), n_random=4, seed=0
    )
    assert record["degenerate"] is False  # used rank 4/6 < 0.9
    zero = record["completeness"]["modes"]["zero"]
    assert zero["ablated_accuracy"] == pytest.approx(record["clean_accuracy"], abs=1e-9)
    assert record["completeness"]["accuracy_retention_fraction"]["zero"] == pytest.approx(
        1.0, abs=1e-9
    )
    # The restricted circuit is bit-identical to the model (nothing outside
    # the used set had any content to ablate), so the KL is exactly zero.
    faithfulness = record["faithfulness"]
    assert faithfulness["kl_model_to_restricted_circuit"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert faithfulness["argmax_agreement"]["mean"] == pytest.approx(1.0)


def test_completeness_holds_for_the_full_faithful_used_set_on_the_two_block_model(
    s3_two_block_model, s3_nontrivial_blocks
):
    model, group = s3_two_block_model
    tokens, targets = unleaked_heldout(group, train_frac=0.8, split_seed=0)
    record = U.used_set_audit(
        model, group, tokens, targets, s3_nontrivial_blocks, n_random=4, seed=0
    )
    assert record["degenerate"] is False  # used rank 5/6 < 0.9
    zero = record["completeness"]["modes"]["zero"]
    assert zero["ablated_accuracy"] == pytest.approx(record["clean_accuracy"], abs=1e-9)


# ---------------------------------------------------------------------------
# Minimality: ablating the sole used block collapses accuracy to <= 1/|G|,
# a training-independent bound (every element collides once it is ablated)
# ---------------------------------------------------------------------------


def test_minimality_ablating_the_sole_used_block_collapses_accuracy(
    s3_single_block_model, s3_minimal_faithful_block
):
    model, group = s3_single_block_model
    tokens = cayley_grid_tokens(group.order)
    targets = torch.as_tensor(group.cayley_table.reshape(-1), dtype=torch.long)
    record = U.used_set_audit(
        model, group, tokens, targets, (s3_minimal_faithful_block,), n_random=4, seed=0
    )
    assert record["clean_accuracy"] > 0.9  # the model actually learned the task
    (block,) = record["minimality"]["blocks"]
    zero = block["modes"]["zero"]
    # Ablating the used set's only block zeroes every element's embedding
    # row identically: for a fixed right argument b, at most one of |G|
    # left arguments can be predicted correctly (targets a*b are pairwise
    # distinct over a by cancellation).
    assert zero["ablated_accuracy"] <= 1.0 / group.order + 1e-9
    assert zero["flip_fraction_over_random"] > 0.0


# ---------------------------------------------------------------------------
# Faithful used set vs. a non-faithful proper subset of it: the headline
# contrast the instrument exists to report
# ---------------------------------------------------------------------------


def test_non_faithful_proper_subset_breaks_completeness_while_the_faithful_set_holds(
    s3_two_block_model, s3_nontrivial_blocks, s3_minimal_faithful_block
):
    """Same trained model (embedding confined to both nontrivial blocks:
    sign + standard). Restricting to the full (faithful) used set preserves
    accuracy exactly; restricting to just the sign block -- a proper subset
    that drops the faithful standard block -- collapses both cosets of the
    sign kernel (A3 and its complement, 3 elements each) into one embedding
    row apiece, bounding accuracy at 2/6 = 1/3: a mathematical consequence
    of the collision, not a training artefact."""
    model, group = s3_two_block_model
    tokens = cayley_grid_tokens(group.order)
    targets = torch.as_tensor(group.cayley_table.reshape(-1), dtype=torch.long)

    faithful = U.used_set_audit(
        model, group, tokens, targets, s3_nontrivial_blocks, n_random=4, seed=0
    )
    assert faithful["clean_accuracy"] > 0.9
    faithful_retained = faithful["completeness"]["modes"]["zero"]["ablated_accuracy"]
    assert faithful_retained == pytest.approx(faithful["clean_accuracy"], abs=1e-9)

    (non_faithful_block,) = [b for b in s3_nontrivial_blocks if b != s3_minimal_faithful_block]
    assert not is_faithful(group, (non_faithful_block,))
    non_faithful = U.used_set_audit(
        model, group, tokens, targets, (non_faithful_block,), n_random=4, seed=0
    )
    non_faithful_retained = non_faithful["completeness"]["modes"]["zero"]["ablated_accuracy"]
    assert non_faithful_retained <= 1.0 / 3.0 + 1e-9
    assert non_faithful_retained < faithful_retained - 0.3


# ---------------------------------------------------------------------------
# Degeneracy guard: a used set spanning every block leaves nothing to ablate
# ---------------------------------------------------------------------------


def test_degenerate_flag_when_used_set_spans_every_block(s3_two_block_model):
    model, group = s3_two_block_model
    all_blocks = tuple(range(len(group.isotypic_blocks)))
    tokens, targets = unleaked_heldout(group, train_frac=0.8, split_seed=0)
    record = U.used_set_audit(model, group, tokens, targets, all_blocks, n_random=4, seed=0)
    assert record["used_rank_fraction"] == pytest.approx(1.0)
    assert record["complement_rank"] == 0
    assert record["degenerate"] is True
    assert record["verdict"] == "degenerate_uninformative"
    # Nothing left to ablate: retention is trivially 1.0, and that triviality
    # is exactly what the flag exists to surface.
    assert record["completeness"]["accuracy_retention_fraction"]["zero"] == pytest.approx(
        1.0, abs=1e-9
    )


def test_not_degenerate_below_the_near_full_rank_threshold(
    s3_two_block_model, s3_nontrivial_blocks
):
    model, group = s3_two_block_model
    tokens, targets = unleaked_heldout(group, train_frac=0.8, split_seed=0)
    record = U.used_set_audit(
        model, group, tokens, targets, s3_nontrivial_blocks, n_random=4, seed=0
    )
    assert record["used_rank_fraction"] == pytest.approx(5.0 / 6.0)
    assert record["degenerate"] is False


def test_near_full_rank_fraction_is_overridable(s3_two_block_model, s3_nontrivial_blocks):
    """A caller who wants a stricter (or looser) degeneracy bar can override
    the default threshold; 5/6 (0.833) is below the module default (0.9) but
    above a threshold of 0.8."""
    model, group = s3_two_block_model
    tokens, targets = unleaked_heldout(group, train_frac=0.8, split_seed=0)
    record = U.used_set_audit(
        model,
        group,
        tokens,
        targets,
        s3_nontrivial_blocks,
        n_random=4,
        seed=0,
        near_full_rank_fraction=0.8,
    )
    assert record["degenerate"] is True
    assert record["verdict"] == "degenerate_uninformative"


# ---------------------------------------------------------------------------
# Empty used set: no blocks measured as used
# ---------------------------------------------------------------------------


def test_empty_used_set_reports_zero_rank_and_its_own_verdict(s3_two_block_model):
    model, group = s3_two_block_model
    tokens, targets = unleaked_heldout(group, train_frac=0.8, split_seed=0)
    record = U.used_set_audit(model, group, tokens, targets, (), n_random=4, seed=0)
    assert record["used_rank"] == 0
    assert record["used_blocks"] == []
    assert record["verdict"] == "empty_used_set"
    assert record["minimality"]["blocks"] == []


# ---------------------------------------------------------------------------
# Verdict field is always one of the documented descriptive labels
# ---------------------------------------------------------------------------


_VALID_VERDICTS = {
    "empty_used_set",
    "degenerate_uninformative",
    "sufficient_and_minimal_circuit",
    "sufficient_but_not_fully_minimal",
    "minimal_but_not_sufficient",
    "neither_sufficient_nor_minimal",
}


@pytest.mark.parametrize("used_blocks_kind", ["single_faithful", "two_block", "non_faithful"])
def test_verdict_is_always_one_of_the_documented_labels(
    used_blocks_kind,
    s3_single_block_model,
    s3_two_block_model,
    s3_nontrivial_blocks,
    s3_minimal_faithful_block,
):
    if used_blocks_kind == "single_faithful":
        model, group = s3_single_block_model
        used_blocks = (s3_minimal_faithful_block,)
    elif used_blocks_kind == "two_block":
        model, group = s3_two_block_model
        used_blocks = s3_nontrivial_blocks
    else:
        model, group = s3_two_block_model
        (non_faithful_block,) = [b for b in s3_nontrivial_blocks if b != s3_minimal_faithful_block]
        used_blocks = (non_faithful_block,)
    tokens, targets = unleaked_heldout(group, train_frac=0.8, split_seed=0)
    record = U.used_set_audit(model, group, tokens, targets, used_blocks, n_random=4, seed=0)
    assert record["verdict"] in _VALID_VERDICTS


# ---------------------------------------------------------------------------
# The individual building blocks are independently usable
# ---------------------------------------------------------------------------


def test_completeness_ablation_and_minimality_ablation_are_independently_callable(
    s3_single_block_model, s3_minimal_faithful_block
):
    model, group = s3_single_block_model
    tokens, targets = unleaked_heldout(group, train_frac=0.8, split_seed=0)
    completeness = U.completeness_ablation(
        model, group, tokens, targets, (s3_minimal_faithful_block,), n_random=4, seed=0
    )
    assert completeness["used_rank"] == 4
    assert completeness["complement_rank"] == 2
    minimality = U.minimality_ablation(
        model, group, tokens, targets, (s3_minimal_faithful_block,), n_random=4, seed=0
    )
    assert [b["block_index"] for b in minimality["blocks"]] == [s3_minimal_faithful_block]


def test_restricted_circuit_faithfulness_is_zero_kl_when_nothing_outside_has_content(
    s3_single_block_model, s3_minimal_faithful_block
):
    model, group = s3_single_block_model
    tokens, _ = unleaked_heldout(group, train_frac=0.8, split_seed=0)
    faithfulness = U.restricted_circuit_faithfulness(
        model, group, tokens, (s3_minimal_faithful_block,)
    )
    assert faithfulness["kl_model_to_restricted_circuit"]["mean"] == pytest.approx(0.0, abs=1e-9)
