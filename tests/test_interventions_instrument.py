"""The intervention harness (instruments/interventions.py): I-07.

Covers the correctness invariants the harness must hold: ablating an empty
neuron set is a no-op (the recompute path reproduces the model's own logits),
zeroing every neuron equals zeroing the MLP component, patching from an
identical source changes nothing, and the ``use_mlp=false`` guard fires. Both
architectures are exercised, since the harness serves both. The numbers here
are structural (no training needed): the harness measures a deterministic
function of the weights.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from group_algorithm_interp.config import ProjectConfig
from group_algorithm_interp.groups.catalog import resolve_group
from group_algorithm_interp.instruments.interventions import (
    _edit_neurons,
    ablate_component,
    ablate_direction,
    ablate_neurons,
    model_correct_mask,
    patch_neurons,
)
from group_algorithm_interp.instruments.nulls import chance_accuracy, random_neuron_control
from group_algorithm_interp.instruments.occupancy import cayley_grid_tokens
from group_algorithm_interp.seed import set_seed
from group_algorithm_interp.training.trainer import build_model

ORDER, INDEX = 8, 3  # D8, in the generated fixture corpus


def _model_and_batch(arch: str = "transformer", use_mlp: bool = True):
    group = resolve_group((ORDER, INDEX))
    config = ProjectConfig(
        device="cpu",
        data={"group": {"order": ORDER, "index": INDEX}},
        model={"arch": arch, "d_model": 16, "d_mlp": 32, "n_heads": 1, "use_mlp": use_mlp},
        logging={"mode": "disabled"},
    )
    set_seed(0, deterministic=False)
    model = build_model(config, group)
    tokens = cayley_grid_tokens(ORDER)
    targets = torch.tensor(group.cayley_table.reshape(-1), dtype=torch.long)
    return model, tokens, targets


@pytest.mark.parametrize("arch", ["transformer", "fc"])
def test_ablating_empty_set_reproduces_baseline(arch):
    # The recompute path must reproduce the model's own logits exactly, so an
    # empty ablation moves nothing -- for both architectures.
    model, tokens, targets = _model_and_batch(arch)
    for mode_block in ablate_neurons(model, tokens, targets, [])["target_set"].values():
        assert mode_block["accuracy_drop"] == pytest.approx(0.0, abs=1e-9)
        assert mode_block["n_label_flips"] == 0


def test_zeroing_all_neurons_equals_zeroing_the_mlp_component():
    model, tokens, targets = _model_and_batch("transformer")
    all_neurons = list(range(model.d_mlp))
    neuron_zero = ablate_neurons(model, tokens, targets, all_neurons)["target_set"]["zero"]
    component = ablate_component(model, tokens, targets, "mlp")
    assert neuron_zero["accuracy_after"] == pytest.approx(component["accuracy_after"], abs=1e-9)


def test_ablation_reports_random_control_of_matched_size():
    model, tokens, targets = _model_and_batch("transformer")
    result = ablate_neurons(model, tokens, targets, [0, 1, 2, 3, 4])
    assert result["n_ablated"] == 5
    assert len(result["random_control_set"]["neuron_indices"]) == 5
    for mode in ("zero", "mean", "resample"):
        assert result["target_set"][mode]["n_eval"] == result["n_eval"]


def test_patching_from_identical_source_is_a_noop():
    model, tokens, targets = _model_and_batch("transformer")
    result = patch_neurons(model, tokens, tokens, [0, 1, 2, 3], targets)
    assert result["accuracy_drop"] == pytest.approx(0.0, abs=1e-9)
    assert result["n_label_flips"] == 0


def test_patching_from_a_different_source_can_move_behaviour():
    model, tokens, targets = _model_and_batch("transformer")
    # A rolled batch is a genuinely different source for most neurons.
    source = torch.roll(tokens, shifts=1, dims=0)
    result = patch_neurons(model, tokens, source, list(range(model.d_mlp)), targets)
    assert result["n_patched"] == model.d_mlp
    assert 0 <= result["accuracy_after"] <= 1.0


def test_ablation_on_unleaked_subset_restricts_the_readout():
    model, tokens, targets = _model_and_batch("transformer")
    n = tokens.shape[0]
    mask = torch.zeros(n, dtype=torch.bool)
    mask[: n // 2] = True
    result = ablate_neurons(model, tokens, targets, [0, 1], unleaked_mask=mask)
    assert result["on_unleaked_subset"] is True
    assert result["n_eval"] == int(mask.sum())


def test_use_mlp_false_guard():
    model, tokens, targets = _model_and_batch("transformer", use_mlp=False)
    with pytest.raises(ValueError):
        ablate_neurons(model, tokens, targets, [0])


def test_component_ablation_rejects_fc():
    model, tokens, targets = _model_and_batch("fc")
    with pytest.raises(ValueError):
        ablate_component(model, tokens, targets, "mlp")


def test_random_neuron_control_matches_the_inline_draw_it_replaced():
    # The shared helper must reproduce, bit for bit, the draw ablate_neurons ran
    # inline before the I-03 battery was consolidated.
    d_mlp, n, seed = 32, 5, 7
    expected = torch.randperm(d_mlp, generator=torch.Generator().manual_seed(seed))[:n]
    got = random_neuron_control(d_mlp, n, seed=seed)
    assert torch.equal(got, expected)


def test_ablate_direction_reports_a_baseline_and_matched_random_controls():
    model, _, _ = _model_and_batch("transformer")
    group = resolve_group((ORDER, INDEX))
    direction = np.zeros(model.d_model)
    direction[0] = 1.0
    record = ablate_direction(model, group, direction, n_controls=3, seed=0)
    assert record["n_scored_pairs"] == group.order * group.order
    assert record["baseline_correct"] == int(model_correct_mask(model, group).sum())
    assert len(record["random_direction_drop_flips"]["per_control"]) == 3
    # a zero direction cannot ablate anything: no correct answers are lost
    assert record["direction_drop_flips"] == 0


def test_ablate_direction_is_deterministic_for_a_fixed_seed():
    model, _, _ = _model_and_batch("transformer")
    group = resolve_group((ORDER, INDEX))
    direction = np.arange(model.d_model, dtype=float) - model.d_model / 2
    first = ablate_direction(model, group, direction, n_controls=4, seed=3)
    second = ablate_direction(model, group, direction, n_controls=4, seed=3)
    assert first == second


def test_ablate_direction_boolean_subset_counts_true_entries_only():
    # A boolean subset only keeps its True entries under mask[subset]; the
    # reported n_scored_pairs must match that, not the full grid.
    model, _, _ = _model_and_batch("transformer")
    group = resolve_group((ORDER, INDEX))
    direction = np.zeros(model.d_model)
    direction[0] = 1.0
    n = group.order * group.order
    subset = np.zeros(n, dtype=bool)
    subset[: n // 3] = True
    record = ablate_direction(model, group, direction, subset=subset, n_controls=2, seed=0)
    assert record["n_scored_pairs"] == int(subset.sum())
    assert record["n_scored_pairs"] < n


# ---------------------------------------------------------------------------
# Finding 1: ablate_component's "attn" scope is direct-path only by default;
# full_removal recomputes the composed embed->attn->mlp->readout path too.
# ---------------------------------------------------------------------------


def test_component_ablation_attn_default_scope_is_direct_path():
    model, tokens, targets = _model_and_batch("transformer")
    result = ablate_component(model, tokens, targets, "attn")
    assert result["scope"] == "direct_path"


def test_component_ablation_mlp_scope_is_full():
    model, tokens, targets = _model_and_batch("transformer")
    result = ablate_component(model, tokens, targets, "mlp")
    assert result["scope"] == "full"


def test_component_ablation_full_removal_matches_w_o_zeroed_forward():
    # full_removal=True on "attn" must be equivalent to genuinely zeroing W_O
    # and rerunning the forward pass -- not just zeroing attn_out's additive
    # term while mlp_post still carries attention's effect.
    model, tokens, targets = _model_and_batch("transformer")
    result = ablate_component(model, tokens, targets, "attn", full_removal=True)

    zeroed = copy.deepcopy(model)
    with torch.no_grad():
        zeroed.W_O.zero_()
    zeroed.eval()
    with torch.no_grad():
        logits = zeroed(tokens)
    preds = logits[:, -1, :].argmax(dim=-1)
    accuracy = (preds == targets).float().mean().item()
    assert result["accuracy_after"] == pytest.approx(accuracy, abs=1e-9)


# ---------------------------------------------------------------------------
# Finding 3: the transformer-path gate must not misreject a use_mlp=False
# transformer, and an "mlp" request on such a model must raise, not report a
# fabricated zero-effect "measured" record.
# ---------------------------------------------------------------------------


def test_component_ablation_attn_is_defined_without_an_mlp():
    model, tokens, targets = _model_and_batch("transformer", use_mlp=False)
    result = ablate_component(model, tokens, targets, "attn")
    # No MLP means no composed path to miss: direct-path removal is complete.
    assert result["scope"] == "full"


def test_component_ablation_mlp_on_no_mlp_model_raises():
    model, tokens, targets = _model_and_batch("transformer", use_mlp=False)
    with pytest.raises(ValueError):
        ablate_component(model, tokens, targets, "mlp")


# ---------------------------------------------------------------------------
# Finding 5: ablate_neurons must reject out-of-bounds or duplicate indices
# rather than silently accepting them (which would desync the matched-size
# random control from what was actually ablated).
# ---------------------------------------------------------------------------


def test_ablate_neurons_rejects_out_of_bounds_indices():
    model, tokens, targets = _model_and_batch("transformer")
    with pytest.raises(ValueError):
        ablate_neurons(model, tokens, targets, [model.d_mlp])


def test_ablate_neurons_rejects_duplicate_indices():
    model, tokens, targets = _model_and_batch("transformer")
    with pytest.raises(ValueError):
        ablate_neurons(model, tokens, targets, [0, 0, 1])


# ---------------------------------------------------------------------------
# Finding 5 / nulls.py: chance_accuracy and random_neuron_control must reject
# malformed input rather than silently misbehaving.
# ---------------------------------------------------------------------------


def test_chance_accuracy_rejects_bool():
    with pytest.raises(TypeError):
        chance_accuracy(True)


def test_chance_accuracy_rejects_non_integral_float():
    with pytest.raises(TypeError):
        chance_accuracy(2.5)


def test_random_neuron_control_rejects_n_greater_than_d_mlp():
    with pytest.raises(ValueError):
        random_neuron_control(10, 11, seed=0)


def test_random_neuron_control_rejects_negative_n():
    with pytest.raises(ValueError):
        random_neuron_control(10, -1, seed=0)


def test_random_neuron_control_does_not_perturb_the_global_torch_stream():
    # The docstring guarantees a private generator; pin it: draw the same
    # global-stream sequence before and after a call, with the global seed
    # reset only in between, and confirm the draw is identical.
    torch.manual_seed(123)
    before = torch.rand(5)
    torch.manual_seed(123)
    random_neuron_control(32, 5, seed=99)
    after = torch.rand(5)
    assert torch.equal(before, after)


# ---------------------------------------------------------------------------
# Finding 6: semantic checks on mean and resample ablation.
# ---------------------------------------------------------------------------


def test_mean_ablation_replaces_with_the_per_position_batch_mean():
    # post: [batch=3, pos=2, neurons=2]. Neuron 0 is ablated; neuron 1 is a
    # control that must be untouched.
    post = torch.tensor(
        [
            [[1.0, 10.0], [4.0, 40.0]],
            [[2.0, 20.0], [5.0, 50.0]],
            [[3.0, 30.0], [6.0, 60.0]],
        ]
    )
    generator = torch.Generator().manual_seed(0)
    edited = _edit_neurons(post, torch.tensor([0]), "mean", generator)
    # Neuron 0's batch mean at position 0 is (1+2+3)/3 = 2, at position 1 is
    # (4+5+6)/3 = 5 -- every batch entry is replaced with its own position's
    # mean, not a single scalar collapsing both positions together.
    assert torch.allclose(edited[:, 0, 0], torch.full((3,), 2.0))
    assert torch.allclose(edited[:, 1, 0], torch.full((3,), 5.0))
    assert torch.equal(edited[:, :, 1], post[:, :, 1])


def test_resample_ablation_is_deterministic_for_a_fixed_seed():
    post = torch.arange(24.0).reshape(4, 2, 3)
    idx = torch.tensor([0, 2])
    edited_first = _edit_neurons(post, idx, "resample", torch.Generator().manual_seed(5))
    edited_second = _edit_neurons(post, idx, "resample", torch.Generator().manual_seed(5))
    assert torch.equal(edited_first, edited_second)


def test_resample_ablation_preserves_the_marginal_multiset_per_position():
    post = torch.arange(24.0).reshape(4, 2, 3)
    idx = torch.tensor([0, 2])
    generator = torch.Generator().manual_seed(1)
    edited = _edit_neurons(post, idx, "resample", generator)
    for pos in range(post.shape[1]):
        for neuron in idx.tolist():
            expected = torch.sort(post[:, pos, neuron]).values
            got = torch.sort(edited[:, pos, neuron]).values
            assert torch.equal(expected, got)
