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

import pytest
import torch

from group_algorithm_interp.config import ProjectConfig
from group_algorithm_interp.groups.catalog import resolve_group
from group_algorithm_interp.instruments.interventions import (
    ablate_component,
    ablate_neurons,
    patch_neurons,
)
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
