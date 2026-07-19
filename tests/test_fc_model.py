"""The FC baseline (architecture confound) -- all FCModel tests live here.

The FC must satisfy the same contract the transformer does so the trainer, eval,
and every analysis instrument work unchanged: logits [batch, n_ctx, d_vocab_out]
read at position -1, a shared W_E, and a cache whose resid_final[:, -1] is the
representation the coset probe reads.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from group_algorithm_interp.groups.catalog import resolve_group
from group_algorithm_interp.groups.group import FiniteGroup
from group_algorithm_interp.model import FCModel, OneLayerTransformer
from group_algorithm_interp.task import build_group_task
from group_algorithm_interp.training.config import (
    DataConfig,
    ExperimentConfig,
    GeneralizationConfig,
    ModelConfig,
    OptimConfig,
)
from group_algorithm_interp.training.trainer import build_model


def _config(arch: str, group: str = "C4") -> GeneralizationConfig:
    return GeneralizationConfig(
        experiment=ExperimentConfig(name="test", seed=0),
        data=DataConfig(group=group),
        model=ModelConfig(arch=arch, d_model=32, d_mlp=64),
        optim=OptimConfig(epochs=10),
    )


def _tokens(group: FiniteGroup) -> tuple[torch.Tensor, torch.Tensor]:
    """All-pairs (a, b, '=') tokens and their a*b targets, via the task builder."""
    task = build_group_task(group)
    eq = np.full((task.inputs.shape[0], 1), group.order)
    tokens = torch.tensor(np.hstack([task.inputs, eq]), dtype=torch.long)
    return tokens, torch.tensor(task.targets, dtype=torch.long)


def _fc_model(d_vocab_in=10, d_vocab_out=10, d_model=16, d_mlp=32, activation="relu"):
    return FCModel(
        d_vocab_in=d_vocab_in,
        d_vocab_out=d_vocab_out,
        n_ctx=3,
        d_model=d_model,
        d_mlp=d_mlp,
        activation=activation,
    )


def test_build_model_selects_arch() -> None:
    g = resolve_group("C4")
    assert isinstance(build_model(_config("fc"), g), FCModel)
    assert isinstance(build_model(_config("transformer"), g), OneLayerTransformer)


def test_build_model_uses_the_derived_d_mlp_when_unset() -> None:
    g = resolve_group("C4")
    for arch in ("fc", "transformer"):
        cfg = GeneralizationConfig(
            experiment=ExperimentConfig(name="test", seed=0),
            data=DataConfig(group="C4"),
            model=ModelConfig(arch=arch, d_model=32),
            optim=OptimConfig(epochs=10),
        )
        assert cfg.model.d_mlp == 64
        model = build_model(cfg, g)
        assert model.d_mlp == 64


def test_fc_forward_shape_and_broadcast_read_position() -> None:
    g = resolve_group("S3")
    n = g.order
    m = FCModel(d_vocab_in=n + 1, d_vocab_out=n, n_ctx=3, d_model=16, d_mlp=32, activation="relu")
    tokens, _ = _tokens(g)
    logits = m(tokens)
    assert logits.shape == (n * n, 3, n)
    # The prediction is broadcast across every position, so every position
    # equals the read position -- the contract the trainer/eval/functional-form
    # rely on.
    for pos in range(1, 3):
        assert torch.equal(logits[:, 0, :], logits[:, pos, :])


def test_fc_cache_exposes_read_representation() -> None:
    g = resolve_group("S3")
    n, d_mlp = g.order, 32
    m = FCModel(
        d_vocab_in=n + 1, d_vocab_out=n, n_ctx=3, d_model=16, d_mlp=d_mlp, activation="relu"
    )
    tokens, _ = _tokens(g)
    cache = m(tokens, return_cache=True)
    assert cache["resid_final"].shape == (n * n, 3, d_mlp)
    assert cache["logits"].shape == (n * n, 3, n)
    assert cache["embed"].shape == (n * n, 3, 16)
    # The coset probe reads resid_final[:, -1]: the post-ReLU hidden, so non-negative.
    assert (cache["resid_final"][:, -1, :] >= 0).all()


def test_fc_logits_at_read_position_depend_on_both_operands() -> None:
    # A model that structurally ignores operand b cannot represent the group
    # operation and sits at exactly chance -- a decreasing loss alone doesn't
    # catch that, so this checks the dependency directly, independent of
    # training.
    g = resolve_group("C4")
    n = g.order
    torch.manual_seed(0)
    m = FCModel(d_vocab_in=n + 1, d_vocab_out=n, n_ctx=3, d_model=16, d_mlp=32, activation="relu")
    tokens, _ = _tokens(g)
    base_logits = m(tokens)[:, -1, :]
    perturbed = tokens.clone()
    perturbed[:, 1] = (perturbed[:, 1] + 1) % n  # perturb only operand b
    new_logits = m(perturbed)[:, -1, :]
    assert not torch.allclose(base_logits, new_logits)


def test_fc_learns_a_batch() -> None:
    g = resolve_group("C4")
    n = g.order
    torch.manual_seed(0)
    m = FCModel(d_vocab_in=n + 1, d_vocab_out=n, n_ctx=3, d_model=16, d_mlp=64, activation="relu")
    tokens, targets = _tokens(g)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = F.cross_entropy(m(tokens)[:, -1, :], targets)
        loss.backward()
        opt.step()
    preds = m(tokens)[:, -1, :].argmax(dim=-1)
    accuracy = (preds == targets).float().mean().item()
    # C4 is 16 pairs -- trivially memorisable by a model that can represent the
    # group operation. A model that structurally ignores operand b sits at
    # exactly chance (0.25) here; only a full 1.0 rules that out.
    assert accuracy == 1.0


# ---------------------------------------------------------------------------
# Cache internals
# ---------------------------------------------------------------------------


def test_fc_cache_has_all_keys():
    m = _fc_model()
    tokens = torch.randint(0, 10, (8, 3))
    cache = m(tokens, return_cache=True)
    expected = {"embed", "attn_pattern", "attn_out", "mlp_pre", "mlp_post", "resid_final", "logits"}
    assert set(cache.keys()) == expected


def test_fc_cache_shapes():
    m = _fc_model(d_model=16, d_mlp=32)  # d_vocab_out=10, n_ctx=3
    tokens = torch.randint(0, 10, (8, 3))
    cache = m(tokens, return_cache=True)
    assert cache["embed"].shape == (8, 3, 16)
    assert cache["attn_pattern"].shape == (8, 1, 3, 3)
    assert cache["attn_out"].shape == (8, 3, 16)
    assert cache["mlp_pre"].shape == (8, 3, 32)
    assert cache["mlp_post"].shape == (8, 3, 32)
    assert cache["resid_final"].shape == (8, 3, 32)
    assert cache["logits"].shape == (8, 3, 10)


def test_fc_cache_zero_attention():
    m = _fc_model()
    tokens = torch.randint(0, 10, (8, 3))
    cache = m(tokens, return_cache=True)
    assert torch.all(cache["attn_pattern"] == 0)
    assert torch.all(cache["attn_out"] == 0)


def test_fc_cache_resid_final_matches_logits():
    m = _fc_model()
    tokens = torch.randint(0, 10, (8, 3))
    cache = m(tokens, return_cache=True)
    assert torch.allclose(cache["resid_final"] @ m.W_U, cache["logits"], atol=1e-6)


def test_fc_activations_produce_different_outputs():
    tokens = torch.randint(0, 10, (8, 3))
    torch.manual_seed(0)
    out_relu = _fc_model(activation="relu")(tokens)
    torch.manual_seed(0)
    out_gelu = _fc_model(activation="gelu")(tokens)
    torch.manual_seed(0)
    out_silu = _fc_model(activation="silu")(tokens)
    assert not torch.allclose(out_relu, out_gelu)
    assert not torch.allclose(out_relu, out_silu)
    assert not torch.allclose(out_gelu, out_silu)


def test_fc_unknown_activation_raises():
    with pytest.raises(KeyError):
        FCModel(
            d_vocab_in=10,
            d_vocab_out=10,
            n_ctx=3,
            d_model=16,
            d_mlp=32,
            activation="tanh",
        )


def test_fc_same_seed_gives_identical_weights_and_logits():
    torch.manual_seed(0)
    m1 = _fc_model()
    torch.manual_seed(0)
    m2 = _fc_model()
    for (name1, p1), (name2, p2) in zip(m1.named_parameters(), m2.named_parameters(), strict=True):
        assert name1 == name2
        assert torch.equal(p1, p2), f"parameter {name1} differs after reseeding"
    tokens = torch.randint(0, 10, (8, 3))
    assert torch.equal(m1(tokens), m2(tokens))


def test_fc_parameter_count_matches_expected_shapes():
    m = _fc_model(d_vocab_in=10, d_vocab_out=10, d_model=16, d_mlp=32)
    expected = (
        10 * 16  # W_E
        + 2 * 16 * 32  # W_in ([2*d_model, d_mlp])
        + 32 * 10  # W_U
    )
    actual = sum(p.numel() for p in m.parameters())
    assert actual == expected


# ---------------------------------------------------------------------------
# Shape / vocab / size / activation sweep
# ---------------------------------------------------------------------------
#
# Every case here would pass against a stub
# `return torch.zeros(batch, n_ctx, d_vocab_out)` -- shape-only, not behaviour.


@pytest.mark.parametrize(
    "d_vocab_in,d_vocab_out,d_model,d_mlp,activation,batch",
    [
        (10, 10, 16, 32, "relu", 8),  # baseline (construction + forward shape)
        (10, 10, 16, 32, "gelu", 8),  # activation variant
        (10, 10, 16, 32, "silu", 8),  # activation variant
        (10, 10, 32, 64, "relu", 4),  # size variant
        (10, 10, 64, 128, "relu", 4),  # size variant
        (10, 10, 8, 32, "relu", 4),  # size variant
        (200, 200, 32, 64, "relu", 8),  # large vocab
    ],
)
def test_fc_forward_shape_across_configurations(
    d_vocab_in, d_vocab_out, d_model, d_mlp, activation, batch
):
    m = FCModel(
        d_vocab_in=d_vocab_in,
        d_vocab_out=d_vocab_out,
        n_ctx=3,
        d_model=d_model,
        d_mlp=d_mlp,
        activation=activation,
    )
    tokens = torch.randint(0, d_vocab_in, (batch, 3))
    assert m(tokens).shape == (batch, 3, d_vocab_out)
