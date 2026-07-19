import einops
import pytest
import torch

from group_algorithm_interp.model import OneLayerTransformer

# Weight names the analysis layer relies on (the "contract").
CONTRACT = {"W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "W_U"}


def _model(use_mlp=True, d_model=8, n_heads=2, activation="relu"):
    return OneLayerTransformer(
        d_vocab_in=6,
        d_vocab_out=5,
        n_ctx=3,
        d_model=d_model,
        n_heads=n_heads,
        use_mlp=use_mlp,
        d_mlp=16,
        activation=activation,
    )


def test_parameter_shapes_match_contract():
    m = _model()
    d_head = 8 // 2
    assert m.W_E.shape == (6, 8)
    assert m.W_pos.shape == (3, 8)
    assert m.W_Q.shape == (2, 8, d_head)
    assert m.W_K.shape == (2, 8, d_head)
    assert m.W_V.shape == (2, 8, d_head)
    assert m.W_O.shape == (2, d_head, 8)
    assert m.W_in.shape == (8, 16)
    assert m.W_out.shape == (16, 8)
    assert m.W_U.shape == (8, 5)


def test_forward_output_shape():
    m = _model()
    tokens = torch.randint(0, 6, (4, 3))  # [batch, n_ctx], int64
    assert m(tokens).shape == (4, 3, 5)  # [batch, n_ctx, d_vocab_out]


def test_gradients_flow_to_every_parameter():
    m = _model()
    tokens = torch.randint(0, 6, (4, 3))
    m(tokens).sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"


def test_no_mlp_omits_mlp_parameters_but_still_runs():
    m = _model(use_mlp=False)
    names = dict(m.named_parameters())
    assert "W_in" not in names and "W_out" not in names
    tokens = torch.randint(0, 6, (4, 3))
    assert m(tokens).shape == (4, 3, 5)


def test_state_dict_exposes_the_weight_contract():
    m = _model()
    assert CONTRACT <= set(m.state_dict().keys())


def test_init_uses_fan_in_scaling():
    # Nanda's generalization init: weights ~ N(0, 1/d_model), unembed ~ N(0, 1/d_vocab_out),
    # fixed to fan-in rather than config-driven.
    torch.manual_seed(0)
    m = OneLayerTransformer(
        d_vocab_in=64,
        d_vocab_out=63,
        n_ctx=3,
        d_model=128,
        n_heads=4,
        use_mlp=True,
        d_mlp=512,
        activation="relu",
    )
    assert m.W_E.std().item() == pytest.approx(128**-0.5, rel=0.1)
    assert m.W_in.std().item() == pytest.approx(128**-0.5, rel=0.1)
    assert m.W_U.std().item() == pytest.approx(63**-0.5, rel=0.1)


def test_indivisible_d_model_raises():
    with pytest.raises(AssertionError):
        _model(d_model=8, n_heads=3)  # 8 % 3 != 0


def test_unknown_activation_raises():
    with pytest.raises(KeyError):
        OneLayerTransformer(
            d_vocab_in=6,
            d_vocab_out=5,
            n_ctx=3,
            d_model=8,
            n_heads=2,
            use_mlp=True,
            d_mlp=16,
            activation="tanh",
        )


def test_return_cache_logits_exactly_match_plain_forward():
    m = _model()
    tokens = torch.randint(0, 6, (4, 3))
    cache = m(tokens, return_cache=True)
    assert torch.equal(cache["logits"], m(tokens))


def test_cache_entry_shapes():
    m = _model()  # d_model=8, n_heads=2, d_mlp=16, d_vocab_out=5, n_ctx=3
    tokens = torch.randint(0, 6, (4, 3))
    cache = m(tokens, return_cache=True)
    assert cache["embed"].shape == (4, 3, 8)
    assert cache["attn_pattern"].shape == (4, 2, 3, 3)
    assert cache["attn_out"].shape == (4, 3, 8)
    assert cache["mlp_pre"].shape == (4, 3, 16)
    assert cache["mlp_post"].shape == (4, 3, 16)
    assert cache["resid_final"].shape == (4, 3, 8)
    assert cache["logits"].shape == (4, 3, 5)


def test_cache_attn_pattern_is_causal_probability():
    m = _model()
    tokens = torch.randint(0, 6, (4, 3))
    pattern = m(tokens, return_cache=True)["attn_pattern"]
    assert torch.allclose(pattern.sum(dim=-1), torch.ones(4, 2, 3), atol=1e-6)
    future = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    assert torch.all(pattern[..., future] == 0)


def test_cached_pattern_and_attn_out_are_self_consistent():
    # Recomputes attn_out from the CACHED pattern, so it can't catch a bug in how
    # the pattern itself was computed (see
    # test_attn_pattern_matches_independently_computed_scores for that). Kept
    # anyway because it pins the W_V/W_O contraction order.
    m = _model()
    tokens = torch.randint(0, 6, (4, 3))
    cache = m(tokens, return_cache=True)
    resid = cache["embed"]
    v = einops.einsum(
        resid, m.W_V, "batch pos d_model, head d_model d_head -> batch head pos d_head"
    )
    z = einops.einsum(
        cache["attn_pattern"],
        v,
        "batch head query_pos key_pos, batch head key_pos d_head -> batch head query_pos d_head",
    )
    reconstructed = einops.einsum(
        z, m.W_O, "batch head pos d_head, head d_head d_model -> batch pos d_model"
    )
    assert torch.allclose(reconstructed, cache["attn_out"], atol=1e-6)


def test_attn_pattern_matches_independently_computed_scores():
    # Recomputes the attention pattern from raw Q/K with a per-head matmul,
    # written separately from forward()'s einsum calls. Catches two bugs the
    # rest of the suite lets through: dropping the 1/sqrt(d_head) score scaling,
    # and swapping W_Q/W_K (the causal mask is asymmetric, so a transposed
    # pre-mask score matrix produces a different pattern after masking).
    torch.manual_seed(0)
    m = _model(d_model=8, n_heads=2)
    tokens = torch.randint(0, 6, (4, 3))
    cache = m(tokens, return_cache=True)
    embed = cache["embed"]  # [batch, pos, d_model]
    n_heads, d_head = m.n_heads, m.d_head
    expected_scores = torch.empty(4, n_heads, 3, 3)
    for h in range(n_heads):
        q_h = embed @ m.W_Q[h]  # [batch, pos, d_head] -- W_Q plays the query role
        k_h = embed @ m.W_K[h]  # [batch, pos, d_head] -- W_K plays the key role
        expected_scores[:, h] = (q_h @ k_h.transpose(-1, -2)) / (d_head**0.5)
    causal = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    expected_scores = expected_scores.masked_fill(causal, float("-inf"))
    expected_pattern = torch.softmax(expected_scores, dim=-1)
    assert torch.allclose(expected_pattern, cache["attn_pattern"], atol=1e-6)


def test_cache_resid_final_reproduces_logits():
    m = _model()
    tokens = torch.randint(0, 6, (4, 3))
    cache = m(tokens, return_cache=True)
    assert torch.allclose(cache["resid_final"] @ m.W_U, cache["logits"], atol=1e-6)


def test_cache_residual_stream_decomposes():
    # resid_final = embed + attn_out + mlp_out, where mlp_out = mlp_post @ W_out
    m = _model()
    tokens = torch.randint(0, 6, (4, 3))
    cache = m(tokens, return_cache=True)
    reconstructed = cache["embed"] + cache["attn_out"] + cache["mlp_post"] @ m.W_out
    assert torch.allclose(reconstructed, cache["resid_final"], atol=1e-6)


def test_cache_without_mlp_has_none_mlp_entries():
    m = _model(use_mlp=False)
    tokens = torch.randint(0, 6, (4, 3))
    cache = m(tokens, return_cache=True)
    assert cache["mlp_pre"] is None
    assert cache["mlp_post"] is None
    assert torch.allclose(cache["embed"] + cache["attn_out"], cache["resid_final"], atol=1e-6)


def test_same_seed_gives_identical_weights_and_logits():
    torch.manual_seed(0)
    m1 = _model()
    torch.manual_seed(0)
    m2 = _model()
    for (name1, p1), (name2, p2) in zip(m1.named_parameters(), m2.named_parameters(), strict=True):
        assert name1 == name2
        assert torch.equal(p1, p2), f"parameter {name1} differs after reseeding"
    tokens = torch.randint(0, 6, (4, 3))
    assert torch.equal(m1(tokens), m2(tokens))


def test_parameter_count_matches_expected_shapes():
    m = _model(d_model=8, n_heads=2, use_mlp=True)  # d_vocab_in=6, d_vocab_out=5, d_mlp=16
    d_head = 8 // 2
    expected = (
        6 * 8  # W_E
        + 3 * 8  # W_pos
        + 2 * 8 * d_head  # W_Q
        + 2 * 8 * d_head  # W_K
        + 2 * 8 * d_head  # W_V
        + 2 * d_head * 8  # W_O
        + 8 * 16  # W_in
        + 16 * 8  # W_out
        + 8 * 5  # W_U
    )
    actual = sum(p.numel() for p in m.parameters())
    assert actual == expected


def test_read_position_logits_depend_on_both_operands():
    # logits[:, -1] is the position every downstream consumer (trainer, eval,
    # analysis) reads, so it must depend on both input tokens, not just one.
    torch.manual_seed(0)
    m = _model()
    eq = 5
    base = torch.tensor([[0, 1, eq]])
    change_a = torch.tensor([[2, 1, eq]])
    change_b = torch.tensor([[0, 3, eq]])
    out_base = m(base)[:, -1, :]
    assert not torch.allclose(out_base, m(change_a)[:, -1, :])
    assert not torch.allclose(out_base, m(change_b)[:, -1, :])


def test_only_the_last_position_has_seen_both_operands():
    # Pins the transformer's readout to the last position: position 0 attends
    # only to itself under the causal mask, so its logits cannot depend on
    # token 1 changing -- only position -1 (which has seen both a and b) does.
    torch.manual_seed(0)
    m = _model()
    eq = 5
    base = torch.tensor([[0, 1, eq]])
    change_b_only = torch.tensor([[0, 3, eq]])
    out_base = m(base)
    out_changed = m(change_b_only)
    assert torch.allclose(out_base[:, 0, :], out_changed[:, 0, :])
    assert not torch.allclose(out_base[:, -1, :], out_changed[:, -1, :])


# ---------------------------------------------------------------------------
# OneLayerTransformer — activation / size / vocab / batch / seq-length sweep
# ---------------------------------------------------------------------------
#
# A shape-only sweep: it would pass against a stub `return torch.zeros(batch,
# n_ctx, d_vocab_out)`. Kept as configuration-coverage rather than a
# correctness check.


@pytest.mark.parametrize(
    "d_vocab_in,d_vocab_out,n_ctx,d_model,n_heads,d_mlp,activation,batch",
    [
        (6, 5, 3, 8, 2, 16, "gelu", 4),  # activation variant
        (6, 5, 3, 8, 2, 16, "silu", 4),  # activation variant
        (6, 5, 3, 16, 4, 16, "relu", 4),  # model size variant
        (6, 5, 3, 32, 8, 16, "relu", 4),  # model size variant
        (6, 5, 3, 12, 3, 16, "relu", 4),  # model size variant
        (100, 100, 3, 16, 4, 32, "relu", 4),  # large vocab
        (6, 5, 3, 8, 2, 16, "relu", 1),  # batch size variant
        (6, 5, 3, 8, 2, 16, "relu", 8),  # batch size variant
        (6, 5, 3, 8, 2, 16, "relu", 32),  # batch size variant
        (6, 5, 5, 8, 2, 16, "relu", 4),  # sequence length variant
        (6, 5, 10, 8, 2, 16, "relu", 4),  # sequence length variant
    ],
)
def test_forward_shape_across_configurations(
    d_vocab_in, d_vocab_out, n_ctx, d_model, n_heads, d_mlp, activation, batch
):
    m = OneLayerTransformer(
        d_vocab_in=d_vocab_in,
        d_vocab_out=d_vocab_out,
        n_ctx=n_ctx,
        d_model=d_model,
        n_heads=n_heads,
        use_mlp=True,
        d_mlp=d_mlp,
        activation=activation,
    )
    tokens = torch.randint(0, d_vocab_in, (batch, n_ctx))
    assert m(tokens).shape == (batch, n_ctx, d_vocab_out)


def test_causal_mask_explicit_structure():
    m = OneLayerTransformer(
        d_vocab_in=6,
        d_vocab_out=5,
        n_ctx=4,
        d_model=8,
        n_heads=2,
        use_mlp=True,
        d_mlp=16,
        activation="relu",
    )
    mask = m.causal_mask
    # triu with diagonal=1: positions above diagonal are True (masked out)
    assert mask[0, 0].item() is False
    assert mask[0, 1].item() is True
    assert mask[0, 2].item() is True
    assert mask[0, 3].item() is True
    assert mask[1, 0].item() is False
    assert mask[1, 1].item() is False
    assert mask[1, 2].item() is True
    assert mask[1, 3].item() is True
    assert mask[2, 3].item() is True
    assert mask[3, 3].item() is False


def test_activations_produce_different_outputs():
    tokens = torch.randint(0, 6, (4, 3))
    torch.manual_seed(0)
    out_relu = _model(activation="relu")(tokens)
    torch.manual_seed(0)
    out_gelu = _model(activation="gelu")(tokens)
    torch.manual_seed(0)
    out_silu = _model(activation="silu")(tokens)
    assert not torch.allclose(out_relu, out_gelu)
    assert not torch.allclose(out_relu, out_silu)
    assert not torch.allclose(out_gelu, out_silu)
