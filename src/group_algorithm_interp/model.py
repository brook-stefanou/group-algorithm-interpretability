from typing import Literal, TypedDict, overload

import torch
import torch.nn as nn


class ActivationCache(TypedDict):
    """Every intermediate of one forward pass, keyed TransformerLens-style.

    ``mlp_pre``/``mlp_post`` are the d_mlp-sized activations before/after the
    nonlinearity (None when use_mlp=False). ``resid_mid`` and ``mlp_out`` are
    omitted as recoverable: resid_mid = embed + attn_out;
    mlp_out = resid_final - embed - attn_out.
    """

    embed: torch.Tensor  # W_E[tokens] + W_pos      [batch, pos, d_model]
    attn_pattern: torch.Tensor  # post-softmax      [batch, head, q_pos, k_pos]
    attn_out: torch.Tensor  # after W_O             [batch, pos, d_model]
    mlp_pre: torch.Tensor | None  # pre-activation  [batch, pos, d_mlp]
    mlp_post: torch.Tensor | None  # post-activation [batch, pos, d_mlp]
    resid_final: torch.Tensor  # final residual     [batch, pos, d_model]
    logits: torch.Tensor  #                         [batch, pos, d_vocab_out]


class OneLayerTransformer(nn.Module):
    """A 1-layer attention + MLP transformer for the group-multiplication task.

    Minimal and interpretability-friendly: no LayerNorm, no biases, learned
    positional embeddings, untied embed/unembed. Weight names follow the
    TransformerLens convention (W_E, W_Q, W_K, W_V, W_O, W_in, W_out, W_U) so
    the state_dict doubles as the analysis weight contract.
    """

    def __init__(
        self,
        d_vocab_in: int,
        d_vocab_out: int,
        n_ctx: int,
        d_model: int,
        n_heads: int,
        use_mlp: bool,
        d_mlp: int,
        activation: str,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"Model dimension ({d_model}) is not evenly divisible by number of heads "
            f"({n_heads}). This means shape won't match residual stream when adding back to it."
        )
        d_head = d_model // n_heads
        self.d_head = d_head
        self.use_mlp = use_mlp
        self.d_mlp = d_mlp
        self.d_vocab_in = d_vocab_in
        self.d_vocab_out = d_vocab_out
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.n_heads = n_heads

        # Initialization follows Nanda's generalization transformer: every weight
        # is scaled by 1/sqrt(d_model) (fan-in), except the unembedding, which
        # uses 1/sqrt(d_vocab_out). This places the model in the weight-norm
        # regime where generalization appears within the training budget; it is
        # fixed, not config-driven.
        scale = d_model**-0.5
        self.W_E = nn.Parameter(torch.randn(d_vocab_in, d_model) * scale)
        self.W_pos = nn.Parameter(torch.randn(n_ctx, d_model) * scale)
        self.W_Q = nn.Parameter(torch.randn(n_heads, d_model, d_head) * scale)
        self.W_K = nn.Parameter(torch.randn(n_heads, d_model, d_head) * scale)
        self.W_V = nn.Parameter(torch.randn(n_heads, d_model, d_head) * scale)
        self.W_O = nn.Parameter(torch.randn(n_heads, d_head, d_model) * scale)
        if use_mlp:
            self.W_in = nn.Parameter(torch.randn(d_model, d_mlp) * scale)
            self.W_out = nn.Parameter(torch.randn(d_mlp, d_model) * scale)
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab_out) * d_vocab_out**-0.5)

        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        self.activation = activations[activation]

        # Causal mask: True above the diagonal (key_pos > query_pos) marks the
        # future positions each query is forbidden from attending to.
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(n_ctx, n_ctx, dtype=torch.bool), diagonal=1),
        )

    @overload
    def forward(self, tokens: torch.Tensor, return_cache: Literal[False] = ...) -> torch.Tensor: ...

    @overload
    def forward(self, tokens: torch.Tensor, return_cache: Literal[True]) -> ActivationCache: ...

    def forward(
        self, tokens: torch.Tensor, return_cache: bool = False
    ) -> torch.Tensor | ActivationCache:
        # tokens: [batch, n_ctx] int64 ids; returns logits [batch, n_ctx, d_vocab_out],
        # or the full ActivationCache when return_cache=True.
        embed = self.W_E[tokens] + self.W_pos
        resid = embed
        q = torch.einsum("b p m, h m d -> b h p d", resid, self.W_Q)
        k = torch.einsum("b p m, h m d -> b h p d", resid, self.W_K)
        v = torch.einsum("b p m, h m d -> b h p d", resid, self.W_V)
        attn_scores = torch.einsum("b h q d, b h k d -> b h q k", q, k)
        attn_scores = attn_scores / (self.d_head**0.5)
        attn_scores = attn_scores.masked_fill(self.causal_mask, float("-inf"))
        pattern = torch.softmax(attn_scores, dim=-1)
        z = torch.einsum("b h q k, b h k d -> b h q d", pattern, v)
        attn_out = torch.einsum("b h p d, h d m -> b p m", z, self.W_O)
        resid = resid + attn_out
        mlp_pre: torch.Tensor | None = None
        mlp_post: torch.Tensor | None = None
        if self.use_mlp:
            mlp_pre = torch.einsum("b p m, m l -> b p l", resid, self.W_in)
            mlp_post = self.activation(mlp_pre)
            mlp_out = torch.einsum("b p l, l m -> b p m", mlp_post, self.W_out)
            resid = resid + mlp_out
        logits = torch.einsum("b p m, m v -> b p v", resid, self.W_U)
        if return_cache:
            return ActivationCache(
                embed=embed,
                attn_pattern=pattern,
                attn_out=attn_out,
                mlp_pre=mlp_pre,
                mlp_post=mlp_post,
                resid_final=resid,
                logits=logits,
            )
        return logits


class FCModel(nn.Module):
    """One-hidden-layer fully-connected baseline for the architecture confound.

    Mirrors the Stander et al. (2024) coset-paper model -- one-hot inputs, a single
    hidden layer with ReLU, linear readout -- with two deviations so it is directly
    comparable to ``OneLayerTransformer`` and its analysis instruments:

    * a SHARED embedding ``W_E`` for both inputs (Stander use separate ``E_l``,
      ``E_r``), so the isotypic-energy instrument reads the same ``W_E`` object as
      the transformer;
    * no biases and fan-in initialisation, matching this repo's clean-analysis
      conventions and the transformer's weight-norm generalization regime.

    Tokens are ``[batch, n_ctx]`` = ``(a, b, '=')``; only ``a`` and ``b`` are
    embedded. ``forward`` returns logits ``[batch, n_ctx, d_vocab_out]`` with the
    prediction broadcast to the read position (-1), so the model is a drop-in for
    the trainer, evaluation, and the functional-form / energy / coset instruments
    (which read ``logits[:, -1]``, ``W_E``, and ``resid_final[:, -1]``). For the
    cache, ``resid_final`` is the post-ReLU hidden representation -- the FC's
    structural analog of the transformer's final residual feeding ``W_U``.
    """

    def __init__(
        self,
        d_vocab_in: int,
        d_vocab_out: int,
        n_ctx: int,
        d_model: int,
        d_mlp: int,
        activation: str,
    ):
        super().__init__()
        self.d_vocab_in = d_vocab_in
        self.d_vocab_out = d_vocab_out
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.d_mlp = d_mlp

        self.W_E = nn.Parameter(torch.randn(d_vocab_in, d_model) * d_model**-0.5)
        self.W_in = nn.Parameter(torch.randn(2 * d_model, d_mlp) * (2 * d_model) ** -0.5)
        self.W_U = nn.Parameter(torch.randn(d_mlp, d_vocab_out) * d_vocab_out**-0.5)

        activations = {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}
        self.activation = activations[activation]

    @overload
    def forward(self, tokens: torch.Tensor, return_cache: Literal[False] = ...) -> torch.Tensor: ...

    @overload
    def forward(self, tokens: torch.Tensor, return_cache: Literal[True]) -> ActivationCache: ...

    def forward(
        self, tokens: torch.Tensor, return_cache: bool = False
    ) -> torch.Tensor | ActivationCache:
        # tokens: [batch, n_ctx] int64 ids (a, b, '='); only a and b are used.
        a = self.W_E[tokens[:, 0]]  # [batch, d_model]
        b = self.W_E[tokens[:, 1]]  # [batch, d_model]
        pre = torch.cat([a, b], dim=-1) @ self.W_in  # [batch, d_mlp]
        post = self.activation(pre)
        pred = post @ self.W_U  # [batch, d_vocab_out]
        # Broadcast to every position so logits[:, -1] (the read position) is the
        # prediction -- keeps the [batch, n_ctx, d_vocab_out] contract.
        logits: torch.Tensor = pred.unsqueeze(1).expand(-1, self.n_ctx, -1)
        if return_cache:
            n = tokens.shape[0]
            seq = self.n_ctx

            def _seq(x: torch.Tensor) -> torch.Tensor:
                return x.unsqueeze(1).expand(-1, seq, -1)

            return ActivationCache(
                embed=self.W_E[tokens],  # [batch, n_ctx, d_model]
                # No attention in an FC; zero placeholders keep the cache contract.
                attn_pattern=torch.zeros(n, 1, seq, seq),
                attn_out=torch.zeros(n, seq, self.d_model),
                mlp_pre=_seq(pre),
                mlp_post=_seq(post),
                resid_final=_seq(post),  # post-ReLU hidden = the read representation
                logits=logits,
            )
        return logits


# The two architectures share the vocab/n_ctx contract, a ``W_E``/``W_U`` weight
# pair, and the ``logits[:, -1]`` read position, so the analysis instruments accept
# either. ``build_model`` returns this union.
GroupModel = OneLayerTransformer | FCModel
