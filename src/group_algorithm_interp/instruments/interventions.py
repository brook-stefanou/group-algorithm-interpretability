"""I-07: the intervention harness (substrate for rungs 3-4).

Hook-free ablation and activation patching of MLP neurons and whole components,
computed post hoc from a model's own activation cache. The object of study is
the MLP neuron ``m`` (``plan.md`` §8, "Architecture facts the instruments
assume"), so the primitives here edit ``mlp_post`` and recompute the readout.
Both model types expose the same readout, ``logits = resid_final @ W_U``, which
is what makes one harness serve both.

The harness provides:

* zero, mean, and resample ablation of a neuron set, reported side by side, in
  behavioural units — label flips and multiples of ``1/n`` — never bare
  percentages (``plan.md`` §3, item 5);
* a random-neuron-set control of matched size (the I-03 permuted-neuron / random-
  subset null), so an ablation effect is read against a matched-size baseline;
* activation patching: the selected neurons' activations from a second forward
  pass (typically the corrupted member of a pair) substituted into the first,
  the rest held fixed, and the readout recomputed.

Everything runs offline and is deterministic for a fixed ``rng_seed``: the mean
and resample edits use a private seeded RNG and never perturb the global torch
stream. The harness measures; the corrupted-pair design, the choice of neuron
set, and any claim about what the effect means belong to the experiment that
consumes it. Two-track rule: a within-model intervention supports a mechanism
claim about that one model, never a between-model difference claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..model import GroupModel

MODES = ("zero", "mean", "resample")
READ_POS = -1


def _require_mlp(model: GroupModel) -> None:
    if getattr(model, "use_mlp", True) is False:
        raise ValueError("model has no MLP (use_mlp=false); neuron interventions are undefined")


def _logits_from_edited_post(
    model: GroupModel, cache: dict[str, Any], edited_post: torch.Tensor
) -> torch.Tensor:
    """Recompute logits after replacing ``mlp_post`` with ``edited_post``.

    Transformer: ``resid_final = embed + attn_out + mlp_post @ W_out``, so the
    edited residual is ``embed + attn_out + edited_post @ W_out`` and the logits
    are that times ``W_U``. Fully-connected: ``resid_final`` is the post-ReLU
    hidden itself, so the edited residual is ``edited_post`` and the logits are
    ``edited_post @ W_U``. Both read out through the same ``W_U``.
    """
    w_u = model.W_U
    if hasattr(model, "W_out"):
        resid = (
            cache["embed"]
            + cache["attn_out"]
            + torch.einsum("b p l, l m -> b p m", edited_post, model.W_out)
        )
        return torch.einsum("b p m, m v -> b p v", resid, w_u)
    return torch.einsum("b p l, l v -> b p v", edited_post, w_u)


def _edit_neurons(
    post: torch.Tensor,
    neuron_indices: torch.Tensor,
    mode: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """A copy of ``post`` with the selected neuron columns replaced per ``mode``.

    ``zero`` sets them to 0; ``mean`` replaces each selected neuron with its mean
    over the batch (per position), i.e. its activation carries no example-specific
    information; ``resample`` permutes each selected neuron's activations across
    the batch, breaking the neuron's alignment with the input while preserving its
    marginal distribution.
    """
    edited = post.clone()
    if mode == "zero":
        edited[:, :, neuron_indices] = 0.0
    elif mode == "mean":
        mean = post[:, :, neuron_indices].mean(dim=0, keepdim=True)
        edited[:, :, neuron_indices] = mean.expand_as(edited[:, :, neuron_indices])
    elif mode == "resample":
        batch = post.shape[0]
        perm = torch.randperm(batch, generator=generator)
        edited[:, :, neuron_indices] = post[perm][:, :, neuron_indices]
    else:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return edited


@dataclass(frozen=True)
class _Behaviour:
    n: int
    n_correct: int
    accuracy: float


def _behaviour(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None
) -> _Behaviour:
    preds = logits[:, READ_POS, :].argmax(dim=-1)
    correct = preds == targets
    if mask is not None:
        correct = correct[mask]
    n = int(correct.shape[0])
    n_correct = int(correct.sum())
    return _Behaviour(n=n, n_correct=n_correct, accuracy=n_correct / n if n else float("nan"))


def _effect_block(
    model: GroupModel,
    cache: dict[str, Any],
    post: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    neuron_indices: torch.Tensor,
    baseline: _Behaviour,
    rng_seed: int,
) -> dict[str, Any]:
    """Ablate ``neuron_indices`` under each mode and report the effect against
    ``baseline`` in behavioural units."""
    out: dict[str, Any] = {}
    for mode in MODES:
        generator = torch.Generator().manual_seed(rng_seed)
        edited = _edit_neurons(post, neuron_indices, mode, generator)
        logits = _logits_from_edited_post(model, cache, edited)
        after = _behaviour(logits, targets, mask)
        baseline_preds = cache["logits"][:, READ_POS, :].argmax(dim=-1)
        after_preds = logits[:, READ_POS, :].argmax(dim=-1)
        flipped = baseline_preds != after_preds
        if mask is not None:
            flipped = flipped[mask]
        out[mode] = {
            "accuracy_after": after.accuracy,
            "accuracy_drop": baseline.accuracy - after.accuracy,
            "n_correct_after": after.n_correct,
            "correct_lost": baseline.n_correct - after.n_correct,
            "n_label_flips": int(flipped.sum()),
            "n_eval": after.n,
        }
    return out


def ablate_neurons(
    model: GroupModel,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    neuron_indices: list[int] | torch.Tensor,
    *,
    unleaked_mask: torch.Tensor | None = None,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Zero/mean/resample ablation of an MLP-neuron set, with a matched-size
    random-neuron control, reported in behavioural units.

    ``tokens`` are ``(a, b, '=')`` triples; ``targets`` are the products; the
    optional ``unleaked_mask`` restricts the behavioural readout to the
    transpose-unleaked subset so the effect is on the endpoint accuracy, not the
    leaked one. The record carries the baseline accuracy, each mode's effect on
    the target set, and the same three modes on a random set of the same size —
    the control an effect must beat to be more than a generic capacity loss. No
    verdict is emitted.
    """
    _require_mlp(model)
    idx = torch.as_tensor(list(neuron_indices), dtype=torch.long)
    model.eval()
    with torch.no_grad():
        cache = model(tokens, return_cache=True)
        post = cache["mlp_post"]
        if post is None:
            raise ValueError("model produced no mlp_post; neuron ablation is undefined")
        baseline = _behaviour(cache["logits"], targets, unleaked_mask)
        d_mlp = post.shape[-1]
        target_effect = _effect_block(
            model, cache, post, targets, unleaked_mask, idx, baseline, rng_seed
        )
        control_generator = torch.Generator().manual_seed(rng_seed + 1)
        control_idx = torch.randperm(d_mlp, generator=control_generator)[: idx.shape[0]]
        control_effect = _effect_block(
            model, cache, post, targets, unleaked_mask, control_idx, baseline, rng_seed
        )
    return {
        "instrument": "intervention-ablation",
        "d_mlp": int(d_mlp),
        "n_ablated": int(idx.shape[0]),
        "neuron_indices": idx.tolist(),
        "baseline_accuracy": baseline.accuracy,
        "baseline_n_correct": baseline.n_correct,
        "n_eval": baseline.n,
        "on_unleaked_subset": unleaked_mask is not None,
        "target_set": target_effect,
        "random_control_set": {"neuron_indices": control_idx.tolist(), **control_effect},
        "note": (
            "Effects in behavioural units (label flips, correct-answer counts "
            "over 1/n). Zero, mean and resample side by side; a matched-size "
            "random-neuron control alongside. Within-model only (two-track rule)."
        ),
    }


def ablate_component(
    model: GroupModel,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    component: str,
    *,
    unleaked_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Zero-ablate a whole component of the transformer's short paths: the MLP
    block (``component="mlp"``) or the attention block (``component="attn"``).
    The 1-layer paths are embed -> readout, embed -> attn -> readout, and
    embed -> mlp -> readout, so zeroing one block isolates its contribution to
    the readout. Reported in behavioural units; transformer only (the FC model
    has a single path)."""
    if not hasattr(model, "W_out"):
        raise ValueError("component ablation is defined for the transformer path only")
    if component not in ("mlp", "attn"):
        raise ValueError(f"component must be 'mlp' or 'attn', got {component!r}")
    model.eval()
    with torch.no_grad():
        cache = model(tokens, return_cache=True)
        baseline = _behaviour(cache["logits"], targets, unleaked_mask)
        embed = cache["embed"]
        attn_out = cache["attn_out"]
        post = cache["mlp_post"]
        mlp_out = (
            torch.einsum("b p l, l m -> b p m", post, model.W_out) if post is not None else 0.0
        )
        if component == "mlp":
            resid = embed + attn_out
        else:
            resid = embed + mlp_out
        logits = torch.einsum("b p m, m v -> b p v", resid, model.W_U)
        after = _behaviour(logits, targets, unleaked_mask)
    return {
        "instrument": "intervention-component-ablation",
        "component": component,
        "baseline_accuracy": baseline.accuracy,
        "accuracy_after": after.accuracy,
        "accuracy_drop": baseline.accuracy - after.accuracy,
        "correct_lost": baseline.n_correct - after.n_correct,
        "n_eval": baseline.n,
        "on_unleaked_subset": unleaked_mask is not None,
    }


def patch_neurons(
    model: GroupModel,
    tokens: torch.Tensor,
    source_tokens: torch.Tensor,
    neuron_indices: list[int] | torch.Tensor,
    targets: torch.Tensor,
    *,
    unleaked_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Activation patching: run ``tokens`` (the clean pass) but substitute the
    selected neurons' ``mlp_post`` from a second pass over ``source_tokens`` (the
    corrupted pass), then recompute the readout. ``tokens`` and ``source_tokens``
    must have the same batch size and be aligned example-for-example (the
    per-experiment corrupted-pair design). Reports the effect on the clean
    behaviour in behavioural units; no verdict."""
    _require_mlp(model)
    if tokens.shape[0] != source_tokens.shape[0]:
        raise ValueError("clean and source passes must have the same batch size for patching")
    idx = torch.as_tensor(list(neuron_indices), dtype=torch.long)
    model.eval()
    with torch.no_grad():
        clean = model(tokens, return_cache=True)
        source = model(source_tokens, return_cache=True)
        clean_post = clean["mlp_post"]
        source_post = source["mlp_post"]
        if clean_post is None or source_post is None:
            raise ValueError("model produced no mlp_post; patching is undefined")
        baseline = _behaviour(clean["logits"], targets, unleaked_mask)
        patched = clean_post.clone()
        patched[:, :, idx] = source_post[:, :, idx]
        logits = _logits_from_edited_post(model, clean, patched)
        after = _behaviour(logits, targets, unleaked_mask)
        baseline_preds = clean["logits"][:, READ_POS, :].argmax(dim=-1)
        after_preds = logits[:, READ_POS, :].argmax(dim=-1)
        flipped = baseline_preds != after_preds
        if unleaked_mask is not None:
            flipped = flipped[unleaked_mask]
    return {
        "instrument": "intervention-patch",
        "n_patched": int(idx.shape[0]),
        "neuron_indices": idx.tolist(),
        "baseline_accuracy": baseline.accuracy,
        "accuracy_after": after.accuracy,
        "accuracy_drop": baseline.accuracy - after.accuracy,
        "correct_lost": baseline.n_correct - after.n_correct,
        "n_label_flips": int(flipped.sum()),
        "n_eval": baseline.n,
        "on_unleaked_subset": unleaked_mask is not None,
        "note": "Corrupted-source activation patching; within-model (two-track rule).",
    }


__all__ = [
    "MODES",
    "ablate_component",
    "ablate_neurons",
    "patch_neurons",
]
