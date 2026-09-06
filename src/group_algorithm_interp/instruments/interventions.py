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

Everything runs offline and is deterministic for a fixed ``rng_seed``: the
resample edit uses a private seeded RNG (mean needs none -- it has no
randomness) and neither perturbs the global torch stream. The harness measures;
the corrupted-pair design, the choice of neuron
set, and any claim about what the effect means belong to the experiment that
consumes it. Two-track rule: a within-model intervention supports a mechanism
claim about that one model, never a between-model difference claim.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..groups.group import FiniteGroup
from ..model import GroupModel
from .nulls import random_neuron_control
from .occupancy import cayley_grid_tokens

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
    information; ``resample`` draws a single random permutation of the batch and
    applies it jointly to every selected neuron (and every position), so the
    same donor example supplies all of them together. This breaks each selected
    neuron's alignment with its own input while preserving both each neuron's
    marginal distribution and the within-set correlations across the selected
    neurons -- not an independent per-neuron permutation.
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
    optional ``unleaked_mask`` restricts the *behavioural readout* to the
    transpose-unleaked subset so the effect is on the endpoint accuracy, not the
    leaked one -- the mean and resample statistics themselves are still computed
    over the full batch (the mean is the full-batch mean; resample's donor
    permutation is drawn over the full batch), regardless of ``unleaked_mask``.
    The record carries the baseline accuracy, each mode's effect on the target
    set, and the same three modes on a random set of the same size — the
    control an effect must beat to be more than a generic capacity loss. No
    verdict is emitted.

    ``neuron_indices`` must be in-bounds for the model's ``d_mlp`` and free of
    duplicates -- rejected outright rather than silently deduped, so the
    matched-size random control's size always agrees with what the caller asked
    to ablate.
    """
    _require_mlp(model)
    idx = torch.as_tensor(list(neuron_indices), dtype=torch.long)
    if idx.numel() > 0:
        d_mlp_model = int(model.d_mlp)
        if int(idx.min()) < 0 or int(idx.max()) >= d_mlp_model:
            raise ValueError(
                f"neuron_indices out of bounds for d_mlp={d_mlp_model}: {idx.tolist()}"
            )
        if int(idx.unique().numel()) != int(idx.numel()):
            raise ValueError(f"neuron_indices contains duplicates: {idx.tolist()}")
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
        control_idx = random_neuron_control(d_mlp, idx.shape[0], seed=rng_seed + 1)
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
    full_removal: bool = False,
) -> dict[str, Any]:
    """Ablate one block of the transformer's residual stream and measure the
    behavioural cost of removing it.

    The 1-layer transformer's readout is reached from ``embed`` by four paths:
    ``embed -> readout`` (the residual skip, always present), ``embed -> attn ->
    readout`` (attention's own additive term), ``embed -> mlp -> readout`` (the
    MLP's additive term), and ``embed -> attn -> mlp -> readout`` (the composed
    path -- attention's output feeds the MLP's input, before the MLP's
    nonlinearity, so attention's effect also reaches the readout through the
    MLP, not only through its own additive term).

    Because attention runs before the MLP, ``component="mlp"`` removes the
    MLP's contribution completely: there is no further block downstream for it
    to reach the readout through, so this already is a true full-block removal
    (``scope: "full"`` in the returned record). ``component="attn"`` by default
    zeroes only attention's own additive term (``scope: "direct_path"``): the
    cached ``mlp_post`` comes from the original forward pass, so it still
    carries attention's effect, and the composed embed->attn->mlp->readout path
    is NOT removed -- do not read this as "attention's contribution to the
    readout". Pass ``full_removal=True`` to also recompute the MLP from
    ``embed`` alone, equivalent to zeroing ``W_O`` and rerunning the forward
    pass, for a true full removal of attention along both paths (``scope:
    "full"``); the default is unchanged. Reported in behavioural units;
    transformer only (the FC model has a single block, with no direct/composed
    path to isolate)."""
    if not hasattr(model, "W_Q"):
        raise ValueError(
            "component ablation requires the transformer architecture, which has "
            "separate attn/mlp blocks to isolate; the FC model has a single block"
        )
    if component not in ("mlp", "attn"):
        raise ValueError(f"component must be 'mlp' or 'attn', got {component!r}")
    if component == "mlp":
        _require_mlp(model)
    model.eval()
    with torch.no_grad():
        cache = model(tokens, return_cache=True)
        baseline = _behaviour(cache["logits"], targets, unleaked_mask)
        embed = cache["embed"]
        attn_out = cache["attn_out"]
        post = cache["mlp_post"]
        if component == "mlp":
            resid = embed + attn_out
            scope = "full"
        elif post is None:
            # No MLP exists at all, so there is no embed->attn->mlp->readout
            # composed path to miss: removing attention's direct additive term
            # already removes its entire contribution to the readout.
            resid = embed
            scope = "full"
        elif full_removal:
            mlp_pre_full = torch.einsum("b p m, m l -> b p l", embed, model.W_in)
            mlp_post_full = model.activation(mlp_pre_full)
            mlp_out_full = torch.einsum("b p l, l m -> b p m", mlp_post_full, model.W_out)
            resid = embed + mlp_out_full
            scope = "full"
        else:
            mlp_out = torch.einsum("b p l, l m -> b p m", post, model.W_out)
            resid = embed + mlp_out
            scope = "direct_path"
        logits = torch.einsum("b p m, m v -> b p v", resid, model.W_U)
        after = _behaviour(logits, targets, unleaked_mask)
    return {
        "instrument": "intervention-component-ablation",
        "component": component,
        "scope": scope,
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


# ---------------------------------------------------------------------------
# Representational-axis ablation: zero-ablate a single direction out of W_E and
# read the behavioural cost in flips, against norm-matched random directions.
# This is the harness substrate I-28b (probes.involution_direction_ablation)
# consumes -- the direction it removes is instrument-specific, the machinery
# that removes it and scores the cost is shared here.
# ---------------------------------------------------------------------------


def model_correct_mask(model: GroupModel, group: FiniteGroup) -> np.ndarray:
    """Boolean ``[order*order]`` mask of whether the model's argmax at the read
    position equals ``a*b``, in Cayley-grid (row-major) order."""
    tokens = cayley_grid_tokens(group.order)
    model.eval()
    with torch.no_grad():
        logits = model(tokens)[:, -1, :]
        predictions = logits.argmax(dim=-1).cpu().numpy()
    targets = group.cayley_table.reshape(-1)
    return predictions == targets


def _accuracy_in_flips(mask: np.ndarray, subset: np.ndarray | None) -> int:
    if subset is None:
        return int(mask.sum())
    return int(mask[subset].sum())


def _subset_size(full_size: int, subset: np.ndarray | None) -> int:
    """The number of pairs ``subset`` actually selects out of ``full_size``:
    every pair when ``subset`` is ``None``, the count of ``True`` entries for a
    boolean mask (``mask[subset]`` only keeps those), or the number of indices
    for an integer index array."""
    if subset is None:
        return full_size
    if subset.dtype == np.bool_:
        return int(subset.sum())
    return int(subset.size)


def _ablated_model(model: GroupModel, direction: np.ndarray) -> GroupModel:
    """A copy of ``model`` with the unit ``direction`` (in ``d_model`` space)
    projected out of every ``W_E`` row -- zero-ablation of one representational
    axis in the *embedding input*, not the residual stream: only the token
    embedding is edited (``W_pos`` is untouched), and any downstream write --
    attention or MLP -- can reintroduce a component along the same axis before
    the readout. This is not a guarantee that the axis is absent from the
    residual stream at every position; it is a lower bound on the model's
    reliance on it as an embedding-input feature."""
    clone = copy.deepcopy(model)
    unit = direction / (np.linalg.norm(direction) + 1e-12)
    unit_t = torch.tensor(unit, dtype=clone.W_E.dtype)
    with torch.no_grad():
        projection = clone.W_E @ unit_t  # [d_vocab_in]
        clone.W_E.sub_(torch.outer(projection, unit_t))
    return clone


def ablate_direction(
    model: GroupModel,
    group: FiniteGroup,
    direction: np.ndarray,
    *,
    subset: np.ndarray | None = None,
    n_controls: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """Zero-ablate one representational axis (``direction``, in ``d_model``
    space) out of ``W_E`` and measure the drop in behavioural accuracy in flips
    -- the change in the number of correctly answered pairs -- against
    norm-matched random-direction controls.

    This is embedding-input ablation, not residual-stream ablation: only
    ``W_E`` is edited (``W_pos`` is untouched), and a downstream write --
    through attention or the MLP -- can reintroduce a component along the same
    axis before the readout (see :func:`_ablated_model`). ``subset`` restricts
    the accuracy count to an unleaked held-out mask when the caller supplies
    one -- a boolean mask counts its ``True`` entries, an integer index array
    counts its length; by default every pair is counted. The random controls
    draw from a private ``numpy`` generator (never the global stream) and share
    the ablated axis's dimensionality, so the effect is read against a matched
    baseline rather than an absolute threshold. Effect sizes only, no verdict
    (within-model, two-track rule)."""
    baseline_mask = model_correct_mask(model, group)
    baseline = _accuracy_in_flips(baseline_mask, subset)
    n_scored = _subset_size(group.order * group.order, subset)

    ablated_mask = model_correct_mask(_ablated_model(model, direction), group)
    ablated = _accuracy_in_flips(ablated_mask, subset)

    rng = np.random.default_rng(seed)
    control_drops: list[int] = []
    for _ in range(n_controls):
        random_direction = np.asarray(rng.standard_normal(direction.shape[0]))
        control_mask = model_correct_mask(_ablated_model(model, random_direction), group)
        control_drops.append(baseline - _accuracy_in_flips(control_mask, subset))

    return {
        "n_scored_pairs": n_scored,
        "baseline_correct": baseline,
        "ablated_correct": ablated,
        "direction_drop_flips": baseline - ablated,
        "random_direction_drop_flips": {
            "mean": float(np.mean(control_drops)),
            "per_control": control_drops,
        },
    }


__all__ = [
    "MODES",
    "ablate_component",
    "ablate_direction",
    "ablate_neurons",
    "model_correct_mask",
    "patch_neurons",
]
