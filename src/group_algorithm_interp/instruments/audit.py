"""Audit instruments: the rung-5 circuit audit (I-34), the architecture-confound
replication (I-36), and the rebuilt direct logit attribution (I-35).

* **I-34 -- circuit audit.** For any rung-5 claim (C5's twisted rule, C3's carry,
  the D32 signed-cyclic circuit), report the three faithfulness criteria on
  held-out pairs, each as an effect size with a bootstrap CI (estimation-first,
  no threshold gates): *faithfulness* (KL between the model and the circuit
  output, not just argmax agreement), *completeness* (ablating outside the
  circuit does not hurt), *minimality* (ablating inside the circuit does hurt),
  plus a resample-ablation (causal-scrubbing) check. The residual and the
  alternative explanations travel with the record; a circuit that fails any of
  the three is not a rung-5 claim.

* **I-36 -- architecture-confound replication.** A headline result re-run on the
  ``FCModel`` as well as the ``OneLayerTransformer`` (they share the ``W_E``/``W_U``
  contract and the ``logits[:, -1]`` read position, so every instrument accepts
  either). Both are reported with CIs; a result on one architecture only is an
  architecture-conditional result and is labelled as one. Scoped in the core to
  the C2 case study.

* **I-35 -- direct logit attribution (rebuilt).** The signed contribution of each
  component (direct ``W_E -> W_U`` path, attention, MLP) to the correct-answer
  logit. The ported version was broken (it divided signed contributions by
  ``|total|``, so the fractions summed to -1 whenever the correct logit was
  negative). Softmax is shift-invariant, so this reports contributions to logit
  *differences*, asserts a constant shift leaves the output unchanged, and (when
  it reports fractions) asserts they sum to 1, preferring ratio-of-means.

Everything is offline and deterministic: ablation recomputes logits from the
one forward pass' components (both architectures satisfy
``logits = resid_final @ W_U``), so no training-time hooks and no re-forward are
needed. Held-out means the transpose-unleaked test subset (rule 1: the only
admissible generalisation set).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .. import stats
from ..groups.group import FiniteGroup
from ..manifest import get_git_commit
from ..model import FCModel, GroupModel, OneLayerTransformer
from ..task import build_group_task, train_test_split, transpose_leaked_mask
from .occupancy import cayley_grid_tokens
from .report import file_sha256, instrument_code_hashes

MeasureFn = Callable[[GroupModel, FiniteGroup], float]

_ABLATION_MODES = ("zero", "mean", "resample")


# ---------------------------------------------------------------------------
# Held-out subset and per-forward component access
# ---------------------------------------------------------------------------


def held_out_rows(
    group: FiniteGroup, *, train_frac: float = 0.8, split_seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Grid-row indices of the transpose-unleaked held-out test pairs, and their
    targets. Rows index the Cayley grid ``k = a * |G| + b``.

    Rule 1: generalisation is measured on the unleaked test subset and nowhere
    else. A pair ``(a, b)`` is leaked when its transpose is in train and it
    commutes; those are dropped.
    """
    n = group.order
    task = build_group_task(group)
    split = train_test_split(task, train_frac, split_seed)
    leaked = transpose_leaked_mask(split, group.cayley_table)
    kept = ~leaked
    a = split.test_inputs[kept, 0].astype(np.int64)
    b = split.test_inputs[kept, 1].astype(np.int64)
    rows = a * n + b
    targets = split.test_targets[kept].astype(np.int64)
    return rows, targets


def _components(
    model: GroupModel, order: int
) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """The read-position MLP post-activations over the whole grid and a closure
    mapping a (possibly ablated) post matrix to logits.

    Both architectures satisfy ``logits = resid_final @ W_U`` with
    ``resid_final = base + mlp_post @ W_out`` (transformer) or
    ``resid_final = mlp_post`` (FC), so a modified ``mlp_post`` recomputes logits
    exactly with no re-forward.
    """
    tokens = cayley_grid_tokens(order)
    model.eval()
    with torch.no_grad():
        cache = model(tokens, return_cache=True)
    post = cache["mlp_post"]
    if post is None:
        raise ValueError("model has no MLP; the circuit audit needs neurons to ablate")
    post_read = post[:, -1, :].detach().to(torch.float64)
    unembed = model.W_U.detach().to(torch.float64)
    if isinstance(model, OneLayerTransformer):
        resid = cache["resid_final"][:, -1, :].detach().to(torch.float64)
        w_out = model.W_out.detach().to(torch.float64)
        base = resid - post_read @ w_out

        def to_logits(post_mod: torch.Tensor) -> torch.Tensor:
            return (base + post_mod @ w_out) @ unembed

    elif isinstance(model, FCModel):

        def to_logits(post_mod: torch.Tensor) -> torch.Tensor:
            return post_mod @ unembed

    else:  # pragma: no cover - defensive
        raise TypeError(f"unsupported model type {type(model)!r}")

    return post_read, to_logits


def _ablate_post(post: torch.Tensor, columns: np.ndarray, mode: str, *, seed: int) -> torch.Tensor:
    """Return a copy of ``post`` with the given neuron ``columns`` ablated."""
    if mode not in _ABLATION_MODES:
        raise ValueError(f"mode must be one of {_ABLATION_MODES}, got {mode!r}")
    out = post.clone()
    if columns.size == 0:
        return out
    cols = torch.as_tensor(columns, dtype=torch.long)
    if mode == "zero":
        out[:, cols] = 0.0
    elif mode == "mean":
        out[:, cols] = post[:, cols].mean(dim=0, keepdim=True)
    else:  # resample: permute each column's values across pairs
        generator = torch.Generator().manual_seed(seed)
        for c in cols.tolist():
            perm = torch.randperm(post.shape[0], generator=generator)
            out[:, c] = post[perm, c]
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def kl_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> np.ndarray:
    """Per-row ``KL(P || Q)`` between two logit matrices (softmaxed first)."""
    p = _softmax(np.asarray(p_logits, dtype=np.float64))
    q = _softmax(np.asarray(q_logits, dtype=np.float64))
    eps = 1e-12
    return (p * (np.log(p + eps) - np.log(q + eps))).sum(axis=1)


def _mean_ci(values: Sequence[float]) -> dict[str, float]:
    arr = [float(v) for v in values]
    mean = float(np.mean(arr)) if arr else float("nan")
    if len(arr) < 2:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": float(len(arr))}
    low, high = stats.bootstrap_ci(arr)
    return {"mean": mean, "ci_low": low, "ci_high": high, "n": float(len(arr))}


# ---------------------------------------------------------------------------
# I-34: circuit audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditRecord:
    """I-34 circuit-audit record: three faithfulness criteria on held-out pairs,
    each an effect size with a bootstrap CI, plus a resample (causal-scrubbing)
    check and the residual. Estimation-first: no verdict label, no threshold."""

    order: int
    index: int
    n_held_out: int
    n_inside: int
    n_outside: int
    ablation_mode: str
    baseline_accuracy: dict[str, float]
    completeness: dict[str, Any]
    minimality: dict[str, Any]
    faithfulness: dict[str, Any]
    causal_scrubbing: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)
    alternatives: str = (
        "A completeness/minimality separation is consistent with the claimed "
        "circuit but does not exclude a distributed alternative that overlaps "
        "the inside set; report the residual KL and the ablation deltas, never a "
        "pass/fail label."
    )

    def to_record(self) -> dict[str, Any]:
        return {
            "instrument": "circuit_audit",
            "group": {"order": self.order, "index": self.index},
            "n_held_out": self.n_held_out,
            "n_inside": self.n_inside,
            "n_outside": self.n_outside,
            "ablation_mode": self.ablation_mode,
            "baseline_accuracy": self.baseline_accuracy,
            "completeness": self.completeness,
            "minimality": self.minimality,
            "faithfulness": self.faithfulness,
            "causal_scrubbing": self.causal_scrubbing,
            "alternatives": self.alternatives,
            "provenance": self.provenance,
        }


def circuit_audit(
    model: GroupModel,
    group: FiniteGroup,
    inside_neurons: Sequence[int],
    *,
    train_frac: float = 0.8,
    split_seed: int = 0,
    ablation_mode: str = "zero",
    seed: int = 0,
    checkpoint_path: Path | None = None,
) -> AuditRecord:
    """I-34: audit a claimed circuit (the ``inside_neurons``) on the held-out set.

    * *completeness* -- ablate the OUTSIDE neurons and report the held-out
      accuracy drop (small when the circuit is complete);
    * *minimality* -- ablate the INSIDE neurons and report the drop (large when
      the circuit is load-bearing);
    * *faithfulness* -- the KL from the full model to the circuit-only output
      (outside ablated) on held-out pairs, with argmax agreement;
    * *causal scrubbing* -- resample-ablate the outside and report the held-out
      accuracy (behaviour preserved under resampling of the non-circuit part).

    Every quantity carries a bootstrap CI over held-out rows.
    """
    d_mlp = model.d_mlp
    inside = np.asarray(sorted(set(int(i) for i in inside_neurons)), dtype=np.int64)
    if inside.size and (inside.min() < 0 or inside.max() >= d_mlp):
        raise ValueError(f"inside_neurons out of range for d_mlp={d_mlp}")
    outside = np.array([i for i in range(d_mlp) if i not in set(inside.tolist())], dtype=np.int64)

    rows, targets = held_out_rows(group, train_frac=train_frac, split_seed=split_seed)
    if rows.size < 2:
        raise ValueError("held-out subset too small to audit (need >= 2 unleaked test pairs)")

    post, to_logits = _components(model, group.order)

    def logits_for(post_mod: torch.Tensor) -> np.ndarray:
        return to_logits(post_mod).cpu().numpy()[rows]

    def correct_vector(logits: np.ndarray) -> np.ndarray:
        return (logits.argmax(axis=1) == targets).astype(np.float64)

    baseline_logits = logits_for(post)
    baseline_correct = correct_vector(baseline_logits)

    outside_ablated = logits_for(_ablate_post(post, outside, ablation_mode, seed=seed))
    inside_ablated = logits_for(_ablate_post(post, inside, ablation_mode, seed=seed))
    resampled = logits_for(_ablate_post(post, outside, "resample", seed=seed))

    completeness_drop = baseline_correct - correct_vector(outside_ablated)
    minimality_drop = baseline_correct - correct_vector(inside_ablated)
    faithfulness_kl = kl_divergence(baseline_logits, outside_ablated)
    faithfulness_agree = (baseline_logits.argmax(axis=1) == outside_ablated.argmax(axis=1)).astype(
        np.float64
    )

    provenance: dict[str, Any] = {
        "analysis_git_commit": get_git_commit(),
        "instrument_code_sha256": instrument_code_hashes(),
        "train_frac": train_frac,
        "split_seed": split_seed,
    }
    if checkpoint_path is not None:
        provenance["checkpoint_sha256"] = file_sha256(checkpoint_path)

    return AuditRecord(
        order=group.order,
        index=group.index,
        n_held_out=int(rows.size),
        n_inside=int(inside.size),
        n_outside=int(outside.size),
        ablation_mode=ablation_mode,
        baseline_accuracy=_mean_ci(baseline_correct.tolist()),
        completeness={
            "accuracy_drop": _mean_ci(completeness_drop.tolist()),
            "note": "small drop -> ablating outside the circuit does not hurt (complete)",
        },
        minimality={
            "accuracy_drop": _mean_ci(minimality_drop.tolist()),
            "note": "large drop -> ablating inside the circuit hurts (load-bearing)",
        },
        faithfulness={
            "kl_model_to_circuit": _mean_ci(faithfulness_kl.tolist()),
            "argmax_agreement": _mean_ci(faithfulness_agree.tolist()),
            "note": "KL from the full model to the circuit-only (outside-ablated) output",
        },
        causal_scrubbing={
            "accuracy_resample_outside": _mean_ci(correct_vector(resampled).tolist()),
            "note": "held-out accuracy with the outside resample-ablated",
        },
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# I-36: architecture-confound replication
# ---------------------------------------------------------------------------


def measure_over_models(
    measure: MeasureFn, models: Sequence[GroupModel], group: FiniteGroup
) -> list[float]:
    """Run a scalar measurement over a list of same-group models (seeds)."""
    return [float(measure(model, group)) for model in models]


@dataclass(frozen=True)
class ReplicationRecord:
    """I-36: one headline scalar measured on both architectures.

    ``architecture_conditional`` is a *descriptive* flag (the two CIs do not
    overlap), never a threshold gate: a result present on one architecture only
    is reported as architecture-conditional, both distributions shown."""

    statistic: str
    transformer: dict[str, float]
    fc: dict[str, float]
    difference: float
    architecture_conditional: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "instrument": "architecture_replication",
            "statistic": self.statistic,
            "transformer": self.transformer,
            "fc": self.fc,
            "difference_transformer_minus_fc": self.difference,
            "architecture_conditional": self.architecture_conditional,
            "note": "architecture_conditional is descriptive (non-overlapping CIs), not a gate",
        }


def replicate_across_architectures(
    transformer_values: Sequence[float],
    fc_values: Sequence[float],
    *,
    statistic: str = "headline",
) -> ReplicationRecord:
    """I-36: report a headline scalar on both architectures with bootstrap CIs
    and their difference. Scoped in the core to the C2 case study."""
    transformer = _mean_ci(transformer_values)
    fc = _mean_ci(fc_values)
    disjoint = transformer["ci_high"] < fc["ci_low"] or fc["ci_high"] < transformer["ci_low"]
    return ReplicationRecord(
        statistic=statistic,
        transformer=transformer,
        fc=fc,
        difference=transformer["mean"] - fc["mean"],
        architecture_conditional=bool(disjoint),
    )


# ---------------------------------------------------------------------------
# I-35: direct logit attribution (rebuilt, shift-invariant)
# ---------------------------------------------------------------------------


def _component_logits(model: GroupModel, order: int) -> dict[str, np.ndarray]:
    """Read-position per-component logit contributions over the grid.

    Transformer: direct (``embed``), attention (``attn_out``), MLP
    (``resid_final - embed - attn_out``). FC: a single ``mlp`` path. Each
    component's contribution is ``component @ W_U``; they sum to the logits."""
    tokens = cayley_grid_tokens(order)
    model.eval()
    with torch.no_grad():
        cache = model(tokens, return_cache=True)
    unembed = model.W_U.detach().to(torch.float64)

    def contrib(x: torch.Tensor) -> np.ndarray:
        return (x[:, -1, :].detach().to(torch.float64) @ unembed).cpu().numpy()

    if isinstance(model, OneLayerTransformer):
        embed = cache["embed"]
        attn_out = cache["attn_out"]
        resid = cache["resid_final"]
        mlp = resid - embed - attn_out
        return {"direct": contrib(embed), "attention": contrib(attn_out), "mlp": contrib(mlp)}
    return {"mlp": contrib(cache["resid_final"])}


def direct_logit_attribution(
    model: GroupModel, group: FiniteGroup, *, train_frac: float = 0.8, split_seed: int = 0
) -> dict[str, Any]:
    """I-35: each component's signed contribution to the correct-answer logit
    *difference* (logit minus the vocabulary mean -- shift-invariant), as a
    fraction of the total, on the held-out subset.

    Asserts the fractions sum to 1 (ratio-of-means) and that a constant logit
    shift leaves the correct-answer logit difference unchanged. Never divides by
    ``|total|`` (the ported bug)."""
    rows, targets = held_out_rows(group, train_frac=train_frac, split_seed=split_seed)
    if rows.size < 2:
        raise ValueError("held-out subset too small for DLA")
    components = _component_logits(model, group.order)
    idx = np.arange(rows.size)

    def correct_diff(logit_matrix: np.ndarray) -> np.ndarray:
        sub = logit_matrix[rows]
        return sub[idx, targets] - sub.mean(axis=1)

    total: np.ndarray = np.add.reduce(list(components.values()))
    total_diff = correct_diff(total)
    mean_total = float(total_diff.mean())

    # Shift invariance: adding a constant to every logit leaves the difference
    # unchanged (the mean shifts by the same constant).
    shifted = total + 7.5
    if not np.allclose(correct_diff(shifted), total_diff):
        raise ValueError("logit-difference attribution is not shift-invariant")

    per_component: dict[str, dict[str, float]] = {}
    fraction_sum = 0.0
    for name, matrix in components.items():
        diff = correct_diff(matrix)
        mean_contrib = float(diff.mean())
        fraction = mean_contrib / mean_total if abs(mean_total) > 1e-12 else float("nan")
        if np.isfinite(fraction):
            fraction_sum += fraction
        per_component[name] = {"mean_contribution": mean_contrib, "fraction": fraction}

    if abs(mean_total) > 1e-12 and abs(fraction_sum - 1.0) > 1e-6:
        raise ValueError(f"component fractions sum to {fraction_sum}, expected 1")

    return {
        "instrument": "direct_logit_attribution",
        "group": {"order": group.order, "index": group.index},
        "n_held_out": int(rows.size),
        "mean_correct_logit_difference": mean_total,
        "components": per_component,
        "note": "contributions to the correct-answer logit difference (shift-invariant); "
        "fractions are ratio-of-means and sum to 1",
    }


__all__ = [
    "AuditRecord",
    "ReplicationRecord",
    "circuit_audit",
    "direct_logit_attribution",
    "held_out_rows",
    "kl_divergence",
    "measure_over_models",
    "replicate_across_architectures",
]
