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
  the three is not a rung-5 claim. In ``ablation_mode="resample"`` the
  completeness draw and the causal-scrubbing draw use distinct seeds (``seed``
  and ``seed + 1``) so they are independent permutations, even though both
  estimate the same held-out quantity (outside neurons resampled) -- whether
  the two criteria should differ more deeply than "independent draw" is an
  open design question, not settled here.

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
  it reports fractions) asserts they sum to 1, preferring ratio-of-means. The
  record carries the same provenance block as I-34: analysis git commit,
  instrument-code hashes, the held-out subset's ``train_frac``/``split_seed``,
  and (when given) the checkpoint's sha256.

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
from .occupancy import (
    cayley_grid_tokens,
    isotypic_energies,
    neuron_activations,
    per_neuron_concentration,
)
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
    """Mean and bootstrap CI over ``values``.

    Rejects an empty ``values``: an empty input has no mean to report, and
    silently returning ``mean=nan, n=0`` produced a complete-looking record
    for a measurement that never happened. At ``n == 1`` the CI is a
    degenerate point (``ci_low == ci_high == mean``); callers that compare two
    such CIs must treat that comparison as undetermined, not a strict
    inequality (see :func:`replicate_across_architectures`).
    """
    if not values:
        raise ValueError("_mean_ci requires at least one value, got an empty sequence")
    arr = [float(v) for v in values]
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": float(len(arr))}
    low, high = stats.bootstrap_ci(arr)
    return {"mean": mean, "ci_low": low, "ci_high": high, "n": float(len(arr))}


def _validate_integral_indices(indices: Sequence[int]) -> list[int]:
    """Reject non-integral neuron indices instead of silently truncating them
    (``int(2.7) == 2`` would otherwise ablate the wrong neuron with no error)."""
    out: list[int] = []
    for value in indices:
        if isinstance(value, bool):
            raise ValueError(f"neuron index must be an integer, got bool {value!r}")
        if isinstance(value, (int, np.integer)):
            out.append(int(value))
            continue
        as_float = float(value)
        if not as_float.is_integer():
            raise ValueError(f"neuron index must be integral, got {value!r}")
        out.append(int(as_float))
    return out


# ---------------------------------------------------------------------------
# Coset-block -> neuron bridge: the only circuit derivation this module
# implements for feeding I-34 (see module docstring and measure_audit_run).
# ---------------------------------------------------------------------------


def coset_circuit_neurons(
    model: GroupModel,
    group: FiniteGroup,
    *,
    argument: str = "left",
    min_top_share: float = 0.5,
) -> list[int] | None:
    """The coset arm's target (``coset.coset_target``, I-17/I-18/I-19) turned
    into a neuron subset :func:`circuit_audit` can ablate: the neurons whose
    activation concentrates (I-09's :func:`per_neuron_concentration`) on the
    coset target's ``occupied_blocks`` (the ``Ind_H^G 1`` support), at least
    ``min_top_share`` of that neuron's own energy.

    This operationalises the "coset-circuit signature" ``plan.md``'s I-09 entry
    already names -- dense coset-indicator neurons over ``Ind_H^G 1``'s support
    -- as a concrete neuron set. It is new wiring built for the audit driver
    (2026-07-26), not a pre-existing instrument output: neither
    ``coset_target`` nor the mechanism probes (``probes.signed_cyclic_instrument``,
    ``probes.carry_digit_instrument``) themselves expose a neuron-level circuit
    -- they fit on the shared embedding or on read-position logits, never on
    ``mlp_post`` neuron indices. This bridge is therefore scoped to claims whose
    circuit *is* the coset/induced-representation subspace (C2's D32
    signed-cyclic/coset case study); it has no analogue for a circuit defined
    some other way (e.g. C3's carry-digit structure, which is not tied to any
    isotypic block anywhere in this codebase -- every C3 member is abelian, so
    ``coset_target`` returns ``None`` for it too, the same theorem as Q32).

    Returns ``None`` when the group has no coset target at all (``coset_target``
    returns ``None``: no nontrivial core-free subgroup exists, true of the Q32
    family and of every abelian group). Returns a (possibly empty) list of
    neuron indices otherwise; an empty result means no neuron cleared
    ``min_top_share`` on an occupied block, itself a measurement, not an error.
    """
    from .coset import coset_target

    target = coset_target(group)
    if target is None:
        return None
    activations = neuron_activations(model, group.order)
    energies = isotypic_energies(activations, group, argument=argument)
    top_share, top_block = per_neuron_concentration(energies)
    occupied = set(int(j) for j in target.occupied_blocks)
    selected = [
        m
        for m in range(energies.shape[0])
        if int(top_block[m]) in occupied
        and np.isfinite(top_share[m])
        and float(top_share[m]) >= min_top_share
    ]
    return selected


# ---------------------------------------------------------------------------
# I-34: circuit audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditRecord:
    """I-34 circuit-audit record: three faithfulness criteria on held-out pairs,
    each an effect size with a bootstrap CI, plus a resample (causal-scrubbing)
    check and the residual. Estimation-first: no verdict label, no threshold.

    ``seed`` is the base seed passed to :func:`circuit_audit`; it drives the
    resample ablation used wherever ``ablation_mode == "resample"`` and the
    causal-scrubbing draw (which always resamples, at ``seed + 1``,
    regardless of ``ablation_mode``). Recording it makes any resample-mode
    number in this record reproducible from the record alone."""

    order: int
    index: int
    n_held_out: int
    n_inside: int
    n_outside: int
    ablation_mode: str
    seed: int
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
            "seed": self.seed,
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

    ``seed`` drives every resample draw: the ``ablation_mode="resample"``
    completeness/minimality ablations use it directly, and the
    causal-scrubbing draw always uses ``seed + 1`` (regardless of
    ``ablation_mode``) so it is an independent permutation from the
    completeness draw rather than a second report of the identical
    computation -- in resample mode the two criteria still estimate the same
    held-out quantity (outside neurons resampled), just from independent
    draws; whether they should differ more deeply is left open.

    Every quantity carries a bootstrap CI over held-out rows.
    """
    d_mlp = model.d_mlp
    inside = np.asarray(sorted(set(_validate_integral_indices(inside_neurons))), dtype=np.int64)
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

    # The causal-scrubbing draw always resamples, so it must not reuse `seed`:
    # when ablation_mode == "resample" that would make the resampled outside
    # ablation below the exact same permutation as `outside_ablated`, reporting
    # one computation twice under two criterion names.
    scrubbing_seed = seed + 1

    outside_ablated = logits_for(_ablate_post(post, outside, ablation_mode, seed=seed))
    inside_ablated = logits_for(_ablate_post(post, inside, ablation_mode, seed=seed))
    resampled = logits_for(_ablate_post(post, outside, "resample", seed=scrubbing_seed))

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
        seed=seed,
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
            "resample_seed": scrubbing_seed,
            "note": (
                "held-out accuracy with the outside resample-ablated, drawn with "
                "seed + 1 -- an independent permutation from the completeness "
                "draw even in ablation_mode='resample', where both estimate the "
                "same held-out quantity"
            ),
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
    is reported as architecture-conditional, both distributions shown.

    At ``n < 2`` on either side the CI is a degenerate point, so "do the CIs
    overlap" collapses to a strict point inequality between two single
    samples -- not a claim about the architectures. ``architecture_conditional``
    is ``None`` (undetermined) rather than ``True``/``False`` in that case."""

    statistic: str
    transformer: dict[str, float]
    fc: dict[str, float]
    difference: float
    architecture_conditional: bool | None

    def to_record(self) -> dict[str, Any]:
        return {
            "instrument": "architecture_replication",
            "statistic": self.statistic,
            "transformer": self.transformer,
            "fc": self.fc,
            "difference_transformer_minus_fc": self.difference,
            "architecture_conditional": self.architecture_conditional,
            "note": (
                "architecture_conditional is descriptive (non-overlapping CIs), not "
                "a gate; None means undetermined (fewer than 2 values on one side, "
                "so the CI is a degenerate point)"
            ),
        }


def replicate_across_architectures(
    transformer_values: Sequence[float],
    fc_values: Sequence[float],
    *,
    statistic: str = "headline",
) -> ReplicationRecord:
    """I-36: report a headline scalar on both architectures with bootstrap CIs
    and their difference. Scoped in the core to the C2 case study.

    ``transformer_values``/``fc_values`` must be non-empty (``_mean_ci``
    raises otherwise). With fewer than 2 values on either side,
    ``architecture_conditional`` is ``None`` rather than a point-inequality
    verdict between two degenerate CIs."""
    transformer = _mean_ci(transformer_values)
    fc = _mean_ci(fc_values)
    conditional: bool | None
    if transformer["n"] < 2 or fc["n"] < 2:
        conditional = None
    else:
        disjoint = transformer["ci_high"] < fc["ci_low"] or fc["ci_high"] < transformer["ci_low"]
        conditional = bool(disjoint)
    return ReplicationRecord(
        statistic=statistic,
        transformer=transformer,
        fc=fc,
        difference=transformer["mean"] - fc["mean"],
        architecture_conditional=conditional,
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
    model: GroupModel,
    group: FiniteGroup,
    *,
    train_frac: float = 0.8,
    split_seed: int = 0,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """I-35: each component's signed contribution to the correct-answer logit
    *difference* (logit minus the vocabulary mean -- shift-invariant), as a
    fraction of the total, on the held-out subset.

    Asserts the fractions sum to 1 (ratio-of-means) and that a constant logit
    shift leaves the correct-answer logit difference unchanged. Never divides by
    ``|total|`` (the ported bug). Carries the same provenance block as I-34:
    analysis git commit, instrument-code hashes, the held-out subset's
    ``train_frac``/``split_seed``, and (when given) the checkpoint's sha256."""
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

    # Self-test, not a faithfulness check: `correct_diff` subtracts the
    # row mean, so it is shift-invariant by construction for ANY logit
    # matrix. This only catches a future edit to `correct_diff` that breaks
    # that construction -- it cannot detect anything about the model or the
    # component decomposition, and is kept purely as a regression guard on
    # the helper above.
    shifted = total + 7.5
    if not np.allclose(correct_diff(shifted), total_diff):
        raise ValueError("logit-difference attribution is not shift-invariant")

    per_component: dict[str, dict[str, float]] = {}
    fraction_sum = 0.0
    max_abs_fraction = 0.0
    for name, matrix in components.items():
        diff = correct_diff(matrix)
        mean_contrib = float(diff.mean())
        fraction = mean_contrib / mean_total if abs(mean_total) > 1e-12 else float("nan")
        if np.isfinite(fraction):
            fraction_sum += fraction
            max_abs_fraction = max(max_abs_fraction, abs(fraction))
        per_component[name] = {"mean_contribution": mean_contrib, "fraction": fraction}

    # The fraction-sum tolerance scales with the conditioning of the
    # decomposition: fraction_i == c_i / mean_total, so components that
    # nearly cancel against the total (|c_i| >> |mean_total|) amplify
    # ordinary float64 rounding by that same ratio. A fixed 1e-6 tolerance
    # rejects a legitimate model whenever `max_abs_fraction` is large; scaling
    # by it keeps the check meaningful at both good and bad conditioning.
    tolerance = 1e-6 * max(1.0, max_abs_fraction)
    if abs(mean_total) > 1e-12 and abs(fraction_sum - 1.0) > tolerance:
        raise ValueError(
            f"component fractions sum to {fraction_sum}, expected 1 (tolerance {tolerance:.3g})"
        )

    provenance: dict[str, Any] = {
        "analysis_git_commit": get_git_commit(),
        "instrument_code_sha256": instrument_code_hashes(),
        "train_frac": train_frac,
        "split_seed": split_seed,
    }
    if checkpoint_path is not None:
        provenance["checkpoint_sha256"] = file_sha256(checkpoint_path)

    return {
        "instrument": "direct_logit_attribution",
        "group": {"order": group.order, "index": group.index},
        "n_held_out": int(rows.size),
        "mean_correct_logit_difference": mean_total,
        "components": per_component,
        "note": "contributions to the correct-answer logit difference (shift-invariant); "
        "fractions are ratio-of-means and sum to 1 within a conditioning-scaled tolerance",
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# Run-directory record builder (offline, dip-aware checkpoint selection):
# I-35 (always) + I-34 (only where this module's coset-block bridge applies).
# ---------------------------------------------------------------------------


def measure_audit_run(
    run_dir: Path,
    *,
    metric: str = "val/accuracy",
    threshold: float = 0.99,
    ablation_mode: str = "zero",
    seed: int = 0,
    min_top_share: float = 0.5,
) -> dict[str, Any]:
    """The audit record for one run: I-35 direct logit attribution (always,
    every architecture supports it) plus I-34 circuit audit wherever this
    module can derive a neuron circuit for the group -- currently only via
    :func:`coset_circuit_neurons` (the D32/QD32-style coset case study).

    Behind the same dip-aware checkpoint rule the other instruments use
    (``checkpoints.select_checkpoint``): a run with no stable checkpoint (a
    censored/never-grokked seed) returns ``status: "skipped"`` with the
    selection record, never a silently-analysed unstable model.

    A measured run always carries ``direct_logit_attribution``. Its
    ``circuit_audit`` sub-record is ``status: "skipped"`` with a reason when
    the group has no coset target (every abelian group -- including every C3
    member -- and the Q32 family; see :func:`coset_circuit_neurons`), and
    ``status: "measured"`` with the full I-34 :class:`AuditRecord` otherwise,
    alongside the derived circuit's provenance (the coset subgroup/occupied
    blocks and the ``min_top_share`` threshold used to select neurons). This
    sub-skip is expected and does not by itself mean the run was skipped for
    checkpoint reasons -- callers should read the top-level ``status`` for
    that.

    Provenance mirrors ``coset.measure_coset_run``: manifest hashes, the
    analysis-time git commit, this module's sha256, the checkpoint's sha256,
    and the analysed group artifact's repo-relative path and sha256.
    """
    import yaml

    from ..config import validate_config
    from ..groups.catalog import resolve_group
    from ..groups.data import artifact_path
    from ..manifest import read_manifest
    from ..training.trainer import build_model
    from .checkpoints import select_checkpoint

    manifest = read_manifest(run_dir)
    config = validate_config(yaml.safe_load((run_dir / "resolved_config.yaml").read_text()))
    selection = select_checkpoint(run_dir, metric=metric, threshold=threshold)
    art_path = artifact_path(config.data.group.order, config.data.group.index)
    repo_root = Path(__file__).resolve().parents[3]
    try:
        art_rel = str(art_path.relative_to(repo_root))
    except ValueError:
        art_rel = art_path.name
    record: dict[str, Any] = {
        "instrument": "audit",
        "run_id": manifest.get("run_id", run_dir.name),
        "seed": config.seed,
        "group": {
            "order": config.data.group.order,
            "index": config.data.group.index,
            "name": config.data.group.canonical_name,
        },
        "model": {
            "arch": config.model.arch,
            "d_model": config.model.d_model,
            "d_mlp": config.model.d_mlp,
            "activation": config.model.activation,
        },
        "checkpoint_selection": selection.to_record(),
        "provenance": {
            "git_commit": manifest.get("provenance", {}).get("git_commit"),
            "config_hash": manifest.get("provenance", {}).get("config_hash"),
            "config_group_hash": manifest.get("provenance", {}).get("config_group_hash"),
            "campaign_id": manifest.get("provenance", {}).get("campaign_id"),
            "dataset_spec_hash": manifest.get("dataset", {}).get("spec_hash"),
            "analysis_git_commit": get_git_commit(),
            "instrument_code_sha256": instrument_code_hashes(),
            "group_artifact": art_rel,
            "group_artifact_sha256": (file_sha256(art_path) if art_path.is_file() else None),
        },
    }
    if selection.path is None:
        record["status"] = "skipped"
        return record

    group = resolve_group(config.data.group)
    checkpoint = torch.load(selection.path, map_location="cpu", weights_only=False)
    model = build_model(config, group)
    model.load_state_dict(checkpoint["model_state_dict"])

    record["status"] = "measured"
    record["provenance"]["checkpoint_sha256"] = file_sha256(selection.path)

    record["direct_logit_attribution"] = direct_logit_attribution(
        model,
        group,
        train_frac=config.data.train_frac,
        split_seed=config.effective_split_seed,
        checkpoint_path=selection.path,
    )

    inside = coset_circuit_neurons(model, group, min_top_share=min_top_share)
    if inside is None:
        record["circuit_audit"] = {
            "status": "skipped",
            "reason": (
                "no coset target for this group (coset.coset_target returns "
                "None: no nontrivial core-free subgroup exists -- true of every "
                "abelian group, including every C3 member, and of the Q32 "
                "family). This driver's only implemented circuit-derivation "
                "bridge is the coset-block one (coset_circuit_neurons); a "
                "carry-digit or other non-coset circuit definition is not "
                "implemented, so I-34 cannot be produced for this run without "
                "one -- see coset_circuit_neurons' docstring."
            ),
        }
    else:
        from .coset import coset_target

        target = coset_target(group)  # not None: coset_circuit_neurons already checked
        assert target is not None
        audit_record = circuit_audit(
            model,
            group,
            inside,
            train_frac=config.data.train_frac,
            split_seed=config.effective_split_seed,
            ablation_mode=ablation_mode,
            seed=seed,
            checkpoint_path=selection.path,
        )
        record["circuit_audit"] = {
            "status": "measured",
            "circuit_source": "coset_occupied_blocks",
            "coset_subgroup_index": target.subgroup_index,
            "coset_occupied_blocks": list(target.occupied_blocks),
            "min_top_share": min_top_share,
            **audit_record.to_record(),
        }
    return record


__all__ = [
    "AuditRecord",
    "ReplicationRecord",
    "circuit_audit",
    "coset_circuit_neurons",
    "direct_logit_attribution",
    "held_out_rows",
    "kl_divergence",
    "measure_audit_run",
    "measure_over_models",
    "replicate_across_architectures",
]
