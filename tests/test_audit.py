"""Audit instruments (instruments/audit.py): the circuit audit (I-34), the
architecture-confound replication (I-36), and direct logit attribution (I-35).

The circuit-audit mechanics are pinned decisively without depending on any real
circuit: with the whole MLP declared the circuit (``inside = all neurons``),
ablating the (empty) outside cannot hurt and the circuit-only output equals the
model exactly (completeness drop == 0, faithfulness KL == 0), while ablating the
inside on a model that has learned the task does hurt (minimality drop > 0). A
small S3 model trained to accuracy 1.0 supplies the load-bearing case.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from group_algorithm_interp.config import ProjectConfig
from group_algorithm_interp.groups.catalog import resolve_group
from group_algorithm_interp.instruments import audit as A
from group_algorithm_interp.instruments.occupancy import cayley_grid_tokens
from group_algorithm_interp.seed import set_seed
from group_algorithm_interp.training.trainer import build_model


def _config(order: int, index: int, arch: str = "transformer") -> ProjectConfig:
    return ProjectConfig(
        device="cpu",
        data={"group": {"order": order, "index": index}},
        model={"arch": arch, "d_model": 32, "d_mlp": 64, "n_heads": 4},
        logging={"mode": "disabled"},
    )


def _trained_model(name: str, order: int, index: int, arch: str, epochs: int = 1500):
    """A small model trained on the full Cayley table to (typically) accuracy 1.0.

    Full-table training is deliberate: it is a deterministic, fast fixture for
    exercising the audit instrument's mechanics, not a research measurement."""
    group = resolve_group(name)
    set_seed(0, deterministic=False)
    model = build_model(_config(order, index, arch), group)
    tokens = cayley_grid_tokens(order)
    targets = torch.as_tensor(group.cayley_table.reshape(-1), dtype=torch.long)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1.0)
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = F.cross_entropy(model(tokens)[:, -1, :], targets)
        loss.backward()
        optimiser.step()
    return model, group


@pytest.fixture(scope="module")
def s3_transformer():
    return _trained_model("S3", 6, 1, "transformer")


@pytest.fixture(scope="module")
def s3_fc():
    return _trained_model("S3", 6, 1, "fc")


@pytest.fixture(scope="module")
def d8_transformer():
    """D8 has core-free subgroups (``coset_target`` is not ``None``), the same
    split as D32 in the C2 case study -- the group :func:`A.coset_circuit_neurons`
    is defined on."""
    return _trained_model("D8", 8, 3, "transformer")


@pytest.fixture(scope="module")
def q8_transformer():
    """Q8's only core-free subgroup is trivial (``coset_target`` is ``None``),
    the same theorem as Q32 -- the negative control for the coset-block bridge."""
    return _trained_model("Q8", 8, 4, "transformer")


@pytest.fixture(scope="module")
def c8_transformer():
    """C8 is abelian with a genuine polycyclic carry structure (radices
    (2, 2, 2), triangular ripple carry) -- the small stand-in for the C128
    member :func:`A.carry_circuit_neurons` is defined on. ``coset_target`` is
    ``None`` (abelian), so only the carry bridge applies."""
    return _trained_model("C8", 8, 1, "transformer")


# ---------------------------------------------------------------------------
# Held-out subset and KL primitives
# ---------------------------------------------------------------------------


def test_held_out_rows_are_the_unleaked_test_subset():
    group = resolve_group("S3")
    rows, targets = A.held_out_rows(group)
    assert rows.size == targets.size
    assert rows.size >= 2
    # Rows index the Cayley grid and their targets match the table.
    a, b = rows // group.order, rows % group.order
    assert np.array_equal(group.cayley_table[a, b], targets)


def test_kl_divergence_hand_values():
    logits = np.array([[2.0, 0.0], [0.0, 3.0]])
    assert np.allclose(A.kl_divergence(logits, logits), 0.0)
    # KL is non-negative and positive between different distributions.
    other = np.array([[0.0, 2.0], [3.0, 0.0]])
    assert (A.kl_divergence(logits, other) > 0.0).all()


def test_ablate_post_mean_mode_hand_values():
    """Mean-mode semantics, exactly: an ablated column collapses to the mean of
    its own original values, unlike zero-mode (collapses to 0)."""
    post = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    columns = np.array([0])
    ablated = A._ablate_post(post, columns, "mean", seed=0)
    assert torch.allclose(ablated[:, 0], torch.full((3,), 2.0))  # mean of [1, 2, 3]
    assert torch.allclose(ablated[:, 1], post[:, 1])  # untouched column
    zeroed = A._ablate_post(post, columns, "zero", seed=0)
    assert torch.allclose(zeroed[:, 0], torch.zeros(3))


def test_mean_ci_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one value"):
        A._mean_ci([])


# ---------------------------------------------------------------------------
# I-34: circuit audit
# ---------------------------------------------------------------------------


def test_full_mlp_circuit_is_complete_and_faithful(s3_transformer):
    """inside = all neurons, outside = empty: ablating the outside cannot change
    anything, so completeness drop == 0, faithfulness KL == 0, and argmax
    agreement == 1.0."""
    model, group = s3_transformer
    record = A.circuit_audit(model, group, inside_neurons=list(range(model.d_mlp)))
    assert record.baseline_accuracy["mean"] > 0.0
    assert record.completeness["accuracy_drop"]["mean"] == pytest.approx(0.0)
    assert record.faithfulness["kl_model_to_circuit"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert record.faithfulness["argmax_agreement"]["mean"] == pytest.approx(1.0)


def test_ablating_the_full_circuit_hurts_minimality(s3_transformer):
    """A model that has learned the task loses accuracy when its whole MLP is
    ablated -- the circuit is load-bearing (minimality drop > 0)."""
    model, group = s3_transformer
    record = A.circuit_audit(model, group, inside_neurons=list(range(model.d_mlp)))
    assert record.minimality["accuracy_drop"]["mean"] > 0.0


def test_empty_circuit_is_incomplete(s3_transformer):
    """inside = empty, outside = all: ablating the (empty) inside cannot hurt
    (minimality drop == 0), while ablating everything outside does."""
    model, group = s3_transformer
    record = A.circuit_audit(model, group, inside_neurons=[])
    assert record.minimality["accuracy_drop"]["mean"] == pytest.approx(0.0)
    assert record.completeness["accuracy_drop"]["mean"] > 0.0


@pytest.mark.parametrize("mode", ["zero", "mean", "resample"])
def test_ablation_modes_are_accepted(s3_transformer, mode):
    model, group = s3_transformer
    record = A.circuit_audit(model, group, inside_neurons=[0, 1, 2], ablation_mode=mode)
    assert record.ablation_mode == mode
    assert record.causal_scrubbing["accuracy_resample_outside"]["n"] == record.n_held_out


def test_circuit_audit_validates_inputs(s3_transformer):
    model, group = s3_transformer
    with pytest.raises(ValueError, match="out of range"):
        A.circuit_audit(model, group, inside_neurons=[model.d_mlp])
    with pytest.raises(ValueError, match="mode must be"):
        A.circuit_audit(model, group, inside_neurons=[0], ablation_mode="scale")


def test_circuit_audit_rejects_non_integral_neuron_indices(s3_transformer):
    """``2.7`` used to be silently truncated to neuron 2 via ``int(2.7)``; it
    must now be rejected rather than ablating the wrong neuron with no error.
    A whole-valued float (``2.0``) is still accepted."""
    model, group = s3_transformer
    with pytest.raises(ValueError, match="integral"):
        A.circuit_audit(model, group, inside_neurons=[0, 2.7])
    record = A.circuit_audit(model, group, inside_neurons=[0, 2.0])
    assert record.n_inside == 2


def test_circuit_audit_records_seed_and_ablation_mode(s3_transformer):
    model, group = s3_transformer
    record = A.circuit_audit(
        model, group, inside_neurons=[0, 1, 2], ablation_mode="resample", seed=7
    ).to_record()
    assert record["seed"] == 7
    assert record["ablation_mode"] == "resample"
    assert record["causal_scrubbing"]["resample_seed"] == 8


def test_resample_mode_scrubbing_draw_is_independent_of_completeness_draw(s3_transformer):
    """In ``ablation_mode="resample"`` the completeness ablation (outside,
    ablation_mode's seed) and the causal-scrubbing draw (outside, seed + 1) must
    be independent permutations, not the same computation reported twice under
    two criterion names -- the historical bug made
    ``baseline - completeness_drop == scrubbing_accuracy`` exactly."""
    model, group = s3_transformer
    record = A.circuit_audit(
        model, group, inside_neurons=[0, 1, 2], ablation_mode="resample", seed=3
    )
    baseline = record.baseline_accuracy["mean"]
    completeness_drop = record.completeness["accuracy_drop"]["mean"]
    scrubbing_accuracy = record.causal_scrubbing["accuracy_resample_outside"]["mean"]
    assert (baseline - completeness_drop) != pytest.approx(scrubbing_accuracy)


def test_resample_mode_is_deterministic_given_the_same_seed(s3_transformer):
    """Same seed -> identical record numbers, so resample-mode results are
    reproducible from the recorded ``seed`` alone."""
    model, group = s3_transformer
    record_a = A.circuit_audit(
        model, group, inside_neurons=[0, 1, 2], ablation_mode="resample", seed=5
    ).to_record()
    record_b = A.circuit_audit(
        model, group, inside_neurons=[0, 1, 2], ablation_mode="resample", seed=5
    ).to_record()
    assert record_a["completeness"] == record_b["completeness"]
    assert record_a["minimality"] == record_b["minimality"]
    assert record_a["causal_scrubbing"] == record_b["causal_scrubbing"]


def test_circuit_audit_rejects_too_small_a_held_out_set():
    """An abelian group's test subset is almost entirely transpose-leaked, so the
    unleaked held-out set can be empty -- the audit refuses rather than reporting
    a number off two pairs."""
    set_seed(0, deterministic=False)
    group = resolve_group("C4")
    model = build_model(_config(4, 1), group)
    with pytest.raises(ValueError, match="too small"):
        A.circuit_audit(model, group, inside_neurons=[0])


def test_circuit_audit_record_carries_provenance(s3_transformer, tmp_path):
    model, group = s3_transformer
    checkpoint = tmp_path / "final.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    record = A.circuit_audit(
        model, group, inside_neurons=[0, 1], checkpoint_path=checkpoint
    ).to_record()
    assert record["instrument"] == "circuit_audit"
    assert "instrument_code_sha256" in record["provenance"]
    assert "checkpoint_sha256" in record["provenance"]
    assert "alternatives" in record


def test_circuit_audit_accepts_the_fc_architecture(s3_fc):
    model, group = s3_fc
    record = A.circuit_audit(model, group, inside_neurons=list(range(model.d_mlp)))
    assert record.baseline_accuracy["mean"] > 0.0
    assert record.faithfulness["kl_model_to_circuit"]["mean"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Coset-block -> neuron bridge (coset_circuit_neurons)
# ---------------------------------------------------------------------------


def test_coset_circuit_neurons_returns_none_without_a_coset_target(q8_transformer):
    """Q8's only core-free subgroup is trivial (the same theorem as Q32), so
    ``coset_target`` is ``None`` and the bridge must not fabricate a circuit."""
    model, group = q8_transformer
    assert A.coset_circuit_neurons(model, group) is None


def test_coset_circuit_neurons_selects_a_valid_neuron_subset_on_d8(d8_transformer):
    """D8 has a coset target, so the bridge returns an in-range neuron list (not
    ``None``) whose indices are valid ``inside_neurons`` for :func:`A.circuit_audit`."""
    model, group = d8_transformer
    inside = A.coset_circuit_neurons(model, group)
    assert inside is not None
    assert all(0 <= i < model.d_mlp for i in inside)
    assert len(set(inside)) == len(inside)  # no duplicates
    # The selection must be usable as circuit_audit's inside_neurons directly.
    record = A.circuit_audit(model, group, inside_neurons=inside)
    assert record.n_inside == len(inside)


def test_coset_circuit_neurons_selection_shrinks_as_the_threshold_rises(d8_transformer):
    """A neuron selected at a high ``min_top_share`` must also be selected at a
    lower one -- the threshold only ever removes candidates, never adds them."""
    model, group = d8_transformer
    loose = A.coset_circuit_neurons(model, group, min_top_share=0.0)
    strict = A.coset_circuit_neurons(model, group, min_top_share=0.9)
    assert loose is not None and strict is not None
    assert set(strict).issubset(set(loose))


# ---------------------------------------------------------------------------
# Carry-digit -> neuron bridge (carry_circuit_neurons): the causal
# digit-attribution circuit for the abelian C3 members.
# ---------------------------------------------------------------------------


def test_carry_circuit_neurons_returns_none_on_a_non_abelian_group(q8_transformer):
    """Q8 is non-abelian: ``polycyclic_digits`` DOES return a chain under this
    artifact's enumeration, but that chain is an enumeration-dependent artefact
    its own docstring warns against reading as structure, so the carry bridge
    must decline (return ``None``) rather than audit an artefact -- keeping Q8
    on the skip path the same way ``coset_circuit_neurons`` does."""
    model, group = q8_transformer
    assert A.carry_circuit_neurons(model, group) is None


def test_carry_circuit_neurons_selects_a_valid_neuron_subset_on_c8(c8_transformer):
    """C8 is abelian with polycyclic digits, so the bridge returns an in-range
    neuron list (not ``None``) usable directly as ``circuit_audit``'s
    ``inside_neurons``."""
    model, group = c8_transformer
    inside = A.carry_circuit_neurons(model, group)
    assert inside is not None
    assert all(0 <= i < model.d_mlp for i in inside)
    assert len(set(inside)) == len(inside)  # no duplicates
    record = A.circuit_audit(model, group, inside_neurons=inside)
    assert record.n_inside == len(inside)


def test_carry_circuit_neurons_selection_shrinks_as_the_threshold_rises(c8_transformer):
    """A neuron selected at a high ``min_top_share`` must also be selected at a
    lower one -- the threshold only removes candidates, never adds them."""
    model, group = c8_transformer
    loose = A.carry_circuit_neurons(model, group, min_top_share=0.0)
    strict = A.carry_circuit_neurons(model, group, min_top_share=0.95)
    assert loose is not None and strict is not None
    assert set(strict).issubset(set(loose))


def test_carry_circuit_neurons_selects_single_digit_and_rejects_diffuse():
    """The causal digit-attribution criterion, pinned decisively: a neuron whose
    output direction is a pure single-output-digit main effect is selected; a
    neuron whose output direction spreads equally across the three digits (top
    share 1/3) is not, at ``min_top_share=0.5``.

    The neuron's causal (zero-ablation) logit direction is exactly its ``W_U``
    row for the FC architecture, so hand-setting two rows controls the two
    neurons' digit attribution while leaving the rest random."""
    set_seed(0, deterministic=False)
    group = resolve_group("C8")
    model = build_model(_config(8, 1, "fc"), group)
    from group_algorithm_interp.instruments.probes import polycyclic_digits

    digits = polycyclic_digits(group).digits  # [8, 3], radix-2 columns
    # A single-bit sign pattern for each digit coordinate (mean zero already).
    bit = [np.where(digits[:, j] == 0, 1.0, -1.0) for j in range(3)]
    pure_digit0 = bit[0]  # a function of output digit 0 only -> concentration 1.0
    diffuse = bit[0] + bit[1] + bit[2]  # equal energy on all three -> share 1/3

    with torch.no_grad():
        model.W_U[0, :] = torch.as_tensor(pure_digit0, dtype=model.W_U.dtype)
        model.W_U[1, :] = torch.as_tensor(diffuse, dtype=model.W_U.dtype)

    selected = set(A.carry_circuit_neurons(model, group, min_top_share=0.5))
    assert 0 in selected  # the pure single-digit neuron clears the bar
    assert 1 not in selected  # the diffuse neuron does not (share 1/3 < 0.5)
    # And the diffuse neuron stays out even at a threshold below 1/3's floor is
    # not asserted; but the pure neuron survives an almost-unit threshold.
    assert 0 in set(A.carry_circuit_neurons(model, group, min_top_share=0.99))


# ---------------------------------------------------------------------------
# I-36: architecture-confound replication
# ---------------------------------------------------------------------------


def test_replication_flags_disjoint_distributions_as_architecture_conditional():
    record = A.replicate_across_architectures([1.0, 1.1, 0.9], [5.0, 5.1, 4.9]).to_record()
    assert record["architecture_conditional"] is True
    assert record["transformer"]["mean"] == pytest.approx(1.0)
    assert record["fc"]["mean"] == pytest.approx(5.0)
    assert record["difference_transformer_minus_fc"] == pytest.approx(-4.0)


def test_replication_does_not_flag_overlapping_distributions():
    record = A.replicate_across_architectures([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
    assert record.architecture_conditional is False


def test_replication_handles_a_single_value_per_arch():
    """The n < 2 branch: a single value gives a degenerate (zero-width) CI, and
    ``architecture_conditional`` is undetermined (``None``), not a point-wise
    inequality between two degenerate CIs."""
    record = A.replicate_across_architectures([0.8], [0.8])
    assert record.transformer["ci_low"] == record.transformer["ci_high"] == pytest.approx(0.8)
    assert record.transformer["n"] == pytest.approx(1.0)
    assert record.architecture_conditional is None


def test_replication_flags_none_at_n1_even_when_values_differ():
    """n=1 per side with clearly different values (1.0 vs 1.000001) must not be
    read as a strict inequality -- the flag is undetermined, not True."""
    record = A.replicate_across_architectures([1.0], [1.000001])
    assert record.transformer["n"] == pytest.approx(1.0)
    assert record.fc["n"] == pytest.approx(1.0)
    assert record.architecture_conditional is None


def test_replication_flags_none_when_only_one_side_is_degenerate():
    record = A.replicate_across_architectures([1.0], [5.0, 5.1, 4.9])
    assert record.architecture_conditional is None


def test_replication_rejects_empty_values():
    with pytest.raises(ValueError, match="at least one value"):
        A.replicate_across_architectures([], [1.0, 2.0])
    with pytest.raises(ValueError, match="at least one value"):
        A.replicate_across_architectures([1.0, 2.0], [])


def test_measure_over_models_runs_a_scalar_measurement(s3_transformer):
    model, group = s3_transformer
    values = A.measure_over_models(
        lambda m, g: float(m.W_U.abs().mean().item()), [model, model], group
    )
    assert len(values) == 2
    assert values[0] == values[1]


# ---------------------------------------------------------------------------
# I-35: direct logit attribution (shift-invariant, rebuilt)
# ---------------------------------------------------------------------------


def test_dla_fractions_sum_to_one_for_the_transformer(s3_transformer):
    model, group = s3_transformer
    record = A.direct_logit_attribution(model, group)
    fractions = [component["fraction"] for component in record["components"].values()]
    assert set(record["components"]) == {"direct", "attention", "mlp"}
    assert sum(fractions) == pytest.approx(1.0)
    assert np.isfinite(record["mean_correct_logit_difference"])


def test_dla_has_a_single_component_for_the_fc(s3_fc):
    model, group = s3_fc
    record = A.direct_logit_attribution(model, group)
    assert set(record["components"]) == {"mlp"}
    assert record["components"]["mlp"]["fraction"] == pytest.approx(1.0)


def test_dla_rejects_too_small_a_held_out_set():
    set_seed(0, deterministic=False)
    group = resolve_group("C4")
    model = build_model(_config(4, 1), group)
    with pytest.raises(ValueError, match="too small"):
        A.direct_logit_attribution(model, group)


def test_dla_record_carries_provenance(s3_transformer, tmp_path):
    """The DLA record must trace to code and data like I-34's: analysis git
    commit, instrument-code hashes, the held-out subset's train_frac/split_seed,
    and (when given) the checkpoint's sha256."""
    model, group = s3_transformer
    checkpoint = tmp_path / "final.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    record = A.direct_logit_attribution(
        model, group, train_frac=0.7, split_seed=2, checkpoint_path=checkpoint
    )
    provenance = record["provenance"]
    assert "instrument_code_sha256" in provenance
    assert "analysis_git_commit" in provenance
    assert provenance["train_frac"] == pytest.approx(0.7)
    assert provenance["split_seed"] == 2
    assert "checkpoint_sha256" in provenance


def test_dla_record_provenance_omits_checkpoint_hash_when_not_given(s3_transformer):
    model, group = s3_transformer
    record = A.direct_logit_attribution(model, group)
    assert "checkpoint_sha256" not in record["provenance"]
