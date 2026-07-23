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
    """The n < 2 branch: a single value gives a degenerate (zero-width) CI."""
    record = A.replicate_across_architectures([0.8], [0.8])
    assert record.transformer["ci_low"] == record.transformer["ci_high"] == pytest.approx(0.8)
    assert record.transformer["n"] == pytest.approx(1.0)
    assert record.architecture_conditional is False


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
