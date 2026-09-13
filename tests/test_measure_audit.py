"""The circuit-audit entry point (scripts/measure_audit.py).

Runs the real ``measure`` command against short training runs on the fixture
corpus -- offline, no W&B, no network. The script is an executable entry point,
not part of the installed package, so it is loaded by file path (mirrors
``tests/test_measure_probes.py``). The decisive end-to-end property: a D8 run
(coset target exists) yields a measured ``circuit_audit`` sub-record, a Q8 run
(coset target is ``None``, the same theorem as Q32) yields a skipped one with a
reason -- while both always carry ``direct_logit_attribution``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from group_algorithm_interp.config import ExperimentConfig, LoggingConfig, ProjectConfig
from group_algorithm_interp.experiment import GroupGeneralizationExperiment

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "measure_audit.py"

EPOCHS = 10


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_audit", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_audit"] = module
    spec.loader.exec_module(module)
    return module


def _train(runs_root: Path, name: str) -> Path:
    config = ProjectConfig(
        device="cpu",
        seed=0,
        data={"group": name, "train_frac": 0.5, "split_seed": 0},
        model={"d_model": 16, "d_mlp": 32, "n_heads": 1},
        optim={"epochs": EPOCHS, "log_every": 1, "print_every": 1},
        snapshot={
            "enabled": True,
            "interval": 4,
            "log_dense_until": 4,
            "event_based": False,
            "final_window_epochs": 3,
        },
        logging=LoggingConfig(mode="disabled"),
        experiment=ExperimentConfig(name="audit-measure"),
    )
    experiment = GroupGeneralizationExperiment(config, runs_root=runs_root)
    experiment.execute()
    return experiment.run_dir


@pytest.fixture(scope="module")
def dihedral_and_quaternion_runs(tmp_path_factory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("audit-runs")
    return _train(root, "D8"), _train(root, "Q8")


@pytest.fixture(scope="module")
def cyclic_run(tmp_path_factory) -> Path:
    """C8: abelian, no coset target, but a genuine polycyclic carry structure --
    the small stand-in for the C3 members the carry-digit bridge is built for."""
    root = tmp_path_factory.mktemp("audit-runs-cyclic")
    return _train(root, "C8")


def test_measure_audit_run_measures_a_carry_digit_circuit_on_a_cyclic_member(cyclic_run):
    """DECISIVE for the new bridge: C8 has no coset target, so the coset bridge
    declines, but it is abelian with polycyclic digits, so the carry-digit
    bridge produces the I-34 circuit audit -- circuit_source is
    ``carry_digit_attribution`` and the record carries the polycyclic
    provenance, not a skip."""
    script = _load_script()
    record = script.measure_audit_run(cyclic_run, threshold=0.0)
    assert record["status"] == "measured"
    assert record["group"] == {"order": 8, "index": 1, "name": "SmallGroup(8,1)"}
    ca = record["circuit_audit"]
    assert ca["status"] == "measured"
    assert ca["circuit_source"] == "carry_digit_attribution"
    assert ca["carry_composition_length"] == 3
    assert ca["carry_radices"] == [2, 2, 2]
    assert "carry_is_diagonal" in ca and "carry_coordinate_directions" in ca
    assert "faithfulness" in ca and "completeness" in ca and "minimality" in ca
    json.dumps(record)  # JSON-serialisable (modulo NaN)


def test_measure_audit_run_is_measured_with_a_circuit_on_the_dihedral_member(
    dihedral_and_quaternion_runs,
):
    d8_run, _ = dihedral_and_quaternion_runs
    script = _load_script()
    record = script.measure_audit_run(d8_run, threshold=0.0)
    assert record["status"] == "measured"
    assert record["group"] == {"order": 8, "index": 3, "name": "SmallGroup(8,3)"}

    dla = record["direct_logit_attribution"]
    assert dla["instrument"] == "direct_logit_attribution"

    ca = record["circuit_audit"]
    assert ca["status"] == "measured"
    assert ca["circuit_source"] == "coset_occupied_blocks"
    assert "coset_subgroup_index" in ca
    assert "faithfulness" in ca and "completeness" in ca and "minimality" in ca

    provenance = record["provenance"]
    assert provenance["checkpoint_sha256"]
    assert provenance["instrument_code_sha256"]
    assert provenance["group_artifact"].endswith("smallgroup_8_3.npz")
    json.dumps(record)  # the whole record must be JSON-serialisable (modulo NaN)


def test_measure_audit_run_skips_the_circuit_on_the_quaternionic_member(
    dihedral_and_quaternion_runs,
):
    """DECISIVE: Q8 has no coset target (the same theorem as Q32) AND, being
    non-abelian, no trusted polycyclic carry-digit structure -- so neither
    circuit bridge applies and circuit_audit reports a skip with a reason,
    rather than fabricating one from an enumeration artefact, while
    direct_logit_attribution still runs regardless."""
    _, q8_run = dihedral_and_quaternion_runs
    script = _load_script()
    record = script.measure_audit_run(q8_run, threshold=0.0)
    assert record["status"] == "measured"
    assert record["direct_logit_attribution"]["instrument"] == "direct_logit_attribution"
    ca = record["circuit_audit"]
    assert ca["status"] == "skipped"
    assert ca["reason"]
    json.dumps(record)


def test_measure_audit_run_skips_a_run_with_no_stable_checkpoint(dihedral_and_quaternion_runs):
    d8_run, _ = dihedral_and_quaternion_runs
    script = _load_script()
    record = script.measure_audit_run(d8_run, threshold=0.99)
    assert record["status"] == "skipped"
    assert record["checkpoint_selection"]["reason"] is not None
    assert "direct_logit_attribution" not in record
    assert "circuit_audit" not in record


def test_measure_audit_run_respects_ablation_mode_and_seed(dihedral_and_quaternion_runs):
    d8_run, _ = dihedral_and_quaternion_runs
    script = _load_script()
    record = script.measure_audit_run(d8_run, threshold=0.0, ablation_mode="resample", seed=7)
    ca = record["circuit_audit"]
    assert ca["ablation_mode"] == "resample"
    assert ca["seed"] == 7
    assert ca["causal_scrubbing"]["resample_seed"] == 8


def test_cli_measure_writes_per_run_and_collected_output(dihedral_and_quaternion_runs, tmp_path):
    script = _load_script()
    d8_run, q8_run = dihedral_and_quaternion_runs
    out = tmp_path / "audit.json"
    code = script.main(
        ["measure", str(d8_run), str(q8_run), "--threshold", "0.0", "--out", str(out)]
    )
    assert code == 0
    payload = json.loads(out.read_text())
    assert len(payload["runs"]) == 2
    for run_dir in (d8_run, q8_run):
        per_run = json.loads((run_dir / "analysis" / "audit.json").read_text())
        assert per_run["status"] == "measured"


def test_cli_measure_exits_nonzero_when_a_run_is_skipped(dihedral_and_quaternion_runs, tmp_path):
    script = _load_script()
    d8_run, _ = dihedral_and_quaternion_runs
    out = tmp_path / "audit-skip.json"
    code = script.main(["measure", str(d8_run), "--threshold", "0.99", "--out", str(out)])
    assert code == 1
    payload = json.loads(out.read_text())
    assert payload["runs"][0]["status"] == "skipped"
