import pytest
import yaml

from group_algorithm_interp.config import (
    ExperimentConfig,
    LoggingConfig,
    OptimConfig,
    ProjectConfig,
    ValidationConfig,
)
from group_algorithm_interp.experiment import BaseExperiment, HoldoutError, _prediction_outcome


def _config(stance="exploratory", prediction=None) -> ProjectConfig:
    if stance == "confirmatory" and prediction is None:
        prediction = {"metric": "accuracy", "direction": "higher", "value": 0.7}
    return ProjectConfig(
        device="cpu",
        logging=LoggingConfig(mode="disabled"),
        optim=OptimConfig(epochs=5),
        experiment=ExperimentConfig(name="t"),
        validation=ValidationConfig(stance=stance, prediction=prediction),
    )


class _Confirmatory(BaseExperiment):
    phases = []

    def setup(self):
        self.declare_holdout(fingerprint="abc123", size=100, spec={"frac": 0.1, "seed": 0})

    def run(self):
        self.phases.append("run")
        return {"train/loss": 0.1}

    def holdout_evaluation(self):
        self.phases.append("holdout")
        return {"accuracy": 0.81}


class _NoDeclare(BaseExperiment):
    def setup(self):
        return None

    def run(self):
        return {}


class _NoEvaluate(BaseExperiment):
    def setup(self):
        self.declare_holdout(fingerprint="abc123", size=100)

    def run(self):
        return {}


class _EarlyEvaluate(BaseExperiment):
    def setup(self):
        self.declare_holdout(fingerprint="abc123", size=100)

    def run(self):
        return self.evaluate_holdout(lambda: {"accuracy": 0.8})


def test_prediction_outcome_pure():
    assert _prediction_outcome(None, {"accuracy": 0.9}) is None
    p = {"metric": "accuracy", "direction": "higher", "value": 0.7}
    assert _prediction_outcome(p, {"accuracy": 0.81}) == "predicted"
    assert _prediction_outcome(p, {"accuracy": 0.5}) == "refuted"
    assert _prediction_outcome(p, {"loss": 0.1}) == "inconclusive"
    low = {"metric": "loss", "direction": "lower", "value": 0.5}
    assert _prediction_outcome(low, {"loss": 0.3}) == "predicted"
    assert _prediction_outcome(low, {"loss": 0.9}) == "refuted"


def test_exploratory_needs_nothing(tmp_path):
    exp = _NoDeclare(_config(), runs_root=tmp_path)
    exp.execute()  # no raise
    m = yaml.safe_load((exp.run_dir / "manifest.yaml").read_text())
    assert m["status"] == "completed"
    assert m["validation"]["stance"] == "exploratory"
    assert m["validation"]["holdout"] is None


def test_confirmatory_records_holdout_and_outcome(tmp_path):
    cfg = _config("confirmatory", {"metric": "accuracy", "direction": "higher", "value": 0.7})
    exp = _Confirmatory(cfg, runs_root=tmp_path)
    exp.phases = []
    summary = exp.execute()
    m = yaml.safe_load((exp.run_dir / "manifest.yaml").read_text())
    assert m["status"] == "completed"
    assert m["validation"]["holdout"]["fingerprint"] == "abc123"
    assert m["validation"]["holdout_result"]["metrics"] == {"accuracy": 0.81}
    assert m["validation"]["holdout_result"]["outcome"] == "predicted"
    assert exp.phases == ["run", "holdout"]
    assert summary == {"train/loss": 0.1, "accuracy": 0.81}


def test_confirmatory_refuted_outcome(tmp_path):
    cfg = _config("confirmatory", {"metric": "accuracy", "direction": "higher", "value": 0.9})
    exp = _Confirmatory(cfg, runs_root=tmp_path)
    exp.execute()
    m = yaml.safe_load((exp.run_dir / "manifest.yaml").read_text())
    assert m["validation"]["holdout_result"]["outcome"] == "refuted"


def test_confirmatory_without_declare_fails(tmp_path):
    exp = _NoDeclare(_config("confirmatory"), runs_root=tmp_path)
    with pytest.raises(HoldoutError, match="declare"):
        exp.execute()
    m = yaml.safe_load((exp.run_dir / "manifest.yaml").read_text())
    assert m["status"] == "failed"


def test_confirmatory_without_evaluate_fails(tmp_path):
    exp = _NoEvaluate(_config("confirmatory"), runs_root=tmp_path)
    with pytest.raises(HoldoutError, match="holdout_evaluation"):
        exp.execute()
    m = yaml.safe_load((exp.run_dir / "manifest.yaml").read_text())
    assert m["status"] == "failed"


def test_holdout_cannot_be_evaluated_during_run(tmp_path):
    exp = _EarlyEvaluate(_config("confirmatory"), runs_root=tmp_path)
    with pytest.raises(HoldoutError, match="only after run"):
        exp.execute()
    m = yaml.safe_load((exp.run_dir / "manifest.yaml").read_text())
    assert m["status"] == "failed"
