import pytest
from pydantic import ValidationError

from group_algorithm_interp.config import ProjectConfig, ValidationConfig

# Pydantic's own mechanisms (Literal allowlists, extra="forbid") are covered once
# in test_config.py; what is project logic -- and tested here -- is the
# pre-registration rule that a confirmatory stance requires a prediction.


def test_defaults_are_exploratory_with_no_prediction():
    v = ProjectConfig().validation
    assert v.stance == "exploratory"
    assert v.prediction is None


def test_confirmatory_with_prediction_validates():
    cfg = ProjectConfig(
        validation={
            "stance": "confirmatory",
            "prediction": {"metric": "accuracy", "direction": "higher", "value": 0.7},
        }
    )
    assert cfg.validation.stance == "confirmatory"
    assert cfg.validation.prediction.metric == "accuracy"
    assert cfg.validation.prediction.direction == "higher"
    assert cfg.validation.prediction.value == 0.7


def test_confirmatory_without_prediction_is_rejected():
    with pytest.raises(ValidationError, match="pre-registered prediction"):
        ValidationConfig(stance="confirmatory")
