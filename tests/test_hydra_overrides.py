"""Hydra composition + override behaviour, validated through Pydantic."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from pydantic import ValidationError

from group_algorithm_interp.config import validate_config

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

# The full-length budget the optim group sets; the short presets must not use it.
FULL_EPOCHS = 10_000


def _compose(overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=overrides)


def test_compose_defaults():
    cfg = validate_config(_compose([]))
    assert cfg.model.arch == "transformer"
    assert cfg.data.group.canonical_name == "SmallGroup(8,1)"
    assert cfg.experiment.name == "run"
    assert cfg.deterministic is False
    assert cfg.logging.mode == "disabled"
    # The default composition is the FULL run: a campaign launched with no
    # `experiment=` override must not be silently shortened.
    assert cfg.optim.epochs == FULL_EPOCHS


@pytest.mark.parametrize("preset,epochs", [("debug", 20), ("smoke", 200)])
def test_short_experiment_presets_actually_shorten_the_run(preset, epochs):
    """A dead field here would silently revert every preset to the full
    optim.epochs -- an operator smoke-testing would pay for the whole run."""
    cfg = validate_config(_compose([f"experiment={preset}"]))
    assert cfg.experiment.name == preset
    assert cfg.optim.epochs == epochs
    assert cfg.optim.epochs < FULL_EPOCHS


def test_core_experiment_preset_sets_the_campaign_epoch_ceiling():
    """`experiment=core` is the pre-registered campaign preset: it must set
    `optim.epochs` to the campaign ceiling (30,000), not the optim group's
    default (10,000) -- the same run-length-has-one-home rule debug/smoke
    follow, just lengthening instead of shortening the run."""
    cfg = validate_config(_compose(["experiment=core"]))
    assert cfg.experiment.name == "core"
    assert cfg.optim.epochs == 30_000
    assert cfg.optim.epochs > FULL_EPOCHS


def test_cli_epochs_override_beats_a_preset():
    """Defaults compose first, command-line overrides last."""
    cfg = validate_config(_compose(["experiment=debug", "optim.epochs=7"]))
    assert cfg.optim.epochs == 7


def test_invalid_override_fails_validation():
    with pytest.raises(ValidationError):
        validate_config(_compose(["optim.epochs=-1"]))
