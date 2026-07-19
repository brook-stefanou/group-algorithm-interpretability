import pytest
from pydantic import ValidationError

from group_algorithm_interp.config import (
    DataConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    OptimConfig,
    ProjectConfig,
    validate_config,
)
from group_algorithm_interp.manifest import compute_group_hash


def test_defaults_validate():
    cfg = ProjectConfig()
    assert cfg.project_name == "group-algorithm-interp"
    assert cfg.deterministic is False
    assert cfg.device == "auto"
    assert cfg.logging.mode == "disabled"
    assert cfg.snapshot.save_final is True
    # The run length lives in exactly one place, and it is what the loop iterates.
    assert cfg.optim.epochs > 0


def test_validate_config_from_dict():
    cfg = validate_config({"seed": 7, "optim": {"epochs": 50}})
    assert isinstance(cfg, ProjectConfig)
    assert cfg.seed == 7
    assert cfg.optim.epochs == 50


# ---------------------------------------------------------------------------
# Schema mechanisms. Pydantic's own constraint machinery is not this project's
# code, so each mechanism gets ONE representative test rather than one per field.
# ---------------------------------------------------------------------------


def test_positive_int_constraint_rejects_zero():
    """Representative of the gt=0 constraints across the schema."""
    with pytest.raises(ValidationError):
        ModelConfig(d_model=0)


def test_literal_constraint_rejects_a_value_outside_the_allowlist():
    """Representative of the Literal allowlists (device, arch, wandb mode, ...)."""
    for device in ("auto", "cpu", "cuda", "mps"):
        ProjectConfig(device=device)
    with pytest.raises(ValidationError):
        ProjectConfig(device="tpu")


def test_unknown_key_rejected():
    """Representative of extra="forbid": a typo'd or retired key fails loudly."""
    with pytest.raises(ValidationError):
        ModelConfig(d_modell=1)
    with pytest.raises(ValidationError):
        validate_config({"model": {"d_modell": 1}})
    # Checkpoint (resume-oriented) and snapshot (analysis-oriented) configs were
    # unified into SnapshotConfig; "checkpoint" is no longer a recognized key.
    with pytest.raises(ValidationError):
        validate_config({"checkpoint": {"enabled": True}})


def test_logging_config_wandb_feature_defaults():
    cfg = LoggingConfig()
    # Feature flags default off / safe so existing behaviour is unchanged.
    assert cfg.group is None
    assert cfg.job_type == "run"
    assert cfg.save_code is False
    assert cfg.watch_model is False
    assert cfg.watch_log_freq == 100
    assert cfg.log_checkpoints is False


def test_unsupported_logging_backend_rejected():
    """W&B is the only backend anything branches on. `backend: tensorboard` used
    to validate and then silently log to W&B anyway; it must now fail loudly."""
    assert LoggingConfig().backend == "wandb"
    with pytest.raises(ValidationError):
        LoggingConfig(backend="tensorboard")


# ---------------------------------------------------------------------------
# d_mlp defaults to 2 * d_model -- a width sweep over d_model must move d_mlp
# with it, or the sweep silently leaves MLP width fixed. The shipped 128/256
# default is this ratio made explicit rather than a coincidence.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d_model,expected_d_mlp", [(64, 128), (128, 256), (32, 64)])
def test_d_mlp_defaults_to_two_times_d_model(d_model, expected_d_mlp):
    assert ModelConfig(d_model=d_model).d_mlp == expected_d_mlp
    # An explicit null is the same "unset" signal as omitting the key --
    # both are what a composed Hydra YAML with `d_mlp: null` produces.
    assert ModelConfig(d_model=d_model, d_mlp=None).d_mlp == expected_d_mlp


def test_d_mlp_defaults_through_full_project_config():
    assert ProjectConfig().model.d_mlp == 256  # the shipped 128 -> 256 default
    assert ProjectConfig(model=ModelConfig(d_model=64)).model.d_mlp == 128


def test_d_mlp_explicit_override_is_respected():
    """An explicit d_mlp is never overwritten by the 2x-d_model derivation,
    even when it does not follow that ratio."""
    assert ModelConfig(d_model=64, d_mlp=999).d_mlp == 999
    assert ModelConfig(d_model=64, d_mlp=64).d_mlp == 64


def test_d_mlp_keeps_the_positive_constraint_whether_explicit_or_derived():
    with pytest.raises(ValidationError):
        ModelConfig(d_model=64, d_mlp=0)
    # d_model=0 is itself invalid, so the derived d_mlp (0) can never arise
    # from a valid config -- the gt=0 constraint on d_model prevents it.
    with pytest.raises(ValidationError):
        ModelConfig(d_model=0)


def test_resolved_config_records_the_integer_d_mlp_never_null():
    """The point of resolving in config.py: every consumer of the *resolved*
    config -- model_dump(), the manifest's resolved_config.yaml -- sees a
    concrete integer, not `null`, regardless of whether d_mlp was set."""
    cfg = validate_config({"model": {"d_model": 64}})
    assert cfg.model.d_mlp == 128
    dumped = cfg.model_dump()
    assert dumped["model"]["d_mlp"] == 128
    assert dumped["model"]["d_mlp"] is not None


def test_unset_and_explicit_two_x_d_mlp_hash_identically():
    """The W&B run-group hash (manifest.compute_group_hash) keys on `model.*`. A
    d_model sweep that leaves d_mlp unset must land on the SAME hash as one that
    pins d_mlp to the equivalent explicit value -- otherwise two configs that
    are, in every way that matters, identical would group apart."""
    cfg_unset = ProjectConfig(model=ModelConfig(d_model=64))
    cfg_explicit = ProjectConfig(model=ModelConfig(d_model=64, d_mlp=128))

    assert compute_group_hash(cfg_unset) == compute_group_hash(cfg_explicit)

    # A genuinely different d_mlp must still produce a different hash.
    cfg_decoupled = ProjectConfig(model=ModelConfig(d_model=64, d_mlp=512))
    assert compute_group_hash(cfg_unset) != compute_group_hash(cfg_decoupled)


# ---------------------------------------------------------------------------
# ProjectConfig._reconcile_and_guard: the derived quantities and the aliases.
# ---------------------------------------------------------------------------


def test_lr_effective_equals_lr_when_width_scaling_is_off():
    cfg = ProjectConfig(optim=OptimConfig(lr=3e-3, scale_lr_with_width=False))
    assert cfg.optim.lr_effective == 3e-3


def test_lr_effective_scales_by_base_width_over_d_model():
    cfg = ProjectConfig(
        model=ModelConfig(d_model=128),
        optim=OptimConfig(lr=1e-3, lr_base_width=64, scale_lr_with_width=True),
    )
    # The defining invariant of the LR every run actually trains at.
    assert cfg.optim.lr_effective == 1e-3 * 64 / 128


def test_doubling_d_model_halves_lr_effective():
    def lr_effective(d_model: int) -> float:
        cfg = ProjectConfig(
            model=ModelConfig(d_model=d_model),
            optim=OptimConfig(lr=1e-3, lr_base_width=64, scale_lr_with_width=True),
        )
        assert cfg.optim.lr_effective is not None
        return cfg.optim.lr_effective

    # Guards against an inverted ratio (d_model / lr_base_width), which would
    # double the LR here instead of halving it.
    assert lr_effective(128) == pytest.approx(lr_effective(64) / 2)
    assert lr_effective(32) == pytest.approx(lr_effective(64) * 2)


def test_lr_effective_is_constant_across_widths_by_default():
    """scale_lr_with_width defaults to False, so lr_effective == lr == 1e-3
    regardless of d_model when no run opts in to width scaling. Relies on the
    OptimConfig default instead of passing scale_lr_with_width explicitly, so
    it catches a future regression that flips the default back to True."""
    for d_model in (64, 128, 256):
        cfg = ProjectConfig(model=ModelConfig(d_model=d_model), optim=OptimConfig())
        assert cfg.optim.scale_lr_with_width is False
        assert cfg.optim.lr_effective == pytest.approx(1e-3)


def test_experiment_seed_alias_overwrites_the_top_level_seed():
    cfg = ProjectConfig(seed=0, experiment=ExperimentConfig(seed=11))
    assert cfg.seed == 11
    # ...and leaving the alias unset does not touch the top-level seed.
    assert ProjectConfig(seed=3).seed == 3


def test_experiment_use_wandb_false_alias_disables_logging():
    cfg = ProjectConfig(
        logging=LoggingConfig(mode="online"), experiment=ExperimentConfig(use_wandb=False)
    )
    assert cfg.logging.mode == "disabled"
    # use_wandb=True is not an instruction to *enable* anything: mode stands.
    assert (
        ProjectConfig(
            logging=LoggingConfig(mode="offline"), experiment=ExperimentConfig(use_wandb=True)
        ).logging.mode
        == "offline"
    )


def test_expensive_run_guard_blocks_oversized_runs():
    # epochs above the eval.max_steps_warn ceiling must fail loudly...
    with pytest.raises(ValidationError):
        ProjectConfig(eval={"max_steps_warn": 10}, optim={"epochs": 100})
    # ...unless the run explicitly opts in.
    cfg = ProjectConfig(allow_expensive=True, eval={"max_steps_warn": 10}, optim={"epochs": 100})
    assert cfg.optim.epochs == 100
    assert ProjectConfig().allow_expensive is False


# ---------------------------------------------------------------------------
# The split guard: a valid-looking config must not produce an empty split side.
# ---------------------------------------------------------------------------


def test_effective_split_seed_prefers_the_pinned_split_seed():
    assert ProjectConfig(seed=7, data=DataConfig(split_seed=2)).effective_split_seed == 2


def test_effective_split_seed_falls_back_to_the_run_seed_when_unpinned():
    assert ProjectConfig(seed=7, data=DataConfig(split_seed=None)).effective_split_seed == 7


def test_train_frac_that_empties_the_test_split_is_rejected():
    """C2 has only 4 multiplication pairs, so round(0.9 * 4) == 4: the test set
    is empty, the loss is nan, and the run still finalises as `completed`.
    0 < train_frac < 1 does not catch this -- the group's size decides."""
    with pytest.raises(ValidationError) as excinfo:
        DataConfig(group="C2", train_frac=0.9)
    message = str(excinfo.value)
    assert "SmallGroup(2,1)" in message  # names the group
    assert "0.9" in message  # ...and the train_frac
    assert "0 train" not in message and "0 test" in message


def test_train_frac_that_empties_the_test_split_of_a_larger_group_is_rejected():
    # C8: 64 pairs, round(0.995 * 64) == 64.
    with pytest.raises(ValidationError, match="SmallGroup\\(8,1\\)"):
        DataConfig(group="C8", train_frac=0.995)


def test_train_frac_that_empties_the_train_split_is_rejected():
    # The mirror image: round(1e-6 * 64) == 0 train examples.
    with pytest.raises(ValidationError, match="0 train"):
        DataConfig(group="C8", train_frac=1e-6)


def test_the_split_guard_fires_through_full_config_validation():
    with pytest.raises(ValidationError, match="train_frac"):
        validate_config({"data": {"group": "C2", "train_frac": 0.9}})


@pytest.mark.parametrize(
    "group,train_frac,expected",
    [
        ("C2", 0.5, (2, 2)),
        ("C2", 0.7, (3, 1)),  # the last train_frac C2 admits
        ("C8", 0.8, (51, 13)),
        ("C8", 0.99, (63, 1)),
    ],
)
def test_valid_group_and_train_frac_pairs_are_accepted(group, train_frac, expected):
    from group_algorithm_interp.config import split_sizes

    cfg = DataConfig(group=group, train_frac=train_frac)
    assert split_sizes(cfg.group.order**2, cfg.train_frac) == expected
