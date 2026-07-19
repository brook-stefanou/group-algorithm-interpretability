"""Unit tests for the W&B wrapper. All run with mode=disabled (conftest also
forces WANDB_MODE=disabled), so nothing touches the network."""

import builtins

import pytest

from group_algorithm_interp.config import LoggingConfig, ProjectConfig
from group_algorithm_interp.manifest import compute_group_hash
from group_algorithm_interp.wandb_utils import WandbLogger


def _config(**logging_kwargs) -> ProjectConfig:
    return ProjectConfig(logging=LoggingConfig(mode="disabled", **logging_kwargs))


def test_group_uses_explicit_arg_first():
    logger = WandbLogger(_config(), run_name="r1", group="hash-abc")
    assert logger.group == "hash-abc"
    assert logger.job_type == "run"


def test_group_falls_back_to_config_then_none():
    assert WandbLogger(_config(group="cfg-grp"), run_name="r2").group == "cfg-grp"
    assert WandbLogger(_config(), run_name="r3").group is None


def test_new_methods_are_safe_noops_before_start(tmp_path):
    """Before start() (and thus with _run=None) every helper must no-op rather
    than raise -- the trainer relies on unconditional calls being safe."""
    logger = WandbLogger(_config(), run_name="r4")
    # No run yet:
    logger.watch(object(), log_freq=10)
    logger.log_table("eval/preds", columns=["a", "b"], rows=[[1, 2], [3, 4]])
    ckpt = tmp_path / "x.pt"
    ckpt.write_bytes(b"not-a-real-checkpoint")
    logger.log_checkpoint_artifact(ckpt, name="demo-model", aliases=["best"])
    logger.finish()  # must also be a no-op


def test_disabled_start_does_not_import_wandb(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "wandb":
            raise AssertionError("disabled mode imported wandb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    logger = WandbLogger(_config(), run_name="no-import")
    logger.start()
    assert logger.run_id is None


def test_start_attaches_to_active_run_and_does_not_finish_it(monkeypatch):
    """In a W&B sweep the agent starts the run; WandbLogger must attach to it
    (not double-init) and must not finish a run it didn't create."""
    import wandb

    finished = {"called": False}

    class _FakeRun:
        id = "agent-run-id"
        url = "https://wandb.ai/x/y/runs/agent-run-id"

        class _Config:
            def update(self, *a: object, **k: object) -> None:
                pass

        config = _Config()

        def define_metric(self, *a, **k):
            pass

        def finish(self):
            finished["called"] = True

    fake = _FakeRun()
    monkeypatch.setattr(wandb, "run", fake, raising=False)

    def _boom(*a, **k):  # wandb.init must NOT be called when a run is active
        raise AssertionError("wandb.init called despite an active run")

    monkeypatch.setattr(wandb, "init", _boom, raising=False)

    config = ProjectConfig(logging=LoggingConfig(mode="online"))
    logger = WandbLogger(config, run_name="trial")
    logger.start()
    assert logger.run_id == "agent-run-id"
    logger.finish()
    assert finished["called"] is False  # we didn't own it, so we don't finish it


def test_finish_calls_run_finish_exactly_once_when_owned(monkeypatch):
    """finish() on a run WandbLogger owns must call run.finish() exactly once --
    this line previously had 0% test coverage, so a regression that reduced
    finish() to a no-op would have passed the whole suite silently."""
    import wandb

    monkeypatch.setattr(wandb, "run", None, raising=False)

    finish_calls = {"count": 0}

    class _FakeRun:
        id = "owned-run-id"
        url = "https://wandb.ai/x/y/runs/owned-run-id"

        def define_metric(self, *a, **k):
            pass

        def finish(self):
            finish_calls["count"] += 1

    monkeypatch.setattr(wandb, "init", lambda *a, **k: _FakeRun(), raising=False)

    config = ProjectConfig(logging=LoggingConfig(mode="online"))
    logger = WandbLogger(config, run_name="owned")
    logger.start()
    assert logger._owns_run is True

    logger.finish()
    assert finish_calls["count"] == 1
    assert logger._run is None

    # finish() must stay idempotent: a second call (e.g. a defensive
    # belt-and-suspenders call from a caller) must not re-finish the run.
    logger.finish()
    assert finish_calls["count"] == 1


def test_finish_calls_run_finish_once_on_the_exception_path(monkeypatch):
    """Mirrors BaseExperiment.execute()'s real call site: logger.finish() runs
    from a ``finally`` block after an exception propagates out of the run body.
    finish() must still call run.finish() exactly once in that path."""
    import wandb

    monkeypatch.setattr(wandb, "run", None, raising=False)

    finish_calls = {"count": 0}

    class _FakeRun:
        id = "owned-run-id"
        url = "https://wandb.ai/x/y/runs/owned-run-id"

        def define_metric(self, *a, **k):
            pass

        def finish(self):
            finish_calls["count"] += 1

    monkeypatch.setattr(wandb, "init", lambda *a, **k: _FakeRun(), raising=False)

    config = ProjectConfig(logging=LoggingConfig(mode="online"))
    logger = WandbLogger(config, run_name="owned-exc")
    logger.start()

    with pytest.raises(RuntimeError, match="boom"):
        try:
            raise RuntimeError("boom")
        finally:
            logger.finish()

    assert finish_calls["count"] == 1


def test_extra_config_carries_seed_invariant_hash_and_seed_into_wandb_init(monkeypatch):
    """The seed-invariant config_group_hash (passed via extra_config) and the
    per-run seed must both land in the dict passed to wandb.init's config=
    kwarg, so a reviewer can group on the hash in the W&B UI while each run's
    own seed is still visible. The hash must be identical across two configs
    that differ only by seed -- compute_group_hash ignores the seed."""
    import wandb

    monkeypatch.setattr(wandb, "run", None, raising=False)

    captured_configs: list[dict] = []

    class _FakeRun:
        id = "fake-run-id"
        url = "https://wandb.ai/x/y/runs/fake-run-id"

        def define_metric(self, *a, **k):
            pass

    def _fake_init(*args, **kwargs):
        captured_configs.append(kwargs["config"])
        return _FakeRun()

    monkeypatch.setattr(wandb, "init", _fake_init, raising=False)

    config_1 = ProjectConfig(seed=1, logging=LoggingConfig(mode="online"))
    config_2 = ProjectConfig(seed=2, logging=LoggingConfig(mode="online"))

    logger_1 = WandbLogger(
        config_1,
        run_name="run-1",
        extra_config={"config_group_hash": compute_group_hash(config_1)},
    )
    logger_2 = WandbLogger(
        config_2,
        run_name="run-2",
        extra_config={"config_group_hash": compute_group_hash(config_2)},
    )
    logger_1.start()
    logger_2.start()

    assert len(captured_configs) == 2
    config_kwargs_1, config_kwargs_2 = captured_configs

    assert config_kwargs_1["seed"] == 1
    assert config_kwargs_2["seed"] == 2
    assert config_kwargs_1["seed"] != config_kwargs_2["seed"]
    assert config_kwargs_1["config_group_hash"] == config_kwargs_2["config_group_hash"]


def test_run_ids_stay_unique_even_when_run_names_collide(monkeypatch):
    """Regression test for the W&B run-id collision hazard: wandb.init must
    never be given an explicit id derived from run_name. If it were, two runs
    that ever shared a run_name (e.g. from a future change to create_run_id)
    would silently overwrite each other's history in W&B. Real W&B assigns a
    fresh unique id per run when none is passed; this fake mimics that and
    asserts two loggers built from the same run_name still get different ids."""
    import itertools

    import wandb

    monkeypatch.setattr(wandb, "run", None, raising=False)
    counter = itertools.count()

    class _FakeRun:
        def __init__(self):
            self.id = f"generated-{next(counter)}"
            self.url = f"https://wandb.ai/x/y/runs/{self.id}"

        def define_metric(self, *a, **k):
            pass

    def _fake_init(*args, **kwargs):
        assert kwargs.get("id") is None, (
            "wandb.init must not receive an explicit id derived from run_name -- "
            "that reintroduces the run-id collision hazard"
        )
        return _FakeRun()

    monkeypatch.setattr(wandb, "init", _fake_init, raising=False)

    config = ProjectConfig(logging=LoggingConfig(mode="online"))
    logger_1 = WandbLogger(config, run_name="same-name")
    logger_2 = WandbLogger(config, run_name="same-name")
    logger_1.start()
    logger_2.start()

    assert logger_1.run_id != logger_2.run_id
