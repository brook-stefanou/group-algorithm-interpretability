"""End-to-end tests of the reusable experiment lifecycle."""

import math
import os
import signal

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from group_algorithm_interp.config import (
    ExperimentConfig,
    LoggingConfig,
    OptimConfig,
    ProjectConfig,
)
from group_algorithm_interp.experiment import (
    BaseExperiment,
    GroupGeneralizationExperiment,
    RunAborted,
)
from group_algorithm_interp.manifest import compute_group_hash
from group_algorithm_interp.seed import CUBLAS_WORKSPACE_CONFIG, set_seed


def _config() -> ProjectConfig:
    return ProjectConfig(
        device="cpu",
        logging=LoggingConfig(mode="disabled"),
        optim=OptimConfig(epochs=20),
        experiment=ExperimentConfig(name="test"),
    )


class _LifecycleExperiment(BaseExperiment):
    """Minimal test-only subclass for exercising the lifecycle contract."""

    def setup(self) -> None:
        return None

    def run(self) -> dict[str, float]:
        self.global_step = self.config.optim.epochs
        value = float(torch.rand(1).item())
        metrics = {"test/value": value}
        self.log_metrics(metrics, step=self.global_step)
        return metrics


def test_smoke_run_completes(tmp_path):
    trainer = _LifecycleExperiment(_config(), runs_root=tmp_path)
    summary = trainer.execute()

    assert "test/value" in summary
    assert trainer.global_step == 20


def test_run_writes_manifest_and_resolved_config(tmp_path):
    trainer = _LifecycleExperiment(_config(), runs_root=tmp_path)
    trainer.execute()

    assert (trainer.run_dir / "manifest.yaml").is_file()
    assert (trainer.run_dir / "resolved_config.yaml").is_file()
    assert (trainer.run_dir / "checkpoints" / "final.pt").is_file()


def test_final_checkpoint_has_step_and_config_no_resume_state(tmp_path):
    """The base state_dict() is a point-in-time snapshot: no rng_state (no
    resume path exists to restore it into), no optimizer_state_dict
    (BaseExperiment has no optimizer)."""
    import torch

    trainer = _LifecycleExperiment(_config(), runs_root=tmp_path)
    trainer.execute()

    ckpt = torch.load(trainer.run_dir / "checkpoints" / "final.pt", weights_only=False)
    assert ckpt["step"] == 20
    assert "config" in ckpt
    assert "rng_state" not in ckpt
    assert "optimizer_state_dict" not in ckpt


def test_manifest_status_completed_with_summary(tmp_path):
    trainer = _LifecycleExperiment(_config(), runs_root=tmp_path)
    trainer.execute()

    manifest = yaml.safe_load((trainer.run_dir / "manifest.yaml").read_text())
    assert manifest["status"] == "completed"
    assert manifest["completed_at"] is not None
    assert manifest["summary"]["completed_steps"] == 20
    assert manifest["summary"]["metrics"]["test/value"] is not None


def test_wandb_disabled_needs_no_account(tmp_path):
    trainer = _LifecycleExperiment(_config(), runs_root=tmp_path)
    trainer.execute()
    manifest = yaml.safe_load((trainer.run_dir / "manifest.yaml").read_text())
    assert manifest["tracking"]["mode"] == "disabled"
    assert manifest["tracking"]["wandb_url"] is None
    assert manifest["tracking"]["wandb_run_id"] is None


class _FailingExperiment(_LifecycleExperiment):
    def setup(self) -> None:
        raise RuntimeError("intentional failure")


def test_failed_run_marks_manifest_failed(tmp_path):
    trainer = _FailingExperiment(_config(), runs_root=tmp_path)
    with pytest.raises(RuntimeError, match="intentional failure"):
        trainer.execute()

    manifest = yaml.safe_load((trainer.run_dir / "manifest.yaml").read_text())
    assert manifest["status"] == "failed"
    assert "RuntimeError" in manifest["error"]
    # Even a crashed run leaves the config behind for reproducibility.
    assert (trainer.run_dir / "resolved_config.yaml").is_file()


# ---------------------------------------------------------------------------
# Interruption: a run killed from outside must still reach a terminal manifest
# status. `except Exception` missed KeyboardInterrupt, and SIGTERM (the
# launcher's drain signal, and how a spot GPU is reclaimed) killed the process
# outright -- both left `status: running, completed_at: null` forever,
# indistinguishable from a run still training.
# ---------------------------------------------------------------------------


class _InterruptedExperiment(_LifecycleExperiment):
    def setup(self) -> None:
        import warnings as w

        w.warn("careful: interrupted run", UserWarning, stacklevel=2)

    def run(self) -> dict[str, float]:
        raise KeyboardInterrupt("operator pressed ctrl-c")


def test_keyboard_interrupt_finalises_manifest_as_aborted(tmp_path):
    trainer = _InterruptedExperiment(_config(), runs_root=tmp_path)
    with pytest.raises(KeyboardInterrupt):
        trainer.execute()

    manifest = yaml.safe_load((trainer.run_dir / "manifest.yaml").read_text())
    assert manifest["status"] == "aborted"
    assert manifest["completed_at"] is not None
    assert "KeyboardInterrupt" in manifest["error"]
    # Warnings before the interrupt are preserved as provenance and re-emitted
    # to the run log, same as the success path and unlike the failure path.
    assert any("careful: interrupted run" in s for s in manifest["warnings"])
    assert (
        "captured warning: UserWarning: careful: interrupted run"
        in (trainer.run_dir / "run.log").read_text()
    )


class _SigtermedExperiment(_LifecycleExperiment):
    def run(self) -> dict[str, float]:
        os.kill(os.getpid(), signal.SIGTERM)
        # A signal handler runs between bytecodes, not inside os.kill; give it
        # somewhere to land, then fail loudly if it never fired.
        for _ in range(10_000):
            pass
        raise AssertionError("SIGTERM did not interrupt the run")


def test_sigterm_finalises_manifest_as_aborted_and_restores_the_handler(tmp_path):
    """The launcher's terminate-grace-kill sequence is only meaningful if the
    child converts SIGTERM into an unwind that runs its `finally`."""
    before = signal.getsignal(signal.SIGTERM)
    trainer = _SigtermedExperiment(_config(), runs_root=tmp_path)
    with pytest.raises(RunAborted, match="SIGTERM"):
        trainer.execute()

    manifest = yaml.safe_load((trainer.run_dir / "manifest.yaml").read_text())
    assert manifest["status"] == "aborted"
    assert manifest["completed_at"] is not None
    assert "RunAborted" in manifest["error"]
    # The handler installation is run-scoped: the previous disposition is
    # restored on the way out.
    assert signal.getsignal(signal.SIGTERM) is before


def test_failing_run_still_reaches_a_terminal_status_and_logs_its_warnings(tmp_path):
    class _WarnThenFail(_LifecycleExperiment):
        def setup(self) -> None:
            import warnings as w

            w.warn("careful: doomed run", UserWarning, stacklevel=2)

        def run(self) -> dict[str, float]:
            raise RuntimeError("boom")

    trainer = _WarnThenFail(_config(), runs_root=tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        trainer.execute()

    manifest = yaml.safe_load((trainer.run_dir / "manifest.yaml").read_text())
    assert manifest["status"] == "failed"
    assert manifest["completed_at"] is not None
    assert any("careful: doomed run" in s for s in manifest["warnings"])
    assert "captured warning:" in (trainer.run_dir / "run.log").read_text()


def test_smoke_run_with_all_wandb_features_enabled(tmp_path):
    """Even with watch/artifacts/tables/grouping enabled, a disabled-mode run
    completes cleanly (the W&B calls are exercised as no-ops)."""
    config = ProjectConfig(
        device="cpu",
        logging=LoggingConfig(
            mode="disabled",
            watch_model=True,
            log_checkpoints=True,
            save_code=True,
        ),
        optim=OptimConfig(epochs=20),
        experiment=ExperimentConfig(name="test"),
    )
    trainer = _LifecycleExperiment(config, runs_root=tmp_path)
    summary = trainer.execute()

    assert "test/value" in summary
    assert (trainer.run_dir / "checkpoints" / "final.pt").is_file()
    # Grouping is view-time in the W&B UI, not forced at init.
    assert trainer.logger.group is None
    assert trainer.logger.extra_config["config_group_hash"] == compute_group_hash(trainer.config)


def test_default_config_group_is_none_so_seeds_are_not_forced_together(tmp_path):
    """BaseExperiment no longer forces every seed of a config into a shared
    W&B group at init time."""
    trainer = _LifecycleExperiment(_config(), runs_root=tmp_path)
    assert trainer.logger.group is None


def test_run_writes_run_log(tmp_path):
    trainer = _LifecycleExperiment(_config(), runs_root=tmp_path)
    trainer.execute()
    assert (trainer.run_dir / "run.log").is_file()
    text = (trainer.run_dir / "run.log").read_text()
    assert text.strip()
    assert "completed" in text


def test_lifecycle_seeds_before_run_so_stub_values_match(tmp_path):
    """NOT a training-determinism test: ``_LifecycleExperiment.run()`` is a
    one-line stub with no model, no data, no optimizer. Only checks that
    ``BaseExperiment.execute()`` calls ``set_seed(...)`` before ``run()`` --
    i.e. the lifecycle's seeding hook fires at the right point. See
    ``test_group_experiment_same_seed_produces_bit_identical_weights`` below
    for the real end-to-end determinism guarantee."""
    s1 = _LifecycleExperiment(_config(), runs_root=tmp_path).execute()
    s2 = _LifecycleExperiment(_config(), runs_root=tmp_path).execute()
    assert s1["test/value"] == s2["test/value"]


def test_campaign_id_recorded_from_env_var(tmp_path, monkeypatch):
    """BaseExperiment reads GAI_CAMPAIGN_ID from the environment and records
    it verbatim as provenance.campaign_id -- the join key back to the
    launching campaign."""
    monkeypatch.setenv("GAI_CAMPAIGN_ID", "deadbeef1234")
    trainer = _LifecycleExperiment(_config(), runs_root=tmp_path)
    trainer.execute()
    manifest = yaml.safe_load((trainer.run_dir / "manifest.yaml").read_text())
    assert manifest["provenance"]["campaign_id"] == "deadbeef1234"


def test_campaign_id_is_none_outside_a_campaign(tmp_path, monkeypatch):
    monkeypatch.delenv("GAI_CAMPAIGN_ID", raising=False)
    trainer = _LifecycleExperiment(_config(), runs_root=tmp_path)
    summary = trainer.execute()
    assert "test/value" in summary
    manifest = yaml.safe_load((trainer.run_dir / "manifest.yaml").read_text())
    assert manifest["status"] == "completed"
    assert manifest["provenance"]["campaign_id"] is None


def test_run_records_warnings_in_manifest(tmp_path):
    import warnings as w

    class _WarnExperiment(_LifecycleExperiment):
        def setup(self) -> None:
            super().setup()
            w.warn("careful: lifecycle run", UserWarning, stacklevel=2)

    trainer = _WarnExperiment(_config(), runs_root=tmp_path)
    trainer.execute()
    manifest = yaml.safe_load((trainer.run_dir / "manifest.yaml").read_text())
    assert any("careful: lifecycle run" in s for s in manifest["warnings"])


def test_group_experiment_runs_real_training_path(tmp_path):
    config = ProjectConfig(
        device="cpu",
        data={"group": "C4", "train_frac": 0.5},
        model={"d_model": 16, "d_mlp": 32, "n_heads": 1},
        optim={"epochs": 2, "log_every": 1, "print_every": 1},
        snapshot={"enabled": False},
        logging=LoggingConfig(mode="disabled"),
        experiment=ExperimentConfig(name="group-smoke"),
    )
    experiment = GroupGeneralizationExperiment(config, runs_root=tmp_path)
    summary = experiment.execute()

    assert experiment.global_step == 2
    assert experiment.current_epoch == 1
    assert set(summary) >= {"train/loss", "val/loss", "val/accuracy"}
    assert (experiment.run_dir / "checkpoints" / "final.pt").is_file()
    assert "epoch 0 |" in (experiment.run_dir / "run.log").read_text()


def test_checkpoints_never_include_optimizer_state(tmp_path):
    """No resume path exists, so neither a trajectory snapshot nor the final
    checkpoint carries optimizer_state_dict."""
    config = ProjectConfig(
        device="cpu",
        data={"group": "C4", "train_frac": 0.5},
        model={"d_model": 16, "d_mlp": 32, "n_heads": 1},
        optim={"epochs": 1, "log_every": 1, "print_every": 1},
        logging=LoggingConfig(mode="disabled"),
        experiment=ExperimentConfig(name="no-optimizer-state"),
    )
    trainer = GroupGeneralizationExperiment(config, runs_root=tmp_path)
    trainer.execute()

    snapshot = torch.load(trainer.run_dir / "checkpoints" / "step_0.pt", weights_only=False)
    final = torch.load(trainer.run_dir / "checkpoints" / "final.pt", weights_only=False)

    assert "optimizer_state_dict" not in snapshot
    assert "optimizer_state_dict" not in final
    assert "model_state_dict" in snapshot
    assert "model_state_dict" in final


def test_snapshot_enabled_and_save_final_are_independent(tmp_path):
    """SnapshotConfig has two independent switches: `enabled` gates periodic
    trajectory capture (step_N.pt / generalized_step_N.pt), `save_final`
    gates the one final.pt snapshot regardless of `enabled`. Disabling
    periodic capture must not silently disable the final checkpoint too."""
    disabled_periodic = ProjectConfig(
        device="cpu",
        data={"group": "C4", "train_frac": 0.5},
        model={"d_model": 16, "d_mlp": 32, "n_heads": 1},
        optim={"epochs": 1, "log_every": 1, "print_every": 1},
        snapshot={"enabled": False},
        logging=LoggingConfig(mode="disabled"),
        experiment=ExperimentConfig(name="snapshot-periodic-off"),
    )
    trainer = GroupGeneralizationExperiment(disabled_periodic, runs_root=tmp_path)
    trainer.execute()
    assert not (trainer.run_dir / "checkpoints" / "step_0.pt").is_file()
    assert (trainer.run_dir / "checkpoints" / "final.pt").is_file()

    no_final = ProjectConfig(
        device="cpu",
        data={"group": "C4", "train_frac": 0.5},
        model={"d_model": 16, "d_mlp": 32, "n_heads": 1},
        optim={"epochs": 1, "log_every": 1, "print_every": 1},
        snapshot={"save_final": False},
        logging=LoggingConfig(mode="disabled"),
        experiment=ExperimentConfig(name="snapshot-no-final"),
    )
    trainer2 = GroupGeneralizationExperiment(no_final, runs_root=tmp_path)
    trainer2.execute()
    assert (trainer2.run_dir / "checkpoints" / "step_0.pt").is_file()
    assert not (trainer2.run_dir / "checkpoints" / "final.pt").is_file()


def test_final_window_snapshots_last_n_epochs(tmp_path):
    """`snapshot.final_window_epochs=N` writes final_epoch_<E>.pt for exactly
    the last N epochs of the ceiling -- independent of `enabled`, like
    `save_final` -- and leaves final.pt untouched. The last window snapshot
    carries the same weights final.pt does (nothing steps after the last
    epoch), so a dipped final.pt can always be swapped for an earlier window
    epoch post hoc."""
    config = ProjectConfig(
        device="cpu",
        data={"group": "C4", "train_frac": 0.5},
        model={"d_model": 16, "d_mlp": 32, "n_heads": 1},
        optim={"epochs": 8, "log_every": 1, "print_every": 8},
        snapshot={"enabled": False, "final_window_epochs": 3},
        logging=LoggingConfig(mode="disabled"),
        experiment=ExperimentConfig(name="final-window"),
    )
    trainer = GroupGeneralizationExperiment(config, runs_root=tmp_path)
    trainer.execute()

    ckpts = trainer.run_dir / "checkpoints"
    window = sorted(p.name for p in ckpts.glob("final_epoch_*.pt"))
    assert window == ["final_epoch_5.pt", "final_epoch_6.pt", "final_epoch_7.pt"]
    assert (ckpts / "final.pt").is_file()

    last = torch.load(ckpts / "final_epoch_7.pt", weights_only=False)
    final = torch.load(ckpts / "final.pt", weights_only=False)
    assert last["epoch"] == 7
    assert "model_state_dict" in last
    assert "optimizer_state_dict" not in last
    for name, tensor in final["model_state_dict"].items():
        assert torch.equal(tensor, last["model_state_dict"][name])


def test_final_window_zero_disables_the_window(tmp_path):
    """`final_window_epochs=0` writes no final_epoch_*.pt at all; final.pt is
    unaffected."""
    config = ProjectConfig(
        device="cpu",
        data={"group": "C4", "train_frac": 0.5},
        model={"d_model": 16, "d_mlp": 32, "n_heads": 1},
        optim={"epochs": 2, "log_every": 1, "print_every": 2},
        snapshot={"enabled": False, "final_window_epochs": 0},
        logging=LoggingConfig(mode="disabled"),
        experiment=ExperimentConfig(name="final-window-off"),
    )
    trainer = GroupGeneralizationExperiment(config, runs_root=tmp_path)
    trainer.execute()

    ckpts = trainer.run_dir / "checkpoints"
    assert list(ckpts.glob("final_epoch_*.pt")) == []
    assert (ckpts / "final.pt").is_file()


class _ScriptedEvalExperiment(GroupGeneralizationExperiment):
    """Real training path with ``_evaluate`` scripted, so stopping/summary
    tests don't depend on whether a tiny model happens to generalize.

    ``generalizes_from`` is the first epoch whose scripted val accuracy
    clears the bar; before it, accuracy is 0.0. The scripted val loss is
    ``1 / (epoch + 1)``, which uniquely identifies which epoch a recorded
    summary came from. Everything else -- model, optimizer step, train-side
    metrics, snapshotting -- is the production code path.
    """

    generalizes_from = 0

    def _evaluate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        accuracy = 1.0 if self.current_epoch >= self.generalizes_from else 0.0
        # Unleaked accuracy mirrors raw accuracy here: the generalisation streak
        # keys on unleaked accuracy (falling back to raw only when the unleaked
        # subset is empty), so both drive the patience counting identically and
        # this stays a test of the streak machinery, not of a real split.
        return (
            torch.tensor(1.0 / (self.current_epoch + 1)),
            torch.tensor(accuracy),
            torch.tensor(accuracy),
        )


def _stop_on_generalize_config(name: str, *, log_every: int, patience: int) -> ProjectConfig:
    return ProjectConfig(
        device="cpu",
        data={"group": "C4", "train_frac": 0.5},
        model={"d_model": 16, "d_mlp": 32, "n_heads": 1},
        optim={
            "epochs": 500,
            "log_every": log_every,
            "print_every": log_every,
            "wandb_every": log_every,
            "stop_on_generalize": True,
            "generalize_test_acc": 0.99,
            "generalize_patience": patience,
        },
        snapshot={"enabled": False},
        logging=LoggingConfig(mode="disabled"),
        experiment=ExperimentConfig(name=name),
    )


def test_generalize_patience_counts_epochs_not_log_events(tmp_path):
    """Regression test: `gen_streak` used to increment only inside the
    `should_log` block, so patience counted log events, not epochs. With
    log_every=10, patience=3, the old behaviour ran to roughly epoch 20 (log
    events at 0, 10, 20) before stopping; the fix stops after 3 qualifying
    epochs (global_step == 3), independent of log_every.

    ``_evaluate`` is scripted to clear the bar from epoch 0, so this tests
    patience counting alone. It previously ran the real model with
    ``generalize_test_acc=0.01`` and passed only on a seed lottery: a single
    epoch of 0.0 test accuracy resets the streak, and 2 of 12 seeds did
    exactly that."""
    config = _stop_on_generalize_config("patience-counts-epochs", log_every=10, patience=3)
    trainer = _ScriptedEvalExperiment(config, runs_root=tmp_path)
    trainer.execute()

    assert trainer.global_step == 3
    assert trainer.current_epoch == 2


def test_recorded_summary_is_the_epoch_the_run_stopped_on(tmp_path):
    """With `stop_on_generalize` armed, the loop evaluates every epoch but
    logs only every `log_every`-th, so `break` can land on a non-logged
    epoch. The summary dict used to be built only inside `if should_log:`, so
    `logger.set_summary()` / `finalize_manifest()` got the last logged epoch,
    up to `log_every` epochs stale.

    Concretely: val accuracy crosses the bar at epoch 10, patience 5 stops it
    at epoch 14, but with log_every=100 the only logged epoch is 0. Pre-fix
    the manifest recorded `completed_steps: 15` alongside `val/accuracy: 0.0`
    (it never generalized on the logged epoch).
    """
    config = _stop_on_generalize_config("summary-tracks-stop-epoch", log_every=100, patience=5)
    trainer = _ScriptedEvalExperiment(config, runs_root=tmp_path)
    trainer.generalizes_from = 10
    summary = trainer.execute()

    # Streak starts at epoch 10 and reaches 5 at epoch 14.
    assert trainer.current_epoch == 14
    assert trainer.global_step == 15

    assert summary["val/accuracy"] == 1.0
    assert summary["val/loss"] == pytest.approx(1.0 / 15.0)  # the scripted epoch-14 value
    # The train-side scalars are refreshed on the same (unlogged) epoch too:
    # val/weight_norm must be the norm of the weights the run actually ended on.
    expected_norm = torch.sqrt(
        sum(p.detach().pow(2).sum() for p in trainer.model.parameters())
    ).item()
    assert summary["val/weight_norm"] == pytest.approx(expected_norm, rel=1e-6)

    manifest = yaml.safe_load((trainer.run_dir / "manifest.yaml").read_text())
    assert manifest["status"] == "completed"
    assert manifest["summary"]["completed_steps"] == 15
    assert manifest["summary"]["metrics"]["val/accuracy"] == 1.0
    assert manifest["summary"]["metrics"]["val/loss"] == pytest.approx(1.0 / 15.0)


# ---------------------------------------------------------------------------
# Real end-to-end training determinism (the reproducibility claim).
#
# The tests above exercise lifecycle/checkpoint machinery. These exercise the
# real path used by scripts/run.py: GroupGeneralizationExperiment trained
# twice with a tiny config on CPU. They check the project's seeding plumbing
# (same seed -> same init, same shuffle, same split) under the default
# `deterministic: false`, not full cudnn/CUDA kernel determinism, which is a
# separate, more expensive guarantee (see the opt-in test at the bottom).
# ---------------------------------------------------------------------------


def _tiny_group_config(
    name: str, *, seed: int, split_seed: int | None = None, deterministic: bool = False
) -> ProjectConfig:
    """Smallest fixture group (C2, order 2) with a minimal model so two full
    training runs finish in well under a second on CPU."""
    return ProjectConfig(
        device="cpu",
        seed=seed,
        deterministic=deterministic,
        data={"group": "C2", "train_frac": 0.5, "split_seed": split_seed},
        model={"d_model": 8, "d_mlp": 16, "n_heads": 1},
        optim={"epochs": 5, "log_every": 1, "print_every": 1},
        snapshot={"enabled": False},
        logging=LoggingConfig(mode="disabled"),
        experiment=ExperimentConfig(name=name),
    )


def _assert_models_bit_identical(lhs: torch.nn.Module, rhs: torch.nn.Module) -> None:
    lhs_params = dict(lhs.named_parameters())
    rhs_params = dict(rhs.named_parameters())
    assert lhs_params.keys() == rhs_params.keys()
    for name, left_value in lhs_params.items():
        right_value = rhs_params[name]
        assert torch.equal(left_value, right_value), f"parameter {name!r} diverged"


def test_group_experiment_same_seed_produces_bit_identical_weights(tmp_path):
    """Same seed, CPU, default `deterministic: false` must produce
    bit-identical final weights (checked both in-memory and via the
    persisted checkpoint), an identical train/test split, and identical
    final metrics."""
    run_a = GroupGeneralizationExperiment(
        _tiny_group_config("determinism-a", seed=1234), runs_root=tmp_path
    )
    summary_a = run_a.execute()
    run_b = GroupGeneralizationExperiment(
        _tiny_group_config("determinism-b", seed=1234), runs_root=tmp_path
    )
    summary_b = run_b.execute()

    assert summary_a == summary_b

    _assert_models_bit_identical(run_a.model, run_b.model)

    assert torch.equal(run_a.train_tokens, run_b.train_tokens)
    assert torch.equal(run_a.train_targets, run_b.train_targets)
    assert torch.equal(run_a.test_tokens, run_b.test_tokens)
    assert torch.equal(run_a.test_targets, run_b.test_targets)

    # The persisted checkpoint too -- what a reviewer would load.
    ckpt_a = torch.load(run_a.run_dir / "checkpoints" / "final.pt", weights_only=False)
    ckpt_b = torch.load(run_b.run_dir / "checkpoints" / "final.pt", weights_only=False)
    assert ckpt_a["model_state_dict"].keys() == ckpt_b["model_state_dict"].keys()
    for key, left_value in ckpt_a["model_state_dict"].items():
        assert torch.equal(left_value, ckpt_b["model_state_dict"][key]), (
            f"checkpoint parameter {key!r} diverged"
        )


class _RecordingExperiment(GroupGeneralizationExperiment):
    """Keeps the initial weights and every logged metrics dict, so a test can
    check that training moved the weights and moved them in the right direction."""

    def setup(self) -> None:
        super().setup()
        self.initial_params = {
            name: parameter.detach().clone() for name, parameter in self.model.named_parameters()
        }
        self.history: list[dict[str, float]] = []

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.history.append(dict(metrics))
        super().log_metrics(metrics, step)


def _training_config(name: str, *, epochs: int) -> ProjectConfig:
    return ProjectConfig(
        device="cpu",
        seed=0,
        data={"group": "C4", "train_frac": 0.5, "split_seed": 0},
        model={"d_model": 16, "d_mlp": 32, "n_heads": 1},
        optim={"epochs": epochs, "log_every": 1, "print_every": epochs, "wandb_every": 1},
        snapshot={"enabled": False},
        logging=LoggingConfig(mode="disabled"),
        experiment=ExperimentConfig(name=name),
    )


def test_training_reduces_the_loss_and_moves_every_weight(tmp_path):
    """Verified gap: patching ``AdamW.step`` to a no-op left the determinism
    tests, both split tests, and the patience test green -- all satisfied by
    initialisation alone. This one is not: it requires the train loss to fall
    substantially and every parameter tensor to differ from its
    initialisation, which a no-op optimizer step cannot produce.
    """
    trainer = _RecordingExperiment(
        _training_config("does-it-train", epochs=200), runs_root=tmp_path
    )
    summary = trainer.execute()

    assert len(trainer.history) == 200
    first_loss = trainer.history[0]["train/loss"]
    last_loss = trainer.history[-1]["train/loss"]
    assert last_loss < first_loss / 2.0, f"train loss did not fall: {first_loss} -> {last_loss}"
    assert summary["train/loss"] == pytest.approx(last_loss)
    assert summary["train/accuracy"] == 1.0  # the train set is memorised by epoch 200

    for name, initial in trainer.initial_params.items():
        current = dict(trainer.model.named_parameters())[name]
        assert not torch.equal(initial, current.detach()), f"parameter {name!r} never moved"


def test_group_experiment_summary_metrics_are_recomputable_from_the_model(tmp_path):
    """Replaces a test that compared metrics to hard-coded magic constants --
    an oracle that could only say "different from last time", with
    ``abs=1e-9`` sitting on a ``val/accuracy`` whose quantum is 1/n_test.
    Each metric here is checked against its defining invariant, recomputed
    from the run's own artifacts:

    * ``val/*`` and ``val/weight_norm`` are post-update, recomputed directly
      from the final model;
    * ``train/*`` are pre-update (the last update's forward pass ran on the
      previous weights), recomputed from the same config with one epoch
      fewer -- whose final weights are, bit for bit, the weights the last
      epoch's train metrics were measured at.
    """
    epochs = 20
    trainer = GroupGeneralizationExperiment(
        _training_config("summary-invariants", epochs=epochs), runs_root=tmp_path
    )
    summary = trainer.execute()

    assert summary.keys() == {
        "train/loss",
        "train/accuracy",
        "val/loss",
        "val/accuracy",
        "val/unleaked_accuracy",
        "val/weight_norm",
    }

    with torch.no_grad():
        val_logits = trainer.model(trainer.test_tokens)[:, -1, :]
        val_loss = F.cross_entropy(val_logits, trainer.test_targets)
        val_correct_t = val_logits.argmax(dim=-1) == trainer.test_targets
        val_correct = int(val_correct_t.sum())
    n_test = len(trainer.test_targets)

    assert summary["val/loss"] == pytest.approx(val_loss.item(), rel=1e-6)
    assert summary["val/accuracy"] == pytest.approx(val_correct / n_test, rel=1e-6)
    # An accuracy is a count over n_test: it can only land on the grid k/n_test.
    assert summary["val/accuracy"] * n_test == pytest.approx(
        round(summary["val/accuracy"] * n_test)
    )

    # val/unleaked_accuracy is the same predictions masked to the transpose-
    # unleaked subset: a masked count over n_unleaked (NaN if that subset is
    # empty, which C4 at train_frac=0.5 is not).
    n_unleaked = int(trainer.unleaked_mask.sum())
    assert n_unleaked > 0
    val_unleaked_correct = int(val_correct_t[trainer.unleaked_mask].sum())
    assert summary["val/unleaked_accuracy"] == pytest.approx(
        val_unleaked_correct / n_unleaked, rel=1e-6
    )

    weight_norm = torch.sqrt(sum(p.detach().pow(2).sum() for p in trainer.model.parameters()))
    assert summary["val/weight_norm"] == pytest.approx(weight_norm.item(), rel=1e-6)

    # Same config, one epoch fewer: its final weights are exactly the weights the
    # full run's last epoch computed its (pre-update) train metrics on.
    previous = GroupGeneralizationExperiment(
        _training_config("summary-invariants-prev", epochs=epochs - 1), runs_root=tmp_path
    )
    previous.execute()
    with torch.no_grad():
        train_logits = previous.model(trainer.train_tokens)[:, -1, :]
        train_loss = F.cross_entropy(train_logits, trainer.train_targets)
        train_correct = int((train_logits.argmax(dim=-1) == trainer.train_targets).sum())
    n_train = len(trainer.train_targets)

    assert summary["train/loss"] == pytest.approx(train_loss.item(), rel=1e-6)
    assert summary["train/accuracy"] == pytest.approx(train_correct / n_train, rel=1e-6)


# ---------------------------------------------------------------------------
# Transpose-leak covariates + the transpose-unleaked eval metric (I-02).
# ---------------------------------------------------------------------------


def _leak_config(name: str, group: str = "C4", train_frac: float = 0.5) -> ProjectConfig:
    return ProjectConfig(
        device="cpu",
        data={"group": group, "train_frac": train_frac, "split_seed": 0},
        model={"d_model": 16, "d_mlp": 32, "n_heads": 1},
        optim={"epochs": 2, "log_every": 1, "print_every": 1},
        snapshot={"enabled": False},
        logging=LoggingConfig(mode="disabled"),
        experiment=ExperimentConfig(name=name),
    )


def test_run_records_leakage_covariates_matching_an_independent_recompute(tmp_path):
    """A real run records dataset.leakage on the realised split, and every number
    matches an independent recomputation from task.py on the same split."""
    from group_algorithm_interp.task import (
        build_group_task,
        commuting_probability,
        train_test_split,
        transpose_leak_fraction,
        transpose_leaked_mask,
    )

    config = _leak_config("leak-covariates")
    exp = GroupGeneralizationExperiment(config, runs_root=tmp_path)
    summary = exp.execute()

    assert "val/unleaked_accuracy" in summary
    assert not math.isnan(summary["val/unleaked_accuracy"])  # C4 0.5 is not degenerate

    manifest = yaml.safe_load((exp.run_dir / "manifest.yaml").read_text())
    leak = manifest["dataset"]["leakage"]

    task = build_group_task(exp.group)
    split = train_test_split(task, config.data.train_frac, config.effective_split_seed)
    mask = transpose_leaked_mask(split, exp.group.cayley_table)

    assert leak["transpose_leak_fraction"] == pytest.approx(
        transpose_leak_fraction(split, exp.group.cayley_table)
    )
    assert leak["commuting_probability"] == pytest.approx(commuting_probability(exp.group))
    assert leak["test_size"] == int(mask.shape[0])
    assert leak["unleaked_test_size"] == int((~mask).sum())
    assert leak["unleaked_empty"] is False
    assert leak["generalize_metric"] == "unleaked_accuracy"
    assert leak["test_size"] == leak["unleaked_test_size"] + int(mask.sum())


def test_empty_unleaked_subset_is_handled_and_recorded_loudly(tmp_path, monkeypatch):
    """Degenerate case: every test pair leaked (patch the leak mask to
    all-True). Must record NaN unleaked accuracy, a loud manifest note, and
    the raw-accuracy fallback, without dividing by zero."""
    import group_algorithm_interp.experiment as experiment_mod

    monkeypatch.setattr(
        experiment_mod,
        "transpose_leaked_mask",
        lambda split, table: np.ones(split.test_inputs.shape[0], dtype=bool),
    )

    exp = GroupGeneralizationExperiment(_leak_config("leak-degenerate"), runs_root=tmp_path)
    summary = exp.execute()

    assert exp.has_unleaked is False
    assert math.isnan(summary["val/unleaked_accuracy"])

    manifest = yaml.safe_load((exp.run_dir / "manifest.yaml").read_text())
    leak = manifest["dataset"]["leakage"]
    assert leak["unleaked_empty"] is True
    assert leak["unleaked_test_size"] == 0
    assert leak["transpose_leak_fraction"] == pytest.approx(1.0)
    assert leak["generalize_metric"] == "raw_test_accuracy"
    assert "EMPTY" in (leak["note"] or "")
    assert any("transpose-unleaked subset is EMPTY" in w for w in manifest["warnings"])


class _RawFallbackScripted(GroupGeneralizationExperiment):
    """Scripts raw accuracy to 1.0 while unleaked accuracy is NaN, so a
    stop-on-generalize run can only stop via the raw-accuracy fallback
    streak -- exercised together with a forced-empty unleaked subset."""

    def _evaluate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(0.5),
            torch.tensor(1.0),
            torch.tensor(float("nan")),
        )


def test_generalize_streak_falls_back_to_raw_accuracy_when_unleaked_is_empty(tmp_path, monkeypatch):
    """With the unleaked subset empty, the streak must fall back to raw
    accuracy: NaN>=bar is always False, so without the fallback the streak
    would never build and the run would never stop."""
    import group_algorithm_interp.experiment as experiment_mod

    monkeypatch.setattr(
        experiment_mod,
        "transpose_leaked_mask",
        lambda split, table: np.ones(split.test_inputs.shape[0], dtype=bool),
    )
    config = _stop_on_generalize_config("raw-fallback-stops", log_every=1, patience=3)
    exp = _RawFallbackScripted(config, runs_root=tmp_path)
    exp.execute()

    assert exp.has_unleaked is False
    assert exp.global_step == 3


def test_group_experiment_different_seeds_produce_different_weights(tmp_path):
    """Different seeds must give different trained weights and final metrics,
    or the seed sweep is not a sweep. This is NOT a control against a
    non-training loop: different seeds give different *initialisations*
    whether or not a gradient step ever runs (verified: stays green with
    ``AdamW.step`` stubbed to a no-op). The real control is
    ``test_training_reduces_the_loss_and_moves_every_weight``."""
    run_a = GroupGeneralizationExperiment(
        _tiny_group_config("determinism-seed-1", seed=1), runs_root=tmp_path
    )
    summary_a = run_a.execute()
    run_b = GroupGeneralizationExperiment(
        _tiny_group_config("determinism-seed-2", seed=2), runs_root=tmp_path
    )
    summary_b = run_b.execute()

    assert summary_a != summary_b
    assert not torch.equal(run_a.model.W_E, run_b.model.W_E)


# ---------------------------------------------------------------------------
# CUDA determinism plumbing. The GPU path itself cannot be executed here (no
# CUDA device on this machine); what IS testable on CPU is that the process is
# configured so the GPU path *can* be made deterministic if an operator opts in
# -- the cuBLAS workspace variable is in the environment before torch can touch
# CUDA -- and that nothing forces that opt-in on. CUDA runs are deliberately not
# bit-reproducible: determinism and TF32-off both cost throughput, and neither is
# imposed (see docs/reproducibility.md).
# ---------------------------------------------------------------------------


def test_cublas_workspace_config_is_set_before_torch_can_initialise_cuda():
    """Without this env var, ``torch.use_deterministic_algorithms(True)``
    makes the model's first cuBLAS einsum raise on CUDA -- every seed of a
    deterministic campaign dies on the pod. Set at import of
    ``group_algorithm_interp.seed`` (and again at the top of scripts/run.py),
    both before any CUDA initialisation."""
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_CONFIG


def test_deterministic_config_enables_deterministic_algorithms_for_the_run(tmp_path):
    """`deterministic: true` must reach torch.use_deterministic_algorithms --
    the call routing the embedding backward's atomicAdd (the model's one
    non-deterministic CUDA op) to a deterministic kernel. Lives in
    `set_seed`, not `GroupGeneralizationExperiment.setup()`, so every entry
    point gets it."""

    class _FlagCapturing(GroupGeneralizationExperiment):
        def setup(self) -> None:
            super().setup()
            self.deterministic_in_setup = torch.are_deterministic_algorithms_enabled()

    trainer = _FlagCapturing(
        _tiny_group_config("deterministic-on", seed=1234, deterministic=True), runs_root=tmp_path
    )
    try:
        trainer.execute()
    finally:
        torch.use_deterministic_algorithms(False)  # process-global; never leak it

    assert trainer.deterministic_in_setup is True
    # TF32 is orthogonal to determinism (TF32 matmuls are reproducible) and
    # stays on even for a deterministic run -- a throughput choice.
    assert torch.backends.cuda.matmul.fp32_precision == "tf32"
    assert torch.backends.cudnn.conv.fp32_precision == "tf32"


def test_tf32_is_enabled_for_matmul(tmp_path):
    """TF32 is turned on deliberately: it trades fp32 mantissa bits for a
    large matmul speedup on Ampere and later, the trade this project wants on
    rented GPUs. torch's own default leaves matmul TF32 off
    (`fp32_precision="none"`), so this needs an explicit call.

    Asserted through the new `fp32_precision` API only: torch 2.12 tracks
    which of the two APIs a process used and raises RuntimeError ("mix of the
    legacy and new APIs") on a legacy read after a new-API write. That is the
    trap this test pins -- the codebase must speak one dialect, the new one."""
    set_seed(0, deterministic=False)
    assert torch.backends.cuda.matmul.fp32_precision == "tf32"
    # cudnn conv TF32 is torch's own default (the models have no
    # convolutions); pinned here so a torch default flip is noticed rather
    # than silently changing numerics.
    assert torch.backends.cudnn.conv.fp32_precision == "tf32"

    with pytest.raises(RuntimeError, match="legacy and new APIs"):
        _ = torch.backends.cuda.matmul.allow_tf32


def test_cuda_run_does_not_force_determinism(tmp_path):
    """A CUDA run is not forced into deterministic kernels and is therefore
    not bit-reproducible: the embedding backward's atomicAdd means the same
    seed produces slightly different weights, a deliberate throughput choice
    (see docs/reproducibility.md). `deterministic` is a pure config opt-in on
    every device, defaulting to false; the harness never overrides it.

    Only the resolution logic is exercised here: this machine has no CUDA
    device."""
    trainer = GroupGeneralizationExperiment(
        _tiny_group_config("cuda-no-forced-determinism", seed=0), runs_root=tmp_path
    )
    assert trainer.config.deterministic is False

    # Same on CUDA as CPU: whatever the config said, nothing more.
    trainer.device = torch.device("cuda")  # not used to launch anything
    assert trainer.config.deterministic is False
    assert not hasattr(trainer, "_resolve_determinism")


def test_same_data_split_seed_gives_same_split_across_different_model_seeds(tmp_path):
    """Observational, not a design endorsement: `data.split_seed`, when set
    explicitly, decouples the split from `seed`. Two runs with different
    `seed` but the same explicit `data.split_seed` currently produce an
    identical split. Whether the default `data.split_seed=None` (which falls
    back to `seed`) should behave differently is a separate, open question
    this test does not take a position on."""
    run_a = GroupGeneralizationExperiment(
        _tiny_group_config("split-seed-a", seed=1, split_seed=999), runs_root=tmp_path
    )
    run_a.execute()
    run_b = GroupGeneralizationExperiment(
        _tiny_group_config("split-seed-b", seed=2, split_seed=999), runs_root=tmp_path
    )
    run_b.execute()

    assert torch.equal(run_a.train_tokens, run_b.train_tokens)
    assert torch.equal(run_a.train_targets, run_b.train_targets)
    assert torch.equal(run_a.test_tokens, run_b.test_tokens)
    assert torch.equal(run_a.test_targets, run_b.test_targets)
    # The split is identical, but the model seed still differs, so weights
    # still diverge -- split_seed only pins the split, nothing else.
    assert not torch.equal(run_a.model.W_E, run_b.model.W_E)


def test_multi_seed_sweep_with_pinned_split_seed_uses_one_identical_split(tmp_path):
    """The invariant a multi-seed campaign cell depends on: a sweep across
    seeds must vary only initialisation, not the train/test split. Before
    `data.split_seed` was pinned and emitted explicitly, each seed fell back
    to `split_seed=seed`, so every seed trained/tested on a different split,
    silently conflating initialisation noise with split noise. Builds four
    configs the way a launched sweep now does: `seed` varies, `data.split_seed`
    is the one pinned value shared by the campaign."""
    runs = [
        GroupGeneralizationExperiment(
            _tiny_group_config(f"sweep-seed-{seed}", seed=seed, split_seed=42),
            runs_root=tmp_path,
        )
        for seed in range(4)
    ]
    for run in runs:
        run.execute()

    first = runs[0]
    for other in runs[1:]:
        assert torch.equal(first.train_tokens, other.train_tokens)
        assert torch.equal(first.train_targets, other.train_targets)
        assert torch.equal(first.test_tokens, other.test_tokens)
        assert torch.equal(first.test_targets, other.test_targets)

    # Negative control: seeds must still differ in initialisation, or the
    # identical-split assertion above would be vacuous.
    assert not torch.equal(runs[0].model.W_E, runs[1].model.W_E)


_FULL_DETERMINISM_ENV_VAR = "RUN_FULL_DETERMINISM_TEST"


@pytest.mark.skipif(
    os.environ.get(_FULL_DETERMINISM_ENV_VAR) != "1",
    reason=(
        "Opt-in: exercises torch.use_deterministic_algorithms(True) end-to-end. "
        "The project does not enable full determinism by default (too slow for "
        f"routine CPU training), so this stays out of the gate. Set "
        f"{_FULL_DETERMINISM_ENV_VAR}=1 to run it as a verified path for if/when "
        "full determinism is turned on for a real run."
    ),
)
def test_group_experiment_same_seed_bit_identical_with_full_determinism(tmp_path):
    """Opt-in companion to the default-settings determinism test above: same
    check, with `deterministic: true` (torch.use_deterministic_algorithms)."""
    try:
        run_a = GroupGeneralizationExperiment(
            _tiny_group_config("full-determinism-a", seed=1234, deterministic=True),
            runs_root=tmp_path,
        )
        summary_a = run_a.execute()
        run_b = GroupGeneralizationExperiment(
            _tiny_group_config("full-determinism-b", seed=1234, deterministic=True),
            runs_root=tmp_path,
        )
        summary_b = run_b.execute()

        assert summary_a == summary_b
        _assert_models_bit_identical(run_a.model, run_b.model)
        assert torch.equal(run_a.train_tokens, run_b.train_tokens)
        assert torch.equal(run_a.test_tokens, run_b.test_tokens)
    finally:
        # process-global; reset it so this opt-in test never leaks state.
        torch.use_deterministic_algorithms(False)
