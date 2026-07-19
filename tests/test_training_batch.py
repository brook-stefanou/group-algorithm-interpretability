"""Tests for the vmapped ensemble trainer (training/ensemble.py).

The batch path trains many seeds of one ``(group, width)`` together through
``torch.func.vmap``. These tests pin the invariants that make it a safe parallel
path to the one-seed-per-process trainer:

* per-model init is bit-identical to the single path (only the init is
  guaranteed identical -- see the trajectory test's comment);
* the paired-seed design is enforced (a pinned split is required);
* every seed gets a complete, terminal, single-run-schema manifest, including on
  the abort path.

All CPU, fast, offline.
"""

from __future__ import annotations

import os
import signal
import types

import pytest
import torch
import yaml

from group_algorithm_interp.config import (
    ExperimentConfig,
    LoggingConfig,
    ProjectConfig,
)
from group_algorithm_interp.experiment import GroupGeneralizationExperiment, RunAborted
from group_algorithm_interp.groups.catalog import resolve_group
from group_algorithm_interp.seed import set_seed
from group_algorithm_interp.training.ensemble import (
    BatchEnsembleTrainer,
    build_stacked_ensemble,
    compute_chunk_size,
    pack_chunks,
)
from group_algorithm_interp.training.trainer import build_model


def _config(
    *,
    epochs: int = 3,
    split_seed: int | None = 0,
    name: str = "batch-test",
) -> ProjectConfig:
    return ProjectConfig(
        device="cpu",
        data={"group": "C4", "train_frac": 0.5, "split_seed": split_seed},
        model={"d_model": 16, "d_mlp": 32, "n_heads": 1},
        optim={"epochs": epochs, "log_every": 1, "print_every": epochs},
        snapshot={"enabled": False},
        logging=LoggingConfig(mode="disabled"),
        experiment=ExperimentConfig(name=name),
    )


# ---------------------------------------------------------------------------
# 1. Per-model init is bit-identical to the single path.
# ---------------------------------------------------------------------------


def test_stacked_init_is_bit_identical_to_single_path():
    """The critical invariant: the k-th slice of the stacked init equals, byte
    for byte, the model ``scripts/run.py`` would build for that seed. Checked for
    a few seeds -- the single-path init is reconstructed here independently
    (set_seed(seed) then build_model), not via the trainer's own helper."""
    config = _config()
    group = resolve_group(config.data.group)
    seeds = [0, 7, 42]

    params, buffers, _ = build_stacked_ensemble(config, group, seeds, torch.device("cpu"))

    for index, seed in enumerate(seeds):
        set_seed(seed, deterministic=config.deterministic)
        single = build_model(config, group)
        for name, tensor in single.named_parameters():
            assert torch.equal(tensor, params[name][index]), (
                f"seed {seed}: parameter {name!r} diverged from the single path"
            )
        # Buffers (the causal mask) must match too.
        for name, tensor in single.named_buffers():
            assert torch.equal(tensor, buffers[name][index])


def test_different_seeds_give_different_stacked_slices():
    """Sanity: the stacked slices are actually different per seed (else the
    bit-identity check above would be vacuous)."""
    config = _config()
    group = resolve_group(config.data.group)
    params, _, _ = build_stacked_ensemble(config, group, [1, 2], torch.device("cpu"))
    assert not torch.equal(params["W_E"][0], params["W_E"][1])


# ---------------------------------------------------------------------------
# 2. The paired-seed design: a pinned split is required.
# ---------------------------------------------------------------------------


def test_requires_pinned_split_seed():
    """run_batch requires a pinned data.split_seed so all K models share one
    split."""
    config = _config(split_seed=None)
    with pytest.raises(ValueError, match="pinned data.split_seed"):
        BatchEnsembleTrainer(config, [0, 1, 2])


# ---------------------------------------------------------------------------
# 3. A K=3 short run completes with 3 valid, distinct, single-run-schema
#    manifests.
# ---------------------------------------------------------------------------


def test_k3_short_run_completes_with_valid_per_model_manifests(tmp_path):
    config = _config(epochs=3)
    trainer = BatchEnsembleTrainer(config, [0, 1, 2], runs_root=tmp_path)
    statuses = trainer.run()

    assert statuses == {0: "completed", 1: "completed", 2: "completed"}
    assert len(trainer._runs) == 3

    val_losses = []
    for run in trainer._runs:
        # Provenance files a single run would leave, so sync_runs / the analysis
        # pipeline read a batch seed unchanged.
        assert (run.run_dir / "resolved_config.yaml").is_file()
        assert (run.run_dir / "checkpoints" / "final.pt").is_file()
        log_text = (run.run_dir / "run.log").read_text()
        assert log_text.strip()
        assert "epoch 0 |" in log_text

        manifest = yaml.safe_load((run.run_dir / "manifest.yaml").read_text())
        assert manifest["status"] == "completed"
        assert manifest["completed_at"] is not None
        assert manifest["summary"]["completed_steps"] == 3
        metrics = manifest["summary"]["metrics"]
        assert set(metrics) == {
            "train/loss",
            "train/accuracy",
            "val/loss",
            "val/accuracy",
            "val/unleaked_accuracy",
            "val/weight_norm",
        }
        # C4 at train_frac=0.5 is not degenerate: the unleaked metric is present
        # and finite.
        assert metrics["val/unleaked_accuracy"] == metrics["val/unleaked_accuracy"]  # not NaN
        assert manifest["dataset"]["split_seed"] == 0  # the one shared, pinned split
        val_losses.append(metrics["val/loss"])

    # Distinct seeds -> distinct trained models -> distinct metrics.
    assert len(set(val_losses)) == 3

    # The shared split really is shared: one spec_hash across the batch.
    spec_hashes = {
        yaml.safe_load((run.run_dir / "manifest.yaml").read_text())["dataset"]["spec_hash"]
        for run in trainer._runs
    }
    assert len(spec_hashes) == 1
    # ...but distinct full-config hashes (seed differs).
    config_hashes = {
        yaml.safe_load((run.run_dir / "manifest.yaml").read_text())["provenance"]["config_hash"]
        for run in trainer._runs
    }
    assert len(config_hashes) == 3


def test_final_checkpoint_has_single_run_schema(tmp_path):
    """Each seed's final.pt carries the same schema a single run writes:
    step/config/epoch/model_state_dict, no optimizer state, no resume state."""
    config = _config(epochs=2)
    trainer = BatchEnsembleTrainer(config, [0, 1], runs_root=tmp_path)
    trainer.run()

    ckpt = torch.load(trainer._runs[0].run_dir / "checkpoints" / "final.pt", weights_only=False)
    assert ckpt["step"] == 2
    assert ckpt["epoch"] == 1
    assert "config" in ckpt
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" not in ckpt
    assert "rng_state" not in ckpt
    # The state dict slice is a real per-model state dict (W_E etc. present).
    assert "W_E" in ckpt["model_state_dict"]


def test_final_window_snapshots_per_model(tmp_path):
    """`snapshot.final_window_epochs=N` writes final_epoch_<E>.pt for exactly
    the last N epochs, per model, with the single path's naming and checkpoint
    schema -- independent of `enabled` -- and each seed's window snapshot holds
    that seed's own weights (its last one matches its final.pt)."""
    config = _config(epochs=4)
    config.snapshot.final_window_epochs = 2
    trainer = BatchEnsembleTrainer(config, [0, 1], runs_root=tmp_path)
    statuses = trainer.run()
    assert statuses == {0: "completed", 1: "completed"}

    for run in trainer._runs:
        ckpts = run.run_dir / "checkpoints"
        window = sorted(p.name for p in ckpts.glob("final_epoch_*.pt"))
        assert window == ["final_epoch_2.pt", "final_epoch_3.pt"]
        last = torch.load(ckpts / "final_epoch_3.pt", weights_only=False)
        assert last["epoch"] == 3
        assert last["config"]["seed"] == run.seed
        final = torch.load(ckpts / "final.pt", weights_only=False)
        for name, tensor in final["model_state_dict"].items():
            assert torch.equal(tensor, last["model_state_dict"][name])

    # Per-model, not shared: the two seeds' window snapshots differ.
    a = torch.load(
        trainer._runs[0].run_dir / "checkpoints" / "final_epoch_3.pt", weights_only=False
    )
    b = torch.load(
        trainer._runs[1].run_dir / "checkpoints" / "final_epoch_3.pt", weights_only=False
    )
    assert not torch.equal(a["model_state_dict"]["W_E"], b["model_state_dict"]["W_E"])


# ---------------------------------------------------------------------------
# 4. The abort path: a SIGTERM mid-run finalises every unfinished manifest.
# ---------------------------------------------------------------------------


class _SigtermMidRun(BatchEnsembleTrainer):
    def _on_epoch_end(self, epoch: int) -> None:
        os.kill(os.getpid(), signal.SIGTERM)
        # Give the handler somewhere to land (it fires between bytecodes).
        for _ in range(10_000):
            pass
        raise AssertionError("SIGTERM did not interrupt the batch")


def test_sigterm_finalises_every_manifest_as_aborted(tmp_path):
    """Mirror of the single-run SIGTERM discipline: a real SIGTERM to the batch
    must convert into an unwind that finalises every seed's manifest to
    ``aborted`` (never stuck at ``running``), and restore the previous handler
    on the way out."""
    before = signal.getsignal(signal.SIGTERM)
    trainer = _SigtermMidRun(_config(epochs=50), [0, 1, 2], runs_root=tmp_path)
    with pytest.raises(RunAborted, match="SIGTERM"):
        trainer.run()

    for run in trainer._runs:
        manifest = yaml.safe_load((run.run_dir / "manifest.yaml").read_text())
        assert manifest["status"] == "aborted"
        assert manifest["completed_at"] is not None
        assert "RunAborted" in manifest["error"]
    assert signal.getsignal(signal.SIGTERM) is before


class _RaiseMidRun(BatchEnsembleTrainer):
    def _on_epoch_end(self, epoch: int) -> None:
        raise KeyboardInterrupt("operator pressed ctrl-c")


def test_keyboard_interrupt_finalises_every_manifest_as_aborted(tmp_path):
    trainer = _RaiseMidRun(_config(epochs=50), [0, 1, 2], runs_root=tmp_path)
    with pytest.raises(KeyboardInterrupt):
        trainer.run()
    for run in trainer._runs:
        manifest = yaml.safe_load((run.run_dir / "manifest.yaml").read_text())
        assert manifest["status"] == "aborted"


# ---------------------------------------------------------------------------
# 5. Trajectory tracks the single path (loosely -- NOT bit-identical).
# ---------------------------------------------------------------------------


def test_batch_trajectory_tracks_single_path_loosely(tmp_path):
    """Same seed and same pinned split as a single run: the batch model's loss
    should closely track the single path's.

    Bit-identity of the trajectory is deliberately not asserted: ``vmap``
    routes the forward/backward through batched matmul kernels whose
    floating-point reduction order differs from the unbatched single path, so
    the two diverge by small amounts that compound over epochs. Only the init
    is bit-identical (see test_stacked_init_is_bit_identical_to_single_path);
    the curves here are checked close, not equal.
    """
    epochs = 8
    seed = 5

    single_config = _config(epochs=epochs, name="single-ref")
    single_config.seed = seed
    single = GroupGeneralizationExperiment(single_config, runs_root=tmp_path)
    single_summary = single.execute()

    batch = BatchEnsembleTrainer(
        _config(epochs=epochs, name="batch-ref"), [seed], runs_root=tmp_path
    )
    batch.run()
    batch_summary = batch._runs[0].summary

    assert batch_summary["train/loss"] == pytest.approx(
        single_summary["train/loss"], rel=0.05, abs=0.05
    )
    assert batch_summary["val/loss"] == pytest.approx(
        single_summary["val/loss"], rel=0.05, abs=0.05
    )


# ---------------------------------------------------------------------------
# 6. Memory-aware chunking: chunk-size arithmetic (injected fake values).
# ---------------------------------------------------------------------------


class TestComputeChunkSize:
    """The chunk-size formula in isolation -- pure arithmetic, no torch/CUDA, so
    it is testable with injected free-memory / per-model figures."""

    def test_floor_division(self):
        # budget = 80 GiB * 0.8 = 64 GiB; 64 / 2.0 = 32 (exact).
        gib = 1024**3
        assert (
            compute_chunk_size(50, per_model_bytes=2.0 * gib, free_bytes=80.0 * gib, safety=0.8)
            == 32
        )

    def test_floor_not_round(self):
        # budget/per_model = 3.9 -> floor 3, never 4.
        assert compute_chunk_size(50, per_model_bytes=10.0, free_bytes=39.0, safety=1.0) == 3

    def test_clamped_to_n_seeds(self):
        # Plenty of room for all seeds: never more than K.
        assert compute_chunk_size(8, per_model_bytes=1.0, free_bytes=10_000.0, safety=0.8) == 8

    def test_at_least_one(self):
        # A single model larger than the whole budget still yields a chunk of 1
        # (it will likely OOM, but chunking cannot help below one model).
        assert compute_chunk_size(50, per_model_bytes=1_000.0, free_bytes=10.0, safety=0.8) == 1

    def test_safety_fraction_shrinks_budget(self):
        # Halving the safety fraction halves the usable budget -> halves the chunk.
        big = compute_chunk_size(100, per_model_bytes=1.0, free_bytes=100.0, safety=0.8)
        small = compute_chunk_size(100, per_model_bytes=1.0, free_bytes=100.0, safety=0.4)
        assert big == 80
        assert small == 40

    def test_degenerate_per_model_falls_back_to_all(self):
        # A non-positive measurement means "no useful estimate" -> one chunk.
        assert compute_chunk_size(50, per_model_bytes=0.0, free_bytes=80.0, safety=0.8) == 50


# ---------------------------------------------------------------------------
# 7. Chunk packing: fullest chunks first, remainder last.
# ---------------------------------------------------------------------------


class TestPackChunks:
    def test_pack_50_at_30(self):
        chunks = pack_chunks(list(range(50)), 30)
        assert [len(c) for c in chunks] == [30, 20]
        assert chunks[0] == list(range(30))
        assert chunks[1] == list(range(30, 50))

    def test_pack_50_at_17(self):
        chunks = pack_chunks(list(range(50)), 17)
        assert [len(c) for c in chunks] == [17, 17, 16]

    def test_pack_exact_multiple(self):
        assert [len(c) for c in pack_chunks(list(range(50)), 25)] == [25, 25]

    def test_pack_single_chunk_when_size_ge_n(self):
        assert pack_chunks(list(range(4)), 4) == [[0, 1, 2, 3]]
        assert pack_chunks(list(range(4)), 99) == [[0, 1, 2, 3]]

    def test_pack_preserves_order(self):
        assert pack_chunks([5, 3, 9, 1], 2) == [[5, 3], [9, 1]]


# ---------------------------------------------------------------------------
# 8. --max-models-per-batch forces multiple chunks; results match unchunked.
# ---------------------------------------------------------------------------


def test_max_models_forces_multiple_chunks_all_complete(tmp_path):
    """K=4 with --max-models-per-batch=2 trains as two consecutive chunks; all
    four seeds reach ``completed`` with their own single-run-schema manifest and
    correct per-seed config."""
    config = _config(epochs=3)
    trainer = BatchEnsembleTrainer(config, [0, 1, 2, 3], runs_root=tmp_path, max_models_per_batch=2)
    statuses = trainer.run()

    assert statuses == {0: "completed", 1: "completed", 2: "completed", 3: "completed"}
    assert len(trainer._runs) == 4

    seeds_on_record = set()
    for run in trainer._runs:
        manifest = yaml.safe_load((run.run_dir / "manifest.yaml").read_text())
        assert manifest["status"] == "completed"
        assert manifest["summary"]["completed_steps"] == 3
        resolved = yaml.safe_load((run.run_dir / "resolved_config.yaml").read_text())
        seeds_on_record.add(resolved["seed"])
        # The final checkpoint is sliced from the correct chunk (W_E present).
        ckpt = torch.load(run.run_dir / "checkpoints" / "final.pt", weights_only=False)
        assert "W_E" in ckpt["model_state_dict"]
    assert seeds_on_record == {0, 1, 2, 3}


def test_chunked_matches_unchunked_per_seed(tmp_path):
    """Seeds are independent, so chunking must not change any seed's result. Init
    bit-identity guarantees it: seed k's stacked init is a function of seed k
    alone, identical whether it trains in a chunk of 2 or of 4 -- asserted here
    for one seed -- and each seed's final metrics match the unchunked run."""
    seeds = [0, 1, 2, 3]
    group = resolve_group(_config().data.group)

    # Init bit-identity for one seed (seed 2): its slice is identical whether the
    # ensemble spans all four seeds or just the [2, 3] chunk.
    full_params, _, _ = build_stacked_ensemble(_config(), group, seeds, torch.device("cpu"))
    chunk_params, _, _ = build_stacked_ensemble(_config(), group, [2, 3], torch.device("cpu"))
    for name in full_params:
        assert torch.equal(full_params[name][2], chunk_params[name][0]), (
            f"seed 2: {name!r} init differs between the 4-seed ensemble and its chunk"
        )

    # Full-run equivalence: unchunked (one chunk of 4) vs chunked (two chunks
    # of 2). Each seed's final metrics must agree.
    unchunked = BatchEnsembleTrainer(
        _config(epochs=4, name="unchunked"), seeds, runs_root=tmp_path / "unchunked"
    )
    unchunked.run()
    chunked = BatchEnsembleTrainer(
        _config(epochs=4, name="chunked"),
        seeds,
        runs_root=tmp_path / "chunked",
        max_models_per_batch=2,
    )
    chunked.run()

    by_seed_unchunked = {run.seed: run.summary for run in unchunked._runs}
    by_seed_chunked = {run.seed: run.summary for run in chunked._runs}
    for seed in seeds:
        for key in ("train/loss", "val/loss", "val/accuracy"):
            assert by_seed_chunked[seed][key] == pytest.approx(
                by_seed_unchunked[seed][key], rel=1e-4, abs=1e-5
            ), f"seed {seed} {key}: chunked diverged from unchunked"


# ---------------------------------------------------------------------------
# 9. Abort mid-chunk finalises the current chunk AND not-yet-started seeds.
# ---------------------------------------------------------------------------


class _SigtermFirstChunk(BatchEnsembleTrainer):
    """Fire a real SIGTERM during the first chunk's first epoch, so the second
    chunk never starts training."""

    def _on_epoch_end(self, epoch: int) -> None:
        os.kill(os.getpid(), signal.SIGTERM)
        for _ in range(10_000):
            pass
        raise AssertionError("SIGTERM did not interrupt the batch")


def test_abort_mid_chunk_finalises_every_seed(tmp_path):
    """With chunking, a SIGTERM during chunk 1 must still leave every seed's
    manifest terminal -- the current chunk's models and the seeds chunk 2
    would have trained -- because all manifests are created up front, before
    any chunk trains."""
    before = signal.getsignal(signal.SIGTERM)
    trainer = _SigtermFirstChunk(
        _config(epochs=50), [0, 1, 2, 3], runs_root=tmp_path, max_models_per_batch=2
    )
    with pytest.raises(RunAborted, match="SIGTERM"):
        trainer.run()

    assert len(trainer._runs) == 4  # all seeds got a run dir up front
    for run in trainer._runs:
        manifest = yaml.safe_load((run.run_dir / "manifest.yaml").read_text())
        assert manifest["status"] == "aborted", f"seed {run.seed} not finalised"
        assert manifest["completed_at"] is not None
        assert "RunAborted" in manifest["error"]
    assert signal.getsignal(signal.SIGTERM) is before


# ---------------------------------------------------------------------------
# 10. Estimation: chunk-size decision arithmetic + no-artefacts property.
# ---------------------------------------------------------------------------


class _FakeMeasureTrainer(BatchEnsembleTrainer):
    """Drives the CUDA chunk-decision branch on a CPU box: pretends to be on a
    CUDA device and returns injected per-model / free-memory figures instead of
    reading real GPU stats."""

    def __init__(self, *args, fake_per_model: int, fake_free: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = types.SimpleNamespace(type="cuda")  # type: ignore[assignment]
        self._fake_per_model = fake_per_model
        self._fake_free = fake_free
        self.measured = False

    def _measure_per_model_cost(self, shared) -> int:  # type: ignore[override]
        self.measured = True
        return self._fake_per_model

    def _free_memory(self) -> int:  # type: ignore[override]
        return self._fake_free


def test_resolve_chunk_size_uses_estimate_on_cuda():
    """The CUDA branch measures per-model cost, reads free memory, and applies
    the formula. A fake measurer stands in for the GPU reads."""
    gib = 1024**3
    logs: list[str] = []
    trainer = _FakeMeasureTrainer(
        _config(),
        list(range(50)),
        fake_per_model=2 * gib,
        fake_free=80 * gib,
        log=logs.append,
    )
    size = trainer._resolve_chunk_size(shared=None)  # shared ignored by the fake
    assert trainer.measured
    assert size == 32  # floor(80 GiB * 0.8 / 2 GiB)
    assert any("chunk size 32" in line for line in logs)


def test_resolve_chunk_size_max_override_skips_estimation():
    """--max-models-per-batch is a hard override: no measurement happens."""
    logs: list[str] = []
    trainer = _FakeMeasureTrainer(
        _config(),
        list(range(50)),
        fake_per_model=1,
        fake_free=1,
        max_models_per_batch=30,
        log=logs.append,
    )
    size = trainer._resolve_chunk_size(shared=None)
    assert size == 30
    assert not trainer.measured  # estimation skipped entirely
    assert any("estimation skipped" in line for line in logs)


def test_resolve_chunk_size_cpu_is_single_chunk():
    """A real CPU trainer does not chunk by default: one chunk of every seed."""
    logs: list[str] = []
    trainer = BatchEnsembleTrainer(_config(), list(range(7)), log=logs.append)
    assert trainer._resolve_chunk_size(shared=None) == 7
    assert any("no chunking" in line for line in logs)


def test_estimation_trial_leaves_no_artefacts(tmp_path):
    """The K=1 measurement trial must not create any run directory, manifest,
    or checkpoint -- it is a purely in-memory probe. Exercised on CPU, since
    the trial itself makes no CUDA calls."""
    config = _config(epochs=3)
    trainer = BatchEnsembleTrainer(config, [0, 1, 2, 3], runs_root=tmp_path)
    shared = trainer._prepare_shared_data()

    trainer._estimation_trial(shared)

    assert list(tmp_path.iterdir()) == [], "the estimation trial left an artefact behind"
    assert trainer._runs == []  # no manifests registered
