"""Vmapped ensemble training: many models of one ``(group, width)`` trained
together in a single batched kernel on one device.

This is a *parallel* path to the one-seed-per-process trainer
(``scripts/run.py`` + ``GroupGeneralizationExperiment``), not a replacement.
The single path stays the bit-identity reference and the fallback; this path
trades exact per-kernel reproducibility for throughput by co-batching ``K``
models through ``torch.func.vmap``. What it deliberately keeps identical:

* Per-model weight init is bit-identical to the single path: model ``k``'s
  initial weights are a function of ``seed_k`` alone -- see
  :func:`build_stacked_ensemble`. The stacked init's ``k``-th slice equals,
  byte for byte, what ``scripts/run.py`` produces for that seed.
* Per-model provenance is the single-run schema: each seed gets its own run
  directory, ``manifest.yaml``, ``resolved_config.yaml``, ``run.log`` eval
  history, and checkpoints, with a distinct run id -- so ``scripts/sync_runs.py``
  and the analysis pipeline read a batch-trained seed exactly as a single run.

What is different, on purpose:

* Early stopping is disabled: every model runs to ``optim.epochs`` (the
  campaign's ceiling). ``optim.stop_on_generalize`` is ignored: there is no
  compaction and no eviction, so a censored seed simply reaches the ceiling.
* Trajectories are not bit-identical to the single path: ``vmap`` routes the
  forward/backward through batched matmul kernels whose reduction order differs
  from the unbatched path, so the same seed and split produce *closely tracking*
  but not identical loss curves. Only the *init* is guaranteed identical.

Design contract (settled): the campaign pins ``data.split_seed``, so every model
in a batch shares ONE dataset / split / transpose-unleaked mask (the paired-seed
design). :class:`BatchEnsembleTrainer` therefore **requires** a pinned split and
asserts it.

Memory scales with ``K``: peak memory is roughly ``K * model_size`` (stacked
params) ``+ K * adam_state`` (two moments per param) ``+ K * activation`` memory
for the full-batch forward. ``K`` is chosen implicitly by the caller through the
length of the seeds list, so a caller picks the batch size by picking how many
seeds to hand one process.
"""

from __future__ import annotations

import math
import os
import signal
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, stack_module_state, vmap

from ..config import ProjectConfig
from ..experiment import RunAborted, _format_warnings, _raise_aborted
from ..groups.group import FiniteGroup
from ..manifest import (
    create_manifest,
    create_run_dir,
    create_run_id,
    finalize_manifest,
    record_leakage,
    record_tracking,
    save_resolved_config,
)
from ..seed import resolve_device, set_seed
from ..task import (
    build_group_task,
    commuting_probability,
    train_test_split,
    transpose_leaked_mask,
)
from .trainer import build_model, should_snapshot

# The per-model summary keys, identical to GroupGeneralizationExperiment.run()'s
# returned metrics dict so a batch-trained manifest is schema-compatible with a
# single-run manifest.
_SUMMARY_KEYS = (
    "train/loss",
    "train/accuracy",
    "val/loss",
    "val/accuracy",
    "val/unleaked_accuracy",
    "val/weight_norm",
)


def build_stacked_ensemble(
    config: ProjectConfig,
    group: FiniteGroup,
    seeds: list[int],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.nn.Module]:
    """Build ``K`` per-seed models and stack their parameters/buffers along a new
    leading ``K`` dimension.

    The critical invariant (per-model RNG): model ``k`` is built after seeding
    every RNG with ``seed_k`` exactly as ``BaseExperiment.execute`` does before a
    single run -- ``set_seed(seed_k)`` then the model's parameter draws in
    construction order. On CPU a freshly seeded default generator produces the
    same ``randn`` stream a dedicated ``torch.Generator(seed_k)`` would, so this
    reseed-then-construct sequence gives each model its own seed-only RNG and
    reproduces the single path's initialisation byte for byte. It reuses the
    single path's ``build_model`` verbatim rather than reimplementing the draw
    order, so the stacked init can never silently diverge from ``scripts/run.py``.

    Returns ``(stacked_params, stacked_buffers, base_module)``. The stacked
    tensors are leaf tensors requiring grad (``stack_module_state`` clones them),
    so they can be optimised directly by a single ``AdamW``; ``base_module`` is a
    structural template for ``functional_call`` (its own parameter values are
    never read -- they are overridden by the stacked params on every call).
    """
    models: list[torch.nn.Module] = []
    for seed in seeds:
        # The same seeding the single path performs, per model: model k's init
        # depends on seed_k alone.
        set_seed(seed, deterministic=config.deterministic)
        models.append(build_model(config, group).to(device))
    params, buffers = stack_module_state(models)
    return params, buffers, models[0]


def _make_forward(base: torch.nn.Module) -> Any:
    """A ``vmap``ped functional forward: maps over the leading ``K`` dimension of
    the stacked params and buffers while broadcasting one shared token batch to
    every model (``in_dims=(0, 0, None)``)."""

    def fmodel(
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        out = functional_call(base, (params, buffers), (tokens,))
        assert isinstance(out, torch.Tensor)
        return out

    return vmap(fmodel, in_dims=(0, 0, None))


@dataclass
class _ModelRun:
    """Per-model bookkeeping: one run directory + manifest + eval history."""

    seed: int
    run_id: str
    run_dir: Path
    config: ProjectConfig
    history: list[str] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)
    status: str | None = None  # None until finalised


@dataclass
class _SharedData:
    """The one dataset / split / leakage endpoint every model in the invocation
    shares (the paired-seed design). Built once, before any chunk trains, so all
    chunks -- and the memory-estimation trial -- see byte-identical inputs."""

    group: FiniteGroup
    train_tokens: torch.Tensor
    test_tokens: torch.Tensor
    train_targets: torch.Tensor
    test_targets: torch.Tensor
    unleaked_mask: torch.Tensor
    has_unleaked: bool
    n_train: int
    n_test: int
    leakage: dict[str, Any]


def compute_chunk_size(
    n_seeds: int,
    *,
    per_model_bytes: float,
    free_bytes: float,
    safety: float,
) -> int:
    """How many models fit in one sequential sub-batch.

    ``floor((free_bytes * safety) / per_model_bytes)``, clamped to ``[1,
    n_seeds]``. The fixed overhead (dataset tensors + CUDA context) is counted
    once inside ``per_model_bytes``'s baseline subtraction, not here; the
    ``safety`` fraction (default 0.8) is the headroom that absorbs that
    overhead plus allocator fragmentation. A non-positive ``per_model_bytes``
    (a degenerate measurement) means "no useful estimate" -> one chunk of every
    seed. This is pure arithmetic with no torch/CUDA dependency, so the chunk
    plan is unit-testable with injected ``free_bytes`` / ``per_model_bytes``."""
    if per_model_bytes <= 0:
        return n_seeds
    budget = free_bytes * safety
    size = int(budget // per_model_bytes)  # floor
    return max(1, min(size, n_seeds))


def pack_chunks(seeds: list[int], chunk_size: int) -> list[list[int]]:
    """Split ``seeds`` into consecutive chunks of at most ``chunk_size``, packed
    to the maximum: every chunk but the last is full, the remainder trails in the
    last (``pack_chunks(range(50), 30) == [0..29, 30..49]``, never a balanced
    ``[0..24, 25..49]``). With early stopping disabled a chunk's wall-clock is
    set by the epoch ceiling and, below compute saturation, is nearly independent
    of ``K``; the fullest possible chunks therefore maximise GPU utilisation and
    minimise the chunk count. Order is preserved."""
    return [seeds[i : i + chunk_size] for i in range(0, len(seeds), chunk_size)]


class BatchEnsembleTrainer:
    """Train ``K`` models of one ``(group, width)`` together on one device.

    ``config`` is the shared experiment configuration; its ``seed`` field is
    overwritten per model (each model's manifest records its own seed and a
    distinct ``config_hash``). ``seeds`` is the batch -- its length is ``K``.

    **Requires a pinned ``data.split_seed``** (the paired-seed design): all ``K``
    models share one dataset, one train/test split, and one transpose-unleaked
    mask, so any per-seed difference is initialisation and training noise, never
    a different split. Constructing with ``data.split_seed is None`` raises.

    **Memory-aware chunking.** Per-model peak memory scales roughly with
    ``|G|^2 * width``, so a large ``(group, width)`` with many seeds can exceed
    one card. ``run()`` therefore splits ``seeds`` into consecutive sub-batches
    ("chunks") and trains each to completion before the next starts. On CUDA the
    chunk size is derived empirically: a K=1 trial for this exact
    ``(group, width)`` measures the true per-model peak, and the chunk size is
    ``floor((free * safety) / per_model)`` (see :func:`compute_chunk_size`).
    ``max_models_per_batch`` is a hard override that skips estimation entirely;
    ``memory_safety_fraction`` (default 0.8) is the headroom. On a non-CUDA
    device (and with no override) there is one chunk of every seed. The
    semantics are otherwise unchanged from a single batch: seeds split in order,
    each chunk runs the full epoch ceiling, per-model manifests/run-dirs are
    identical, the run exits ``completed`` iff every seed did, and a SIGTERM
    finalises the current chunk *and* every not-yet-started seed to ``aborted``
    (all seeds' manifests are created up front, before any chunk trains).
    """

    def __init__(
        self,
        config: ProjectConfig,
        seeds: list[int],
        runs_root: Path | str = "runs",
        *,
        max_models_per_batch: int | None = None,
        memory_safety_fraction: float = 0.8,
        log: Callable[[str], None] = print,
    ):
        if config.data.split_seed is None:
            raise ValueError(
                "BatchEnsembleTrainer requires a pinned data.split_seed (the paired-seed "
                "design): every model in a batch must share one train/test split, so that "
                "per-seed differences are initialisation/training noise and never a different "
                "split. Got data.split_seed=None."
            )
        if not seeds:
            raise ValueError("seeds must be a non-empty list")
        if max_models_per_batch is not None and max_models_per_batch < 1:
            raise ValueError(f"max_models_per_batch must be at least 1, got {max_models_per_batch}")
        if not 0.0 < memory_safety_fraction <= 1.0:
            raise ValueError(
                f"memory_safety_fraction must be in (0, 1], got {memory_safety_fraction}"
            )
        self.config = config
        self.seeds = list(seeds)
        self.runs_root = Path(runs_root)
        self.device = resolve_device(config.device)
        self.campaign_id = os.environ.get("GAI_CAMPAIGN_ID") or None
        self.max_models_per_batch = max_models_per_batch
        self.safety = memory_safety_fraction
        self.log = log
        self.global_step = 0
        self._current_epoch = 0
        self._runs: list[_ModelRun] = []
        self._run_by_seed: dict[int, _ModelRun] = {}
        # The chunk currently training (a slice of ``self._runs``); the training
        # helpers index the stacked params by position within this chunk.
        self._chunk_seeds: list[int] = []
        self._chunk_runs: list[_ModelRun] = []
        self._params: dict[str, torch.Tensor] = {}
        self._buffers: dict[str, torch.Tensor] = {}

    # --- lifecycle hooks (test seams) --------------------------------------

    def _on_epoch_end(self, epoch: int) -> None:
        """Called at the end of every training epoch. A no-op in production; a
        test overrides it to inject an interruption and exercise the abort path."""

    # --- provenance --------------------------------------------------------

    def _init_runs(self, leakage: dict[str, Any]) -> None:
        """Create one run directory + manifest per seed, before training can
        fail, so even a crashed batch leaves a terminal manifest for every seed.
        Records the shared leakage covariates on each."""
        for seed in self.seeds:
            per_config = self.config.model_copy(deep=True)
            per_config.seed = seed
            run_id = create_run_id(per_config.experiment.name)
            run_dir = create_run_dir(run_id, root=self.runs_root)
            save_resolved_config(per_config, run_dir)
            create_manifest(per_config, run_dir, campaign_id=self.campaign_id)
            record_tracking(run_dir, None, None)
            record_leakage(run_dir, **leakage)
            model_run = _ModelRun(seed=seed, run_id=run_id, run_dir=run_dir, config=per_config)
            self._runs.append(model_run)
            self._run_by_seed[seed] = model_run

    def _finalize(
        self,
        run: _ModelRun,
        status: str,
        warnings_list: list[str],
        error: str | None = None,
    ) -> None:
        """Finalise one model's manifest to a terminal status, writing its run.log
        and (on success) its final.pt first. Idempotent per model."""
        if run.status is not None:
            return
        if status == "completed":
            run.history.append(f"completed {run.run_id} | {run.summary}")
        (run.run_dir / "run.log").write_text("\n".join(run.history) + "\n")
        final_ckpt: str | None = None
        if status == "completed" and self.config.snapshot.save_final:
            final_ckpt = str(self._save_checkpoint(run, "final"))
        finalize_manifest(
            run.run_dir,
            status=status,
            summary=self._summary_block(run.summary) if status == "completed" else None,
            final_checkpoint=final_ckpt,
            error=error,
            warnings=warnings_list,
        )
        run.status = status

    def _summary_block(self, summary: dict[str, float]) -> dict[str, Any]:
        return {"completed_steps": self.global_step, "metrics": dict(summary)}

    def _model_state_dict(self, index: int) -> dict[str, torch.Tensor]:
        """The ``index``-th model's state dict, sliced out of the stacked params
        and buffers -- keys and shapes identical to a single model's
        ``state_dict()`` (params + the ``causal_mask`` buffer)."""
        state: dict[str, torch.Tensor] = {}
        for name, tensor in self._params.items():
            state[name] = tensor[index].detach().clone()
        for name, tensor in self._buffers.items():
            state[name] = tensor[index].detach().clone()
        return state

    def _save_checkpoint(self, run: _ModelRun, name: str) -> Path:
        """Write one model's checkpoint, matching GroupGeneralizationExperiment's
        snapshot schema (step/config/epoch/model_state_dict).

        The stacked params (``self._params``) hold only the *current chunk*, so
        the slice index is the seed's position within the chunk, not within the
        full seed list."""
        index = self._chunk_seeds.index(run.seed)
        state = {
            "step": self.global_step,
            "config": run.config.model_dump(),
            "epoch": self._current_epoch,
            "model_state_dict": self._model_state_dict(index),
        }
        path = run.run_dir / "checkpoints" / f"{name}.pt"
        torch.save(state, path)
        return path

    # --- training ----------------------------------------------------------

    def run(self) -> dict[int, str]:
        """Train every seed to the epoch ceiling (in one or more sequential
        memory-sized chunks) and finalise every manifest.

        Returns ``{seed: terminal_status}``. All seeds' manifests are created up
        front, before any chunk trains, so on SIGTERM / Ctrl-C every unfinished
        model's manifest -- the chunk currently training *and* every chunk not
        yet started -- is finalised to ``aborted`` (mirroring ``BaseExperiment``'s
        handler discipline) before the exception propagates, so a drained batch
        never leaves a manifest stuck at ``running``.
        """
        previous_sigterm: Any = None
        sigterm_installed = False
        try:
            previous_sigterm = signal.signal(signal.SIGTERM, _raise_aborted)
            sigterm_installed = True
        except ValueError:
            pass  # not the main thread; the host keeps whatever handler it has

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                self._run_chunked(caught)
                return {run.seed: run.status or "completed" for run in self._runs}
            except BaseException as exc:
                status = "aborted" if isinstance(exc, RunAborted | KeyboardInterrupt) else "failed"
                captured = _format_warnings(list(caught))
                error = f"{type(exc).__name__}: {exc}"
                for run in self._runs:
                    self._finalize(run, status, captured, error=error)
                raise
            finally:
                if sigterm_installed:
                    signal.signal(signal.SIGTERM, previous_sigterm or signal.SIG_DFL)

    def _run_chunked(self, caught: list[warnings.WarningMessage]) -> None:
        """Prepare the shared data, create every seed's manifest up front, then
        train each memory-sized chunk to completion in order.

        Each chunk's manifests are finalised to ``completed`` immediately after
        that chunk trains, while ``self._params`` still holds that chunk's stacked
        weights (the final checkpoint is sliced out of them). A failure or SIGTERM
        propagates to :meth:`run`, which finalises everything still unfinished."""
        shared = self._prepare_shared_data()

        # Provenance first: one run dir + manifest per seed -- for EVERY seed,
        # across every chunk -- before any training can fail, so even a batch
        # drained during its first chunk leaves a terminal manifest for the
        # seeds later chunks would have trained.
        self._init_runs(shared.leakage)

        chunk_size = self._resolve_chunk_size(shared)
        # Packed to the maximum -- fullest chunks first, remainder last (see
        # :func:`pack_chunks` for the utilisation rationale).
        chunks = pack_chunks(self.seeds, chunk_size)

        for position, chunk_seeds in enumerate(chunks, start=1):
            chunk_runs = [self._run_by_seed[seed] for seed in chunk_seeds]
            self.log(
                f"[run_batch] chunk {position}/{len(chunks)}: "
                f"seeds {chunk_seeds[0]}..{chunk_seeds[-1]} ({len(chunk_seeds)} model(s))"
            )
            self._train_chunk(chunk_seeds, chunk_runs, shared)
            captured = _format_warnings(list(caught))
            for run in chunk_runs:
                self._finalize(run, "completed", captured)

    def _prepare_shared_data(self) -> _SharedData:
        """Build the one dataset / split / leakage endpoint every model shares.

        Shared across all chunks and the memory-estimation trial, so any per-seed
        difference is initialisation and training noise, never a different split.
        Emits the empty-unleaked-subset warning once (it is a property of the
        split, not of any seed)."""
        device = self.device
        group = _resolve_group(self.config)
        task = build_group_task(group)
        split_seed = self.config.data.split_seed
        assert split_seed is not None  # guaranteed by __init__
        split = train_test_split(task, self.config.data.train_frac, split_seed)

        def tokens(pairs: np.ndarray) -> torch.Tensor:
            equals = np.full((len(pairs), 1), group.order)
            values = np.concatenate([pairs, equals], axis=1)
            return torch.tensor(values, dtype=torch.long, device=device)

        train_tokens = tokens(split.train_inputs)
        test_tokens = tokens(split.test_inputs)
        train_targets = torch.tensor(split.train_targets, dtype=torch.long, device=device)
        test_targets = torch.tensor(split.test_targets, dtype=torch.long, device=device)

        # The shared transpose-unleaked endpoint (fixed once, shared by every
        # model in the batch), computed exactly as GroupGeneralizationExperiment.
        leaked = transpose_leaked_mask(split, group.cayley_table)
        unleaked = ~leaked
        test_size = int(unleaked.shape[0])
        unleaked_mask = torch.tensor(unleaked, dtype=torch.bool, device=device)
        has_unleaked = bool(unleaked.any())
        n_unleaked = int(unleaked.sum())
        generalize_metric = "unleaked_accuracy" if has_unleaked else "raw_test_accuracy"
        note: str | None = None
        if not has_unleaked:
            note = (
                f"transpose-unleaked subset is EMPTY: all {test_size} test pairs of "
                f"{group.canonical_name} are transpose-leaked. Unleaked accuracy is "
                "undefined (NaN); raw test accuracy is recorded instead."
            )
            warnings.warn(note, stacklevel=2)
        leakage = {
            "transpose_leak_fraction": float(leaked.mean()),
            "commuting_probability": commuting_probability(group),
            "test_size": test_size,
            "unleaked_test_size": n_unleaked,
            "unleaked_empty": not has_unleaked,
            "generalize_metric": generalize_metric,
            "note": note,
        }
        return _SharedData(
            group=group,
            train_tokens=train_tokens,
            test_tokens=test_tokens,
            train_targets=train_targets,
            test_targets=test_targets,
            unleaked_mask=unleaked_mask,
            has_unleaked=has_unleaked,
            n_train=int(train_targets.shape[0]),
            n_test=int(test_targets.shape[0]),
            leakage=leakage,
        )

    # --- memory-aware chunk sizing -----------------------------------------

    def _resolve_chunk_size(self, shared: _SharedData) -> int:
        """Decide how many models train together in one sub-batch, and log the
        decision loudly.

        ``max_models_per_batch`` (a hard operator override) short-circuits
        estimation. On a non-CUDA device with no override there is one chunk of
        every seed. Otherwise a K=1 trial measures the true per-model peak for
        this exact ``(group, width)`` and the chunk size is
        ``floor((free * safety) / per_model)`` (see :func:`compute_chunk_size`)."""
        n = len(self.seeds)
        mib = 1024 * 1024

        if self.max_models_per_batch is not None:
            size = max(1, min(self.max_models_per_batch, n))
            self.log(
                f"[run_batch] chunk plan: --max-models-per-batch="
                f"{self.max_models_per_batch} -> chunk size {size}, "
                f"{math.ceil(n / size)} chunk(s) over {n} seed(s) "
                f"(memory estimation skipped)"
            )
            return size

        if self.device.type != "cuda":
            self.log(
                f"[run_batch] chunk plan: device={self.device.type}, no chunking "
                f"-> 1 chunk of all {n} seed(s)"
            )
            return n

        per_model_bytes = self._measure_per_model_cost(shared)
        free_bytes = self._free_memory()
        size = compute_chunk_size(
            n, per_model_bytes=per_model_bytes, free_bytes=free_bytes, safety=self.safety
        )
        self.log(
            f"[run_batch] chunk plan: estimated per-model peak "
            f"{per_model_bytes / mib:.0f} MiB, free {free_bytes / mib:.0f} MiB, "
            f"safety {self.safety:.2f} (budget {free_bytes * self.safety / mib:.0f} MiB) "
            f"-> chunk size {size}, {math.ceil(n / size)} chunk(s) over {n} seed(s)"
        )
        return size

    def _free_memory(self) -> int:
        """Free device memory in bytes, honouring other tenants on the card."""
        free, _total = torch.cuda.mem_get_info(self.device)
        return int(free)

    def _measure_per_model_cost(self, shared: _SharedData) -> int:
        """Empirically measure one model's peak memory cost, in bytes, for this
        exact ``(group, width)``: peak-during-trial minus the already-resident
        baseline (dataset tensors + CUDA context), so the fixed overhead is
        counted once and excluded from the per-model figure. Leaves no run
        directory or manifest -- the trial is purely in-memory."""
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)
        baseline = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._estimation_trial(shared)
        torch.cuda.synchronize(self.device)
        peak = torch.cuda.max_memory_allocated(self.device)
        torch.cuda.empty_cache()
        return max(1, int(peak - baseline))

    def _estimation_trial(self, shared: _SharedData) -> None:
        """Exercise the memory-heavy path for a single model (K=1): build the
        stacked ensemble for one seed, then run ~3 full training steps + one eval
        forward so grads, AdamW state, and activations are all resident at once.

        Deliberately creates NO run directory, manifest, or checkpoint -- it does
        not touch ``_init_runs`` / ``_evaluate_and_record`` -- so a measurement
        never leaves an artefact behind. CPU-safe (no CUDA calls of its own), so
        the no-artefacts property is unit-testable without a GPU."""
        cfg = self.config.optim
        trial_seed = self.seeds[0]
        params, buffers, base = build_stacked_ensemble(
            self.config, shared.group, [trial_seed], self.device
        )
        forward = _make_forward(base)
        optimizer = torch.optim.AdamW(
            list(params.values()),
            lr=cfg.lr_effective or cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
        n_train = shared.n_train
        for _ in range(3):
            optimizer.zero_grad()
            train_logits = forward(params, buffers, shared.train_tokens)[:, :, -1, :]
            v = train_logits.shape[-1]
            loss = (
                F.cross_entropy(
                    train_logits.reshape(n_train, v),
                    shared.train_targets,
                    reduction="sum",
                )
                / n_train
            )
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            forward(params, buffers, shared.test_tokens)
        del params, buffers, base, forward, optimizer

    # --- per-chunk training ------------------------------------------------

    def _train_chunk(
        self,
        chunk_seeds: list[int],
        chunk_runs: list[_ModelRun],
        shared: _SharedData,
    ) -> None:
        """Train one chunk (``len(chunk_seeds)`` models of one ``(group, width)``)
        to the epoch ceiling. ``self._params`` / ``self._chunk_seeds`` are set to
        this chunk so the eval and checkpoint helpers slice the stacked weights by
        position within the chunk."""
        cfg = self.config.optim
        snap = self.config.snapshot

        self._chunk_seeds = list(chunk_seeds)
        self._chunk_runs = chunk_runs

        # Per-model init, bit-identical to the single path (see docstring): a
        # seed's stacked init is a function of that seed alone, so it is identical
        # whether the seed is trained in a chunk of 1 or of many.
        params, buffers, base = build_stacked_ensemble(
            self.config, shared.group, chunk_seeds, self.device
        )
        self._params = params
        self._buffers = buffers
        forward = _make_forward(base)

        # Plain AdamW over the stacked tensors: updates are elementwise, so each
        # K-slice evolves exactly as an independent AdamW would (the summed loss
        # keeps each slice's gradient a function of that model alone). The
        # hyperparameters match the single path's optimiser exactly, including the
        # default eps.
        optimizer = torch.optim.AdamW(
            list(params.values()),
            lr=cfg.lr_effective or cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

        train_tokens = shared.train_tokens
        test_tokens = shared.test_tokens
        train_targets = shared.train_targets
        test_targets = shared.test_targets
        n_train = shared.n_train
        n_test = shared.n_test
        k = len(chunk_seeds)
        last_epoch = cfg.epochs - 1

        for epoch in range(cfg.epochs):
            self._current_epoch = epoch
            self.global_step = epoch + 1

            optimizer.zero_grad()
            train_logits = forward(params, buffers, train_tokens)[:, :, -1, :]  # [K, n_train, V]
            v = train_logits.shape[-1]
            # Summed over models; each per-model term is the mean CE over its
            # batch, so dividing the summed reduction by n_train recovers
            # sum_k mean_batch(CE_k) and each slice's gradient matches the single
            # path (which minimises the per-model mean CE).
            loss = (
                F.cross_entropy(
                    train_logits.reshape(k * n_train, v),
                    train_targets.repeat(k),
                    reduction="sum",
                )
                / n_train
            )
            loss.backward()
            optimizer.step()

            need_eval = epoch % cfg.log_every == 0 or epoch == 0 or epoch == last_epoch
            if need_eval:
                self._evaluate_and_record(
                    epoch=epoch,
                    train_logits=train_logits.detach(),
                    forward=forward,
                    params=params,
                    buffers=buffers,
                    train_targets=train_targets,
                    test_tokens=test_tokens,
                    test_targets=test_targets,
                    unleaked_mask=shared.unleaked_mask,
                    has_unleaked=shared.has_unleaked,
                    n_train=n_train,
                    n_test=n_test,
                )

            if snap.enabled and should_snapshot(epoch, snap):
                for run in chunk_runs:
                    self._save_checkpoint(run, f"step_{epoch}")

            # Final-window capture, per model, independent of `enabled` (like
            # save_final): each of the last `final_window_epochs` epochs of the
            # ceiling gets its own final_epoch_<E>.pt -- same policy and naming
            # as the single path, so a post-grok "slingshot" dip at the last
            # epoch never leaves final.pt as the only end-of-training record.
            if snap.final_window_epochs > 0 and epoch >= cfg.epochs - snap.final_window_epochs:
                for run in chunk_runs:
                    self._save_checkpoint(run, f"final_epoch_{epoch}")

            self._on_epoch_end(epoch)

    def _evaluate_and_record(
        self,
        *,
        epoch: int,
        train_logits: torch.Tensor,
        forward: Any,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        train_targets: torch.Tensor,
        test_tokens: torch.Tensor,
        test_targets: torch.Tensor,
        unleaked_mask: torch.Tensor,
        has_unleaked: bool,
        n_train: int,
        n_test: int,
    ) -> None:
        """Compute per-model metrics (same semantics as
        GroupGeneralizationExperiment._evaluate, vectorised over the ensemble)
        and record them into each model's summary + eval-history log line.

        Train-side metrics are PRE-update (read off ``train_logits``, the forward
        that produced this epoch's step); val-side metrics are POST-update (a
        fresh forward on the just-stepped weights) -- the same one-epoch offset
        the single path documents. ``k`` is the *current chunk* size (the stacked
        weights hold only this chunk), and the rows are zipped with the chunk's
        runs, not the full seed list."""
        k = len(self._chunk_runs)
        v = train_logits.shape[-1]

        # Train side (pre-update).
        train_ce = F.cross_entropy(
            train_logits.reshape(k * n_train, v), train_targets.repeat(k), reduction="none"
        ).reshape(k, n_train)
        train_loss = train_ce.mean(dim=1)
        train_acc = (train_logits.argmax(dim=-1) == train_targets).float().mean(dim=1)

        # Val side (post-update).
        with torch.no_grad():
            test_logits = forward(params, buffers, test_tokens)[:, :, -1, :]  # [K, n_test, V]
            test_ce = F.cross_entropy(
                test_logits.reshape(k * n_test, v), test_targets.repeat(k), reduction="none"
            ).reshape(k, n_test)
            test_loss = test_ce.mean(dim=1)
            correct = test_logits.argmax(dim=-1) == test_targets  # [K, n_test]
            test_acc = correct.float().mean(dim=1)
            if has_unleaked:
                unleaked_acc = correct[:, unleaked_mask].float().mean(dim=1)
            else:
                unleaked_acc = torch.full((k,), float("nan"), device=test_logits.device)

        # One host sync for the whole ensemble this epoch.
        weight_norm = torch.zeros(k, device=train_logits.device)
        for tensor in params.values():
            weight_norm = weight_norm + tensor.detach().reshape(k, -1).pow(2).sum(dim=1)
        weight_norm = weight_norm.sqrt()

        stacked = torch.stack(
            [train_loss, train_acc, test_loss, test_acc, unleaked_acc, weight_norm], dim=1
        ).tolist()  # [K, 6]
        for run, row in zip(self._chunk_runs, stacked, strict=True):
            metrics = dict(zip(_SUMMARY_KEYS, row, strict=True))
            run.summary = metrics
            run.history.append(f"epoch {epoch} | {metrics}")


def _resolve_group(config: ProjectConfig) -> FiniteGroup:
    from ..groups.catalog import resolve_group

    return resolve_group(config.data.group)


__all__ = [
    "BatchEnsembleTrainer",
    "build_stacked_ensemble",
    "compute_chunk_size",
    "pack_chunks",
]
