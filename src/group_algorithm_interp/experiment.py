"""Experiment lifecycle for finite-group learning experiments.

``BaseExperiment`` owns the boring, reusable parts of a run -- run directory,
manifest, seeding, device, W&B, checkpointing, and exception-safe finalisation.
It does not care whether you train, evaluate an API model, or probe released
weights: subclasses implement three project-specific hooks:

    setup()               build whatever the run needs (data/model, API client, hooks)
    run()                 do exploratory work; return the summary metrics dict
    holdout_evaluation()  confirmatory-only evaluation, invoked by the lifecycle
    state_dict()          what to checkpoint (may be trivial for a non-training run)

``GroupGeneralizationExperiment`` provides the project's current full-batch
group-multiplication training implementation. Other finite-group experiments can
subclass :class:`BaseExperiment` while retaining the same provenance contract.
"""

from __future__ import annotations

import logging
import os
import signal
import warnings
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import ProjectConfig
from .groups.catalog import resolve_group
from .groups.group import FiniteGroup
from .logging_utils import setup_logging
from .manifest import (
    compute_group_hash,
    create_manifest,
    create_run_dir,
    create_run_id,
    finalize_manifest,
    record_holdout,
    record_holdout_result,
    record_leakage,
    record_tracking,
    save_resolved_config,
)
from .model import GroupModel
from .seed import resolve_device, set_seed
from .task import (
    build_group_task,
    commuting_probability,
    train_test_split,
    transpose_leaked_mask,
)
from .training.trainer import build_model, should_snapshot
from .wandb_utils import WandbLogger


class HoldoutError(RuntimeError):
    """A held-out sealing or confirmatory-guard violation."""


class RunAborted(BaseException):
    """Raised from the SIGTERM handler so an externally-terminated run unwinds
    through the same cleanup path as a crash or an operator's Ctrl-C.

    A bare SIGTERM kills the process outright: no ``except``, no ``finally``, no
    manifest finalisation, so the run's manifest is frozen at ``status:
    running`` forever -- indistinguishable from a run that is still training.
    That is the *normal* death on a rented/spot GPU and the exact signal a
    campaign runner (``scripts/run_batch.py``) sends to drain a run, so it must
    be recoverable.
    Deriving from ``BaseException`` (like ``KeyboardInterrupt``) keeps it from
    being swallowed by an ``except Exception`` inside a subclass's ``run()``."""


def _prediction_outcome(prediction: dict | None, metrics: dict[str, float]) -> str | None:
    """``predicted`` | ``refuted`` | ``inconclusive``, or ``None`` when no prediction.
    Pure: given the pre-registered prediction and the held-out metrics."""
    if not prediction:
        return None
    metric = prediction["metric"]
    if metric not in metrics:
        return "inconclusive"
    observed = metrics[metric]
    if prediction["direction"] == "higher":
        hit = observed > prediction["value"]
    else:
        hit = observed < prediction["value"]
    return "predicted" if hit else "refuted"


def _raise_aborted(signum: int, _frame: FrameType | None) -> None:
    """Raise ``RunAborted``, and disarm on the way out so the cleanup it triggers
    cannot be interrupted by a *second* SIGTERM.

    The disarm is load-bearing, not defensive. ``launcher.signal_process_group``
    sends SIGTERM to the child's whole process *group*, and that group holds both
    ``uv`` and the trainer underneath it -- and ``uv`` **also forwards** the signal
    to its own child. The trainer therefore receives SIGTERM twice, milliseconds
    apart. With the handler still armed, the second delivery raised ``RunAborted``
    from inside ``finalize_manifest``'s read-modify-write, the write was abandoned,
    and the run's manifest was left at ``status: running`` forever -- precisely the
    failure this handler exists to prevent. Measured against a real ``uv run``
    child, a single ``killpg`` lost the manifest in about half of all trials, while
    a single delivery (to either process alone) never did.

    ``SIG_IGN`` rather than ``SIG_DFL``: the default disposition would let the
    second signal kill the process outright mid-finalise, which is the same bug by
    a different route. Nothing is leaked by ignoring it -- the launcher escalates to
    SIGKILL once the termination grace period expires, and ``execute()``'s
    ``finally`` restores the previous disposition once the manifest is safely on
    disk.
    """
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise RunAborted(f"received signal {signal.Signals(signum).name}")


def _format_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    """De-duplicated ``Category: message`` strings for the manifest."""
    seen: set[str] = set()
    out: list[str] = []
    for w in caught:
        msg = f"{w.category.__name__}: {w.message}"
        if msg not in seen:
            seen.add(msg)
            out.append(msg)
    return out


class BaseExperiment:
    """Run lifecycle scaffolding. See module docstring for the contract."""

    def __init__(self, config: ProjectConfig, runs_root: Path | str = "runs"):
        self.config = config
        self.run_id = create_run_id(config.experiment.name)
        self.run_dir = create_run_dir(self.run_id, root=runs_root)
        self.device = resolve_device(config.device)
        self.logger = WandbLogger(
            config,
            run_name=self.run_id,
            extra_config={"config_group_hash": compute_group_hash(config)},
        )
        self._run_log: logging.Logger | None = None
        self.global_step = 0
        self._holdout_declared = False
        self._holdout_evaluated = False
        self._holdout_phase = False
        # The launching campaign's identifier (from GAI_CAMPAIGN_ID), or None
        # for a standalone run outside any campaign. The `or None` also
        # normalises an empty-string env var to None, so an unset/blank var
        # both mean "no campaign".
        self.campaign_id = os.environ.get("GAI_CAMPAIGN_ID") or None

    # --- subclass hooks -----------------------------------------------------

    def setup(self) -> None:
        """Build data, model, optimiser. Called after seeding, before training."""
        raise NotImplementedError

    def run(self) -> dict[str, float]:
        """Do the work (train / evaluate / probe) and return the final summary
        metrics dict."""
        raise NotImplementedError

    def holdout_evaluation(self) -> dict[str, float]:
        """Evaluate the declared holdout for a confirmatory run.

        ``BaseExperiment`` invokes this hook once, after :meth:`run` returns. A
        confirmatory subclass must override it; exploratory subclasses need not.
        """
        raise HoldoutError(
            "confirmatory run did not implement holdout_evaluation(); "
            "the base lifecycle invokes it once after run()"
        )

    def state_dict(self) -> dict[str, Any]:
        """What to write into a checkpoint. This project keeps no resume path --
        runs always restart from scratch -- so a checkpoint is a point-in-time
        snapshot for post-hoc analysis, not a resumable training state.
        Subclasses extend this with model state via ``super().state_dict()``.
        """
        return {
            "step": self.global_step,
            "config": self.config.model_dump(),
        }

    def model_to_watch(self) -> torch.nn.Module | None:
        """Return the model to pass to ``wandb.watch`` when
        ``logging.watch_model`` is set. Default: nothing to watch."""
        return None

    # --- lifecycle ----------------------------------------------------------

    def execute(self) -> dict[str, float]:
        """The full lifecycle, exception-safe.

        Order matters: config + manifest are persisted *before* anything can
        fail, so even a crashed run leaves a `failed` manifest behind.

        Every exit path -- success, crash, Ctrl-C, SIGTERM -- finalises the
        manifest to a terminal status. SIGTERM is handled explicitly (see
        ``RunAborted``) because the default disposition kills the process without
        running ``finally``, which is how a spot reclaim, or the launcher's own
        drain, used to leave a manifest stuck at ``status: running``.
        """
        save_resolved_config(self.config, self.run_dir)
        create_manifest(self.config, self.run_dir, campaign_id=self.campaign_id)
        self._run_log = setup_logging(self.run_dir)
        self._run_log.info(
            "run %s | device=%s | mode=%s",
            self.run_id,
            self.device,
            self.config.logging.mode,
        )

        # Only the main thread may install a signal handler; a run driven from a
        # worker thread keeps whatever handler the host process installed.
        previous_sigterm: Any = None
        sigterm_installed = False
        try:
            previous_sigterm = signal.signal(signal.SIGTERM, _raise_aborted)
            sigterm_installed = True
        except ValueError:
            self._run_log.warning(
                "not the main thread: SIGTERM will not finalise this run's manifest"
            )

        # Capture warnings raised during the run so they land in the manifest as
        # provenance; re-emit them to the run log so they stay visible.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                # A pure opt-in, on every device: nothing forces determinism on
                # for CUDA. CUDA runs are not bit-reproducible by default, which
                # is a deliberate throughput choice (see seed.py).
                deterministic = self.config.deterministic
                set_seed(self.config.seed, deterministic=deterministic)
                self._run_log.info(
                    "seed=%s | deterministic_algorithms=%s | CUBLAS_WORKSPACE_CONFIG=%s",
                    self.config.seed,
                    deterministic,
                    os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                )

                self.logger.start()
                record_tracking(self.run_dir, self.logger.run_id, self.logger.url)

                self.setup()
                if self.config.logging.watch_model:
                    model = self.model_to_watch()
                    if model is not None:
                        self.logger.watch(model, log_freq=self.config.logging.watch_log_freq)
                if self.config.validation.stance == "confirmatory" and not self._holdout_declared:
                    raise HoldoutError(
                        "confirmatory run declared no held-out set; call "
                        "self.declare_holdout(...) in setup(), or set "
                        "validation.stance=exploratory for exploratory work"
                    )

                summary = self.run()

                if self.config.validation.stance == "confirmatory":
                    self._holdout_phase = True
                    try:
                        holdout_metrics = self.evaluate_holdout(self.holdout_evaluation)
                    finally:
                        self._holdout_phase = False
                    summary = {**summary, **holdout_metrics}

                final_ckpt = None
                if self.config.snapshot.save_final:
                    final_ckpt = str(self.save_checkpoint("final"))

                self.logger.set_summary(summary)
                captured = _format_warnings(list(caught))
                finalize_manifest(
                    self.run_dir,
                    status="completed",
                    summary=self._summary_block(summary),
                    final_checkpoint=final_ckpt,
                    warnings=captured,
                )
                for msg in captured:
                    self._run_log.warning("captured warning: %s", msg)
                self._run_log.info("completed %s | %s", self.run_id, self._summary_block(summary))
                return summary
            # BaseException, not Exception: a KeyboardInterrupt or a SIGTERM
            # (RunAborted) must still reach a terminal manifest state. Catching
            # only Exception is what left interrupted runs recorded as `running`
            # forever -- and an interrupted run is the normal outcome on a spot
            # GPU, not an edge case.
            except BaseException as exc:
                status = "aborted" if isinstance(exc, RunAborted | KeyboardInterrupt) else "failed"
                captured = _format_warnings(list(caught))
                finalize_manifest(
                    self.run_dir,
                    status=status,
                    error=f"{type(exc).__name__}: {exc}",
                    warnings=captured,
                )
                for msg in captured:
                    self._run_log.warning("captured warning: %s", msg)
                self._run_log.exception("run %s %s", self.run_id, status)
                raise
            finally:
                self.logger.finish()
                if sigterm_installed:
                    signal.signal(signal.SIGTERM, previous_sigterm or signal.SIG_DFL)

    def _summary_block(self, summary: dict[str, float]) -> dict[str, Any]:
        """Map the run's final metrics into the manifest summary schema. Shape-
        neutral: ``metrics`` is whatever dict ``run()`` returned, plus the
        completed step count. Override only if you want a different summary
        shape.

        ``best_metric``/``best_step`` are no longer populated here: selecting a
        "best" checkpoint by a test/eval-set metric is model selection on the
        evaluation set, which this project does not do. The manifest still
        carries those two keys (see ``manifest.create_manifest``) as permanent
        ``null`` placeholders for schema stability.
        """
        return {
            "completed_steps": self.global_step,
            "metrics": dict(summary),
        }

    # --- shared helpers -----------------------------------------------------

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Push metrics to W&B (the only metrics backend)."""
        self.logger.log(metrics, step=step)

    def save_checkpoint(self, name: str, aliases: list[str] | None = None) -> Path:
        path = self.run_dir / "checkpoints" / f"{name}.pt"
        torch.save(self.state_dict(), path)
        if self.config.logging.log_checkpoints:
            self.logger.log_checkpoint_artifact(
                path,
                name=f"{self.config.experiment.name}-{name}",
                aliases=aliases or [name],
                metadata={"step": self.global_step},
            )
        return path

    def declare_holdout(
        self, fingerprint: str, size: int, spec: dict[str, Any] | None = None
    ) -> None:
        """Register the held-out set (a hash the subclass computes over its held-out
        ids/content) for provenance. Call this in ``setup()``. The harness records
        the fingerprint and never sees the data. Required for confirmatory runs."""
        self._holdout_declared = True
        record_holdout(self.run_dir, fingerprint=fingerprint, size=size, spec=spec)

    def evaluate_holdout(self, fn: Callable[[], dict[str, float]]) -> dict[str, float]:
        """Execute and record the single lifecycle-controlled holdout evaluation."""
        if not self._holdout_phase:
            raise HoldoutError(
                "evaluate_holdout() is controlled by BaseExperiment and may run only "
                "after run(); implement holdout_evaluation() instead"
            )
        if self._holdout_evaluated:
            raise HoldoutError("evaluate_holdout() may be called only once per run")
        self._holdout_evaluated = True
        metrics = fn()
        prediction = (
            self.config.validation.prediction.model_dump()
            if self.config.validation.prediction is not None
            else None
        )
        record_holdout_result(self.run_dir, metrics, _prediction_outcome(prediction, metrics))
        return metrics


class GroupGeneralizationExperiment(BaseExperiment):
    """Full-batch group-multiplication training on the shared lifecycle spine."""

    group: FiniteGroup
    model: GroupModel
    optimizer: torch.optim.Optimizer
    current_epoch: int
    # The transpose-unleaked subset of the test set, fixed once at setup: a
    # boolean mask over the test tokens (True == keep, i.e. NOT reachable by the
    # commuting-transpose shortcut). ``has_unleaked`` is False in the degenerate
    # case where every test pair is leaked, in which case the generalisation
    # streak falls back to raw test accuracy (recorded loudly in the manifest).
    unleaked_mask: torch.Tensor
    has_unleaked: bool

    def setup(self) -> None:
        # Determinism is not configured here: BaseExperiment.execute() resolves
        # it and hands it to set_seed(), which must run *before* any weight is
        # initialised (and, for CUDA, before the first kernel launches).
        self.current_epoch = 0
        self.group = resolve_group(self.config.data.group)
        self.model = build_model(self.config, self.group).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optim.lr_effective or self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )
        task = build_group_task(self.group)
        split_seed = self.config.data.split_seed
        if split_seed is None:
            # Loud, not silent: this is the exact footgun that conflates
            # initialisation noise with split noise in a multi-seed sweep (a
            # sweep over config.seed then also sweeps the split). The warning
            # is captured into the run manifest's provenance record (see
            # execute()'s warnings.catch_warnings), so it is inspectable even
            # for an unattended run.
            warnings.warn(
                "data.split_seed is unset (None); falling back to config.seed "
                f"({self.config.seed}) for the train/test split. A sweep that "
                "varies config.seed across runs will then also vary the split "
                "per run. Pin data.split_seed to a fixed integer to keep one "
                "split across a seed sweep.",
                stacklevel=2,
            )
            split_seed = self.config.seed
        split = train_test_split(task, self.config.data.train_frac, split_seed)

        def tokens(pairs: np.ndarray) -> torch.Tensor:
            equals = np.full((len(pairs), 1), self.group.order)
            values = np.concatenate([pairs, equals], axis=1)
            return torch.tensor(values, dtype=torch.long, device=self.device)

        self.train_tokens = tokens(split.train_inputs)
        self.test_tokens = tokens(split.test_inputs)
        self.train_targets = torch.tensor(split.train_targets, dtype=torch.long, device=self.device)
        self.test_targets = torch.tensor(split.test_targets, dtype=torch.long, device=self.device)

        # The transpose-unleaked endpoint. Computed ONCE here on the realised
        # split (it is a fixed subset for the whole run): a test pair is leaked
        # when its transpose is in train AND commutes, so its label is already
        # memorised. Raw test accuracy is inflated for high-Pr(G) groups and is
        # not comparable across groups; the unleaked subset removes that shortcut
        # and is the accuracy the generalisation/grok definition and every
        # cross-group comparison read. The mask is kept as a device tensor
        # alongside the test tokens so per-eval cost is one extra masked mean --
        # never a second forward pass.
        leaked = transpose_leaked_mask(split, self.group.cayley_table)
        unleaked = ~leaked
        test_size = int(unleaked.shape[0])
        self.unleaked_mask = torch.tensor(unleaked, dtype=torch.bool, device=self.device)
        self.has_unleaked = bool(unleaked.any())
        n_unleaked = int(unleaked.sum())
        generalize_metric = "unleaked_accuracy" if self.has_unleaked else "raw_test_accuracy"
        note: str | None = None
        if not self.has_unleaked:
            # Degenerate: a high-commuting group at a high train_frac can leave
            # every test pair leaked. Never divide by the zero unleaked count;
            # record NaN unleaked accuracy with a loud manifest note, and let the
            # grok streak fall back to raw accuracy under an explicit flag.
            note = (
                f"transpose-unleaked subset is EMPTY: all {test_size} test pairs of "
                f"{self.group.canonical_name} are transpose-leaked (each has its "
                "commuting transpose in train). Unleaked accuracy is undefined "
                "(NaN) and the generalisation streak falls back to raw test "
                "accuracy for this run."
            )
            warnings.warn(note, stacklevel=2)
        record_leakage(
            self.run_dir,
            transpose_leak_fraction=float(leaked.mean()),
            commuting_probability=commuting_probability(self.group),
            test_size=test_size,
            unleaked_test_size=n_unleaked,
            unleaked_empty=not self.has_unleaked,
            generalize_metric=generalize_metric,
            note=note,
        )

    def model_to_watch(self) -> torch.nn.Module | None:
        return self.model

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
            }
        )
        return state

    def run(self) -> dict[str, float]:
        """Full-batch training. Returns the metrics of the epoch the run actually
        stopped on -- the last epoch, or the epoch the stop-on-generalize rule
        fired on.

        Metric timing, which matters for reading a grokking gap: ``train/loss``
        and ``train/accuracy`` at epoch *e* are computed from the forward pass
        that *produced* update *e*, so they are PRE-update; ``val/*`` and
        ``val/weight_norm`` are read after ``optimizer.step()``, so they are
        POST-update. The train/val curves are therefore one update out of step
        with each other. Left as-is deliberately (making them agree costs a
        second full-batch forward per evaluated epoch, and shifts every
        previously-recorded metric), but any train-vs-val gap onset read off
        these curves carries a one-epoch offset.
        """
        cfg = self.config.optim
        snap = self.config.snapshot
        last_epoch = cfg.epochs - 1
        previous_test_loss = float("inf")
        gen_streak = 0
        final: dict[str, float] = {}

        for epoch in range(self.global_step, cfg.epochs):
            self.current_epoch = epoch
            self.global_step = epoch + 1
            self.optimizer.zero_grad()
            logits = self.model(self.train_tokens)[:, -1, :]
            loss = F.cross_entropy(logits, self.train_targets)
            loss.backward()
            self.optimizer.step()

            should_print = epoch % cfg.print_every == 0 or epoch == last_epoch
            should_wandb = epoch % cfg.wandb_every == 0 or epoch == last_epoch
            should_log = (
                epoch % cfg.log_every == 0 or should_print or should_wandb or epoch == last_epoch
            )
            # generalize_patience must count EPOCHS meeting the generalisation
            # bar, not LOG events: evaluate every epoch whenever
            # stop_on_generalize is armed, so raising log_every for GPU
            # efficiency can't silently multiply the effective patience. The bar
            # is measured on the transpose-UNLEAKED accuracy (raw test accuracy is
            # inflated by the commuting-transpose shortcut and not comparable
            # across groups), falling back to raw accuracy only in the degenerate
            # empty-subset case (self.has_unleaked is False) -- see setup().
            need_eval = should_log or cfg.stop_on_generalize
            event_snapshot = False

            if need_eval:
                test_loss_t, test_accuracy_t, test_unleaked_t = self._evaluate()
                train_accuracy_t = (logits.argmax(dim=-1) == self.train_targets).float().mean()
                weight_norm_t = (
                    torch.stack([p.detach().pow(2).sum() for p in self.model.parameters()])
                    .sum()
                    .sqrt()
                )
                # Single GPU->CPU sync for the whole epoch: stack every scalar
                # this epoch needs and pull them all across with one `.tolist()`
                # instead of five separate `.item()` calls. `loss` still carries
                # its autograd graph from `backward()` above (which is no longer
                # needed), so detach it before stacking alongside the no-grad
                # eval tensors.
                #
                # Every metric is refreshed on EVERY evaluated epoch, not only on
                # logged ones. With stop_on_generalize armed the loop evaluates
                # every epoch but logs every log_every-th, so the `break` below
                # usually lands on a non-logged epoch: building `final` only
                # under `should_log` left the returned summary -- which becomes
                # `summary.metrics` in the manifest and the W&B summary -- stale
                # by up to log_every epochs. A seed that stopped *because* it hit
                # 99% val accuracy was recorded with the val accuracy of some
                # earlier epoch, i.e. as a seed that never generalised. Folding
                # the train-side scalars into the same stack keeps the batched
                # transfer at one sync per evaluated epoch.
                (
                    train_loss_v,
                    train_accuracy_v,
                    weight_norm_v,
                    test_loss_v,
                    test_accuracy_v,
                    test_unleaked_v,
                ) = torch.stack(
                    [
                        loss.detach(),
                        train_accuracy_t,
                        weight_norm_t,
                        test_loss_t,
                        test_accuracy_t,
                        test_unleaked_t,
                    ]
                ).tolist()
                final = {
                    "train/loss": train_loss_v,
                    "train/accuracy": train_accuracy_v,
                    "val/loss": test_loss_v,
                    "val/accuracy": test_accuracy_v,
                    # The endpoint accuracy: NaN when the unleaked subset is empty
                    # (see setup()); a cross-group-comparable measurement otherwise.
                    "val/unleaked_accuracy": test_unleaked_v,
                    "val/weight_norm": weight_norm_v,
                }

                if should_log:
                    if should_print and self._run_log is not None:
                        self._run_log.info("epoch %s | %s", epoch, final)
                    if should_wandb:
                        self.log_metrics(final, step=epoch)

                    # The event-snapshot trigger stays on the LOGGED cadence: it
                    # compares consecutive logged epochs, so `previous_test_loss`
                    # is only advanced here.
                    if snap.enabled and snap.event_based and previous_test_loss < float("inf"):
                        relative_drop = (previous_test_loss - test_loss_v) / previous_test_loss
                        event_snapshot = relative_drop > snap.event_rel_drop
                    previous_test_loss = test_loss_v

                if cfg.stop_on_generalize:
                    # Grok is defined on unleaked accuracy; only the degenerate
                    # empty-subset run falls back to raw accuracy (flagged in the
                    # manifest). test_unleaked_v is finite whenever has_unleaked.
                    gen_accuracy_v = test_unleaked_v if self.has_unleaked else test_accuracy_v
                    gen_streak = gen_streak + 1 if gen_accuracy_v >= cfg.generalize_test_acc else 0

            if snap.enabled and (should_snapshot(epoch, snap) or event_snapshot):
                self.save_checkpoint(f"step_{epoch}")

            # Final-window capture, independent of `enabled` (like save_final):
            # each of the last `final_window_epochs` epochs of the configured
            # ceiling gets its own final_epoch_<E>.pt, so a post-grok
            # "slingshot" dip at the very last epoch never leaves final.pt as
            # the only end-of-training record. final.pt itself is unchanged.
            if snap.final_window_epochs > 0 and epoch >= cfg.epochs - snap.final_window_epochs:
                self.save_checkpoint(f"final_epoch_{epoch}")

            if cfg.stop_on_generalize and gen_streak >= cfg.generalize_patience:
                if snap.enabled:
                    self.save_checkpoint(f"generalized_step_{epoch}")
                break

        return final

    def _evaluate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute validation loss, raw accuracy, and transpose-unleaked accuracy
        as device-resident tensors, with no host sync. Callers decide when (and in
        what batch) to pull values across to the CPU -- see ``run()``, which
        combines these with other per-log-step scalars into a single transfer.

        The unleaked accuracy is a masked mean over the SAME forward pass as raw
        accuracy (one masked reduction, never a second forward). In the degenerate
        empty-subset case (``has_unleaked`` is False) it is NaN -- the mask would
        select zero elements -- so no division by zero occurs."""
        with torch.no_grad():
            readout = self.model(self.test_tokens)[:, -1, :]
            loss = F.cross_entropy(readout, self.test_targets)
            correct = readout.argmax(dim=-1) == self.test_targets
            accuracy = correct.float().mean()
            if self.has_unleaked:
                unleaked_accuracy = correct[self.unleaked_mask].float().mean()
            else:
                unleaked_accuracy = torch.full((), float("nan"), device=readout.device)
        return loss, accuracy, unleaked_accuracy
