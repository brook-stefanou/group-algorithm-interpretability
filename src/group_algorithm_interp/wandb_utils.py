"""Thin wrapper around Weights & Biases.

W&B is the project's primary metrics logger; there is no
local JSONL metrics path. The wrapper exists so the trainer never imports
``wandb`` directly and so ``mode=disabled`` is a clean no-op that needs no W&B
account, network, or even the package to be logged in.

``mode`` comes from ``logging.mode`` in the config:
    online   -> sync to wandb.ai (needs auth)
    offline  -> write to ./wandb locally, sync later with `wandb sync`
    disabled -> drop everything on the floor
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import ProjectConfig

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class WandbLogger:
    """Owns the W&B run lifecycle. Safe to use unconditionally: when
    ``mode=disabled`` every method is a no-op and ``run_id``/``url`` return None."""

    def __init__(
        self,
        config: ProjectConfig,
        run_name: str,
        group: str | None = None,
        extra_config: dict[str, Any] | None = None,
    ):
        self.config = config
        self.run_name = run_name
        self.mode = config.logging.mode
        # Resolution order: explicit arg (the trainer passes the config hash) >
        # config.logging.group > None.
        self.group = group or config.logging.group
        self.job_type = config.logging.job_type
        self.extra_config = extra_config or {}
        self._run: Run | None = None
        self._owns_run: bool = False
        self._watch_active: bool = False
        self._watched_model: Any = None

    @property
    def run_id(self) -> str | None:
        if self._run is None:
            return None
        return self._run.id

    @property
    def url(self) -> str | None:
        # offline/disabled runs have no remote URL.
        return None if self._run is None else self._run.url

    def start(self) -> None:
        """Initialise the W&B run. Raises a clear error if online mode is asked
        for but W&B is not usable -- the message points at offline/disabled."""
        if self.mode == "disabled":
            return
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - exercised only without wandb
            raise RuntimeError(
                "wandb is not installed but logging.backend=wandb. "
                "Install it (`uv add wandb`) or set logging.mode=disabled."
            ) from exc

        try:
            active = getattr(wandb, "run", None)
            if active is not None and not getattr(active, "disabled", False):
                # Attach to the externally-started run (e.g. a W&B sweep agent).
                # The wandb run API is untyped; route through Any at this one
                # boundary (same pattern as analysis.py's wandb.Api()).
                active_run: Any = active
                self._run = active_run
                self._owns_run = False
                active_run.config.update(
                    {**self.config.model_dump(), **self.extra_config}, allow_val_change=True
                )
            else:
                self._run = wandb.init(
                    project=self.config.logging.project,
                    entity=self.config.logging.entity,
                    name=self.run_name,
                    mode=self.mode,
                    group=self.group,
                    job_type=self.job_type,
                    tags=self.config.logging.tags or None,
                    notes=self.config.logging.notes,
                    save_code=self.config.logging.save_code,
                    config={**self.config.model_dump(), **self.extra_config},
                )
                self._owns_run = True
        except Exception as exc:
            raise RuntimeError(
                f"W&B failed to initialise in mode={self.mode!r}: {exc}. "
                "Run `wandb login`, set WANDB_API_KEY, or use "
                "logging.mode=offline / logging.mode=disabled."
            ) from exc

    def define_metric(self, name: str, summary: str = "max") -> None:
        """Declare how a metric reduces in the W&B runs table (``summary="max"``
        for a score, ``"min"`` for a loss), so the table shows the best value
        rather than the last. No-op without a live run. The project hardcodes no
        metric names -- call this from your experiment for the metrics you log."""
        if self._run is not None:
            self._run.define_metric(name, summary=summary)

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._run is not None:
            self._run.log(metrics, step=step)

    def set_summary(self, summary: dict[str, Any]) -> None:
        """Pin final/headline numbers to the run summary (the values W&B shows
        in the runs table)."""
        if self._run is not None:
            for key, value in summary.items():
                self._run.summary[key] = value

    def watch(self, model: Any, log_freq: int = 100) -> None:
        """Log gradient/parameter histograms for ``model`` every ``log_freq``
        steps. No-op when there is no live run."""
        if self._run is None:
            return
        import wandb

        wandb.watch(model, log="all", log_freq=log_freq)
        self._watch_active = True
        self._watched_model = model

    def log_table(
        self,
        key: str,
        columns: Sequence[str],
        rows: Iterable[Sequence[Any]],
        step: int | None = None,
    ) -> None:
        """Log a sample-level table (one row per example). Cells may hold
        scalars, strings, or W&B media objects. No-op without a live run."""
        if self._run is None:
            return
        import wandb

        table = wandb.Table(columns=list(columns))
        for row in rows:
            table.add_data(*row)
        self._run.log({key: table}, step=step)

    def log_checkpoint_artifact(
        self,
        path: Path | str,
        name: str,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a checkpoint file as a versioned W&B *model* artifact (lineage +
        registry). Content-addressed, so unchanged files don't re-upload.
        No-op without a live run."""
        if self._run is None:
            return
        import wandb

        artifact = wandb.Artifact(name=name, type="model", metadata=metadata or {})
        artifact.add_file(str(path))
        self._run.log_artifact(artifact, aliases=aliases or ["latest"])

    def finish(self) -> None:
        if self._run is not None:
            import wandb

            if self._watch_active:
                wandb.unwatch(self._watched_model)
                self._watch_active = False
                self._watched_model = None
            if self._owns_run:
                self._run.finish()
            self._run = None
            self._owns_run = False
