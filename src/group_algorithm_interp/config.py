"""Typed, validated configuration for a run.

Hydra composes the config (defaults list + CLI overrides) but does almost no
validation: a typo'd ``optim.epochs=-1`` or ``logging.mode=onine`` sails
straight through. Pydantic turns the resolved dict into a typed object with real
constraints, so an invalid config fails before a model is built or W&B is
touched.

The flow is::

    Hydra compose -> resolved OmegaConf -> plain dict -> ProjectConfig (validated)

``ProjectConfig`` is the single object a run consumes. Each sub-config maps 1:1
onto a Hydra config group under ``configs/`` (model/, data/, logging/, ...).
"""

from __future__ import annotations

from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Device = Literal["auto", "cpu", "cuda", "mps"]
WandbMode = Literal["online", "offline", "disabled"]


class _Strict(BaseModel):
    """Base for every config model: reject unknown keys so a typo in a YAML file
    or a CLI override (``model.hidden_dimm=...``) fails validation instead of
    being silently ignored."""

    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Strict):
    """Analysis-friendly group-operation model."""

    arch: Literal["transformer", "fc"] = "transformer"
    d_model: int = Field(128, gt=0)
    n_heads: int = Field(4, gt=0)
    # Unset (null/omitted) derives d_mlp = 2 * d_model, the ratio the shipped
    # 128/256 default already used. Resolved in `_default_d_mlp_to_two_x_d_model`
    # before field validation, so the declared type stays plain `int` and every
    # downstream reader (model construction, resolved_config.yaml,
    # manifest.compute_group_hash) sees a concrete integer, never `null` -- which
    # also means an unset d_mlp and an explicit d_mlp = 2 * d_model hash
    # identically. Set d_mlp explicitly to decouple it from a d_model sweep. The
    # Field default below is never read on a normal construction (the
    # before-validator injects the real value first); it exists for mypy/schema.
    d_mlp: int = Field(256, gt=0)
    use_mlp: bool = True
    activation: Literal["relu", "gelu", "silu"] = "relu"

    @model_validator(mode="before")
    @classmethod
    def _default_d_mlp_to_two_x_d_model(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("d_mlp") is None:
            d_model = data.get("d_model", cls.model_fields["d_model"].default)
            data = {**data, "d_mlp": 2 * int(d_model)}
        return data


class GroupSpec(_Strict):
    """Canonical source-of-truth identity of a finite group: its SmallGroup
    ``(order, index)`` in the Sage/GAP catalogue."""

    order: int = Field(gt=0)
    index: int = Field(gt=0)

    @property
    def canonical_id(self) -> tuple[int, int]:
        return self.order, self.index

    @property
    def canonical_name(self) -> str:
        return f"SmallGroup({self.order},{self.index})"


# Name -> SmallGroup(order, index) for config shorthands.
_GROUP_NAME_TO_ID: dict[str, tuple[int, int]] = {
    "C2": (2, 1),
    "C3": (3, 1),
    "C4": (4, 1),
    "C5": (5, 1),
    "C6": (6, 2),
    "C7": (7, 1),
    "C8": (8, 1),
    "S3": (6, 1),
    "D8": (8, 3),
    "Q8": (8, 4),
}


# The split is useless -- and silently produces nan metrics -- if either side is
# empty (an empty tensor makes ``F.cross_entropy`` return nan, an accuracy mean
# over zero elements is nan, and a nan never trips the generalisation bar, so the
# run burns its full epoch budget and finalises as ``completed`` with nan
# numbers). ``0 < train_frac < 1`` does NOT prevent this: with a small group,
# ``round(train_frac * |G|^2)`` can still equal (or reach zero out of) |G|^2 --
# e.g. C2 has 4 pairs, so train_frac=0.9 rounds to 4 train / 0 test.
MIN_TRAIN_EXAMPLES = 1
MIN_TEST_EXAMPLES = 1


def split_sizes(n_examples: int, train_frac: float) -> tuple[int, int]:
    """(n_train, n_test) for ``n_examples`` split at ``train_frac``.

    The single definition of how a fraction becomes counts. ``task.train_test_split``
    and :class:`DataConfig`'s guard must agree, or the guard would validate a
    split the splitter never produces.
    """
    n_train = round(train_frac * n_examples)
    return n_train, n_examples - n_train


class DataConfig(_Strict):
    """Finite-group multiplication dataset and deterministic split."""

    group: GroupSpec = Field(default_factory=lambda: GroupSpec(order=8, index=1))
    train_frac: float = Field(0.8, gt=0.0, lt=1.0)
    # Pinned by default (not None) so a multi-seed sweep that only varies
    # ``config.seed`` reuses the same train/test split across seeds. ``None``
    # is still accepted for callers that explicitly want the split tied to
    # ``config.seed``; see the warning in GroupGeneralizationExperiment.setup()
    # and ``ProjectConfig.effective_split_seed``, which is what actually reaches
    # the splitter and the dataset hash.
    split_seed: int | None = 0

    @field_validator("group", mode="before")
    @classmethod
    def _resolve_group(cls, value: object) -> object:
        if isinstance(value, str):
            if value in _GROUP_NAME_TO_ID:
                order, index = _GROUP_NAME_TO_ID[value]
                return {"order": order, "index": index}
            if "," in value:
                order_str, index_str = value.split(",", 1)
                return {"order": int(order_str.strip()), "index": int(index_str.strip())}
        return value

    @model_validator(mode="after")
    def _split_leaves_both_sides_populated(self) -> DataConfig:
        """Reject a (group, train_frac) pair whose split empties either side.

        This is the config-level twin of the guard in ``task.train_test_split``:
        it fails at validation time, before a run directory, a model, or a W&B
        run exists, and it names both the group and the train_frac so the fix is
        obvious.
        """
        n_examples = self.group.order**2
        n_train, n_test = split_sizes(n_examples, self.train_frac)
        if n_train < MIN_TRAIN_EXAMPLES or n_test < MIN_TEST_EXAMPLES:
            raise ValueError(
                f"data.train_frac={self.train_frac} splits the "
                f"{n_examples} multiplication pairs of "
                f"{self.group.canonical_name} into {n_train} train / {n_test} "
                f"test, but at least {MIN_TRAIN_EXAMPLES} train and "
                f"{MIN_TEST_EXAMPLES} test example(s) are required (an empty "
                "side yields nan loss/accuracy and a run that still reports "
                "itself completed). Choose a train_frac that leaves both sides "
                f"non-empty for this group -- for order {self.group.order}, "
                f"train_frac must round to between {MIN_TRAIN_EXAMPLES} and "
                f"{n_examples - MIN_TEST_EXAMPLES} of {n_examples} pairs."
            )
        return self


class OptimConfig(_Strict):
    lr: float = Field(1e-3, gt=0.0)
    lr_base_width: int = Field(64, gt=0)
    scale_lr_with_width: bool = False
    lr_effective: float | None = None
    betas: tuple[float, float] = (0.9, 0.98)
    weight_decay: float = Field(1.0, ge=0.0)
    epochs: int = Field(10_000, gt=0)
    log_every: int = Field(100, gt=0)
    print_every: int = Field(1000, gt=0)
    wandb_every: int = Field(100, gt=0)
    stop_on_generalize: bool = False
    generalize_test_acc: float = Field(0.99, gt=0.0, le=1.0)
    generalize_patience: int = Field(5, gt=0)

    @field_validator("betas")
    @classmethod
    def _betas_in_unit_interval(cls, value: tuple[float, float]) -> tuple[float, float]:
        if not all(0.0 <= beta < 1.0 for beta in value):
            raise ValueError(f"each beta must be in [0, 1), got {value}")
        return value


class SnapshotConfig(_Strict):
    """Trajectory-snapshot policy: the single concept governing everything
    written under ``runs/<run_id>/checkpoints/``.

    This project never resumes a run -- restarting from scratch is cheap, so
    there is no separate resume-oriented checkpoint config. Two independent
    switches control what gets written under ``checkpoints/``:

    * ``enabled`` gates the periodic trajectory capture: ``step_N.pt`` (step 0,
      dense powers-of-two up to ``log_dense_until``, then every ``interval``
      steps, plus an event trigger on a relative test-loss drop of more than
      ``event_rel_drop``) and ``generalized_step_N.pt`` when the
      stop-on-generalize rule fires.
    * ``save_final`` independently gates one ``final.pt`` snapshot of the last
      step, written whether or not periodic capture is enabled.
    * ``final_window_epochs`` independently gates a per-epoch capture of the
      final window: each of the last N epochs of the configured ceiling
      (``optim.epochs``) is written as ``final_epoch_<E>.pt``, whether or not
      periodic capture is enabled (0 disables the window). ``final.pt`` is
      unchanged by this window; the window exists so post-hoc analysis can
      recover a stable model when a post-grok "slingshot" dip destabilises
      the very last epoch.
    """

    enabled: bool = True
    log_dense_until: int = Field(1024, ge=0)
    interval: int = Field(1000, gt=0)
    event_based: bool = True
    event_rel_drop: float = Field(0.1, gt=0.0)
    save_final: bool = True
    final_window_epochs: int = Field(5, ge=0)


class LoggingConfig(_Strict):
    # W&B is the only metrics backend: nothing branches on this value, it is only
    # recorded in the manifest. A free-form ``str`` would let ``backend:
    # tensorboard`` validate and then silently log to W&B anyway; the Literal
    # makes an unsupported backend fail validation. A second backend needs a real
    # branch in wandb_utils, so keep this single-valued until that branch exists.
    backend: Literal["wandb"] = "wandb"
    # Keep programmatic construction safe in CI and local analysis. Campaigns
    # that intentionally sync metrics must opt in with ``logging.mode=online``.
    mode: WandbMode = "disabled"
    project: str = "group-algorithm-interp"
    entity: str | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    # --- W&B feature flags -------------------------------------------------
    # group: runs sharing a group are aggregated (mean ± std) in the W&B UI.
    # None -> the trainer fills it with a hash of the config that ignores the
    # seed, so all seeds of one config collapse into a single comparable group.
    group: str | None = None
    job_type: str = "run"
    save_code: bool = False
    # watch_model: log gradient/parameter histograms via wandb.watch. Off by
    # default -- it adds hooks + overhead you only want when debugging training.
    watch_model: bool = False
    watch_log_freq: int = Field(100, gt=0)
    # log_checkpoints: also log best/final checkpoints as versioned W&B model
    # artifacts (lineage + model registry). Off by default to respect storage.
    log_checkpoints: bool = False


class ExperimentConfig(_Strict):
    """Run identity and lifecycle controls for a finite-group experiment."""

    name: str = "run"
    # Constructor-only compatibility for migrated analysis/tests. New configs
    # keep these lifecycle fields at the top level and in LoggingConfig. Each is
    # *wired* in ProjectConfig._reconcile_and_guard: seed -> ProjectConfig.seed,
    # use_wandb=False -> logging.mode="disabled". Run length has exactly one
    # home -- ``optim.epochs`` (the knob the training loop iterates); the
    # ``experiment/*.yaml`` presets set it directly.
    seed: int | None = Field(default=None, exclude=True)
    use_wandb: bool | None = Field(default=None, exclude=True)


class EvalConfig(_Strict):
    """Scoring and run-budget settings."""

    # run-budget guard: a run whose optim.epochs exceeds this fails validation
    # unless ProjectConfig.allow_expensive is set. Set this to a
    # project-appropriate ceiling before launching very long runs.
    max_steps_warn: int = Field(100_000, gt=0)


Direction = Literal["higher", "lower"]


class Prediction(_Strict):
    """A pre-registered prediction with a direction (anti-Goodhart). Read as:
    the held-out ``metric`` will be ``direction`` (higher/lower) than ``value``."""

    metric: str
    direction: Direction
    value: float


class ValidationConfig(_Strict):
    """Held-out / pre-registration discipline.

    ``exploratory`` (default) imposes nothing -- rapid iteration, but the run is
    recorded as exploratory so its numbers can't later be passed off as validated.
    ``confirmatory`` turns the gate on: the harness then requires a declared
    held-out set and a single sealed evaluation (see ``BaseExperiment``)."""

    stance: Literal["exploratory", "confirmatory"] = "exploratory"
    prediction: Prediction | None = None

    @model_validator(mode="after")
    def _require_confirmatory_prediction(self) -> ValidationConfig:
        if self.stance == "confirmatory" and self.prediction is None:
            raise ValueError("confirmatory validation requires a pre-registered prediction")
        return self


class ProjectConfig(_Strict):
    """The fully validated config a trainer consumes."""

    project_name: str = "group-algorithm-interp"
    seed: int = 0
    deterministic: bool = False
    device: Device = "auto"
    # opt-in escape hatch for the expensive-run guard below.
    allow_expensive: bool = False

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    snapshot: SnapshotConfig = Field(default_factory=SnapshotConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @property
    def effective_split_seed(self) -> int:
        """The seed the train/test split is actually built from.

        ``data.split_seed`` is pinned by default, but ``None`` is accepted and
        then falls back to ``config.seed`` (see
        ``GroupGeneralizationExperiment.setup``). Everything that has to describe
        the data a run consumed -- the dataset hash, the manifest's dataset block
        -- must use *this* value, not the configured one, or it records ``null``
        for a split that was in fact seeded.
        """
        return self.seed if self.data.split_seed is None else self.data.split_seed

    @model_validator(mode="after")
    def _reconcile_and_guard(self) -> ProjectConfig:
        """Resolve the compatibility aliases and derived quantities, then fail
        loudly on a dangerously large/expensive run unless it explicitly opts in
        with ``allow_expensive=true``.
        """
        if self.experiment.seed is not None:
            self.seed = self.experiment.seed
        if self.experiment.use_wandb is False:
            self.logging.mode = "disabled"

        if not self.allow_expensive and self.optim.epochs > self.eval.max_steps_warn:
            raise ValueError(
                f"optim.epochs={self.optim.epochs} exceeds "
                f"eval.max_steps_warn={self.eval.max_steps_warn}. Set "
                "allow_expensive=true to proceed with a long run."
            )

        # The learning rate every run actually trains at (consumed in
        # GroupGeneralizationExperiment.setup). scale_lr_with_width defaults to
        # False, so lr_effective == lr == 1e-3 by default regardless of
        # d_model -- the learning rate is constant across widths unless a run
        # explicitly opts in. Defining invariant, asserted in
        # tests/test_config.py: lr_effective == lr when scale_lr_with_width is
        # off, and lr * lr_base_width / d_model when it is on -- so doubling
        # d_model halves the learning rate.
        effective_lr = self.optim.lr
        if self.optim.scale_lr_with_width:
            effective_lr *= self.optim.lr_base_width / self.model.d_model
        self.optim.lr_effective = effective_lr
        return self


# Historical alias for analysis code and old checkpoint readers: the same schema
# object, not a separate training config.
GeneralizationConfig = ProjectConfig


def validate_config(cfg: DictConfig | dict[str, Any]) -> ProjectConfig:
    """Convert a composed Hydra/OmegaConf config (or plain dict) into a validated
    ``ProjectConfig``. This is the one boundary where untyped config becomes a
    typed object; everything downstream takes ``ProjectConfig``.
    """
    if isinstance(cfg, DictConfig):
        # resolve=True expands any ${interpolations}; the result is a plain dict.
        raw = OmegaConf.to_container(cfg, resolve=True)
    else:
        raw = cfg
    if not isinstance(raw, dict):
        raise TypeError(f"config must resolve to a mapping, got {type(raw).__name__}")
    # Hydra keeps its own bookkeeping under a top-level `hydra:` key when present;
    # it is not part of the experiment config, so drop it before validation.
    container: dict[str, Any] = {str(k): v for k, v in raw.items() if k != "hydra"}
    return ProjectConfig(**container)
