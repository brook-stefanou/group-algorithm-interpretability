#!/usr/bin/env python3
"""Live-observability sidecar: tail ``runs/`` and stream each run to W&B.

Training never talks to W&B on the campaign path (in-process uploads were the
first campaign's failure point). This sidecar is a separate process that only
reads what training already writes under ``runs/<run_id>/`` -- ``run.log``,
``manifest.yaml``, ``resolved_config.yaml``, ``checkpoints/`` -- and streams it
to one live W&B run per training run. It can be killed and restarted at any
moment without touching training; a restart resumes each W&B run in place
because the W&B run id is the training run's own ``run_id``
(``resume="allow"``).

Each discovered run directory gets a W&B run with ``job_type="live"``, named
``"<cell> (<order>,<index>) w<width> s<seed>"`` (zero-padded seed, so the
name sorts correctly), a matching set of config keys (including
``config_group_hash``, ``campaign_id``, and ``split_seed`` for joining back to
manifests/analysis files), and a W&B ``group`` set to that same
``config_group_hash`` so every seed of one cell/width pools together in the
UI -- the same naming and grouping scheme ``instruments/publish.py`` uses for
occupancy runs, so live, occupancy, and pooled runs line up. Into that run go:

* the ``run.log`` eval rows, throttled to every ``--log-every``-th epoch, plus
  the newest row on every poll cycle so charts track the frontier. All metrics
  are plotted against an ``epoch`` step metric rather than the W&B internal
  step, so restarts and re-sends cannot corrupt the x-axis.
* ``progress/snapshot_epoch``, parsed from checkpoint filenames as snapshots
  appear. Both training paths stream curves mid-training: the single-seed
  path appends ``run.log`` row by row, and the vmapped ensemble path flushes
  its buffered eval rows every ``snapshot.history_flush_epochs`` epochs
  (default 250; ``training/ensemble.py``), so an ensemble seed's curve grows
  in flush-sized steps while it trains and completes within one poll of the
  seed finalising. With the flush disabled (``history_flush_epochs: 0``) an
  ensemble seed's ``run.log`` appears only at finalisation, and the growing
  ``checkpoints/`` directory is its only mid-training liveness signal.
* with ``--occupancy-every N`` (minutes; off by default), headline occupancy
  numbers measured on the run's newest stable snapshot using the occupancy
  instrument's own primitives (CPU), logged as ``occupancy/*`` against the
  same epoch axis. One run is measured per tick, round-robin, so the compute
  cost is bounded no matter how many seeds are live.
* on a terminal manifest: the final row, a ``grokked``/``censored``/
  ``epochs_to_grok`` summary computed from the run's own generalisation
  thresholds, and ``run.finish()``.

Concurrent W&B runs in one process use ``wandb.init(reinit="create_new")``,
the supported multi-run pattern in the installed SDK (wandb 0.28.0,
``wandb/sdk/wandb_init.py``: when a run is active and ``reinit`` is
``"create_new"``, ``init`` continues and returns an independent handle).
Every W&B call goes through one injectable seam (:class:`WandbSink`) with
per-run exponential backoff, so a W&B outage delays streaming but never
tailing. The sidecar writes nothing inside any run directory and uploads no
W&B artifacts.

Examples::

    uv run python scripts/stream_runs.py                      # watch runs/
    uv run python scripts/stream_runs.py --log-every 100 --poll-seconds 30
    uv run python scripts/stream_runs.py --occupancy-every 10
    uv run python scripts/stream_runs.py --once               # single pass
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DEFAULT_PROJECT = "group-algorithm-interp"
DEFAULT_CAMPAIGN_CONFIG = Path(__file__).resolve().parent.parent / "configs/campaign/core.yaml"
LIVE_JOB_TYPE = "live"
TERMINAL_STATUSES = frozenset({"completed", "failed", "aborted"})

# Both run.log dialects (same contract as instruments/checkpoints.py): the
# ensemble path writes bare "epoch 3 | {...}" lines, the single-seed path
# prefixes them with the logging format.
_EPOCH_LINE = re.compile(r"epoch (\d+) \| (\{.*\})\s*$")
_COMPLETED_LINE = re.compile(r"completed \S+ \| (\{.*\})\s*$")
_NONFINITE = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}

# Snapshot filename families, epoch encoded in the name (see SnapshotConfig).
_SNAPSHOT_PREFIXES = ("final_epoch_", "generalized_step_", "step_")

# Documented project defaults (OptimConfig), used only when a run's resolved
# config is missing a field.
_FALLBACK_GENERALIZE_TEST_ACC = 0.99
_FALLBACK_GENERALIZE_PATIENCE = 5

_BACKOFF_BASE_S = 5.0
_BACKOFF_CAP_S = 300.0
_OCCUPANCY_SCAN_LIMIT = 8


def _warn(message: str) -> None:
    print(f"[stream_runs] {message}", file=sys.stderr)


# --- run.log parsing ----------------------------------------------------------


def _parse_dict(text: str) -> dict[str, Any]:
    """One run.log dict repr. Bare ``nan``/``inf`` tokens (written by Python's
    repr for non-finite floats) are quoted before ``literal_eval`` and mapped
    back -- the metric keys are fixed names that never contain those words."""
    substituted = re.sub(r"(?<![\w.'-])(nan|inf|-inf)(?![\w.'])", r"'\1'", text)
    raw = ast.literal_eval(substituted)
    if not isinstance(raw, dict):
        raise ValueError(f"run.log line is not a dict: {text!r}")
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, str) and value in _NONFINITE:
            out[str(key)] = _NONFINITE[value]
        else:
            out[str(key)] = value
    return out


def _numeric_only(mapping: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in mapping.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def parse_metrics(text: str) -> dict[str, float]:
    """The numeric metrics of one run.log dict repr."""
    return _numeric_only(_parse_dict(text))


def parse_epoch_line(line: str) -> tuple[int, dict[str, float]] | None:
    """``(epoch, metrics)`` for an eval row, ``None`` for any other line."""
    match = _EPOCH_LINE.search(line)
    if match is None:
        return None
    return int(match.group(1)), parse_metrics(match.group(2))


def parse_completed_line(line: str) -> dict[str, float] | None:
    """The final metrics from a ``completed <run_id> | {...}`` trailer. The
    ensemble path writes the flat metrics dict; the single-seed path writes a
    summary block with the metrics nested under ``"metrics"`` -- both land
    here as one flat numeric dict."""
    match = _COMPLETED_LINE.search(line)
    if match is None:
        return None
    raw = _parse_dict(match.group(1))
    nested = raw.get("metrics")
    if isinstance(nested, dict):
        return {**_numeric_only(raw), **_numeric_only(dict(nested))}
    return _numeric_only(raw)


@dataclass
class LogTail:
    """Incremental reader for one ``run.log``: remembers the byte offset and
    holds any trailing partial line until its newline arrives. Tolerates the
    file not existing yet (the ensemble path's first periodic flush -- every
    ``snapshot.history_flush_epochs`` epochs -- may not have happened, and
    with the flush disabled the file appears only at finalisation) and starts
    over if the file shrinks."""

    path: Path
    offset: int = 0
    _partial: str = ""

    def read_new_lines(self) -> list[str]:
        if not self.path.is_file():
            return []
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
            self._partial = ""
        if size == self.offset:
            return []
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()
        text = self._partial + chunk.decode("utf-8", errors="replace")
        lines = text.split("\n")
        self._partial = lines.pop()
        return [line for line in lines if line.strip()]


# --- naming and config (the scheme also used by instruments/publish.py) -------


def load_campaign_lookup(path: Path) -> dict[tuple[int, int, int], dict[str, Any]]:
    """``(order, index, width) -> cell`` from a campaign cell list. A missing
    or unreadable file degrades to cosmetic-metadata-free naming."""
    try:
        data = yaml.safe_load(path.read_text())
    except OSError:
        return {}
    lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
    for cell in (data or {}).get("cells", []):
        lookup.setdefault((cell["order"], cell["index"], cell["width"]), cell)
    return lookup


def sanitise_run_id(run_id: str) -> str:
    """W&B run ids allow word characters and dashes; this project's run ids
    already fit, but sanitise defensively."""
    return re.sub(r"[^A-Za-z0-9_-]", "-", run_id)[:120]


_PAIR_PARTNER_RE = re.compile(r"with\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?")


def parse_pair_partner(note: str | None) -> tuple[int, int] | None:
    """Best-effort ``(order, index)`` of a cell's declared CT-pair partner,
    parsed from its ``note`` free text (e.g. ``"...pair with (64,80)..."``).

    ``note`` is documentation only (``configs/campaign/core.yaml``'s own
    header: "never read by the runner") and its wording is not contractual,
    so this is a display/filter convenience, not an authoritative pairing --
    a cell with no declared partner (an anchor, a solo probe) correctly
    yields ``None``, and a >2-way case study (the D32/QD32/Q32 triple)
    yields only the first-mentioned partner."""
    if not note:
        return None
    match = _PAIR_PARTNER_RE.search(note)
    return None if match is None else (int(match.group(1)), int(match.group(2)))


def format_seed(seed: Any) -> str:
    """Zero-padded seed for the display name: campaign cells run seeds 0-49,
    and unpadded ``s10`` sorts before ``s2`` in W&B's lexicographic name
    sort, which scrambles same-cell seed ordering in the UI run list."""
    try:
        return f"{int(seed):02d}"
    except (TypeError, ValueError):
        return str(seed)


@dataclass(frozen=True)
class RunSpec:
    """Everything needed to open (or resume) one training run's live W&B run."""

    run_id: str
    name: str
    config: dict[str, Any]
    tags: list[str]
    wandb_group: str | None
    generalize_metric_key: str
    generalize_test_acc: float
    generalize_patience: int


def build_run_spec(
    run_dir: Path, campaign_lookup: dict[tuple[int, int, int], dict[str, Any]]
) -> RunSpec:
    """Read manifest + resolved config and assemble the W&B identity: the
    display name, config, tags, and pooling group that ``instruments/
    publish.py`` mirrors for occupancy runs, so live streams and occupancy
    publishes line up in the UI."""
    manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text()) or {}
    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text()) or {}

    dataset = manifest.get("dataset") or {}
    leakage = dataset.get("leakage") or {}
    generalize_metric = leakage.get("generalize_metric", "unleaked_accuracy")
    metric_key = "val/accuracy" if generalize_metric == "raw_test_accuracy" else generalize_metric
    if metric_key not in ("val/accuracy", "val/unleaked_accuracy"):
        metric_key = "val/unleaked_accuracy"

    group = (resolved.get("data") or {}).get("group") or {}
    order = group.get("order")
    index = group.get("index")
    width = (resolved.get("model") or {}).get("d_model")
    seed = resolved.get("seed")
    optim = resolved.get("optim") or {}
    experiment_name = (resolved.get("experiment") or {}).get("name") or manifest.get(
        "experiment_name"
    )
    provenance = manifest.get("provenance") or {}
    config_group_hash = provenance.get("config_group_hash")

    cell = None
    if order is not None and index is not None and width is not None:
        cell = campaign_lookup.get((int(order), int(index), int(width)))
    phase = cell["phase"] if cell else "unknown"
    cell_name = cell["name"] if cell else None
    pair_partner = parse_pair_partner(cell.get("note")) if cell else None
    run_id = str(manifest.get("run_id") or run_dir.name)

    label = cell_name or f"({order},{index})"
    tags = [phase]
    if phase.startswith("bonus-"):
        tags.append("bonus")
    group_tag = cell_name or dataset.get("name")
    if group_tag:
        tags.append(group_tag)
    tags.append("campaign-v2")

    return RunSpec(
        run_id=run_id,
        name=f"{label} ({order},{index}) w{width} s{format_seed(seed)}",
        config={
            "group_order": order,
            "group_index": index,
            "group_canonical_name": dataset.get("name"),
            "cell_name": cell_name,
            "width": width,
            "seed": seed,
            "phase": phase,
            "cell_note": cell.get("note") if cell else None,
            "pair_partner_order": pair_partner[0] if pair_partner else None,
            "pair_partner_index": pair_partner[1] if pair_partner else None,
            "experiment_name": experiment_name,
            "generalize_metric": generalize_metric,
            "generalize_test_acc": optim.get("generalize_test_acc", _FALLBACK_GENERALIZE_TEST_ACC),
            "generalize_patience": optim.get("generalize_patience", _FALLBACK_GENERALIZE_PATIENCE),
            "raw_run_id": run_id,
            "config_hash": provenance.get("config_hash"),
            "config_group_hash": config_group_hash,
            "campaign_id": provenance.get("campaign_id"),
            "split_seed": dataset.get("split_seed"),
            "dataset_spec_hash": dataset.get("spec_hash"),
        },
        tags=list(dict.fromkeys(tags)),
        wandb_group=config_group_hash,
        generalize_metric_key=metric_key,
        generalize_test_acc=float(optim.get("generalize_test_acc", _FALLBACK_GENERALIZE_TEST_ACC)),
        generalize_patience=int(optim.get("generalize_patience", _FALLBACK_GENERALIZE_PATIENCE)),
    )


# --- grok summary -------------------------------------------------------------


@dataclass
class GrokTracker:
    """Streaming form of the pre-registered epochs-to-grok rule: first epoch
    beginning a run of ``metric >= threshold`` sustained for ``patience``
    consecutive evaluated epochs; a run that never sustains it is censored.
    Observes every parsed row (throttling never affects the summary)."""

    metric_key: str
    threshold: float
    patience: int
    _streak: int = 0
    _streak_start: int | None = None
    grokked: bool = False
    epochs_to_grok: int | None = None

    def observe(self, epoch: int, metrics: dict[str, float]) -> None:
        if self.grokked:
            return
        value = metrics.get(self.metric_key)
        if value is not None and not math.isnan(value) and value >= self.threshold:
            if self._streak == 0:
                self._streak_start = epoch
            self._streak += 1
            if self._streak >= self.patience:
                self.grokked = True
                self.epochs_to_grok = self._streak_start
        else:
            self._streak = 0
            self._streak_start = None

    def summary(self) -> dict[str, Any]:
        return {
            "grokked": self.grokked,
            "censored": not self.grokked,
            "epochs_to_grok": self.epochs_to_grok,
        }


# --- snapshots and live occupancy ---------------------------------------------


def snapshot_epochs(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    """``(epoch, path)`` for every epoch-named snapshot, ascending by epoch.
    ``final.pt`` carries no epoch in its name and is excluded."""
    found: dict[int, Path] = {}
    if not checkpoint_dir.is_dir():
        return []
    for path in checkpoint_dir.glob("*.pt"):
        for prefix in _SNAPSHOT_PREFIXES:
            if path.name.startswith(prefix):
                tail = path.name[len(prefix) : -len(".pt")]
                if tail.isdigit():
                    found.setdefault(int(tail), path)
                break
    return sorted(found.items())


def newest_snapshot_epoch(run_dir: Path) -> int | None:
    epochs = snapshot_epochs(run_dir / "checkpoints")
    return epochs[-1][0] if epochs else None


def measure_snapshot_occupancy(
    run_dir: Path,
    *,
    val_acc_by_epoch: dict[int, float],
    threshold: float,
) -> tuple[int, dict[str, float]] | None:
    """Headline occupancy numbers for the run's newest stable snapshot,
    computed with the occupancy instrument's primitives
    (``instruments/occupancy.py`` -- the same maths as
    ``scripts/measure_occupancy.py``, without its run.log-dependent dip rule,
    which cannot apply while the ensemble path buffers run.log in memory).

    Newest-first over the last few snapshots, a snapshot is taken if its
    ``run.log`` row (when one exists at that epoch) has ``val/accuracy >=
    threshold``; with no row it cannot be assessed and is accepted as-is.
    Returns ``(epoch, metrics)`` or ``None`` when there is nothing to measure.
    Import of torch and the model stack is deferred to here, so the tailing
    core never needs them."""
    candidates = snapshot_epochs(run_dir / "checkpoints")[-_OCCUPANCY_SCAN_LIMIT:]
    chosen: tuple[int, Path] | None = None
    for epoch, path in reversed(candidates):
        value = val_acc_by_epoch.get(epoch)
        if value is None or value >= threshold:
            chosen = (epoch, path)
            break
    if chosen is None:
        return None
    epoch, path = chosen

    import torch

    from group_algorithm_interp.config import validate_config
    from group_algorithm_interp.groups.catalog import resolve_group
    from group_algorithm_interp.instruments.occupancy import (
        analytic_null,
        dirichlet_noise_floor,
        isotypic_energies,
        neuron_activations,
        population_occupancy,
        restrict_to_nontrivial,
        total_variation,
        trivial_block_index,
    )
    from group_algorithm_interp.training.trainer import build_model

    config = validate_config(yaml.safe_load((run_dir / "resolved_config.yaml").read_text()))
    group = resolve_group(config.data.group)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(config, group)
    model.load_state_dict(payload["model_state_dict"])

    pi0 = analytic_null(group)
    trivial = trivial_block_index(group)
    pi0_nt = restrict_to_nontrivial(pi0, trivial)
    activations = neuron_activations(model, group.order)
    metrics: dict[str, float] = {"occupancy/snapshot_epoch": float(epoch)}
    for argument in ("left", "right"):
        energies = isotypic_energies(activations, group, argument=argument)
        occupancy = population_occupancy(energies)
        occupancy_nt = restrict_to_nontrivial(occupancy, trivial)
        tv = total_variation(occupancy_nt, pi0_nt)
        floor = dirichlet_noise_floor(pi0_nt, config.model.d_mlp)
        metrics[f"occupancy/{argument}/nontrivial/tv_to_null"] = tv
        metrics[f"occupancy/{argument}/nontrivial/tv_over_floor"] = tv / floor
        metrics[f"occupancy/{argument}/trivial_share"] = float(occupancy[trivial])
    return epoch, metrics


# --- the W&B seam -------------------------------------------------------------


class WandbSink:
    """The one boundary that touches the W&B SDK. Every method may raise; the
    caller treats any exception as "back off and retry later". Tests inject a
    fake with the same surface."""

    def __init__(self, project: str, entity: str | None = None):
        import wandb

        self._wandb = wandb
        self.project = project
        self.entity = entity

    def init_run(self, spec: RunSpec) -> Any:
        run = self._wandb.init(
            project=self.project,
            entity=self.entity,
            id=sanitise_run_id(spec.run_id),
            name=spec.name,
            job_type=LIVE_JOB_TYPE,
            group=spec.wandb_group,
            config=spec.config,
            tags=spec.tags or None,
            resume="allow",
            reinit="create_new",
            settings=self._wandb.Settings(
                console="off", quiet=True, x_disable_stats=True, x_disable_meta=True
            ),
        )
        run.define_metric("epoch")
        for pattern in ("train/*", "val/*", "occupancy/*", "progress/*"):
            run.define_metric(pattern, step_metric="epoch")
        return run

    def last_epoch(self, run: Any) -> int:
        """The last streamed epoch a resumed run already holds (from its
        restored summary), or -1 for a fresh run."""
        try:
            value = run.summary.get("epoch")
            return -1 if value is None else int(value)
        except Exception:  # noqa: BLE001 -- a summary quirk must not stop streaming
            return -1

    def stream_complete(self, run: Any) -> bool:
        try:
            return bool(run.summary.get("stream/complete"))
        except Exception:  # noqa: BLE001
            return False

    def log(self, run: Any, payload: dict[str, float]) -> None:
        run.log(payload)

    def set_summary(self, run: Any, mapping: dict[str, Any]) -> None:
        run.summary.update(mapping)

    def finish(self, run: Any) -> None:
        run.finish()


# --- per-run streaming state --------------------------------------------------


@dataclass
class RunStreamer:
    """One training run's tail-and-stream state machine.

    ``poll`` always tails and parses (pure local reads); ``_flush`` is the only
    part that talks to W&B, and any failure there backs off per run without
    losing rows -- pending rows stay queued until a send succeeds."""

    run_dir: Path
    spec: RunSpec
    sink: Any
    log_every: int
    tail: LogTail = field(init=False)
    grok: GrokTracker = field(init=False)
    val_acc_by_epoch: dict[int, float] = field(default_factory=dict)
    pending: list[tuple[int, dict[str, float]]] = field(default_factory=list)
    latest_row: tuple[int, dict[str, float]] | None = None
    final_row: dict[str, float] | None = None
    extra_rows: list[dict[str, float]] = field(default_factory=list)
    handle: Any = None
    finished: bool = False
    last_sent_epoch: int = -1
    last_selected_epoch: int = -1
    snapshot_epoch_sent: int = -1
    occupancy_epoch_sent: int = -1
    _fail_count: int = 0
    _retry_at: float = 0.0

    def __post_init__(self) -> None:
        self.tail = LogTail(self.run_dir / "run.log")
        self.grok = GrokTracker(
            metric_key=self.spec.generalize_metric_key,
            threshold=self.spec.generalize_test_acc,
            patience=self.spec.generalize_patience,
        )

    # -- local side (never raises out, never talks to the network) ------------

    def poll(self, now: float) -> None:
        if self.finished:
            return
        try:
            terminal = self._ingest()
        except Exception as exc:  # noqa: BLE001 -- one run's bad file must not stop the loop
            _warn(f"{self.spec.run_id}: ingest failed ({type(exc).__name__}: {exc})")
            return
        self._flush(now, terminal)

    def _ingest(self) -> bool:
        for line in self.tail.read_new_lines():
            try:
                row = parse_epoch_line(line)
                if row is not None:
                    epoch, metrics = row
                    self.grok.observe(epoch, metrics)
                    if "val/accuracy" in metrics:
                        self.val_acc_by_epoch[epoch] = metrics["val/accuracy"]
                    self.latest_row = (epoch, metrics)
                    if epoch % self.log_every == 0 and epoch > self.last_selected_epoch:
                        self.pending.append((epoch, metrics))
                        self.last_selected_epoch = epoch
                    continue
                completed = parse_completed_line(line)
                if completed is not None:
                    self.final_row = completed
            except (ValueError, SyntaxError, TypeError) as exc:
                _warn(f"{self.spec.run_id}: skipping malformed line {line[:100]!r}: {exc}")

        snapshot_epoch = newest_snapshot_epoch(self.run_dir)
        if snapshot_epoch is not None and snapshot_epoch > self.snapshot_epoch_sent:
            self.extra_rows.append({"progress/snapshot_epoch": float(snapshot_epoch)})
            self.snapshot_epoch_sent = snapshot_epoch

        manifest = yaml.safe_load((self.run_dir / "manifest.yaml").read_text()) or {}
        return str(manifest.get("status")) in TERMINAL_STATUSES

    # -- W&B side (all failures back off) --------------------------------------

    def _flush(self, now: float, terminal: bool) -> None:
        if now < self._retry_at:
            return
        try:
            if self.handle is None:
                self.handle = self.sink.init_run(self.spec)
                already = self.sink.last_epoch(self.handle)
                if already >= 0:
                    self.pending = [(e, m) for e, m in self.pending if e > already]
                    self.last_sent_epoch = already
                if terminal and self.sink.stream_complete(self.handle):
                    # A previous sidecar already streamed and closed this run.
                    self.sink.finish(self.handle)
                    self.finished = True
                    return

            while self.pending:
                epoch, metrics = self.pending[0]
                self.sink.log(self.handle, {"epoch": float(epoch), **metrics})
                self.last_sent_epoch = max(self.last_sent_epoch, epoch)
                self.pending.pop(0)

            if self.latest_row is not None and self.latest_row[0] > self.last_sent_epoch:
                epoch, metrics = self.latest_row
                self.sink.log(self.handle, {"epoch": float(epoch), **metrics})
                self.last_sent_epoch = epoch

            while self.extra_rows:
                self.sink.log(self.handle, self.extra_rows[0])
                self.extra_rows.pop(0)

            if terminal:
                final_metrics = self.final_row or (self.latest_row[1] if self.latest_row else {})
                self.sink.set_summary(
                    self.handle,
                    {**final_metrics, **self.grok.summary(), "stream/complete": True},
                )
                self.sink.finish(self.handle)
                self.finished = True

            self._fail_count = 0
            self._retry_at = 0.0
        except Exception as exc:  # noqa: BLE001 -- W&B failures must never stop tailing
            self._fail_count += 1
            delay = min(_BACKOFF_BASE_S * (2 ** (self._fail_count - 1)), _BACKOFF_CAP_S)
            self._retry_at = now + delay
            _warn(
                f"{self.spec.run_id}: W&B error ({type(exc).__name__}: {exc}); "
                f"retrying in {delay:.0f}s"
            )


# --- the sidecar loop ---------------------------------------------------------


class Sidecar:
    """Watches a runs root, streams every run directory that appears, and
    optionally measures live occupancy round-robin."""

    def __init__(
        self,
        runs_root: Path,
        sink: Any,
        *,
        campaign_lookup: dict[tuple[int, int, int], dict[str, Any]] | None = None,
        log_every: int = 50,
        occupancy_every_s: float | None = None,
        occupancy_threshold: float = 0.99,
        now: Callable[[], float] = time.monotonic,
    ):
        self.runs_root = runs_root
        self.sink = sink
        self.campaign_lookup = campaign_lookup or {}
        self.log_every = log_every
        self.occupancy_every_s = occupancy_every_s
        self.occupancy_threshold = occupancy_threshold
        self.now = now
        self.streams: dict[str, RunStreamer] = {}
        self._occupancy_next = 0.0
        self._occupancy_cursor = 0

    def discover(self) -> None:
        if not self.runs_root.is_dir():
            return
        for manifest_path in sorted(self.runs_root.glob("*/manifest.yaml")):
            run_dir = manifest_path.parent
            if run_dir.name in self.streams:
                continue
            if not (run_dir / "resolved_config.yaml").is_file():
                continue  # still being created; pick it up next cycle
            try:
                spec = build_run_spec(run_dir, self.campaign_lookup)
            except Exception as exc:  # noqa: BLE001 -- retried on the next cycle
                _warn(f"{run_dir.name}: cannot read run metadata yet ({exc})")
                continue
            self.streams[run_dir.name] = RunStreamer(
                run_dir=run_dir, spec=spec, sink=self.sink, log_every=self.log_every
            )
            _warn(f"{spec.run_id}: discovered ({spec.name})")

    def _occupancy_tick(self) -> None:
        if self.occupancy_every_s is None:
            return
        now = self.now()
        if now < self._occupancy_next:
            return
        self._occupancy_next = now + self.occupancy_every_s
        active = [s for s in self.streams.values() if not s.finished]
        if not active:
            return
        active.sort(key=lambda s: s.spec.run_id)
        stream = active[self._occupancy_cursor % len(active)]
        self._occupancy_cursor += 1
        try:
            result = measure_snapshot_occupancy(
                stream.run_dir,
                val_acc_by_epoch=stream.val_acc_by_epoch,
                threshold=self.occupancy_threshold,
            )
        except Exception as exc:  # noqa: BLE001 -- a bad checkpoint must not stop the loop
            _warn(f"{stream.spec.run_id}: occupancy measurement failed ({exc})")
            return
        if result is None:
            return
        epoch, metrics = result
        if epoch == stream.occupancy_epoch_sent:
            return
        stream.occupancy_epoch_sent = epoch
        stream.extra_rows.append({"epoch": float(epoch), **metrics})

    def poll_once(self) -> None:
        self.discover()
        # Tick before the streams flush, so a measurement queued this cycle
        # goes out this cycle (and a single `--once` pass delivers it).
        self._occupancy_tick()
        for stream in self.streams.values():
            stream.poll(self.now())

    def run_forever(self, poll_seconds: float, *, once: bool = False) -> None:
        while True:
            self.poll_once()
            if once:
                return
            time.sleep(poll_seconds)


# --- CLI ----------------------------------------------------------------------


def streaming_skip_reason() -> str | None:
    """Why live streaming cannot proceed (mirrors the occupancy publisher's
    environmental gate), or ``None`` when it can."""
    mode = os.environ.get("WANDB_MODE", "").strip().lower()
    if mode in ("disabled", "dryrun", "offline"):
        return f"WANDB_MODE={mode}; live streaming is online-only"
    if not os.environ.get("WANDB_API_KEY"):
        return "WANDB_API_KEY is not set; nothing to stream with"
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT))
    parser.add_argument("--entity", default=None)
    parser.add_argument(
        "--campaign-config",
        type=Path,
        default=DEFAULT_CAMPAIGN_CONFIG,
        help="cell list used for run naming (cosmetic metadata only)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="stream every Nth epoch row (the newest row always goes out each poll)",
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--occupancy-every",
        type=float,
        default=None,
        metavar="MINUTES",
        help="measure live occupancy on one run's newest stable snapshot every N minutes "
        "(round-robin over live runs; off by default)",
    )
    parser.add_argument(
        "--occupancy-threshold",
        type=float,
        default=0.99,
        help="val/accuracy bar for calling a snapshot stable when a run.log row exists",
    )
    parser.add_argument("--once", action="store_true", help="one poll cycle, then exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    skip = streaming_skip_reason()
    if skip is not None:
        _warn(f"not streaming: {skip}")
        return 2
    if args.log_every < 1:
        _warn("--log-every must be at least 1")
        return 2

    sink = WandbSink(args.project, args.entity)
    sidecar = Sidecar(
        args.runs_root,
        sink,
        campaign_lookup=load_campaign_lookup(args.campaign_config),
        log_every=args.log_every,
        occupancy_every_s=None if args.occupancy_every is None else args.occupancy_every * 60.0,
        occupancy_threshold=args.occupancy_threshold,
    )
    try:
        sidecar.run_forever(args.poll_seconds, once=args.once)
    except KeyboardInterrupt:
        _warn("interrupted; open W&B runs resume on the next start (id=run_id)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
