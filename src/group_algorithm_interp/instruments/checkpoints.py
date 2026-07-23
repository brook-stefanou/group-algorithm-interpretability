"""Dip-aware checkpoint selection for post-hoc analysis.

Post-grok "slingshot" dips are real destabilisations intrinsic to the training
configuration, so ``final.pt`` can land inside a dip and misrepresent the model
the run actually settled on. An analysis therefore never reads ``final.pt``
blindly; it selects a checkpoint through the rule below, and every deviation
from the naive choice is recorded in the analysis output as data (the
literature-supported practice: report the substitution, never hide it).

The rule, applied per run:

* **Runs whose ``resolved_config.yaml`` carries ``snapshot.final_window_epochs``**
  (the restarted campaign onward): among the final-window snapshots
  ``checkpoints/final_epoch_<E>.pt``, select the last epoch whose ``run.log``
  row at exactly that epoch has ``val/accuracy >= threshold``. If every window
  epoch is dipped, fall back to the nearest (latest) stable trajectory snapshot
  (``step_<E>.pt`` / ``generalized_step_<E>.pt``).
* **Older runs** (the field absent from the resolved config): gate ``final.pt``
  on the final ``run.log`` row; when dipped, substitute the nearest stable
  trajectory snapshot.

A checkpoint written at epoch ``E`` holds post-update weights, and the
``val/*`` metrics in the epoch-``E`` ``run.log`` row are post-update too (both
training paths document this), so the row at ``E`` describes exactly the
weights in the epoch-``E`` snapshot. A snapshot with no ``run.log`` row at its
own epoch cannot be assessed and is skipped, recorded as rejected.

A run with no stable checkpoint anywhere (for example a censored seed that
never reached the bar) selects nothing: ``path`` is ``None`` and ``reason``
says why. The caller records that outcome rather than silently analysing an
unstable model.
"""

from __future__ import annotations

import ast
import gzip
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Filenames shared with the ship hook's curated layout. A shipped run keeps its
# checkpoints FLAT at the run-dir root (no ``checkpoints/`` subdirectory), its
# training log gzipped as ``run.log.gz`` (or absent on the earliest flat-schema
# runs), and a ``selection.json`` recording the dip-aware pick made at prune
# time. A freshly trained local run instead has a ``checkpoints/`` subdirectory,
# a plain ``run.log``, and no ``selection.json``. The resolvers below read
# either layout so the instruments run against both.
RUN_LOG_NAME = "run.log"
RUN_LOG_GZ_NAME = "run.log.gz"
CHECKPOINTS_DIRNAME = "checkpoints"
SELECTION_FILENAME = "selection.json"
RESOLVED_CONFIG_NAME = "resolved_config.yaml"

# Matches both run.log dialects: the ensemble path writes bare
# "epoch 3 | {...}" lines, the single-seed path prefixes them with the logging
# format ("HH:MM:SS | INFO | group_algorithm_interp | epoch 3 | {...}").
_EPOCH_LINE = re.compile(r"epoch (\d+) \| (\{.*\})\s*$")

# Python's repr of the metric dicts writes bare ``nan``/``inf`` for non-finite
# floats (a degenerate run records NaN unleaked accuracy), which
# ``ast.literal_eval`` rejects. The metric keys are fixed names that never
# contain these words, so a bare-token substitution is safe.
_NONFINITE = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}


def _parse_metrics(text: str) -> dict[str, float]:
    substituted = re.sub(r"(?<![\w.'-])(nan|inf|-inf)(?![\w.'])", r"'\1'", text)
    raw = ast.literal_eval(substituted)
    if not isinstance(raw, dict):
        raise ValueError(f"run.log metrics line is not a dict: {text!r}")
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, str) and value in _NONFINITE:
            out[str(key)] = _NONFINITE[value]
        else:
            out[str(key)] = float(value)
    return out


@dataclass(frozen=True)
class EpochRow:
    """One evaluated epoch's metrics as recorded in ``run.log``."""

    epoch: int
    metrics: dict[str, float]


def parse_run_log(path: Path) -> list[EpochRow]:
    """Every ``epoch N | {...}`` metrics row of a ``run.log``, in file order.

    Both training paths' dialects are accepted (bare and logging-prefixed
    lines); non-metric lines (startup banners, warnings, the ``completed ...``
    trailer) are ignored. When the same epoch appears twice the later row wins:
    rows are appended in training order, so the last occurrence is the
    authoritative one. A ``.gz`` path is read transparently through gzip (the
    ship hook gzips the log), so both the curated ``run.log.gz`` and a plain
    ``run.log`` parse the same way.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    rows: dict[int, EpochRow] = {}
    with opener(path, "rt") as handle:
        for line in handle:
            match = _EPOCH_LINE.search(line)
            if match is None:
                continue
            epoch = int(match.group(1))
            rows[epoch] = EpochRow(epoch=epoch, metrics=_parse_metrics(match.group(2)))
    return [rows[e] for e in sorted(rows)]


def resolve_run_log(run_dir: Path) -> Path | None:
    """The run's training log, preferring a plain ``run.log`` (a freshly trained
    local run) and falling back to the ship hook's gzipped ``run.log.gz``.
    ``None`` when neither exists (the earliest flat-schema shipped runs kept no
    log at all)."""
    plain = run_dir / RUN_LOG_NAME
    if plain.is_file():
        return plain
    gz = run_dir / RUN_LOG_GZ_NAME
    if gz.is_file():
        return gz
    return None


def run_log_rows(run_dir: Path) -> list[EpochRow]:
    """Parsed epoch rows for whichever training log the run carries (plain or
    gzipped), or ``[]`` when it carries none."""
    log_path = resolve_run_log(run_dir)
    return parse_run_log(log_path) if log_path is not None else []


def resolve_checkpoint(run_dir: Path, filename: str | None) -> Path | None:
    """A checkpoint file by name, tried FLAT at the run-dir root first (the
    curated shipped layout) and then under ``checkpoints/`` (the old training
    layout). ``None`` when the name is empty or the file is on neither path."""
    if not filename:
        return None
    flat = run_dir / filename
    if flat.is_file():
        return flat
    nested = run_dir / CHECKPOINTS_DIRNAME / filename
    if nested.is_file():
        return nested
    return None


@dataclass(frozen=True)
class CheckpointSelection:
    """The outcome of the dip-aware rule for one run, substitutions included.

    ``substitution`` is ``None`` when the rule's primary choice was stable (the
    last window epoch, or ``final.pt``), ``"window"`` when an earlier
    final-window epoch was selected because later ones were dipped, and
    ``"trajectory"`` when the rule fell back to a trajectory snapshot.
    ``rejected`` lists every dipped or unassessable candidate that was passed
    over, newest first, each with the metric value that disqualified it
    (``None`` when no ``run.log`` row exists at that epoch). ``path`` is
    ``None`` -- with ``reason`` set -- when no stable checkpoint exists at all.
    """

    run_dir: Path
    rule: str  # "final_window" | "final_gate"
    metric: str
    threshold: float
    path: Path | None
    epoch: int | None
    metric_value: float | None
    substitution: str | None
    rejected: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None

    def to_record(self) -> dict[str, Any]:
        """JSON-serialisable form for embedding in an analysis output."""
        return {
            "rule": self.rule,
            "metric": self.metric,
            "threshold": self.threshold,
            "checkpoint": None if self.path is None else self.path.name,
            "epoch": self.epoch,
            "metric_value": self.metric_value,
            "substitution": self.substitution,
            "rejected": list(self.rejected),
            "reason": self.reason,
        }


def _snapshot_epochs(ckpt_dir: Path, patterns: tuple[str, ...]) -> list[tuple[int, Path]]:
    """``(epoch, path)`` pairs for every snapshot matching ``patterns``, sorted
    by ascending epoch. ``generalized_step_*.pt`` does not match ``step_*.pt``
    under fnmatch (the prefix differs), so each family is globbed explicitly;
    when two files share an epoch they hold identical weights and either
    serves."""
    found: dict[int, Path] = {}
    for pattern in patterns:
        prefix = pattern[: pattern.index("*")]
        for path in ckpt_dir.glob(pattern):
            epoch = int(path.name[len(prefix) : -len(".pt")])
            found.setdefault(epoch, path)
    return sorted(found.items())


def _stability(
    rows: dict[int, dict[str, float]], epoch: int, metric: str, threshold: float
) -> tuple[bool, float | None]:
    """Whether the ``run.log`` row at exactly ``epoch`` clears the bar, and the
    value it recorded (``None`` when no row exists at that epoch)."""
    metrics = rows.get(epoch)
    if metrics is None or metric not in metrics:
        return False, None
    value = metrics[metric]
    if math.isnan(value):
        return False, value
    return value >= threshold, value


def _rule_from_config(run_dir: Path) -> str:
    """The dip-aware rule name (``"final_window"`` | ``"final_gate"``) implied by
    a run's resolved config, used to label a selection reconstructed from
    ``selection.json`` where the rule was not itself recorded. Presence of
    ``snapshot.final_window_epochs`` (with a positive value) marks a
    restarted-campaign window run; anything else takes the final-gate rule.
    Defaults to ``"final_window"`` when the config cannot be read (the shipped
    campaign is entirely window runs)."""
    try:
        resolved = yaml.safe_load((run_dir / RESOLVED_CONFIG_NAME).read_text())
    except (OSError, yaml.YAMLError):
        return "final_window"
    if not isinstance(resolved, dict):
        return "final_window"
    snapshot_cfg = resolved.get("snapshot") or {}
    window_epochs = int(snapshot_cfg.get("final_window_epochs") or 0)
    has_window = "final_window_epochs" in snapshot_cfg and window_epochs > 0
    return "final_window" if has_window else "final_gate"


_NO_STABLE_REASON = "selection.json recorded no stable checkpoint (censored/never-stable run)"


def _stable_end_anomaly(data: dict[str, Any]) -> str | None:
    """The ``stable_end`` explanation the ship hook recorded in ``anomalies``,
    used as the reason when the curated ``stable_end`` is ``None``."""
    for entry in data.get("anomalies") or []:
        if isinstance(entry, str) and entry.startswith("stable_end"):
            return entry
    return None


def selection_from_json(
    run_dir: Path,
    *,
    metric: str = "val/accuracy",
    threshold: float = 0.99,
) -> CheckpointSelection | None:
    """Reconstruct the dip-aware selection from a run's ``selection.json`` when
    present, so a curated shipped run (flat checkpoints, log gzipped or gone) is
    read straight from the pick the ship hook already computed and recorded --
    no ``checkpoints/`` glob and no ``run.log`` needed. ``None`` when the file is
    absent, unreadable, or carries neither recognised ``stable_end`` block, so
    the caller falls back to recomputing over the on-disk training layout.

    Two ``selection.json`` shapes coexist in the campaign and both are read
    (mirroring :func:`publish._grok_fields_from_selection`):

    * The flat shape's ``selections.stable_end`` is a full
      :meth:`CheckpointSelection.to_record` dict (rule, metric, threshold,
      checkpoint filename, epoch, substitution, rejected, reason) -- mapped
      through verbatim, with the checkpoint file resolved flat-first.
    * ``scripts/ship_runs.py``'s ``categories.stable_end`` is a lighter curated
      entry (``epoch``/``metric_key``/``metric_value``/``filename``, or ``None``
      when no stable checkpoint existed) -- the missing fields are filled from
      the config-implied rule and the top-level threshold, and a ``None`` entry
      becomes a no-checkpoint selection carrying the recorded anomaly reason.
    """
    path = run_dir / SELECTION_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    rule = _rule_from_config(run_dir)

    selections = data.get("selections")
    if isinstance(selections, dict) and "stable_end" in selections:
        return _selection_from_full_record(
            run_dir, selections.get("stable_end"), rule, metric, threshold
        )

    categories = data.get("categories")
    if isinstance(categories, dict) and "stable_end" in categories:
        return _selection_from_curated_entry(
            run_dir, data, categories.get("stable_end"), rule, threshold
        )

    return None


def _selection_from_full_record(
    run_dir: Path,
    record: Any,
    rule: str,
    metric: str,
    threshold: float,
) -> CheckpointSelection | None:
    """Rebuild a selection from the flat shape's full ``stable_end`` record."""
    if not isinstance(record, dict):
        return None
    filename = record.get("checkpoint")
    ckpt_path = resolve_checkpoint(run_dir, filename)
    reason = record.get("reason")
    if filename and ckpt_path is None:
        reason = f"selection.json names {filename!r} but the checkpoint is not on disk"
    elif ckpt_path is None and not reason:
        reason = _NO_STABLE_REASON
    return CheckpointSelection(
        run_dir=run_dir,
        rule=str(record.get("rule") or rule),
        metric=str(record.get("metric") or metric),
        threshold=float(record["threshold"]) if record.get("threshold") is not None else threshold,
        path=ckpt_path,
        epoch=record.get("epoch") if ckpt_path is not None else None,
        metric_value=record.get("metric_value") if ckpt_path is not None else None,
        substitution=record.get("substitution"),
        rejected=list(record.get("rejected") or []),
        reason=None if ckpt_path is not None else reason,
    )


def _selection_from_curated_entry(
    run_dir: Path,
    data: dict[str, Any],
    entry: Any,
    rule: str,
    threshold: float,
) -> CheckpointSelection:
    """Rebuild a selection from ``ship_runs.py``'s lighter curated entry."""
    top_threshold = data.get("threshold")
    resolved_threshold = float(top_threshold) if top_threshold is not None else threshold
    leaked_key = str(data.get("leaked_metric_key") or "val/accuracy")
    if not isinstance(entry, dict):
        return CheckpointSelection(
            run_dir=run_dir,
            rule=rule,
            metric=leaked_key,
            threshold=resolved_threshold,
            path=None,
            epoch=None,
            metric_value=None,
            substitution=None,
            rejected=[],
            reason=_stable_end_anomaly(data) or _NO_STABLE_REASON,
        )
    filename = entry.get("filename")
    ckpt_path = resolve_checkpoint(run_dir, filename)
    reason = None
    if filename and ckpt_path is None:
        reason = f"selection.json names {filename!r} but the checkpoint is not on disk"
    return CheckpointSelection(
        run_dir=run_dir,
        rule=rule,
        metric=str(entry.get("metric_key") or leaked_key),
        threshold=resolved_threshold,
        path=ckpt_path,
        epoch=entry.get("epoch") if ckpt_path is not None else None,
        metric_value=entry.get("metric_value") if ckpt_path is not None else None,
        substitution=None,
        rejected=[],
        reason=reason,
    )


def select_checkpoint(
    run_dir: Path,
    *,
    metric: str = "val/accuracy",
    threshold: float = 0.99,
) -> CheckpointSelection:
    """Apply the dip-aware rule (module docstring) to one run directory.

    When the run carries a ``selection.json`` (every shipped run does), the pick
    is taken straight from the recorded ``stable_end`` -- the ship hook ran this
    same rule at prune time, and the curated layout has no ``checkpoints/`` dir
    or plain ``run.log`` to recompute over. A freshly trained local run has no
    ``selection.json`` and falls through to the recomputation below over its
    ``checkpoints/`` snapshots and ``run.log`` rows.
    """
    from_json = selection_from_json(run_dir, metric=metric, threshold=threshold)
    if from_json is not None:
        return from_json

    ckpt_dir = run_dir / CHECKPOINTS_DIRNAME
    log_path = resolve_run_log(run_dir)

    resolved = yaml.safe_load((run_dir / RESOLVED_CONFIG_NAME).read_text())
    if not isinstance(resolved, dict):
        raise ValueError(f"resolved_config.yaml in {run_dir} is not a mapping")
    snapshot_cfg = resolved.get("snapshot") or {}
    # Presence of the field is what distinguishes a restarted-campaign run from
    # a terminated-campaign one; a run that explicitly configured 0 window
    # epochs has the field but no window files, and takes the final-gate rule.
    window_epochs = int(snapshot_cfg.get("final_window_epochs") or 0)
    has_window = "final_window_epochs" in snapshot_cfg and window_epochs > 0
    rule = "final_window" if has_window else "final_gate"

    def _empty(reason: str, rejected: list[dict[str, Any]] | None = None) -> CheckpointSelection:
        return CheckpointSelection(
            run_dir=run_dir,
            rule=rule,
            metric=metric,
            threshold=threshold,
            path=None,
            epoch=None,
            metric_value=None,
            substitution=None,
            rejected=rejected or [],
            reason=reason,
        )

    if log_path is None:
        return _empty("run.log not found; checkpoint stability cannot be assessed")
    rows = {row.epoch: row.metrics for row in parse_run_log(log_path)}
    if not rows:
        return _empty("run.log contains no epoch metrics rows")

    rejected: list[dict[str, Any]] = []

    if has_window:
        window = _snapshot_epochs(ckpt_dir, ("final_epoch_*.pt",))
        for position, (epoch, path) in enumerate(reversed(window)):
            stable, value = _stability(rows, epoch, metric, threshold)
            if stable:
                return CheckpointSelection(
                    run_dir=run_dir,
                    rule=rule,
                    metric=metric,
                    threshold=threshold,
                    path=path,
                    epoch=epoch,
                    metric_value=value,
                    substitution=None if position == 0 else "window",
                    rejected=rejected,
                )
            rejected.append({"checkpoint": path.name, "epoch": epoch, "value": value})
    else:
        final = ckpt_dir / "final.pt"
        last_epoch = max(rows)
        if final.is_file():
            stable, value = _stability(rows, last_epoch, metric, threshold)
            if stable:
                return CheckpointSelection(
                    run_dir=run_dir,
                    rule=rule,
                    metric=metric,
                    threshold=threshold,
                    path=final,
                    epoch=last_epoch,
                    metric_value=value,
                    substitution=None,
                    rejected=rejected,
                )
            rejected.append({"checkpoint": final.name, "epoch": last_epoch, "value": value})
        else:
            rejected.append({"checkpoint": "final.pt", "epoch": None, "value": None})

    # Fallback shared by both rules: the nearest (latest) stable trajectory
    # snapshot, assessed against the run.log row at its own epoch.
    trajectory = _snapshot_epochs(ckpt_dir, ("step_*.pt", "generalized_step_*.pt"))
    for epoch, path in reversed(trajectory):
        stable, value = _stability(rows, epoch, metric, threshold)
        if stable:
            return CheckpointSelection(
                run_dir=run_dir,
                rule=rule,
                metric=metric,
                threshold=threshold,
                path=path,
                epoch=epoch,
                metric_value=value,
                substitution="trajectory",
                rejected=rejected,
            )
        rejected.append({"checkpoint": path.name, "epoch": epoch, "value": value})

    return _empty(
        f"no checkpoint has a run.log row with {metric} >= {threshold} at its own epoch "
        "(a censored or never-stable run)",
        rejected,
    )


__all__ = [
    "CheckpointSelection",
    "EpochRow",
    "parse_run_log",
    "resolve_checkpoint",
    "resolve_run_log",
    "run_log_rows",
    "select_checkpoint",
    "selection_from_json",
]
