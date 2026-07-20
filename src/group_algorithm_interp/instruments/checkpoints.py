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
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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
    authoritative one.
    """
    rows: dict[int, EpochRow] = {}
    with path.open() as handle:
        for line in handle:
            match = _EPOCH_LINE.search(line)
            if match is None:
                continue
            epoch = int(match.group(1))
            rows[epoch] = EpochRow(epoch=epoch, metrics=_parse_metrics(match.group(2)))
    return [rows[e] for e in sorted(rows)]


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


def select_checkpoint(
    run_dir: Path,
    *,
    metric: str = "val/accuracy",
    threshold: float = 0.99,
) -> CheckpointSelection:
    """Apply the dip-aware rule (module docstring) to one run directory."""
    ckpt_dir = run_dir / "checkpoints"
    log_path = run_dir / "run.log"

    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
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

    if not log_path.is_file():
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


__all__ = ["CheckpointSelection", "EpochRow", "parse_run_log", "select_checkpoint"]
