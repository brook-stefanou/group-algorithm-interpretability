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
from dataclasses import dataclass, field, replace
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
    ``substitution_recorded`` is ``False`` only for a curated ``categories``-shape
    reconstruction where the underlying substitution genuinely could not be
    determined (an unreadable/malformed config, or an unrecognised checkpoint
    filename) -- in that case ``substitution`` is ``None`` as a placeholder, not
    a claim that no substitution occurred; every other path (a live recompute,
    or a flat-shape/full-record reconstruction) knows the answer, so it stays
    ``True``. ``rejected`` lists every dipped or unassessable candidate that was
    passed over, newest first, each with the metric value that disqualified it
    (``None`` when no ``run.log`` row exists at that epoch). ``path`` is
    ``None`` -- with ``reason`` set -- when no stable checkpoint exists at all.
    ``requested_metric``/``requested_threshold`` are set only when a curated
    run's ``selection.json`` short-circuit returned a pick gated on a different
    metric/threshold than the caller asked for (see :func:`select_checkpoint`):
    the divergence between what was asked and what was recorded is then visible
    as data rather than silently substituted.
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
    substitution_recorded: bool = True
    requested_metric: str | None = None
    requested_threshold: float | None = None

    def to_record(self) -> dict[str, Any]:
        """JSON-serialisable form for embedding in an analysis output."""
        record: dict[str, Any] = {
            "rule": self.rule,
            "metric": self.metric,
            "threshold": self.threshold,
            "checkpoint": None if self.path is None else self.path.name,
            "epoch": self.epoch,
            "metric_value": self.metric_value,
            "substitution": self.substitution,
            "substitution_recorded": self.substitution_recorded,
            "rejected": list(self.rejected),
            "reason": self.reason,
        }
        if self.requested_metric is not None or self.requested_threshold is not None:
            record["requested_metric"] = self.requested_metric
            record["requested_threshold"] = self.requested_threshold
        return record


def _snapshot_epochs(run_dir: Path, patterns: tuple[str, ...]) -> list[tuple[int, Path]]:
    """``(epoch, path)`` pairs for every snapshot matching ``patterns``, sorted
    by ascending epoch, discovered both FLAT at ``run_dir`` (the curated
    shipped layout) and under ``run_dir/checkpoints/`` (the local training
    layout) -- mirroring :func:`resolve_checkpoint`'s flat-first resolution, so
    the recompute fallback (used when ``selection.json`` is missing or
    unreadable) still finds a curated run's snapshots rather than seeing an
    empty ``checkpoints/`` directory that was never there. ``generalized_step_*.pt``
    does not match ``step_*.pt`` under fnmatch (the prefix differs), so each
    family is globbed explicitly; when two files share an epoch they hold
    identical weights and either serves (the flat one is kept when both exist)."""
    found: dict[int, Path] = {}
    for directory in (run_dir / CHECKPOINTS_DIRNAME, run_dir):  # flat overrides nested
        if not directory.is_dir():
            continue
        for pattern in patterns:
            prefix = pattern[: pattern.index("*")]
            for path in directory.glob(pattern):
                epoch = int(path.name[len(prefix) : -len(".pt")])
                found[epoch] = path
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


def _read_resolved_config(run_dir: Path) -> dict[str, Any] | None:
    """The parsed ``resolved_config.yaml``, or ``None`` when it is missing,
    unreadable, or not a mapping. A single degrade point so a missing config
    behaves the same way on every path that reads it, rather than silently
    defaulting on one and raising an uncaught ``FileNotFoundError`` on
    another."""
    try:
        resolved = yaml.safe_load((run_dir / RESOLVED_CONFIG_NAME).read_text())
    except (OSError, yaml.YAMLError):
        return None
    return resolved if isinstance(resolved, dict) else None


def _snapshot_window(resolved: dict[str, Any] | None) -> tuple[bool, int]:
    """``(has_window, final_window_epochs)`` from a parsed resolved config's
    ``snapshot`` block. Tolerates a malformed value -- a scalar ``snapshot:
    true`` (not a mapping) or a non-numeric ``final_window_epochs`` -- by
    degrading to "no window" rather than raising ``AttributeError``/``ValueError``."""
    if resolved is None:
        return False, 0
    raw_snapshot = resolved.get("snapshot")
    snapshot_cfg = raw_snapshot if isinstance(raw_snapshot, dict) else {}
    raw_window = snapshot_cfg.get("final_window_epochs")
    try:
        window_epochs = int(raw_window) if raw_window is not None else 0
    except (TypeError, ValueError):
        return False, 0
    has_window = "final_window_epochs" in snapshot_cfg and window_epochs > 0
    return has_window, window_epochs


def _optim_epochs(resolved: dict[str, Any] | None) -> int | None:
    """``optim.epochs`` (the run's epoch ceiling) from a parsed resolved
    config, or ``None`` when absent, unreadable, or non-numeric."""
    if resolved is None:
        return None
    raw_optim = resolved.get("optim")
    optim_cfg = raw_optim if isinstance(raw_optim, dict) else {}
    raw_epochs = optim_cfg.get("epochs")
    try:
        return int(raw_epochs) if raw_epochs is not None else None
    except (TypeError, ValueError):
        return None


def _rule_from_config(run_dir: Path) -> str:
    """The dip-aware rule name (``"final_window"`` | ``"final_gate"``) implied by
    a run's resolved config, used to label a selection reconstructed from
    ``selection.json`` where the rule was not itself recorded. Presence of
    ``snapshot.final_window_epochs`` (with a positive value) marks a
    restarted-campaign window run; anything else takes the final-gate rule.
    Defaults to ``"final_window"`` when the config cannot be read (the shipped
    campaign is entirely window runs)."""
    resolved = _read_resolved_config(run_dir)
    if resolved is None:
        return "final_window"
    has_window, _ = _snapshot_window(resolved)
    return "final_window" if has_window else "final_gate"


def _epoch_from_filename(filename: str) -> tuple[str, int | None]:
    """``(kind, epoch)`` parsed from a checkpoint filename: ``kind`` is one of
    ``"final_epoch"``, ``"step"``, ``"generalized_step"``, ``"final"`` (no
    epoch in the name), or ``"other"`` when the name matches none of the known
    snapshot-naming families."""
    if filename == "final.pt":
        return "final", None
    for prefix, kind in (
        ("final_epoch_", "final_epoch"),
        ("generalized_step_", "generalized_step"),
        ("step_", "step"),
    ):
        if filename.startswith(prefix) and filename.endswith(".pt"):
            tail = filename[len(prefix) : -len(".pt")]
            if tail.isdigit():
                return kind, int(tail)
    return "other", None


def _recompute_curated_substitution(
    run_dir: Path,
    entry: dict[str, Any],
    rule: str,
) -> tuple[str | None, list[dict[str, Any]], bool]:
    """Best-effort honest recomputation of a ``categories``-shape ``stable_end``
    entry's ``substitution``/``rejected`` fields for a shipped run whose
    ``selection.json`` predates the ship hook recording them directly (finding:
    the curated ``categories`` shape used to store no substitution info at
    all, so every reconstruction fabricated ``substitution=None, rejected=[]``
    -- indistinguishable from a genuine no-substitution pick).

    Uses only the deterministic final-window epoch arithmetic
    (``{ceiling - N, ..., ceiling - 1}``, from ``optim.epochs`` and
    ``snapshot.final_window_epochs`` in ``resolved_config.yaml`` --
    :mod:`training.ensemble`'s snapshot condition is exact arithmetic, no
    runtime state) plus ``run.log``/``run.log.gz`` lookups -- never the pruned
    ``checkpoints/`` directory, which the curated layout does not keep these
    files under.

    Returns ``(substitution, rejected, recomputed)``. ``recomputed`` is
    ``False`` when the substitution genuinely cannot be determined (an
    unreadable/malformed config needed to place a ``final_epoch_*`` pick within
    its window, or a filename outside the known snapshot-naming families); the
    caller then records ``substitution_recorded=False`` rather than presenting
    a guess as fact. A ``step_*``/``generalized_step_*`` pick is always a
    trajectory substitution regardless of config (a trajectory file is never a
    final-window one), so that classification needs no config at all -- only
    its ``rejected`` window trail does. When the pick fell to a trajectory
    snapshot from the ``final_gate`` rule, only the single ``final.pt``
    rejection is recoverable: which other trajectory candidates were tried
    first is data-dependent, event-based snapshot timing that a filename-plus-
    config computation cannot recover, so ``rejected`` stays a partial (never
    fabricated) trail there.
    """
    filename = entry.get("filename")
    if not filename:
        return None, [], True  # no checkpoint recorded; no substitution question applies
    kind, epoch = _epoch_from_filename(str(filename))
    metric_key = str(entry.get("metric_key") or "val/accuracy")

    if rule == "final_gate":
        if kind == "final":
            return None, [], True
        if kind in ("step", "generalized_step"):
            rows = {row.epoch: row.metrics for row in run_log_rows(run_dir)}
            last_epoch = max(rows) if rows else None
            rejected: list[dict[str, Any]] = []
            if last_epoch is not None:
                rejected.append(
                    {
                        "checkpoint": "final.pt",
                        "epoch": last_epoch,
                        "value": rows.get(last_epoch, {}).get(metric_key),
                    }
                )
            return "trajectory", rejected, True
        return None, [], False

    if rule == "final_window":
        if kind == "final_epoch" and epoch is not None:
            resolved = _read_resolved_config(run_dir)
            ceiling = _optim_epochs(resolved)
            has_window, window_n = _snapshot_window(resolved)
            if ceiling is None or not has_window:
                return None, [], False  # can't place this epoch within the window
            window_epochs = list(range(max(ceiling - window_n, 0), ceiling))
            if epoch not in window_epochs:
                return None, [], False  # inconsistent with the config; don't guess
            if epoch == window_epochs[-1]:
                return None, [], True
            rows = {row.epoch: row.metrics for row in run_log_rows(run_dir)}
            rejected = [
                {
                    "checkpoint": f"final_epoch_{e}.pt",
                    "epoch": e,
                    "value": rows.get(e, {}).get(metric_key),
                }
                for e in sorted((e for e in window_epochs if e > epoch), reverse=True)
            ]
            return "window", rejected, True
        if kind in ("step", "generalized_step"):
            rejected = []
            resolved = _read_resolved_config(run_dir)
            ceiling = _optim_epochs(resolved)
            has_window, window_n = _snapshot_window(resolved)
            if ceiling is not None and has_window:
                rows = {row.epoch: row.metrics for row in run_log_rows(run_dir)}
                window_epochs = list(range(max(ceiling - window_n, 0), ceiling))
                rejected = [
                    {
                        "checkpoint": f"final_epoch_{e}.pt",
                        "epoch": e,
                        "value": rows.get(e, {}).get(metric_key),
                    }
                    for e in sorted(window_epochs, reverse=True)
                ]
            return "trajectory", rejected, True
        return None, [], False

    return None, [], False


def _final_epoch_prune_fallback(
    run_dir: Path, filename: Any, epoch: Any
) -> tuple[Path | None, str | None]:
    """When a recorded ``stable_end`` pick names ``final_epoch_<E>.pt`` for
    ``E`` exactly the run's final training epoch (``optim.epochs - 1``, read
    from ``resolved_config.yaml``) and that exact file is absent from a
    re-downloaded, pruned run dir, substitutes ``final.pt``.

    Weight-safe, not a heuristic: :mod:`training.ensemble`'s per-epoch loop
    writes ``final_epoch_<epoch>.pt`` for the final-window epochs from the
    post-optimiser-step in-memory model state, and immediately after the loop
    ``_finalize`` writes ``final.pt`` from that *same* still-unmodified state
    -- no optimiser step runs in between. At the ceiling epoch the two files
    are two ``torch.save`` calls of bit-identical tensors. That equivalence
    holds only there: an earlier ``final_epoch_<E>.pt`` (``E < ceiling - 1``)
    or a ``step_*.pt``/``generalized_step_*.pt`` trajectory snapshot shares no
    such guarantee, so this never substitutes for those -- they stay
    unresolved (``None``) when absent, and the caller's usual "not on disk"
    handling applies.

    Returns ``(final_path, note)`` when every condition holds -- ``filename``
    starts with ``final_epoch_``, its recorded ``epoch`` is exactly
    ``optim.epochs - 1``, and ``final.pt`` is present on disk (flat or under
    ``checkpoints/``) -- else ``(None, None)``."""
    if not isinstance(filename, str) or not filename.startswith("final_epoch_"):
        return None, None
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        return None, None
    resolved = _read_resolved_config(run_dir)
    ceiling = _optim_epochs(resolved)
    if ceiling is None or epoch != ceiling - 1:
        return None, None
    final_path = resolve_checkpoint(run_dir, "final.pt")
    if final_path is None:
        return None, None
    note = (
        f"selection.json named {filename} (epoch {epoch} == optim.epochs - 1, the run's "
        "final training epoch) but the curated layout kept only final.pt; substituted "
        "final.pt, which training saves from the identical in-memory model state at the "
        "same epoch with no optimiser step in between -- a documented weight-safe "
        "equivalence, not a guess"
    )
    return final_path, note


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
    """Rebuild a selection from the flat shape's full ``stable_end`` record.

    ``record`` is ``None`` for a run that recorded no stable checkpoint at all
    -- an explicit JSON ``null``, not a missing key. That is itself a complete
    answer, not an unrecognised shape, so it returns the no-checkpoint
    selection directly rather than returning ``None`` here and sending the
    caller to the curated-layout-blind recompute fallback (which used to
    misclassify these as censored for the wrong reason). Any other
    non-``dict`` value is still treated as an unrecognised shape."""
    if record is None:
        return CheckpointSelection(
            run_dir=run_dir,
            rule=rule,
            metric=metric,
            threshold=threshold,
            path=None,
            epoch=None,
            metric_value=None,
            substitution=None,
            substitution_recorded=True,
            rejected=[],
            reason=_NO_STABLE_REASON,
        )
    if not isinstance(record, dict):
        return None
    filename = record.get("checkpoint")
    ckpt_path = resolve_checkpoint(run_dir, filename)
    reason = record.get("reason")
    substituted_note: str | None = None
    if filename and ckpt_path is None:
        ckpt_path, substituted_note = _final_epoch_prune_fallback(
            run_dir, filename, record.get("epoch")
        )
        if ckpt_path is None:
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
        reason=substituted_note if ckpt_path is not None else reason,
    )


def _selection_from_curated_entry(
    run_dir: Path,
    data: dict[str, Any],
    entry: Any,
    rule: str,
    threshold: float,
) -> CheckpointSelection:
    """Rebuild a selection from ``ship_runs.py``'s lighter curated entry.

    A ship predating the fixed :func:`scripts.ship_runs.build_curated_selection`
    recorded no substitution info at all for this category -- reconstructing
    that older entry used to fabricate ``substitution=None, rejected=[]``,
    indistinguishable from a genuine no-substitution pick (70 of 2,290 shipped
    runs affected). A ship that already carries ``substitution``/``rejected``
    keys (the fixed hook writes them for every future ship) is read straight
    through instead. Only when neither is available does
    :func:`_recompute_curated_substitution` attempt an honest best-effort
    recomputation from ``run.log`` plus the deterministic window arithmetic,
    marking ``substitution_recorded=False`` when even that is not feasible.
    """
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
            substitution_recorded=True,
            rejected=[],
            reason=_stable_end_anomaly(data) or _NO_STABLE_REASON,
        )
    filename = entry.get("filename")
    ckpt_path = resolve_checkpoint(run_dir, filename)
    reason = None
    substituted_note: str | None = None
    if filename and ckpt_path is None:
        ckpt_path, substituted_note = _final_epoch_prune_fallback(
            run_dir, filename, entry.get("epoch")
        )
        if ckpt_path is None:
            reason = f"selection.json names {filename!r} but the checkpoint is not on disk"

    if "substitution" in entry:
        substitution = entry.get("substitution")
        rejected = list(entry.get("rejected") or [])
        substitution_recorded = True
    else:
        substitution, rejected, substitution_recorded = _recompute_curated_substitution(
            run_dir, entry, rule
        )

    return CheckpointSelection(
        run_dir=run_dir,
        rule=rule,
        metric=str(entry.get("metric_key") or leaked_key),
        threshold=resolved_threshold,
        path=ckpt_path,
        epoch=entry.get("epoch") if ckpt_path is not None else None,
        metric_value=entry.get("metric_value") if ckpt_path is not None else None,
        substitution=substitution,
        substitution_recorded=substitution_recorded,
        rejected=rejected,
        reason=substituted_note if substituted_note is not None else reason,
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
    ``checkpoints/`` snapshots and ``run.log`` rows (also discovering a curated
    run's FLAT snapshots, so a missing/corrupt ``selection.json`` does not
    misclassify a grokked curated run as censored).

    This short-circuit is gated on whatever metric/threshold the ship hook
    used at prune time (leaked ``val/accuracy``), not on ``metric``/``threshold``
    as passed here -- an open call on whether to recompute the pick for a
    different metric (e.g. the unleaked one) instead. Until that is decided,
    a caller asking for a different metric/threshold still gets the recorded
    pick, but the divergence is made visible: the returned selection's
    ``requested_metric``/``requested_threshold`` carry what was actually asked
    for whenever it differs from what got used.
    """
    from_json = selection_from_json(run_dir, metric=metric, threshold=threshold)
    if from_json is not None:
        if metric != from_json.metric or threshold != from_json.threshold:
            from_json = replace(from_json, requested_metric=metric, requested_threshold=threshold)
        return from_json

    log_path = resolve_run_log(run_dir)

    resolved = _read_resolved_config(run_dir)
    if resolved is None:
        return CheckpointSelection(
            run_dir=run_dir,
            rule="final_window",
            metric=metric,
            threshold=threshold,
            path=None,
            epoch=None,
            metric_value=None,
            substitution=None,
            reason="resolved_config.yaml missing, unreadable, or not a mapping; "
            "cannot determine the dip-aware rule",
        )
    # Presence of the field is what distinguishes a restarted-campaign run from
    # a terminated-campaign one; a run that explicitly configured 0 window
    # epochs has the field but no window files, and takes the final-gate rule.
    # A malformed value (a scalar `snapshot: true`, a non-numeric
    # final_window_epochs) degrades to "no window" rather than raising.
    has_window, _window_epochs = _snapshot_window(resolved)
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
        window = _snapshot_epochs(run_dir, ("final_epoch_*.pt",))
        ceiling = _optim_epochs(resolved)
        # Weight-safe prune fallback for this path's own layout (mirrors
        # _final_epoch_prune_fallback, used by the selection.json-driven
        # paths): a local run dir re-downloaded/pruned to just checkpoints/
        # final.pt has no final_epoch_<E>.pt files at all, so the loop below
        # -- which only ever looks at files present on disk -- would never
        # even consider the ceiling epoch. Check it explicitly: only when the
        # window's OWN top epoch (optim.epochs - 1) is both what the dip rule
        # would pick (its run.log row is stable) and missing on disk does
        # final.pt substitute -- a genuine dip that lands on an earlier,
        # non-ceiling epoch never triggers this, so it stays unresolved when
        # that epoch's file is also absent (different, non-substitutable
        # weights).
        if ceiling is not None and not any(epoch == ceiling - 1 for epoch, _ in window):
            ceiling_epoch = ceiling - 1
            stable, value = _stability(rows, ceiling_epoch, metric, threshold)
            if stable:
                fallback_path, note = _final_epoch_prune_fallback(
                    run_dir, f"final_epoch_{ceiling_epoch}.pt", ceiling_epoch
                )
                if fallback_path is not None:
                    return CheckpointSelection(
                        run_dir=run_dir,
                        rule=rule,
                        metric=metric,
                        threshold=threshold,
                        path=fallback_path,
                        epoch=ceiling_epoch,
                        metric_value=value,
                        substitution=None,
                        rejected=rejected,
                        reason=note,
                    )
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
        final = resolve_checkpoint(run_dir, "final.pt")
        last_epoch = max(rows)
        if final is not None:
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
    trajectory = _snapshot_epochs(run_dir, ("step_*.pt", "generalized_step_*.pt"))
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
