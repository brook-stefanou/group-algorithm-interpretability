"""Local run directory + manifest.

Every run gets a self-contained directory::

    runs/<run_id>/
      manifest.yaml         provenance + tracking + summary (this module)
      resolved_config.yaml  the exact validated config used
      checkpoints/          model weights
      analysis/             post-hoc analysis outputs (analysis.py)

The manifest is the source of truth for *what happened*: it starts as ``running``
and is finalised to ``completed`` or ``failed``. W&B holds the metric curves; the
manifest holds the headline numbers and the pointers (git commit, config hash,
W&B run id/url) needed to find everything else.
"""

from __future__ import annotations

import hashlib
import json
import platform
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from omegaconf import OmegaConf

from .config import ProjectConfig
from .seed import detect_device

MANIFEST_NAME = "manifest.yaml"
RESOLVED_CONFIG_NAME = "resolved_config.yaml"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_config_hash(config: ProjectConfig) -> str:
    """A stable hash of the *whole* resolved config -- two runs with the same hash
    used the same knobs, down to the W&B tags and the requested device. This is
    the run-provenance hash (recorded as ``provenance.config_hash``); it is
    deliberately sensitive to everything, including ``seed``."""
    sorted_json = json.dumps(config.model_dump(), sort_keys=True)
    return hashlib.sha256(sorted_json.encode()).hexdigest()[:16]


def compute_group_hash(config: ProjectConfig) -> str:
    """A stable hash of the *scientific* configuration: the model, the data, and
    the optimiser -- and nothing else.

    Contract: two configs share this hash iff they define the same experiment
    apart from the initialisation ``seed``. That is what its two consumers need:

    * every seed of one config collapses into one comparable W&B group
      (``config_group_hash``, logged into each run's config).

    It deliberately excludes the *operational* fields -- ``logging`` (mode, tags,
    notes), ``device``, ``deterministic``, ``experiment.name``, ``snapshot``,
    ``eval``, ``validation``, ``allow_expensive``, ``project_name``. Folding those
    in (the previous behaviour: the entire config minus ``seed``) meant that
    adding a W&B tag, or naming ``device: cuda`` instead of ``auto``, changed the
    "group" of a run that was scientifically identical. The full-config hash is
    still recorded separately as ``provenance.config_hash``.
    """
    payload = {
        "model": config.model.model_dump(),
        "data": config.data.model_dump(),
        "optim": config.optim.model_dump(),
    }
    sorted_json = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(sorted_json.encode()).hexdigest()[:16]


def compute_dataset_hash(config: ProjectConfig) -> str:
    """A stable hash identifying the dataset a run consumed: the data spec, with
    the *effective* split seed (see ``ProjectConfig.effective_split_seed``).

    Contract: two runs with the same hash saw byte-identical data. This is why
    ``config.seed`` is not folded in: with ``data.split_seed`` pinned -- which it
    is by default, and which every multi-seed campaign cell pins across its
    seeds -- the initialisation seed does not touch the data at all, so
    including it gave a sweep of N seeds N different "dataset" hashes for one
    shared split, and an analysis grouping runs by ``spec_hash`` to confirm they
    trained on the same split would have concluded the opposite. The effective
    seed *is* folded in, so an unpinned ``split_seed: null`` (which falls back to
    ``config.seed``) still hashes the split it actually produced.

    Real projects that load external data should fold in the source revision /
    file hashes here too, and note provenance in the research log.
    """
    data = config.data.model_dump()
    # Record what the splitter was actually given, not what was configured: a
    # null split_seed means "fall back to config.seed", and two runs that reach
    # the same effective seed by different routes did see identical data.
    data["split_seed"] = config.effective_split_seed
    payload = {"data": data}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def compute_lockfile_hash(path: Path | str = "uv.lock") -> str | None:
    """sha256 of the dependency lockfile, or ``None`` if it isn't present -- a
    cheap fingerprint of the exact environment a run was reproduced in."""
    lock = Path(path)
    if not lock.is_file():
        return None
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def _git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_commit() -> str | None:
    return _git(["rev-parse", "--short", "HEAD"])


def get_git_dirty() -> bool | None:
    status = _git(["status", "--porcelain"])
    return None if status is None else len(status) > 0


def create_run_id(experiment_name: str) -> str:
    """A sortable run id that remains unique for fast or concurrent runs."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%f")
    return f"{timestamp}_{experiment_name}_{secrets.token_hex(3)}"


def create_run_dir(run_id: str, root: Path | str = "runs") -> Path:
    run_dir = Path(root) / run_id
    # Never attach a new run to an existing directory: that would mix provenance
    # and allow checkpoints or manifests to overwrite one another.
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "analysis").mkdir()
    return run_dir


def save_resolved_config(config: ProjectConfig, run_dir: Path) -> Path:
    """Write the exact validated config used for this run to
    ``resolved_config.yaml`` (post-Hydra-composition, post-validation)."""
    path = run_dir / RESOLVED_CONFIG_NAME
    text = OmegaConf.to_yaml(OmegaConf.create(config.model_dump()))
    path.write_text(text)
    return path


def create_manifest(
    config: ProjectConfig, run_dir: Path, campaign_id: str | None = None
) -> dict[str, Any]:
    """Write the initial manifest with status ``running``. Tracking and summary
    fields are placeholders until the run finishes (see the update helpers).

    ``campaign_id`` is the launching campaign's identifier, read from the
    ``GAI_CAMPAIGN_ID`` environment variable by the experiment/ensemble runner,
    or ``None`` for a standalone run -- it is never merged into ``config``, so it
    cannot affect ``compute_config_hash`` or ``compute_group_hash``."""
    manifest: dict[str, Any] = {
        "run_id": run_dir.name,
        "status": "running",
        "created_at": _utcnow(),
        "completed_at": None,
        "project_name": config.project_name,
        "experiment_name": config.experiment.name,
        "provenance": {
            "git_commit": get_git_commit(),
            "git_dirty": get_git_dirty(),
            "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
            # Two hashes, two questions: config_hash answers "was this the same
            # run, in every detail?"; config_group_hash answers "was this the
            # same experiment, apart from the seed?" (model + data + optim only).
            "config_hash": compute_config_hash(config),
            "config_group_hash": compute_group_hash(config),
            "lockfile_hash": compute_lockfile_hash(),
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),  # TorchVersion isn't yaml-safe
            "platform": platform.platform(),
            # What hardware was actually present (independent of the device the
            # config requested) -- helps diagnose platform-specific failures.
            "available_device": detect_device(),
            # The launching campaign's identifier (from GAI_CAMPAIGN_ID), or
            # None for a standalone run outside any campaign.
            "campaign_id": campaign_id,
        },
        "dataset": {
            "name": config.data.group.canonical_name,
            "spec_hash": compute_dataset_hash(config),
            "train_frac": config.data.train_frac,
            # The seed the split was ACTUALLY built from. Recording the raw
            # config value wrote `null` whenever `data.split_seed` was unset,
            # even though the splitter had silently fallen back to `config.seed`
            # -- so the seed that produced the split was recorded nowhere.
            "split_seed": config.effective_split_seed,
        },
        "tracking": {
            "backend": config.logging.backend,
            "mode": config.logging.mode,
            "wandb_entity": config.logging.entity,
            "wandb_project": config.logging.project,
            "wandb_run_id": None,
            "wandb_url": None,
        },
        "artifacts": {
            "checkpoint_dir": str(run_dir / "checkpoints"),
            "final_checkpoint": None,
            "analysis_dir": str(run_dir / "analysis"),
        },
        "summary": {
            "completed_steps": None,
            "best_metric": None,
            "best_step": None,
            "metrics": {},  # the run's returned metrics dict (shape-neutral)
        },
        "validation": {
            "stance": config.validation.stance,
            "prediction": (
                config.validation.prediction.model_dump()
                if config.validation.prediction is not None
                else None
            ),
            "holdout": None,
            "holdout_result": None,
        },
        "warnings": [],
    }
    write_manifest(run_dir, manifest)
    return manifest


def read_manifest(run_dir: Path) -> dict[str, Any]:
    data = yaml.safe_load((run_dir / MANIFEST_NAME).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"manifest at {run_dir} is not a mapping")
    return data


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    (run_dir / MANIFEST_NAME).write_text(yaml.safe_dump(manifest, sort_keys=False))


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def update_manifest(run_dir: Path, updates: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``updates`` into the manifest and write it back. Prior fields
    are preserved unless explicitly overwritten."""
    manifest = read_manifest(run_dir)
    _deep_update(manifest, updates)
    write_manifest(run_dir, manifest)
    return manifest


def record_tracking(run_dir: Path, wandb_run_id: str | None, wandb_url: str | None) -> None:
    """Fill in the W&B run id / url once the run is live (may be None in
    offline/disabled mode)."""
    update_manifest(
        run_dir,
        {"tracking": {"wandb_run_id": wandb_run_id, "wandb_url": wandb_url}},
    )


def finalize_manifest(
    run_dir: Path,
    status: str,
    summary: dict[str, Any] | None = None,
    final_checkpoint: str | None = None,
    error: str | None = None,
    warnings: list[str] | None = None,
) -> None:
    """Mark the run ``completed`` or ``failed`` and record final summary numbers.
    On failure, ``error`` (exception type + message) is stored for triage.
    ``warnings`` records any warnings captured during the run."""
    updates: dict[str, Any] = {"status": status, "completed_at": _utcnow()}
    if summary is not None:
        updates["summary"] = summary
    if final_checkpoint is not None:
        updates["artifacts"] = {"final_checkpoint": final_checkpoint}
    if error is not None:
        updates["error"] = error
    if warnings is not None:
        updates["warnings"] = warnings
    update_manifest(run_dir, updates)


def record_leakage(
    run_dir: Path,
    *,
    transpose_leak_fraction: float,
    commuting_probability: float,
    test_size: int,
    unleaked_test_size: int,
    unleaked_empty: bool,
    generalize_metric: str,
    note: str | None = None,
) -> None:
    """Record the transpose-leak covariates for the run's realised split under
    ``dataset.leakage``.

    ``transpose_leak_fraction`` is measured on the ACTUAL split (fraction of test
    pairs derivable via the commuting-transpose shortcut); ``commuting_probability``
    is the group-level ``Pr(G) = k(G)/|G|``. ``generalize_metric`` names which
    accuracy the generalisation/grok streak keys on -- normally
    ``unleaked_accuracy``, and ``raw_test_accuracy`` only in the degenerate
    empty-subset case (``unleaked_empty``), where ``note`` carries the loud
    explanation. Added via the existing deep-merge update path, so it extends the
    manifest without disturbing any other block."""
    update_manifest(
        run_dir,
        {
            "dataset": {
                "leakage": {
                    "transpose_leak_fraction": transpose_leak_fraction,
                    "commuting_probability": commuting_probability,
                    "test_size": test_size,
                    "unleaked_test_size": unleaked_test_size,
                    "unleaked_empty": unleaked_empty,
                    "generalize_metric": generalize_metric,
                    "note": note,
                }
            }
        },
    )


def record_holdout(
    run_dir: Path, fingerprint: str, size: int, spec: dict[str, Any] | None = None
) -> None:
    """Record the declared held-out set's fingerprint + size (provenance). The
    harness never sees the data -- only this hash the subclass computed."""
    update_manifest(
        run_dir,
        {"validation": {"holdout": {"fingerprint": fingerprint, "size": size, "spec": spec}}},
    )


def record_holdout_result(run_dir: Path, metrics: dict[str, float], outcome: str | None) -> None:
    """Record the single held-out evaluation's metrics + the prediction outcome
    (``predicted`` | ``refuted`` | ``inconclusive`` | ``None``), stamped with the time."""
    update_manifest(
        run_dir,
        {
            "validation": {
                "holdout_result": {
                    "metrics": metrics,
                    "evaluated_at": _utcnow(),
                    "outcome": outcome,
                }
            }
        },
    )
