"""Opt-in W&B publisher for occupancy records, so runs can be viewed and
compared in the W&B UI.

Strictly additive to the offline instrument: ``measure`` computes and writes
JSON with no network, and this module pushes those already-computed records to
W&B afterwards. Hard constraints, in order of importance:

* **Metrics, config, and summary on runs only -- never W&B artifacts.**
  Artifact storage is being abandoned (it was the failure point of the first
  campaign's uploads), so nothing here touches ``wandb.Artifact``. Logged
  ``wandb.Table``s are avoided for the same reason: current wandb SDKs back
  them with implicit artifacts. Per-block occupancy therefore goes out as
  plain metrics plotted against a block-index step metric
  (``occupancy/block``), which the UI renders as comparable per-block curves.
* **Publishing is a no-op without credentials.** The entry point requires
  ``WANDB_API_KEY`` and an unset (or ``online``) ``WANDB_MODE``; otherwise it
  prints why and does nothing, so the offline gate and every test stay
  network-free.
* **Published runs are identifiable.** Each measured record becomes -- or, via
  ``resume="allow"``, updates -- a W&B run whose id is the training run's own
  ``run_id`` and whose name is ``"<cell> (<order>,<index>) w<width>
  s<seed>"`` (zero-padded seed), with ``job_type="occupancy"``, grouped by
  the manifest's ``config_group_hash`` so every seed of one experiment lands
  in one W&B group -- the same naming and grouping scheme
  ``scripts/stream_runs.py`` uses for live runs, so the two line up. Pooled
  records use ``job_type="occupancy-pooled"`` with the id
  ``occupancy-pooled-<config_group_hash>``. Cell name, phase, and declared
  pair partner are looked up from ``configs/campaign/core.yaml`` by
  ``(order, index, width)`` and added to config and tags; a
  ``"campaign-v2"`` tag marks every run this publisher writes as restart
  data.

``wandb`` is imported lazily and only past the credential gate; the module
import itself never touches the network or requires the package.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .report import ARGUMENTS, FORMS

OCCUPANCY_JOB_TYPE = "occupancy"
POOLED_JOB_TYPE = "occupancy-pooled"

# The same pre-registered cell list `scripts/stream_runs.py` reads, used here
# only to enrich published runs with cosmetic-but-useful campaign metadata
# (phase, cell name, declared pair partner) -- never to change what gets
# measured or published. A missing or unreadable file degrades gracefully to
# metadata-free naming/tags, same as the live sidecar.
DEFAULT_CAMPAIGN_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "campaign" / "core.yaml"

_PAIR_PARTNER_RE = re.compile(r"with\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?")


def publish_skip_reason() -> str | None:
    """Why publishing would do nothing, or ``None`` when it may proceed.

    The gate is deliberately environmental, mirroring ``scripts/sync_runs.py``:
    no ``WANDB_API_KEY`` means no credentials to publish with, and a
    ``disabled``/``offline`` ``WANDB_MODE`` (the test suite forces
    ``disabled`` process-wide) means the operator asked for no syncing."""
    mode = os.environ.get("WANDB_MODE", "").strip().lower()
    if mode in ("disabled", "dryrun", "offline"):
        return f"WANDB_MODE={mode}; publishing is online-only"
    if not os.environ.get("WANDB_API_KEY"):
        return "WANDB_API_KEY is not set; the publisher does nothing without it"
    return None


def _import_wandb() -> Any:
    """The one seam that reaches the real SDK; tests monkeypatch this."""
    import wandb

    return wandb


def _sanitise_run_id(run_id: str) -> str:
    """W&B run ids allow word characters and dashes; the project's run ids
    already fit, but sanitise defensively so a publish never fails on an id."""
    return re.sub(r"[^A-Za-z0-9_-]", "-", run_id)[:120]


def _load_campaign_lookup(path: Path) -> dict[tuple[int, int, int], dict[str, Any]]:
    """``(order, index, width) -> cell``, mirroring ``scripts/stream_runs.py``'s
    ``load_campaign_lookup`` so both publishers enrich runs from the same
    source."""
    try:
        data = yaml.safe_load(path.read_text())
    except OSError:
        return {}
    lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
    for cell in (data or {}).get("cells", []):
        lookup.setdefault((cell["order"], cell["index"], cell["width"]), cell)
    return lookup


def _parse_pair_partner(note: str | None) -> tuple[int, int] | None:
    """Best-effort ``(order, index)`` of a cell's declared CT-pair partner,
    parsed from its ``note`` free text -- mirrors ``scripts/stream_runs.py``'s
    ``parse_pair_partner``; see its docstring for the caveats (documentation
    only, not authoritative, first-mentioned partner only in a >2-way note)."""
    if not note:
        return None
    match = _PAIR_PARTNER_RE.search(note)
    return None if match is None else (int(match.group(1)), int(match.group(2)))


def _format_seed(seed: Any) -> str:
    """Zero-padded seed so published names sort the same way the live
    sidecar's do (seeds run 0-49 across the campaign)."""
    try:
        return f"{int(seed):02d}"
    except (TypeError, ValueError):
        return str(seed)


def _campaign_context(
    order: Any,
    index: Any,
    width: Any,
    lookup: dict[tuple[int, int, int], dict[str, Any]],
) -> dict[str, Any]:
    """Cosmetic campaign metadata for one ``(order, index, width)``, or the
    ``"unknown"``-phase, partner-free degradation when there is no match."""
    cell = None
    if order is not None and index is not None and width is not None:
        cell = lookup.get((int(order), int(index), int(width)))
    phase = cell["phase"] if cell else "unknown"
    cell_name = cell["name"] if cell else None
    partner = _parse_pair_partner(cell.get("note")) if cell else None
    return {
        "phase": phase,
        "cell_name": cell_name,
        "pair_partner_order": partner[0] if partner else None,
        "pair_partner_index": partner[1] if partner else None,
    }


def _display_name(context: dict[str, Any], order: Any, index: Any, width: Any, tail: str) -> str:
    label = context["cell_name"] or f"({order},{index})"
    return f"{label} ({order},{index}) w{width} {tail}"


def _computed_tags(context: dict[str, Any], group_label: str | None) -> list[str]:
    tags = ["campaign-v2", context["phase"]]
    if context["phase"].startswith("bonus-"):
        tags.append("bonus")
    if group_label:
        tags.append(group_label)
    return tags


def _flat_summary(record: dict[str, Any]) -> dict[str, Any]:
    """The headline scalars for the W&B runs table: TVs with their floors and
    ratios for both arguments and both statistic forms, the trivial share, and
    the full checkpoint-selection outcome (substitutions are data)."""
    selection = record["checkpoint_selection"]
    summary: dict[str, Any] = {
        "status": record["status"],
        "n_units": record.get("n_units"),
        "checkpoint/name": selection.get("checkpoint"),
        "checkpoint/epoch": selection.get("epoch"),
        "checkpoint/rule": selection.get("rule"),
        "checkpoint/substitution": selection.get("substitution"),
        "checkpoint/metric_value": selection.get("metric_value"),
    }
    for argument in ARGUMENTS:
        block = record["occupancy"][argument]
        summary[f"occupancy/{argument}/trivial_share"] = block["trivial_block_share"]
        for form in FORMS:
            stats = block[form] if form == "full" else block["nontrivial"]
            summary[f"occupancy/{argument}/{form}/tv_to_null"] = stats["tv_to_null"]
            summary[f"occupancy/{argument}/{form}/noise_floor"] = stats["noise_floor"]
            summary[f"occupancy/{argument}/{form}/tv_over_floor"] = stats["tv_over_floor"]
    return summary


def _run_config(record: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    provenance = record.get("provenance", {})
    return {
        "group": record["group"],
        "model": record.get("model"),
        "seed": record.get("seed"),
        "config_hash": provenance.get("config_hash"),
        "config_group_hash": provenance.get("config_group_hash"),
        "dataset_spec_hash": provenance.get("dataset_spec_hash"),
        "campaign_id": provenance.get("campaign_id"),
        "checkpoint_sha256": provenance.get("checkpoint_sha256"),
        "selection_metric": record["checkpoint_selection"].get("metric"),
        "selection_threshold": record["checkpoint_selection"].get("threshold"),
        "trivial_block_index": record.get("trivial_block_index"),
        "block_ranks": record.get("block_ranks"),
        "phase": context["phase"],
        "cell_name": context["cell_name"],
        "pair_partner_order": context["pair_partner_order"],
        "pair_partner_index": context["pair_partner_index"],
    }


def _log_per_block_metrics(run: Any, record: dict[str, Any]) -> None:
    """Per-block occupancy as plain metrics against a block-index step metric
    -- the comparable-curves view, with no Table and no artifact behind it."""
    run.define_metric("occupancy/block")
    run.define_metric("occupancy/*", step_metric="occupancy/block")
    null = record["analytic_null"]
    ranks = record["block_ranks"]
    left = record["occupancy"]["left"]["occupancy"]
    right = record["occupancy"]["right"]["occupancy"]
    for j in range(len(null)):
        run.log(
            {
                "occupancy/block": j,
                "occupancy/block_rank": ranks[j],
                "occupancy/null": null[j],
                "occupancy/left": left[j],
                "occupancy/right": right[j],
            }
        )


def _publish_one(
    wandb: Any,
    *,
    run_id: str,
    name: str,
    job_type: str,
    group: str | None,
    project: str,
    entity: str | None,
    tags: list[str] | None,
    config: dict[str, Any],
    record: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    run = wandb.init(
        project=project,
        entity=entity,
        id=_sanitise_run_id(run_id),
        name=name,
        job_type=job_type,
        group=group,
        tags=tags or None,
        resume="allow",
        config=config,
    )
    try:
        _log_per_block_metrics(run, record)
        for key, value in summary.items():
            run.summary[key] = value
    finally:
        run.finish()


def publish_records(
    records: list[dict[str, Any]],
    pooled: list[dict[str, Any]] | None = None,
    *,
    project: str,
    entity: str | None = None,
    tags: list[str] | None = None,
    wandb_module: Any | None = None,
    campaign_config: Path | None = None,
) -> dict[str, int]:
    """Publish measured occupancy records (and optionally pooled records) as
    metrics-only W&B runs. Skipped records are counted, never published --
    they carry no numbers to compare. Returns
    ``{"published", "pooled_published", "skipped_records"}``.

    Every published run's name, tags, and config are enriched with campaign
    metadata (phase, cell name, declared pair partner) looked up from
    ``campaign_config`` (default: ``configs/campaign/core.yaml``) by
    ``(order, index, width)`` -- the same lookup and naming scheme
    ``scripts/stream_runs.py`` uses for live runs, plus a ``"campaign-v2"``
    tag marking every run this publisher writes as restart data (the 129
    quarantined v1 replays carry ``"quarantined-v1"``/``"campaign-v1"``
    instead, from a separate one-off tagging pass).

    ``wandb_module`` is an injectable seam for tests (mirroring the injectable
    boundaries of ``scripts/sync_runs.py``); production callers leave it
    ``None`` and get the real SDK via :func:`_import_wandb`."""
    wandb = wandb_module if wandb_module is not None else _import_wandb()
    lookup = _load_campaign_lookup(campaign_config or DEFAULT_CAMPAIGN_CONFIG)
    published = 0
    skipped = 0
    for record in records:
        if record.get("status") != "measured":
            skipped += 1
            continue
        group_info = record["group"]
        width = (record.get("model") or {}).get("d_model")
        context = _campaign_context(group_info.get("order"), group_info.get("index"), width, lookup)
        name = _display_name(
            context,
            group_info.get("order"),
            group_info.get("index"),
            width,
            f"s{_format_seed(record.get('seed'))}",
        )
        computed_tags = _computed_tags(context, context["cell_name"] or group_info.get("name"))
        _publish_one(
            wandb,
            run_id=record["run_id"],
            name=name,
            job_type=OCCUPANCY_JOB_TYPE,
            group=record.get("provenance", {}).get("config_group_hash"),
            project=project,
            entity=entity,
            tags=list(dict.fromkeys([*(tags or []), *computed_tags])),
            config=_run_config(record, context),
            record=record,
            summary=_flat_summary(record),
        )
        published += 1

    pooled_published = 0
    for entry in pooled or []:
        key = entry["config_group_hash"]
        group_info = entry["group"]
        width = (entry.get("model") or {}).get("d_model")
        context = _campaign_context(group_info.get("order"), group_info.get("index"), width, lookup)
        name = _display_name(
            context,
            group_info.get("order"),
            group_info.get("index"),
            width,
            f"pooled ({entry['n_runs']} seeds)",
        )
        computed_tags = _computed_tags(context, context["cell_name"] or group_info.get("name"))
        summary: dict[str, Any] = {
            "status": "pooled",
            "n_runs": entry["n_runs"],
            "n_units": entry["n_units"],
            "seeds": entry["seeds"],
        }
        for argument in ARGUMENTS:
            block = entry["occupancy"][argument]
            summary[f"occupancy/{argument}/trivial_share"] = block["trivial_block_share"]
            for form in FORMS:
                stats = block[form] if form == "full" else block["nontrivial"]
                summary[f"occupancy/{argument}/{form}/tv_to_null"] = stats["tv_to_null"]
                summary[f"occupancy/{argument}/{form}/noise_floor"] = stats["noise_floor"]
                summary[f"occupancy/{argument}/{form}/tv_over_floor"] = stats["tv_over_floor"]
        _publish_one(
            wandb,
            run_id=f"occupancy-pooled-{key}",
            name=name,
            job_type=POOLED_JOB_TYPE,
            group=key,
            project=project,
            entity=entity,
            tags=list(dict.fromkeys([*(tags or []), *computed_tags])),
            config={
                "group": entry["group"],
                "model": entry["model"],
                "config_group_hash": key,
                "run_ids": entry["run_ids"],
                "trivial_block_index": entry["trivial_block_index"],
                "phase": context["phase"],
                "cell_name": context["cell_name"],
                "pair_partner_order": context["pair_partner_order"],
                "pair_partner_index": context["pair_partner_index"],
            },
            record=entry,
            summary=summary,
        )
        pooled_published += 1

    return {
        "published": published,
        "pooled_published": pooled_published,
        "skipped_records": skipped,
    }


__all__ = [
    "OCCUPANCY_JOB_TYPE",
    "POOLED_JOB_TYPE",
    "publish_records",
    "publish_skip_reason",
]
