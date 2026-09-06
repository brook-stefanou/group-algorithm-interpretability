"""Mechanism-probe entry point (the I-20/I-26/I-27/I-28 mechanism arm).

Runs the mechanism probes on one or more trained runs, entirely offline -- no
network, no W&B, no Sage/GAP. One subcommand::

    uv run python scripts/measure_probes.py measure runs/<id> [runs/<id> ...] \
        [--out results/probes.json] [--threshold 0.99] [--metric val/accuracy] \
        [--source embed] [--artifacts-dir data/group_artifacts]

Each run is loaded through the same dip-aware checkpoint rule the occupancy
instrument uses (runs whose resolved config carries ``snapshot.final_window_epochs``
pick the last stable final-window epoch, older runs gate ``final.pt``; every
substitution and every run skipped for want of a stable checkpoint is recorded as
data). For the selected model the applicable probes are run:

* **I-28 / I-28b** power-map probe (element order, involution-ness, square/cube
  map -- chance-corrected) and the involution-direction ablation;
* **I-27** the signed-cyclic probe and twisted-rule fit -- ``UNDEFINED`` on the
  quaternionic (non-split) member, by construction;
* **I-26** the polycyclic digit probe, direction count and carry structure --
  ``UNDEFINED`` where no mixed-radix coordinate system exists.

``--source`` picks the feature space (``embed``, ``left`` or ``right``) for the
classification/fit probes above. The involution-direction ablation always
ablates in ``embed`` space regardless of ``--source``: zero-ablation removes a
direction from ``W_E`` (``d_model`` space), so a ``d_mlp``-space direction
(``left``/``right``) has no valid ablation there.

Each structural probe is compared against a random-init model of the same shape
(seeded exactly as training would seed it), so the untrained null the rule-1
regression requires travels in the record. Every run's record is written to
``<run_dir>/analysis/probes.json``; ``--out`` additionally collects them into one
file. Exit code is ``0`` iff every requested run was measured, ``1`` when a run
was skipped for want of a stable checkpoint (so a campaign wrapper notices).
Provenance carries the checkpoint's sha256, the analysed group artifact's
repo-relative path and sha256, and every instrument module's sha256.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402

from group_algorithm_interp.config import validate_config  # noqa: E402
from group_algorithm_interp.groups.catalog import resolve_group  # noqa: E402
from group_algorithm_interp.groups.data import artifact_path  # noqa: E402
from group_algorithm_interp.instruments.checkpoints import select_checkpoint  # noqa: E402
from group_algorithm_interp.instruments.probes import (  # noqa: E402
    carry_digit_instrument,
    involution_direction_ablation,
    power_map_probe,
    signed_cyclic_instrument,
)
from group_algorithm_interp.instruments.report import (  # noqa: E402
    file_sha256,
    instrument_code_hashes,
)
from group_algorithm_interp.manifest import get_git_commit, read_manifest  # noqa: E402
from group_algorithm_interp.seed import set_seed  # noqa: E402
from group_algorithm_interp.training.trainer import build_model  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative_to_repo(path: Path) -> str:
    """``path`` relative to the repo root when possible, else its absolute
    form (e.g. an ``--artifacts-dir`` override or a test's tmp directory
    outside the repo) -- so the provenance block is always a valid path,
    never an assumption about where artifacts live."""
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def probe_run(
    run_dir: Path,
    *,
    metric: str = "val/accuracy",
    threshold: float = 0.99,
    source: str = "embed",
) -> dict[str, Any]:
    """One run's mechanism-probe record: dip-aware checkpoint selection (recorded,
    substitutions included), then the applicable I-26/I-27/I-28 probes against a
    random-init null of the same shape. A run with no stable checkpoint comes
    back ``status: "skipped"`` with the full selection record -- reported as
    data, never silently dropped."""
    manifest = read_manifest(run_dir)
    config = validate_config(yaml.safe_load((run_dir / "resolved_config.yaml").read_text()))
    selection = select_checkpoint(run_dir, metric=metric, threshold=threshold)
    record: dict[str, Any] = {
        "instrument": "mechanism-probes",
        "run_id": manifest.get("run_id", run_dir.name),
        "seed": config.seed,
        "group": {
            "order": config.data.group.order,
            "index": config.data.group.index,
            "name": config.data.group.canonical_name,
        },
        "model": {
            "arch": config.model.arch,
            "d_model": config.model.d_model,
            "d_mlp": config.model.d_mlp,
            "activation": config.model.activation,
        },
        "source": source,
        "checkpoint_selection": selection.to_record(),
        "provenance": {
            "git_commit": manifest.get("provenance", {}).get("git_commit"),
            "config_hash": manifest.get("provenance", {}).get("config_hash"),
            "config_group_hash": manifest.get("provenance", {}).get("config_group_hash"),
            "campaign_id": manifest.get("provenance", {}).get("campaign_id"),
            "dataset_spec_hash": manifest.get("dataset", {}).get("spec_hash"),
            "analysis_git_commit": get_git_commit(),
            "analysed_at": _utcnow(),
            "instrument_code_sha256": instrument_code_hashes(),
        },
    }
    if selection.path is None:
        record["status"] = "skipped"
        return record

    group = resolve_group(config.data.group)
    checkpoint = torch.load(selection.path, map_location="cpu", weights_only=False)
    model = build_model(config, group)
    model.load_state_dict(checkpoint["model_state_dict"])

    # The random-init null: the model training would have started from (same
    # seeding as the training paths), for the rule-1 regression.
    set_seed(config.seed, deterministic=False)
    null_model = build_model(config, group)

    group_artifact = artifact_path(config.data.group.order, config.data.group.index)
    record["status"] = "measured"
    record["provenance"]["checkpoint_sha256"] = file_sha256(selection.path)
    record["provenance"]["group_artifact_path"] = _relative_to_repo(group_artifact)
    record["provenance"]["group_artifact_sha256"] = file_sha256(group_artifact)
    record["probes"] = {
        "power_map": power_map_probe(model, group, source=source, seed=config.seed),
        # The involution-direction ablation always ablates out of W_E (d_model
        # space, interventions.ablate_direction's only supported space) --
        # unlike the other probes here, it does not follow the CLI's general
        # --source choice, which can also name a d_mlp feature space
        # (neuron_features) that has no valid ablation analogue. See
        # probes.involution_direction_ablation's docstring.
        "involution_ablation": involution_direction_ablation(
            model, group, source="embed", seed=config.seed
        ),
        "signed_cyclic": signed_cyclic_instrument(
            model, group, source=source, seed=config.seed, null_model=null_model
        ),
        "carry_digit": carry_digit_instrument(
            model, group, source=source, seed=config.seed, null_model=null_model
        ),
    }
    return record


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _cmd_measure(args: argparse.Namespace) -> int:
    if args.artifacts_dir is not None:
        os.environ["GROUP_ARTIFACTS_DIR"] = str(args.artifacts_dir)
    records = []
    n_skipped = 0
    for run_dir in args.run_dirs:
        record = probe_run(
            Path(run_dir), metric=args.metric, threshold=args.threshold, source=args.source
        )
        records.append(record)
        _write_json(Path(run_dir) / "analysis" / "probes.json", record)
        if record["status"] == "measured":
            sc = record["probes"]["signed_cyclic"]["status"]
            cd = record["probes"]["carry_digit"]["status"]
            print(
                f"[probes] {record['run_id']}: {record['group']['name']} "
                f"checkpoint={record['checkpoint_selection']['checkpoint']} "
                f"signed_cyclic={sc} carry_digit={cd}"
            )
        else:
            n_skipped += 1
            print(
                f"[probes] {record['run_id']}: SKIPPED -- "
                f"{record['checkpoint_selection']['reason']}"
            )
    if args.out is not None:
        _write_json(Path(args.out), {"runs": records})
        print(f"[probes] wrote {len(records)} run record(s) to {args.out}")
    return 0 if n_skipped == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser("measure", help="run the mechanism probes on one or more runs")
    measure.add_argument("run_dirs", nargs="+", help="run directories to measure")
    measure.add_argument("--out", default=None, help="write all records here")
    measure.add_argument("--metric", default="val/accuracy", help="stability metric in run.log")
    measure.add_argument(
        "--threshold", type=float, default=0.99, help="stability bar on --metric (default 0.99)"
    )
    measure.add_argument(
        "--source",
        default="embed",
        choices=["embed", "left", "right"],
        help=(
            "per-element feature source for the classification/fit probes "
            "(power_map, signed_cyclic, carry_digit; default: embed). The "
            "involution-direction ablation always uses 'embed' regardless of "
            "this flag -- ablation is only defined in W_E's d_model space."
        ),
    )
    measure.add_argument(
        "--artifacts-dir", default=None, help="override the group-artifact directory"
    )
    measure.set_defaults(func=_cmd_measure)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
