"""Occupancy instrument entry point (the I-08..I-14 spectral core + I-11 gate).

Measurement runs entirely offline from run-directory artifacts; no network, no
W&B, no Sage/GAP. Three subcommands::

    # Measure one or more runs and pool seeds within each experiment config.
    uv run python scripts/measure_occupancy.py measure runs/<id> [runs/<id> ...] \
        [--out results/occupancy.json] [--threshold 0.99] [--metric val/accuracy] \
        [--artifacts-dir data/group_artifacts]

    # I-11 null-calibration gate: the identical pipeline on random-init models
    # of two groups (typically a contrast pair's two members).
    uv run python scripts/measure_occupancy.py null-gate --group-a 8,3 --group-b 8,4 \
        --seeds 0:10 [--d-model 128] [--d-mlp 256] [--arch transformer] \
        [--activation relu] [--out results/null_gate.json]

    # Opt-in publisher: push already-measured analysis/occupancy.json records
    # to W&B for viewing/comparing in the UI. Metrics/summary on runs ONLY --
    # never W&B artifacts (artifact storage is abandoned). Requires
    # WANDB_API_KEY and an online WANDB_MODE; without them it prints why and
    # does nothing, so the core instrument and the gate stay network-free.
    WANDB_API_KEY=... uv run python scripts/measure_occupancy.py publish \
        runs/<id> [runs/<id> ...] [--project group-algorithm-interp] \
        [--entity NAME] [--tags tag1,tag2] [--no-pooled]

``publish`` attaches each record to an identifiable W&B run: the run id and
name are the training run's own ``run_id`` (``resume="allow"``, so a W&B run
already existing under that id -- a replayed training run -- is updated in
place, and a metrics-only run with ``job_type="occupancy"`` is created
otherwise), grouped by the manifest's ``config_group_hash``. Pooled records
publish as ``job_type="occupancy-pooled"`` unless ``--no-pooled``.

``measure`` applies the dip-aware checkpoint rule per run: runs whose resolved
config carries ``snapshot.final_window_epochs`` select the last final-window
epoch with ``val/accuracy >= threshold`` per ``run.log`` (falling back to the
nearest stable trajectory snapshot when the whole window is dipped); older runs
gate ``final.pt`` on its own final ``run.log`` row and substitute the nearest
stable trajectory snapshot when dipped. Every substitution -- and every run
skipped because nothing stable exists -- is recorded in the output as data.

Each measured run's record is written to ``<run_dir>/analysis/occupancy.json``;
``--out`` additionally writes one file holding every per-run record plus the
seed-pooled records (pooling key: the manifest's ``config_group_hash``, so
seeds never pool across groups or widths).

Exit code: ``0`` iff every requested run was measured (``measure``), the gate
record was written (``null-gate``), or publishing completed or was cleanly
skipped for lack of credentials (``publish``); ``1`` when a run was skipped by
``measure`` or ``publish`` finds a run not yet measured, so a campaign wrapper
notices missing measurements loudly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from group_algorithm_interp.instruments.device import resolve_device  # noqa: E402
from group_algorithm_interp.instruments.publish import (  # noqa: E402
    publish_records,
    publish_skip_reason,
)
from group_algorithm_interp.instruments.report import (  # noqa: E402
    measure_run,
    null_gate,
    pool_records,
    sanitise_nonfinite,
)


def parse_seeds(spec: str) -> list[int]:
    """Same grammar as ``scripts/run_batch.py``: a comma-separated mix of
    half-open ranges ``a:b`` and single integers, order-preserving,
    de-duplicated."""
    seeds: list[int] = []
    seen: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            start_text, end_text = token.split(":", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"empty/reversed seed range {token!r}: need a:b with b >= a")
            values = list(range(start, end))
        else:
            values = [int(token)]
        for value in values:
            if value not in seen:
                seen.add(value)
                seeds.append(value)
    if not seeds:
        raise ValueError(f"--seeds {spec!r} selected no seeds")
    return seeds


def _parse_group(spec: str) -> tuple[int, int]:
    order_text, index_text = spec.split(",", 1)
    return int(order_text.strip()), int(index_text.strip())


def _write_json(path: Path, payload: object) -> None:
    """Write ``payload`` as strict RFC-8259 JSON. Non-finite floats (NaN, +/-Inf
    -- reachable via a rejected checkpoint candidate's metric value) are
    sanitised to ``null`` first, and ``allow_nan=False`` makes any missed case a
    hard error rather than a bare ``NaN`` token that ``jq`` and strict parsers
    reject. ``null`` therefore reads as "non-finite" in the written record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(sanitise_nonfinite(payload), indent=2, sort_keys=False, allow_nan=False)
    path.write_text(text + "\n")


def _cmd_measure(args: argparse.Namespace) -> int:
    if args.artifacts_dir is not None:
        os.environ["GROUP_ARTIFACTS_DIR"] = str(args.artifacts_dir)
    device = resolve_device(args.device)
    records = []
    n_skipped = 0
    for run_dir in args.run_dirs:
        record = measure_run(
            Path(run_dir), metric=args.metric, threshold=args.threshold, device=device
        )
        records.append(record)
        _write_json(Path(run_dir) / "analysis" / "occupancy.json", record)
        if record["status"] == "measured":
            left = record["occupancy"]["left"]["nontrivial"]
            print(
                f"[occupancy] {record['run_id']}: {record['group']['name']} "
                f"checkpoint={record['checkpoint_selection']['checkpoint']} "
                f"(substitution={record['checkpoint_selection']['substitution']}) "
                f"TV(left, nontrivial)={left['tv_to_null']:.4f} "
                f"floor={left['noise_floor']:.4f}"
            )
        else:
            n_skipped += 1
            print(
                f"[occupancy] {record['run_id']}: SKIPPED -- "
                f"{record['checkpoint_selection']['reason']}"
            )
    pooled = pool_records(records)
    if args.out is not None:
        _write_json(Path(args.out), {"runs": records, "pooled": pooled})
        print(
            f"[occupancy] wrote {len(records)} run record(s), {len(pooled)} pooled, to {args.out}"
        )
    return 0 if n_skipped == 0 else 1


def _cmd_null_gate(args: argparse.Namespace) -> int:
    if args.artifacts_dir is not None:
        os.environ["GROUP_ARTIFACTS_DIR"] = str(args.artifacts_dir)
    model_config: dict[str, object] = {
        "arch": args.arch,
        "d_model": args.d_model,
        "activation": args.activation,
        "n_heads": args.n_heads,
    }
    if args.d_mlp is not None:
        model_config["d_mlp"] = args.d_mlp
    record = null_gate(
        _parse_group(args.group_a),
        _parse_group(args.group_b),
        parse_seeds(args.seeds),
        model_config=model_config,
    )
    if args.out is not None:
        _write_json(Path(args.out), record)
        print(f"[null-gate] wrote {args.out}")
    else:
        print(json.dumps(record, indent=2))
    return 0


def _cmd_publish(args: argparse.Namespace) -> int:
    skip = publish_skip_reason()
    if skip is not None:
        print(f"[publish] not publishing: {skip}")
        return 0
    records = []
    for run_dir in args.run_dirs:
        record_path = Path(run_dir) / "analysis" / "occupancy.json"
        if not record_path.is_file():
            print(f"[publish] {run_dir}: no analysis/occupancy.json -- run `measure` first")
            return 1
        records.append(json.loads(record_path.read_text()))
    pooled = None if args.no_pooled else pool_records(records)
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()] or None
    counts = publish_records(records, pooled, project=args.project, entity=args.entity, tags=tags)
    print(
        f"[publish] published {counts['published']} run(s), "
        f"{counts['pooled_published']} pooled, "
        f"{counts['skipped_records']} skipped record(s) not published"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser("measure", help="measure occupancy for one or more runs")
    measure.add_argument("run_dirs", nargs="+", help="run directories to measure")
    measure.add_argument("--out", default=None, help="write all records + pooled records here")
    measure.add_argument("--metric", default="val/accuracy", help="stability metric in run.log")
    measure.add_argument(
        "--threshold", type=float, default=0.99, help="stability bar on --metric (default 0.99)"
    )
    measure.add_argument(
        "--artifacts-dir", default=None, help="override the group-artifact directory"
    )
    measure.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cpu"],
        help="device for the I-08 neuron-activation forward pass; 'auto' is MPS when "
        "available, else CPU (default: auto). The occupancy analysis always stays "
        "CPU float64.",
    )
    measure.set_defaults(func=_cmd_measure)

    gate = sub.add_parser("null-gate", help="I-11 null-calibration gate on random-init models")
    gate.add_argument("--group-a", required=True, help="first group as 'order,index'")
    gate.add_argument("--group-b", required=True, help="second group as 'order,index'")
    gate.add_argument("--seeds", required=True, help="seed spec, e.g. 0:10 or 0,1,2")
    gate.add_argument("--arch", default="transformer", choices=["transformer", "fc"])
    gate.add_argument("--d-model", type=int, default=128)
    gate.add_argument("--d-mlp", type=int, default=None, help="default: 2 * d_model")
    gate.add_argument("--n-heads", type=int, default=4)
    gate.add_argument("--activation", default="relu", choices=["relu", "gelu", "silu"])
    gate.add_argument("--out", default=None, help="write the gate record here (default: stdout)")
    gate.add_argument("--artifacts-dir", default=None, help="override the group-artifact directory")
    gate.set_defaults(func=_cmd_null_gate)

    publish = sub.add_parser(
        "publish",
        help="opt-in: push measured records to W&B as metrics-only runs (no artifacts)",
    )
    publish.add_argument("run_dirs", nargs="+", help="run directories already measured")
    publish.add_argument("--project", default="group-algorithm-interp")
    publish.add_argument("--entity", default=None)
    publish.add_argument("--tags", default=None, help="comma-separated W&B tags")
    publish.add_argument(
        "--no-pooled", action="store_true", help="do not publish seed-pooled records"
    )
    publish.set_defaults(func=_cmd_publish)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
