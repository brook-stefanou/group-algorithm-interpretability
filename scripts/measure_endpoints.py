"""Tier-0 endpoint instrument entry point (I-01/I-02/I-04/I-06).

Measurement runs entirely offline from run-directory artefacts -- ``run.log``,
``manifest.yaml``, ``resolved_config.yaml`` -- with no network, no W&B, and no
re-training. Two subcommands::

    # Per-run measurement vector (I-06): the epochs-to-grok endpoint (censored
    # or not), the transpose-unleaked and raw accuracy at the dip-aware
    # checkpoint and the final epoch, the realised-leak covariate (I-02), and
    # the chance-accuracy anchor. Each run's record is written to
    # <run_dir>/analysis/endpoints.json; --out additionally writes one file with
    # every per-run record.
    uv run python scripts/measure_endpoints.py measure runs/<id> [runs/<id> ...] \
        [--out results/endpoints.json] [--threshold 0.99] [--sustain 5]

    # Pair-level estimates (I-04): the paired within-pair epochs-to-grok
    # difference (censored) and the paired final-unleaked-accuracy difference,
    # with a bootstrap CI and descriptive supplements. Members are two sets of
    # run dirs; seeds are matched across members, unmatched seeds dropped.
    uv run python scripts/measure_endpoints.py pair \
        --member-a runs/a0 runs/a1 --member-b runs/b0 runs/b1 \
        [--out results/pair.json]

Exit code: 0 iff every requested run was measured (``measure``) or the pair
record was written (``pair``); 1 when a run was skipped by ``measure`` (no
``run.log`` rows), so a campaign wrapper notices missing measurements loudly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from group_algorithm_interp.instruments.endpoints import (  # noqa: E402
    measurement_vector,
    within_pair_endpoints,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _cmd_measure(args: argparse.Namespace) -> int:
    records = []
    n_skipped = 0
    for run_dir in args.run_dirs:
        record = measurement_vector(Path(run_dir), threshold=args.threshold, sustain=args.sustain)
        records.append(record)
        _write_json(Path(run_dir) / "analysis" / "endpoints.json", record)
        if record["status"] == "measured":
            grok = record["epochs_to_grok"]
            grok_str = "censored" if grok["censored"] else f"epoch {grok['epoch']}"
            final = record["accuracy"]["at_final_epoch"]["unleaked"]
            print(
                f"[endpoints] {record['run_id']}: {record['group']['name']} "
                f"grok={grok_str} final_unleaked={final} "
                f"leak={record['leak_covariate']['transpose_leak_fraction']}"
            )
        else:
            n_skipped += 1
            print(f"[endpoints] {record['run_id']}: SKIPPED -- {record.get('reason')}")
    if args.out is not None:
        _write_json(Path(args.out), {"runs": records})
        print(f"[endpoints] wrote {len(records)} run record(s) to {args.out}")
    return 0 if n_skipped == 0 else 1


def _cmd_pair(args: argparse.Namespace) -> int:
    records_a = [
        measurement_vector(Path(d), threshold=args.threshold, sustain=args.sustain)
        for d in args.member_a
    ]
    records_b = [
        measurement_vector(Path(d), threshold=args.threshold, sustain=args.sustain)
        for d in args.member_b
    ]
    record = within_pair_endpoints(records_a, records_b)
    if args.out is not None:
        _write_json(Path(args.out), record)
        print(f"[endpoints] wrote pair record to {args.out}")
    else:
        print(json.dumps(record, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser("measure", help="per-run measurement vector (I-06)")
    measure.add_argument("run_dirs", nargs="+", help="run directories to measure")
    measure.add_argument("--out", default=None, help="write all records here")
    measure.add_argument(
        "--threshold", type=float, default=0.99, help="grok accuracy bar (default 0.99)"
    )
    measure.add_argument(
        "--sustain", type=int, default=5, help="consecutive epochs above the bar (default 5)"
    )
    measure.set_defaults(func=_cmd_measure)

    pair = sub.add_parser("pair", help="pair-level paired endpoint estimates (I-04)")
    pair.add_argument("--member-a", nargs="+", required=True, help="member A run dirs")
    pair.add_argument("--member-b", nargs="+", required=True, help="member B run dirs")
    pair.add_argument("--out", default=None, help="write the pair record here (default: stdout)")
    pair.add_argument("--threshold", type=float, default=0.99, help="grok accuracy bar")
    pair.add_argument("--sustain", type=int, default=5, help="consecutive epochs above the bar")
    pair.set_defaults(func=_cmd_pair)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
