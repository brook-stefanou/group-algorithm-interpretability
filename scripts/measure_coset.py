"""Coset-arm + isotypic-block-usage entry point for the C2 case study.

The mechanism half of the D32/QD32/Q32 case study (plan §2, §5): the I-17/I-18/
I-19 coset arm and the I-15 isotypic-block usage check, all scored on the
transpose-unleaked held-out set behind the dip-aware checkpoint rule. Runs
entirely offline from run-directory artifacts -- no network, no W&B, no
Sage/GAP -- mirroring ``scripts/measure_occupancy.py``.

    uv run python scripts/measure_coset.py measure runs/<id> [runs/<id> ...] \
        [--out results/coset.json] [--threshold 0.99] [--metric val/accuracy] \
        [--n-random 16] [--n-null 500] [--artifacts-dir data/group_artifacts]

The coset arm is ``UNDEFINED`` by theorem on any group with no nontrivial
core-free subgroup (Q32 and its family): the record says so explicitly rather
than reporting a number -- that undefinedness is the C2 result, not a gap. Each
measured run's record is written to ``<run_dir>/analysis/coset.json``; ``--out``
additionally writes one file holding every per-run record.

Exit code: ``0`` iff every requested run was measured; ``1`` when any run was
skipped for want of a stable checkpoint, so a campaign wrapper notices loudly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from group_algorithm_interp.instruments.coset import measure_coset_run  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _cmd_measure(args: argparse.Namespace) -> int:
    if args.artifacts_dir is not None:
        os.environ["GROUP_ARTIFACTS_DIR"] = str(args.artifacts_dir)
    records = []
    n_skipped = 0
    for run_dir in args.run_dirs:
        record = measure_coset_run(
            Path(run_dir),
            metric=args.metric,
            threshold=args.threshold,
            n_random=args.n_random,
            n_null=args.n_null,
        )
        records.append(record)
        _write_json(Path(run_dir) / "analysis" / "coset.json", record)
        if record["status"] == "measured":
            arm = record["coset_arm"]
            defined = arm["coset_defined"]
            print(
                f"[coset] {record['run_id']}: {record['group']['name']} "
                f"checkpoint={record['checkpoint_selection']['checkpoint']} "
                f"coset_defined={defined}"
            )
        else:
            n_skipped += 1
            print(
                f"[coset] {record['run_id']}: SKIPPED -- {record['checkpoint_selection']['reason']}"
            )
    if args.out is not None:
        _write_json(Path(args.out), {"runs": records})
        print(f"[coset] wrote {len(records)} run record(s) to {args.out}")
    return 0 if n_skipped == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser("measure", help="measure the coset arm + I-15 for one or more runs")
    measure.add_argument("run_dirs", nargs="+", help="run directories to measure")
    measure.add_argument("--out", default=None, help="write all records here")
    measure.add_argument("--metric", default="val/accuracy", help="stability metric in run.log")
    measure.add_argument(
        "--threshold", type=float, default=0.99, help="stability bar on --metric (default 0.99)"
    )
    measure.add_argument(
        "--n-random", type=int, default=16, help="matched random subspaces per ablation"
    )
    measure.add_argument(
        "--n-null", type=int, default=500, help="random partitions for the coset-collapse null"
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
