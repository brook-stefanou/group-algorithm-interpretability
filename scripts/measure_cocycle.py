"""C5 cocycle-arm entry point (the I-21/I-22/I-22b/I-22c/I-22d instrument).

The mechanism measurement for C5 -- ``(48,29)`` GL(2,3) [split] vs ``(48,28)``
SL(2,3).C2 [non-split] -- on one or more trained runs, entirely offline: no
network, no W&B, no Sage/GAP. One subcommand::

    uv run python scripts/measure_cocycle.py measure runs/<id> [runs/<id> ...] \
        [--out results/cocycle.json] [--threshold 0.99] [--metric val/accuracy] \
        [--normal-order 24] [--n-shuffle 500] [--artifacts-dir data/group_artifacts]

Each run is loaded through the same dip-aware checkpoint rule the other
instruments use (every substitution and every run skipped for want of a stable
checkpoint is recorded as data). For the selected model the full C5 arm runs:

* **I-21** the extension precompute (transversal / action / cocycle; identities
  asserted, ``f == 1`` achievable iff split) and the split-vs-non-split fit;
* **I-22 / I-22b / I-22c** the cocycle-value probe, the ``f``-direction ablation
  and the twisted-rule FVE fit -- each carrying its ``|Q| = 2`` structural-
  degeneracy guard, so on the registered ``Q = C2`` contrast they self-report
  ``STRUCTURALLY_DEGENERATE`` / confound-flagged rather than shipping a confounded
  number;
* **I-22d** the intervention-based predicted-error test -- the ``|Q| = 2``
  replacement: knock out the model's winning prediction on twist-active rows and
  score whether the induced errors land on the row-specific untwisted product,
  against a shuffled-correspondence null and (where ``|N| >= 3``) counterfactual
  cocycle-value targets. The split member coordinatises with its trivialising
  transversal, so I-22/I-22b/I-22c report their split controls and I-22d reports
  ``twist_inactive`` -- the built-in negative-control arm.

``--normal-order`` is the order of the normal subgroup ``N`` to coordinatise over
(default ``order // 2``, the index-2 ``N`` of the C5 pair); the chosen subgroup's
membership is pinned in the record. Each measured run's record is written to
``<run_dir>/analysis/cocycle.json``; ``--out`` additionally writes one file
holding every per-run record, NaN-safe (non-finite floats serialise as ``null``).

Exit code is ``0`` iff every requested run was measured, ``1`` when any run was
skipped (no stable checkpoint, or no normal subgroup of the requested order), so
a campaign wrapper notices loudly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from group_algorithm_interp.instruments.cocycle import measure_cocycle_run  # noqa: E402
from group_algorithm_interp.instruments.report import sanitise_nonfinite  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitise_nonfinite(payload), indent=2, allow_nan=False) + "\n")


def _cmd_measure(args: argparse.Namespace) -> int:
    if args.artifacts_dir is not None:
        os.environ["GROUP_ARTIFACTS_DIR"] = str(args.artifacts_dir)
    records = []
    n_skipped = 0
    for run_dir in args.run_dirs:
        record = measure_cocycle_run(
            Path(run_dir),
            metric=args.metric,
            threshold=args.threshold,
            normal_order=args.normal_order,
            n_shuffle=args.n_shuffle,
        )
        records.append(record)
        _write_json(Path(run_dir) / "analysis" / "cocycle.json", record)
        if record["status"] == "measured":
            pe = record["i22d_predicted_error"]
            print(
                f"[cocycle] {record['run_id']}: {record['group']['name']} "
                f"checkpoint={record['checkpoint_selection']['checkpoint']} "
                f"splits={record['f_equiv_one_fit']['splits']} "
                f"i22d={pe['status']}"
                + (
                    f" effect_vs_shuffle={pe['effect_vs_shuffle']:+.3f}"
                    if pe["status"] == "measured"
                    else ""
                )
            )
        else:
            n_skipped += 1
            reason = record.get("skip_reason") or record["checkpoint_selection"]["reason"]
            print(f"[cocycle] {record['run_id']}: SKIPPED -- {reason}")
    if args.out is not None:
        _write_json(Path(args.out), {"runs": records})
        print(f"[cocycle] wrote {len(records)} run record(s) to {args.out}")
    return 0 if n_skipped == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser("measure", help="measure the C5 cocycle arm for one or more runs")
    measure.add_argument("run_dirs", nargs="+", help="run directories to measure")
    measure.add_argument("--out", default=None, help="write all records here")
    measure.add_argument("--metric", default="val/accuracy", help="stability metric in run.log")
    measure.add_argument(
        "--threshold", type=float, default=0.99, help="stability bar on --metric (default 0.99)"
    )
    measure.add_argument(
        "--normal-order",
        type=int,
        default=None,
        help="order of the normal subgroup N to coordinatise over (default: order // 2)",
    )
    measure.add_argument(
        "--n-shuffle",
        type=int,
        default=500,
        help="shuffled-correspondence null permutations for the I-22d effect (default 500)",
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
