"""I-34 circuit-audit + I-35 direct-logit-attribution entry point.

Runs the rung-5 circuit audit and the direct logit attribution on one or more
trained runs, entirely offline: no network, no W&B, no Sage/GAP. One
subcommand::

    uv run python scripts/measure_audit.py measure runs/<id> [runs/<id> ...] \
        [--out results/audit.json] [--threshold 0.99] [--metric val/accuracy] \
        [--ablation-mode zero] [--seed 0] [--min-top-share 0.5] \
        [--artifacts-dir data/group_artifacts]

Each run is loaded through the same dip-aware checkpoint rule the other
instruments use (every substitution and every run skipped for want of a
stable checkpoint is recorded as data). For the selected model:

* **I-35** ``direct_logit_attribution`` always runs -- every architecture this
  suite supports decomposes into components, so it needs no circuit
  definition.
* **I-34** ``circuit_audit`` runs only where this module can derive a neuron
  circuit for the group. The only bridge implemented is
  ``audit.coset_circuit_neurons``: the neurons whose activation concentrates
  on the coset arm's occupied isotypic blocks (``Ind_H^G 1``'s support),
  scoped to groups with a coset target (the D32/QD32-style case study). Groups
  with no coset target -- every abelian group (so every C3 carry-structure
  member) and the Q32 family -- get a ``circuit_audit.status == "skipped"``
  sub-record with the reason: there is no carry-digit-to-neuron (or other
  non-coset) circuit derivation implemented anywhere in this codebase, and
  this driver does not invent one silently. This sub-skip is expected and
  distinct from a checkpoint-selection skip; the top-level ``status`` still
  reads ``"measured"`` whenever a stable checkpoint was found, since I-35 ran
  regardless.

Each measured run's record is written to ``<run_dir>/analysis/audit.json``;
``--out`` additionally writes one file holding every per-run record, NaN-safe
(non-finite floats serialise as ``null``, matching ``measure_cocycle.py``).

Exit code is ``0`` iff every requested run had a stable checkpoint (I-35 ran);
``1`` when any run was skipped for want of one, so a campaign wrapper notices.
A group with no coset target does NOT count as skipped for this purpose --
that is an expected, documented circuit-audit sub-skip, not a checkpoint
failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from group_algorithm_interp.instruments.audit import measure_audit_run  # noqa: E402
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
        record = measure_audit_run(
            Path(run_dir),
            metric=args.metric,
            threshold=args.threshold,
            ablation_mode=args.ablation_mode,
            seed=args.seed,
            min_top_share=args.min_top_share,
        )
        records.append(record)
        _write_json(Path(run_dir) / "analysis" / "audit.json", record)
        if record["status"] == "measured":
            ca = record["circuit_audit"]
            if ca["status"] == "measured":
                kl = ca["faithfulness"]["kl_model_to_circuit"]["mean"]
                comp = ca["completeness"]["accuracy_drop"]["mean"]
                mini = ca["minimality"]["accuracy_drop"]["mean"]
                circuit_note = (
                    f"circuit_audit=measured n_inside={ca['n_inside']}/{ca['n_inside'] + ca['n_outside']} "
                    f"kl={kl:.4g} completeness_drop={comp:+.4g} minimality_drop={mini:+.4g}"
                )
            else:
                circuit_note = f"circuit_audit=skipped ({ca['reason'][:60]}...)"
            print(
                f"[audit] {record['run_id']}: {record['group']['name']} "
                f"checkpoint={record['checkpoint_selection']['checkpoint']} {circuit_note}"
            )
        else:
            n_skipped += 1
            print(
                f"[audit] {record['run_id']}: SKIPPED -- {record['checkpoint_selection']['reason']}"
            )
    if args.out is not None:
        _write_json(Path(args.out), {"runs": records})
        print(f"[audit] wrote {len(records)} run record(s) to {args.out}")
    return 0 if n_skipped == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser(
        "measure", help="run the I-34 circuit audit + I-35 DLA for one or more runs"
    )
    measure.add_argument("run_dirs", nargs="+", help="run directories to measure")
    measure.add_argument("--out", default=None, help="write all records here")
    measure.add_argument("--metric", default="val/accuracy", help="stability metric in run.log")
    measure.add_argument(
        "--threshold", type=float, default=0.99, help="stability bar on --metric (default 0.99)"
    )
    measure.add_argument(
        "--ablation-mode",
        default="zero",
        choices=["zero", "mean", "resample"],
        help="I-34 ablation mode for the completeness/minimality draws (default zero)",
    )
    measure.add_argument(
        "--seed",
        type=int,
        default=0,
        help="I-34 base seed driving any resample-mode ablation (default 0)",
    )
    measure.add_argument(
        "--min-top-share",
        type=float,
        default=0.5,
        help=(
            "the coset-block circuit bridge's neuron-selection threshold: a "
            "neuron is 'inside' the circuit when at least this share of its "
            "own isotypic energy sits in an occupied coset block (default 0.5)"
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
