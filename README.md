# Group Algorithm Interp

Two non-isomorphic finite groups can share a character table.  Every account
that reads a network's learned circuit through representation theory (the
Fourier account, the tensor-rank account, the coset account) predicts
identical behaviour on both members of such a pair, so a reproducible
within-pair difference falsifies that account.

This repository is a pre-registered, reproducible study of what algorithm a
small neural network learns for finite-group multiplication, built on that
instrument.  [The core study](docs/core-study.md) registers the tests, fixed
before any run, over 26 groups in 31 paired-seed cells (1,550 runs), reported
estimation-first.

The design was fixed on 2026-07-15 and the epoch ceiling on 2026-07-20; the
campaign has since run to completion.  Every measurement is in
[`results/`](results/), computed post-hoc from committed snapshots by the
instruments in `src/group_algorithm_interp/instruments/`, and the
[reviewer notebook](notebooks/reviewer_walkthrough.ipynb) reproduces the main
results end to end from those files.

## The study

- [The core study](docs/core-study.md): the question, the pre-registered
  tests, the groups, the endpoints, and the run plan.
- [Extensions](docs/extensions.md): the modules the study could grow into,
  each with the measured trigger that would prompt running it.
- [Methodology](docs/methodology.md): group selection and the
  estimation-first reporting discipline.
- [Reproducibility](docs/reproducibility.md): what the repository guarantees
  about data, runs, and provenance.
- [The invariant dataset](docs/dataset.md): the full schema of
  `data/group_properties_full.jsonl`: 153 columns for each of the 6,958
  groups of orders 21–255, with null semantics and gotchas.
- [Research log](docs/research-log.md): research-direction decisions as
  dated entries.

## The pre-registered tests

Each is specified in full in [the core study](docs/core-study.md) and
reported estimation-first: effect sizes with confidence intervals, no
significance threshold, and every measurement ships, null outcomes included.

| test | what it does |
|---|---|
| Character-table-equivalent pairs (headline) | eleven pairs of groups identical in character table, Frobenius–Schur data, and coset-template structure; measures the within-pair difference in epochs-to-grok and irrep-occupancy, which the Fourier, tensor-rank, and coset accounts all forbid |
| The coset case study | the coset route is removed entirely on Q32 (no faithful action below \|G\|); probes whether the network learns a coset-based circuit on D32, with Q32 as a structural control |
| Satellite tiers and direction tests | satellite tiers over the same character-table-equal pair family; falsification-grade and direction tests at their best exemplars |
| A non-split extension | GL(2,3) against SL(2,3)·C2, a split/non-split extension pair; probes, ablates, and fits the 2-cocycle a non-split extension is predicted to force |

The pair family behind the headline test comes from a re-runnable screen:
`scripts/derive_falsifiers.py` reduced 7,274 same-fingerprint candidate pairs
to 370 clean pairs, eleven of which the core study selects under a rule
fixed before any run.  The screen's output is committed at
[`results/falsifier_screen_results_full.json`](results/falsifier_screen_results_full.json),
and [`results/README.md`](results/README.md) records the input and output
hashes that make it reproducible byte-for-byte.

## Findings

The campaign is complete: 2,100 canonical one-run-per-seed cells across the
core and its width rescues (`results/canonical_runs.json`, 1,544 grokked /
556 censored).

| test | finding | records |
|---|---|---|
| Character-table-equivalent pairs (headline) | the within-pair difference includes zero on both endpoints (epochs-to-grok and the irrep-occupancy contrast) for every pair with an informative sample, with occupancy well above its analytic floor, so the null reads as genuine equality | [`results/endpoints/`](results/endpoints/), [`results/occupancy/`](results/occupancy/) |
| The coset case study | no coset circuit survives audit: D32's induced representation spans rank 30 of 32, the block-neuron selector tags 81–87% of the MLP so its width-256 "pass" is a set-size artifact, and ablating the coset subspace flips about as many held-out predictions as a rank-matched random subspace. Neither probe separates a coset route from the bulk, so no coset mechanism is established beyond the relevant representation blocks being occupied | [`results/coset/`](results/coset/), [`results/audit/`](results/audit/) |
| Satellite tiers and direction tests | the signed-cyclic coordinate is undefined on the quaternionic C13:Q8 (all 50 seeds) and defined on D104, where it decodes cleanly on about half the seeds (27 of 49), the direction the character-decode account predicts | [`results/probes/`](results/probes/) |
| A non-split extension | with a two-element quotient the pre-registered cocycle probe is structurally degenerate; a replacement intervention finds the untwisted-product fallback in a minority of seeds | [`results/cocycle/`](results/cocycle/) |

Two tier-1 pairs sat below the grok floor at their pinned width and were
measured wider (order-27 at width 512, and one order-64 pair at width 256 with
a 400,000-epoch ceiling), with the pair family unchanged.  Five heavier tier-1
pairs are staged behind a pre-registered width probe that did not clear, so
they stand as a documented deferral described in the core study.

## Quickstart: run the training loop without SageMath

The production group corpus is Sage/GAP-generated and stays local.  So that
the offline test suite does not depend on Sage, `tests/support_artifacts.py`
builds a small deterministic corpus of group artefacts in pure Python, and
every code path that loads a group resolves its directory from the
`GROUP_ARTIFACTS_DIR` environment variable.  The same fixtures run the
training loop with no SageMath installed:

```bash
uv sync --dev

# Write the fixture artefacts: C2, C4, C6, C7, C8, C21, C32, S3, D4, Q8.
uv run python -c "import sys; sys.path.insert(0, 'tests'); from pathlib import Path; from support_artifacts import write_test_artifacts; write_test_artifacts(Path('tests/.artifacts/quickstart'))"

# Train on S3 = SmallGroup(6,1). W&B defaults to disabled; no network, no keys.
GROUP_ARTIFACTS_DIR=tests/.artifacts/quickstart \
  uv run python scripts/run.py data.group.order=6 data.group.index=1 optim.epochs=500
```

The run logs its run directory.  Inside it: `manifest.yaml` (git commit and
dirty flag, config hash, dataset hash, lockfile hash, environment, command,
final metrics), `resolved_config.yaml`, `run.log`, and `checkpoints/` holding
trajectory snapshots (`step_N.pt`, `final.pt`).  Runs always restart from
scratch, and a snapshot is a point-in-time record of the model weights for
post-hoc analysis; see [the reproducibility contract](docs/reproducibility.md)
for why there is no resume path and what a run does and does not guarantee.

This shows the plumbing works; the fixture groups are tiny and the numbers
mean nothing.

## Project shape

```text
SageMath/GAP export         per-group .npz artefact       training run
scripts/export_group.py  →  re-validated on load       →  scripts/run.py
                            (group axioms, characters,     Hydra compose,
                             projectors)                   Pydantic validate
                                                                 ↓
                                                          runs/<run_id>/
                                                          manifest.yaml
                                                          resolved_config.yaml
                                                          run.log
                                                          checkpoints/ (trajectory
                                                            snapshots)
```

`src/group_algorithm_interp/` holds the group data interfaces (`groups/`,
`representations/`), the task and the two models (a one-layer transformer
and a fully-connected control, `task.py`, `model.py`), training
(`training/`, including the vmapped multi-seed trainer `training/ensemble.py`,
and `experiment.py`), and run provenance (`manifest.py`, `seed.py`,
`config.py`, `stats.py`).  `scripts/` holds the executable entry points,
`configs/` their Hydra configuration, `docs/` the public study documents, and
`results/` committed derivation outputs.

## Running the study

A single run is `scripts/run.py` plus Hydra overrides, as in the quickstart;
`just train` wraps it, and `experiment=smoke` gives a 200-epoch end-to-end
smoke run.

`scripts/run_batch.py` trains many seeds of one (group, width) as a single
`torch.func.vmap` kernel.  `--seeds 0:50` selects the batch, and every other
argument is a Hydra override composed exactly as `scripts/run.py` takes it.
It writes one `runs/<run_id>/` per seed in the shape `scripts/run.py`
produces, and exits 0 only if every seed reached a `completed` manifest.  On
CUDA it measures true per-model peak memory with a one-model trial and splits
the seeds into memory-sized chunks; `--max-models-per-batch` caps a chunk
directly and `--memory-safety-fraction` (default 0.8) sets the headroom.
Across a batch only the initialisation seed varies; `data.split_seed` is
pinned, so every seed trains on one split.

`scripts/run_campaign.py` drives the pre-registered core study.  It reads the
ordered cell list from `configs/campaign/core.yaml`, where a cell is one
(group, width, seed-range) block re-checked against the group catalogue at
run time. It skips any cell whose seeds already have a `completed` manifest
under `runs/`, and invokes `scripts/run_batch.py` once per remaining cell.
Re-running it after an interruption picks up where it left off:

```bash
uv run python scripts/run_campaign.py --dry-run
uv run python scripts/run_campaign.py --only-phase pilot
uv run python scripts/run_campaign.py --keep-going --override logging.mode=online
```

Two opt-in bonus phases (extension supply for E2 and E6, excluded from the
pre-registered core counts) sit in the campaign file and run only under
`--include-bonus`.  On a multi-GPU pod,
`--shard i/n` gives each worker a deterministic, disjoint, roughly
cost-balanced slice of the cell list, so `CUDA_VISIBLE_DEVICES=$i uv run
python scripts/run_campaign.py --shard $i/8` across eight GPUs covers every
cell once.

`uv run python scripts/preflight.py` checks that a machine has what a run
needs: a CUDA device torch can see, the group artefact the config names
present and loadable, disk for `runs/`, and a W&B credential when
`logging.mode=online`.  Pass the campaign's own `--override` flags so it
checks the config that will run; `--mode export` checks SageMath instead,
needed only to regenerate group artefacts.  After a campaign,
`scripts/sync_runs.py` ships completed runs off the pod (as W&B artifacts, to
an rclone target, or both) before the pod is destroyed.

## Group data

SageMath is an optional extra, installed with pip, so no system Sage is needed:

```bash
uv sync --extra sage                                       # passagemath 10.8
uv run python scripts/export_group.py --order 8 --index 3
```

An upstream SageMath install works too (`sage -python
scripts/export_group.py --order 8 --index 3`).  The extra requires
Python >= 3.11; plain `uv sync` (what CI runs) does not install it, and the
test suite never imports Sage.  The exporter is still exercised by the
offline gate: `tests/data/group_artifacts/` holds committed exporter output
for S3, D4, Q8, and C8, and `tests/test_representation_ground_truth.py`
checks the exported characters and projectors, for those golden artefacts
and the Python fixtures alike, against structure re-derived from the Cayley
table alone.

Group selection rests on the group-properties dataset,
`data/group_properties_full.jsonl` (see [its schema](docs/dataset.md)),
generated by a two-stage pipeline committed alongside it:

```bash
gap -b -q -T scripts/enumerate_groups.g > data/group_properties.jsonl
uv run python scripts/enumerate_groups.py --output data/group_properties_full.jsonl
```

Per-group `.npz` artefacts, run directories, and snapshots are regenerable
local research artefacts and stay out of the
repository; their generation metadata and hashes are recorded in each
artefact's provenance block and in the per-run manifest.

## Development

```bash
uv sync --dev
just gate
```

`just gate` runs the documentation-link and research-log format checks
(`scripts/check_docs.py`, `scripts/check_research_log.py`), `ruff check`,
`ruff format --check`, `mypy`, `uv lock --check`, and `pytest -n auto`.  It
runs offline, with W&B disabled and no API keys, and enforces a coverage
floor of 80%.  An autouse fixture reseeds the Python, NumPy, and torch RNGs
before every test, so the suite is deterministic and order-independent.  CI
runs the same checks under Python 3.12, with a few additions: `--base` and
`--render-dir` on `check_research_log.py` (the log is diffed against the pull
request's base commit and its indexes uploaded as a build artefact), a
sequential `pytest` with an XML coverage report, and `uv build --no-sources`.
A second job reruns `pytest` alone on Python 3.10, 3.11, and 3.13, so the two
jobs together cover every Python version this project supports
(`requires-python`, `>=3.10,<3.14`).

## Licence

See [LICENSE](LICENSE).
