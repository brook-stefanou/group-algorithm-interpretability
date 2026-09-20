# Group Algorithm Interp

Project write-up: [brookstefanou.com/projects/finite-group-interp](https://brookstefanou.com/projects/finite-group-interp/)

Small models learn to multiply in finite groups. This repository studies which
part of a group's representation-theoretic structure a trained model uses to do
it. The thesis is that a model settles on a near-minimal faithful subset of that
structure and recovers the product in the readout, without carrying the whole
regular representation. The design controls the comparison with matched pairs of
groups that share a character table.

For the argument and the design in full, see
[the core study](docs/core-study.md) and [the methodology](docs/methodology.md).

## The instruments

The trained models are read offline against committed snapshots, by the
instruments in
[`src/group_algorithm_interp/instruments/`](src/group_algorithm_interp/instruments/).
Each reads a single trained model.

- `occupancy.py` measures how a model spreads its weight across the group's
  irreducible representations.
- `recruited_dimension.py` compares the model's effective dimension against the
  minimal faithful real dimension and the group order.
- `used_set_audit.py` checks whether the blocks a model occupies are the ones its
  computation needs.
- `gcr_matmul.py` tests whether the hidden layer forms the shared-index matrix
  product of the two argument matrices or only a generic bilinear form.
- `readout_characterisation.py` tests whether the output stage needs anything
  beyond the scalar character of the product.
- `coset_quotient.py` reads the coset route as one instance of the thesis, on
  groups where a subgroup quotient is well-formed.

## The study docs

- [The core study](docs/core-study.md): the question, the design, the groups,
  and the run plan.
- [Extensions](docs/extensions.md): the modules the study could grow into, each
  with the measured trigger that would prompt running it.
- [Methodology](docs/methodology.md): group selection and the estimation-first
  reporting discipline.
- [Reproducibility](docs/reproducibility.md): what the repository guarantees
  about data, runs, and provenance.
- [The invariant dataset](docs/dataset.md): the schema of
  `data/group_properties_full.jsonl`, 153 columns for each of the 6,958 groups
  of orders 21 to 255, with null semantics and gotchas.
- [Research log](docs/research-log.md): research-direction decisions as dated
  entries.

The [reviewer notebook](notebooks/reviewer_walkthrough.ipynb) reproduces the
main results from the committed files, offline.

## Quickstart: run the training loop without SageMath

The production group corpus is Sage/GAP-generated and stays local. So that the
offline test suite does not depend on Sage, `tests/support_artifacts.py` builds a
small deterministic corpus of group artefacts in pure Python, and every code
path that loads a group resolves its directory from the `GROUP_ARTIFACTS_DIR`
environment variable. The same fixtures run the training loop with no SageMath
installed:

```bash
uv sync --dev

# Write the fixture artefacts: C2, C4, C6, C7, C8, C21, C32, S3, D4, Q8.
uv run python -c "import sys; sys.path.insert(0, 'tests'); from pathlib import Path; from support_artifacts import write_test_artifacts; write_test_artifacts(Path('tests/.artifacts/quickstart'))"

# Train on S3 = SmallGroup(6,1). W&B defaults to disabled, no network, no keys.
GROUP_ARTIFACTS_DIR=tests/.artifacts/quickstart \
  uv run python scripts/run.py data.group.order=6 data.group.index=1 optim.epochs=500
```

The run logs its run directory. Inside it are `manifest.yaml` (git commit and
dirty flag, config hash, dataset hash, lockfile hash, environment, command,
final metrics), `resolved_config.yaml`, `run.log`, and `checkpoints/` holding
trajectory snapshots (`step_N.pt`, `final.pt`). Runs always restart from
scratch, and a snapshot is a point-in-time record of the model weights for
post-hoc analysis. See [the reproducibility contract](docs/reproducibility.md)
for why there is no resume path and what a run does and does not guarantee.

This shows the plumbing works. The fixture groups are tiny and the numbers mean
nothing.

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
`representations/`), the task and the two models (a one-layer transformer and a
fully-connected control, `task.py`, `model.py`), training (`training/`,
including the vmapped multi-seed trainer `training/ensemble.py`, and
`experiment.py`), and run provenance (`manifest.py`, `seed.py`, `config.py`,
`stats.py`). `scripts/` holds the executable entry points, `configs/` their
Hydra configuration, `docs/` the public study documents, and `results/`
committed derivation outputs.

## Running the study

A single run is `scripts/run.py` plus Hydra overrides, as in the quickstart.
`just train` wraps it, and `experiment=smoke` gives a 200-epoch end-to-end smoke
run.

`scripts/run_batch.py` trains many seeds of one (group, width) as a single
`torch.func.vmap` kernel. `--seeds 0:50` selects the batch, and every other
argument is a Hydra override composed exactly as `scripts/run.py` takes it. It
writes one `runs/<run_id>/` per seed in the shape `scripts/run.py` produces, and
exits 0 only if every seed reached a `completed` manifest. On CUDA it measures
true per-model peak memory with a one-model trial and splits the seeds into
memory-sized chunks. `--max-models-per-batch` caps a chunk directly and
`--memory-safety-fraction` (default 0.8) sets the headroom. Across a batch only
the initialisation seed varies, and `data.split_seed` is pinned, so every seed
trains on one split.

`scripts/run_campaign.py` drives the core study. It reads the ordered cell list
from `configs/campaign/core.yaml`, where a cell is one (group, width,
seed-range) block re-checked against the group catalogue at run time. It skips
any cell whose seeds already have a `completed` manifest under `runs/`, and
invokes `scripts/run_batch.py` once per remaining cell. Re-running it after an
interruption picks up where it left off:

```bash
uv run python scripts/run_campaign.py --dry-run
uv run python scripts/run_campaign.py --only-phase pilot
uv run python scripts/run_campaign.py --keep-going --override logging.mode=online
```

On a multi-GPU pod, `--shard i/n` gives each worker a deterministic, disjoint,
roughly cost-balanced slice of the cell list, so `CUDA_VISIBLE_DEVICES=$i uv run
python scripts/run_campaign.py --shard $i/8` across eight GPUs covers every cell
once.

`uv run python scripts/preflight.py` checks that a machine has what a run needs,
a CUDA device torch can see, the group artefact the config names present and
loadable, disk for `runs/`, and a W&B credential when `logging.mode=online`.
Pass the campaign's own `--override` flags so it checks the config that will run.
`--mode export` checks SageMath instead, needed only to regenerate group
artefacts. After a campaign, `scripts/sync_runs.py` ships completed runs off the
pod, as W&B artifacts, to an rclone target, or both, before the pod is
destroyed.

## Group data

SageMath is an optional extra, installed with pip, so no system Sage is needed:

```bash
uv sync --extra sage                                       # passagemath 10.8
uv run python scripts/export_group.py --order 8 --index 3
```

An upstream SageMath install works too (`sage -python scripts/export_group.py
--order 8 --index 3`). The extra requires Python >= 3.11. Plain `uv sync` (what
CI runs) does not install it, and the test suite never imports Sage. The exporter
is still exercised by the offline gate. `tests/data/group_artifacts/` holds
committed exporter output for S3, D4, Q8, and C8, and
`tests/test_representation_ground_truth.py` checks the exported characters and
projectors, for those golden artefacts and the Python fixtures alike, against
structure re-derived from the Cayley table alone.

Group selection rests on the group-properties dataset,
`data/group_properties_full.jsonl` (see [its schema](docs/dataset.md)),
generated by a two-stage pipeline committed alongside it:

```bash
gap -b -q -T scripts/enumerate_groups.g > data/group_properties.jsonl
uv run python scripts/enumerate_groups.py --output data/group_properties_full.jsonl
```

Per-group `.npz` artefacts, run directories, and snapshots are regenerable local
research artefacts and stay out of the repository. Their generation metadata and
hashes are recorded in each artefact's provenance block and in the per-run
manifest.

## Development

```bash
uv sync --dev
just gate
```

`just gate` runs the documentation-link and research-log format checks
(`scripts/check_docs.py`, `scripts/check_research_log.py`), `ruff check`, `ruff
format --check`, `mypy`, `uv lock --check`, and `pytest -n auto`. It runs
offline, with W&B disabled and no API keys, and enforces a coverage floor of
80%. An autouse fixture reseeds the Python, NumPy, and torch RNGs before every
test, so the suite is deterministic and order-independent. CI runs the same
checks under Python 3.12, with a few additions, `--base` and `--render-dir` on
`check_research_log.py` (the log is diffed against the pull request's base commit
and its indexes uploaded as a build artefact), a sequential `pytest` with an XML
coverage report, and `uv build --no-sources`. A second job reruns `pytest` alone
on Python 3.10, 3.11, and 3.13, so the two jobs together cover every Python
version this project supports (`requires-python`, `>=3.10,<3.14`).

## Licence

See [LICENSE](LICENSE).
