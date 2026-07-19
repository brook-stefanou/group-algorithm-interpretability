# Reproducibility

What the repository guarantees today. Only machinery that has actually been
built is covered here.

## Group data

Every group the models train on enters through one boundary: a Sage/GAP export
(`scripts/export_group.py`) written to a per-group `.npz` artefact. The
artefact carries the Cayley table, conjugacy classes, character table,
Frobenius–Schur indicators, concrete irreps, isotypic projectors, subgroups
and left cosets, together with a metadata block recording its format version,
its canonical SmallGroups `(order, index)` identity, and its provenance (the
backend that produced it). The Python training code contains no finite-group
algorithms and never imports Sage or GAP.

Loading an artefact re-validates it: the Cayley table is checked exhaustively
against the group axioms (closure, Latin square, identity, inverses,
associativity), and the isotypic blocks are checked against their own
projectors (`block_rank == trace(P)`, `block_rank == sum(d^2)` over the merged
irreps, and block ranks summing to `|G|`). A corrupt or mismatched artefact
fails at load, naming the invariant it violated. Artefacts are regenerable
from the exporter, and the exporter is the versioned thing.

Artefacts are located via the `GROUP_ARTIFACTS_DIR` environment variable,
defaulting to `data/group_artifacts/`. The offline test suite — and a reviewer
without SageMath — runs the full training path against a fixture corpus
generated in pure Python (`tests/support_artifacts.py`); see the
[README](../README.md).

## The invariant dataset

Group selection and the clean-pair screen work from a committed dataset,
`data/group_properties_full.jsonl`: 6,958 groups spanning orders 21–255, with
153 columns of invariants per group. It's produced by a two-stage pipeline,
both stages runnable from the repository:

```bash
gap -b -q -T scripts/enumerate_groups.g > data/group_properties.jsonl
uv run python scripts/enumerate_groups.py --output data/group_properties_full.jsonl
```

Stage 1 (GAP) enumerates the base invariants into an immutable source
catalogue; stage 2 (Sage, available via `uv sync --extra sage` or an upstream
SageMath's `sage -python`) enriches it and refuses to write over its own
input.
Column semantics that are easy to misread — the columns where null means "the
property does not hold" and the diagnostic-only columns that must never be
used as features — are documented in the header of
`scripts/enumerate_groups.py`. The full per-column schema, with types, null
meanings, and gotchas, is in [`docs/dataset.md`](dataset.md).

## Runs

Every run writes a self-contained `runs/<run_id>/` directory, created fresh; a
run never attaches to an existing one. It contains:

- `manifest.yaml` — the record of what happened. It is written with status
  `running` before anything can fail and finalised to `completed`, `failed`
  (with the exception type and message), or `aborted` (Ctrl-C or SIGTERM, the
  normal end of a run on a reclaimed spot GPU), so even a crashed or
  interrupted run leaves a terminal manifest behind. Its provenance block
  records the git commit and a dirty flag, the exact command, a hash of the
  dependency lockfile, the Python, torch, and platform versions, the detected
  device, and two config hashes: `config_hash` over the whole resolved config
  (two runs with the same hash used the same knobs, down to the W&B tags) and
  `config_group_hash` over the scientific configuration alone — model, data,
  optimiser — shared by all seeds of one experiment. Its dataset block records
  the group's canonical name, the train fraction, the split seed actually used
  (`data.split_seed` when pinned, otherwise the run seed it falls back to,
  with a captured warning), a dataset hash over the data spec and that
  effective split seed, and a leakage block: the realised transpose-leak
  fraction, the group's commuting probability `k(G)/|G|`, the test and
  unleaked-subset sizes, and which accuracy the generalisation streak keyed
  on. `provenance.campaign_id` carries the launching campaign's identifier (from
  the `GAI_CAMPAIGN_ID` environment variable) when the run was started under a
  campaign, and is null for a standalone run. Warnings raised during the run are
  captured into the manifest.
- `resolved_config.yaml` — the exact validated config the run used, after
  Hydra composition and Pydantic validation.
- `run.log` — the run's log.
- `checkpoints/` — trajectory snapshots.

Reproducing a run means checking out the recorded commit, restoring the
environment the lockfile hash identifies, regenerating the group artefact, and
re-running with the resolved config. The seed is part of the config.

## Determinism

A CUDA run isn't bit-reproducible by default, and that's deliberate. The
model's embedding backward is a scatter-add (`atomicAdd`) whose reduction
order varies between runs, so on CUDA the same seed produces slightly
different weights; TF32 matmul is enabled, trading fp32 mantissa bits for a
large speedup on Ampere and later. Bit-reproducible kernels are an operator
opt-in: `deterministic: true` drives
`torch.use_deterministic_algorithms(True)`, costs real throughput, and nothing
in the harness overrides it. A run is reproducible in its configuration and
provenance — same commit, lockfile, resolved config, seed, and data split —
and its results are expected to be statistically equivalent across repeats.
Claims are read as distributions over seeds.

`CUBLAS_WORKSPACE_CONFIG` is set at package import and at the top of
`scripts/run.py`, before the first CUDA BLAS call can read it, which is what
makes the opt-in work (cuBLAS reads it when it creates its handle). The opt-in
has been exercised on CPU only — the development machine has no CUDA device —
so anyone relying on it on a GPU should first confirm that
`deterministic: true` does not raise at the first cuBLAS call.

The locked environment (torch 2.12.1) resolves a CUDA 12.9 wheel on Linux and
the ordinary PyPI/MPS wheel elsewhere. `uv.lock` pins both x86_64 and aarch64
Linux torch wheels, so an ARM NVIDIA host (GH200/GB200) resolves and runs. The
cu129 wheel carries kernels for Blackwell (`sm_120`) as well as Ampere and
Hopper; a cu124 or cu126 wheel does not, and dies on Blackwell with "no kernel
image is available for execution on the device".

## Run length

The composed default config trains for 10,000 epochs (`optim.epochs`). The
campaign preset `experiment=core` sets the pre-registered 30,000-epoch ceiling;
a seed that has not grokked by the ceiling is analysed as censored (see
[core-study.md](core-study.md)) and is never extended or resumed.

## Snapshots

One policy governs everything written to `checkpoints/`: a snapshot at step 0,
at powers of two up to a dense-logging horizon, at a fixed step interval, on a
large relative drop in test loss (when event snapshots are enabled), on the
step at which a run's generalisation streak reaches its patience (when early
stopping on generalisation is armed), on each of the last few epochs of the
ceiling (`final_epoch_<E>.pt`, `snapshot.final_window_epochs` of them, so a
late training-loss spike never leaves the final snapshot as the only
end-of-training record), and once at the end (`final.pt`).

A snapshot holds the step, the epoch, the config, and the model state, with no
optimiser state. There is no resume path and no best-checkpoint selection:
runs always restart from scratch, and a snapshot is a point-in-time record for
post-hoc analysis. Selecting a "best" checkpoint by an evaluation-set metric
would be model selection on the evaluation set, which this project does not
do; the manifest's `best_metric` and `best_step` keys remain permanently null
for schema stability.

The `analysis/` directory inside a run is an empty placeholder: nothing in
this repository reads a snapshot back yet.

## Multi-seed campaigns

`scripts/run_campaign.py` drives the pre-registered core study from a campaign
file (`configs/campaign/core.yaml`). Each cell in that file is one
(group, width, seed-range) block, and each cell's `(order, index)` is checked
against the group-properties ground truth so a typo cannot silently train a
group nobody pre-registered. The runner iterates the cells in order, skips any
cell whose full seed range already has a `completed` manifest under `runs/`, and
invokes `scripts/run_batch.py` once per remaining cell — so re-running it after
an interruption resumes where it stopped without re-training a finished cell. It
continues past a failed cell under `--keep-going`, and exits non-zero if any
cell failed.

`scripts/run_batch.py` trains a cell's seeds, co-batching the models of one
(order, width) into a single `torch.func.vmap` kernel. It writes one
`runs/<run_id>/manifest.yaml` + `resolved_config.yaml` per seed, in the same
shape `scripts/run.py` produces, and exits 0 only when every seed reached a
`completed` manifest. Across a cell it varies only the initialisation seed;
`data.split_seed` is pinned so every seed of a cell trains on one identical
split. On SIGINT or SIGTERM it drains its running seeds so each one's own
cleanup can finalise its manifest, rather than leaving a manifest frozen at
`status: running`.

On a multi-GPU pod, `--shard i/n` partitions the cell list deterministically
across `n` workers — the shards are disjoint, cover every cell once, and are
roughly cost-balanced — so `CUDA_VISIBLE_DEVICES=$i uv run python
scripts/run_campaign.py --shard $i/8` on eight GPUs runs the whole campaign in
parallel. A campaign identifier can be exported as `GAI_CAMPAIGN_ID` and is then
recorded on each run's `provenance.campaign_id`.

`scripts/preflight.py` checks a machine against the config that is about to
run: CUDA visible to torch, the named group artefact present and loadable, disk
headroom under `runs/`, and a W&B key when the config logs online (`--mode
export` checks for a working Sage/GAP instead). Pass the campaign's own
`--override` flags so it checks the config the campaign will run.
`scripts/sync_runs.py` copies finished runs off a pod (as W&B artifacts, to an
rclone target, or both) before the pod is destroyed; only runs whose manifest
has reached a terminal status are shipped.

## Confirmatory runs

A run declares a validation stance. An `exploratory` run may look at whatever
it likes. A `confirmatory` run must declare a held-out set in `setup()` — the
harness records only a fingerprint and a size, never the data — and the
lifecycle itself invokes the held-out evaluation exactly once, after the
exploratory phase has returned. Calling it twice, or early, is an error, as is
a confirmatory run that declares no held-out set. If a prediction was
pre-registered in the config, the manifest records the outcome as `predicted`,
`refuted`, or `inconclusive` against it.

## The gate

`just gate` is what CI runs: documentation-link and research-log checks,
`ruff check`, `ruff format --check`, `mypy`, and `pytest`. It runs with no
network access, no W&B, and no API keys, and enforces a coverage floor of 80%.
CI additionally builds the package (`uv build --no-sources`) and runs the test
suite on Python 3.10 through 3.13. The RNGs are reseeded before every test, so
the suite is deterministic and independent of test order.

## The research log

`research-log.md` records the project's decisions, results, and corrections as
dated entries, appended in order. A dated bullet may carry an invisible
`decision` or `learning` tag, which marks it for exact copy into a decision
view and a learning view. CI (`scripts/check_research_log.py`) verifies the
file's format and its append-only history, then renders those two views as a
workflow artefact; the views are regenerated on each CI run and stay out of
the repository.
