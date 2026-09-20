# notebooks

`reviewer_walkthrough.ipynb` is a read-only tour of the study for a reviewer. It
states the research question, then walks the main results: what the
character-table-equivalent pairs share and where they differ, the occupancy
departure from the analytic null across the panel, a live run of the occupancy,
recruited-dimension and matrix-product instruments on one shipped checkpoint, and
the full thesis chain (isotypic-block ablation, recruited dimension against its
minimal-faithful and regular-representation anchors, the used-set audit as an
honest limitation, the coset-quotient route as a concrete instance, and the
readout). Every number outside the live-checkpoint section loads from a committed
`results/` file; the live-checkpoint section recomputes by importing the
project's own instruments from `src/group_algorithm_interp` against the one
checkpoint shipped under `sample_runs/`. It trains nothing, hits no network, and
touches no W&B or S3. Launch it interactively with
`uv run jupyter lab notebooks/reviewer_walkthrough.ipynb`, or run it headless with
`uv run --with nbconvert --with jupyter --with ipykernel --with matplotlib --with
pandas jupyter nbconvert --to notebook --execute --inplace
notebooks/reviewer_walkthrough.ipynb`.

`sample_runs/2026-07-20_143051_631691_core_db19e1/` ships the checkpoint the
live-checkpoint section runs against (`final.pt`, `resolved_config.yaml`,
`selection.json`, `manifest.yaml`): SmallGroup(64,228), seed 0, width 128, a
seed that grokked cleanly to accuracy 1.0 over the full 60,000-epoch ceiling,
chosen because its degree-4 irreducible representation gives the matrix-product
test something to distinguish. It needs
`data/group_artifacts/smallgroup_64_228.npz` alongside it for the group object.
`sample_runs/2026-07-20_152155_823626_core_cfdf0c/` is an earlier sample run kept
for its own committed `analysis/*.json` records (no checkpoint shipped with it).
