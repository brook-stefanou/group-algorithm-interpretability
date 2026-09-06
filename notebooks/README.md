# notebooks

`reviewer_walkthrough.ipynb` is a read-only tour of the core study for a
reviewer: it states the research question, then walks the main results (the C1
character-table-equivalent null, the C3 carry asymmetry, the C2 coset case
study, the C5 cocycle partial) and a live grok curve, loading every number from
a committed `results/` file or recomputing it by importing the project's own
instruments from `src/group_algorithm_interp` — it trains nothing, hits no
network, and touches no W&B or S3. Launch it interactively with
`uv run jupyter lab notebooks/reviewer_walkthrough.ipynb`, or run it headless
with `uv run --with nbconvert --with jupyter --with ipykernel --with matplotlib
--with pandas jupyter nbconvert --to notebook --execute --inplace
notebooks/reviewer_walkthrough.ipynb`. The C3 and grok-curve cells recompute
from checkpoints already on disk under `runs/`; if a checkpoint is absent they
degrade to the instrument's docstring plus the committed record rather than
erroring.
