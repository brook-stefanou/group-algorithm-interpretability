# Common tasks. Run `just --list` to see them. Recipes wrap `uv run`.

# install runtime + dev deps and the pre-commit hook
setup:
    uv sync --dev
    uv run pre-commit install

# run the fast test suite (parallel workers; plain `uv run pytest` stays
# sequential for easier debugging/output ordering)
test:
    uv run pytest -n auto

# lint + format check
lint:
    uv run ruff check .
    uv run ruff format --check .

# pragmatic type check (intentionally not mypy --strict)
typecheck:
    uv run mypy

# verify that every local Markdown link resolves to a repository path
docs:
    uv run python scripts/check_docs.py
    uv run python scripts/check_research_log.py

# docs + lint + typecheck + uv.lock freshness + tests -- network-free, no W&B,
# no API keys. CI runs these plus `uv build --no-sources`, a sequential
# `pytest --cov-report=xml`, `--base`/`--render-dir` on check_research_log.py,
# and Python 3.10/3.11/3.13 (CI's `uv sync --locked` already enforces lock
# freshness, so it does not repeat the standalone check below).
gate: docs lint typecheck
    uv lock --check
    uv run pytest -n auto

# test coverage report for local inspection
cov:
    uv run pytest -n auto --cov=group_algorithm_interp --cov-report=term-missing

# run the training loop; the default composition is the full budget
# (`optim.epochs`, 10,000 by default) -- pass overrides for a short run, e.g.
# `just train experiment=smoke` or `just train experiment=debug seed=1`
train *ARGS:
    uv run python scripts/run.py {{ARGS}}

# can this machine do what is about to be asked of it? Defaults to the training
# checks (CUDA, group artifacts, disk, W&B). Pass the campaign's own overrides so
# it checks the config that will run, e.g. `just preflight --seeds 0-19 --override
# data.group.order=32 --override data.group.index=20`. `just preflight --mode
# export` checks SageMath instead.
preflight *ARGS:
    uv run python scripts/preflight.py {{ARGS}}

# ship completed runs/ off this machine as W&B artifacts and/or to an object
# store. Idempotent (a ledger in runs/ tracks what already went out) and safe
# on a cron. Pass extra flags
# through, e.g. `just sync-runs --dry-run` or
# `just sync-runs --no-wandb --rclone-target s3remote:bucket/prefix`.
sync-runs *ARGS:
    uv run python scripts/sync_runs.py {{ARGS}}

# run (or resume) the pre-registered core-study campaign: iterate the cells of
# configs/campaign/core.yaml in order, skip cells whose seeds all have a
# `completed` manifest under runs/, and invoke scripts/run_batch.py per cell.
# Pass flags through, e.g. `just campaign --dry-run`,
# `just campaign --only-phase pilot`, or
# `just campaign --keep-going --override logging.mode=online`.
campaign *ARGS:
    uv run python scripts/run_campaign.py {{ARGS}}

# Mac-side: pull synced run results from the object store into a local
# results-archive/ (gitignored). rclone only -- this Mac's /usr/bin/rsync is
# Apple's openrsync and hangs silently on tree copies. e.g.
# `just pull-results target=s3remote:my-bucket/gai-runs` (a bare
# `just pull-results s3remote:my-bucket/gai-runs` also works -- the optional
# `target=` prefix is stripped).
pull-results target:
    rclone copy "{{ trim_start_match(target, 'target=') }}" results-archive/ --progress
