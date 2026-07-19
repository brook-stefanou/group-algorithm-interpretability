import pytest
import yaml

from group_algorithm_interp.config import LoggingConfig, ProjectConfig
from group_algorithm_interp.manifest import (
    compute_config_hash,
    compute_dataset_hash,
    compute_group_hash,
    compute_lockfile_hash,
    create_manifest,
    create_run_dir,
    create_run_id,
    finalize_manifest,
    read_manifest,
    record_tracking,
    save_resolved_config,
)


def _config() -> ProjectConfig:
    return ProjectConfig(logging=LoggingConfig(mode="disabled"))


def test_create_manifest_starts_running(tmp_path):
    run_dir = create_run_dir("r1", root=tmp_path)
    create_manifest(_config(), run_dir)
    manifest = read_manifest(run_dir)

    assert manifest["status"] == "running"
    assert manifest["completed_at"] is None
    assert manifest["provenance"]["config_hash"] == compute_config_hash(_config())
    assert manifest["provenance"]["python_version"]
    assert manifest["provenance"]["torch_version"]
    assert manifest["provenance"]["available_device"] in {"cpu", "cuda", "mps"}
    assert manifest["summary"]["completed_steps"] is None
    assert manifest["provenance"]["campaign_id"] is None


def test_create_manifest_records_campaign_id(tmp_path):
    run_dir = create_run_dir("r1-campaign", root=tmp_path)
    create_manifest(_config(), run_dir, campaign_id="abc123")
    manifest = read_manifest(run_dir)
    assert manifest["provenance"]["campaign_id"] == "abc123"


def test_run_ids_are_unique_for_fast_same_name_runs():
    first = create_run_id("smoke")
    second = create_run_id("smoke")
    assert first != second
    assert "smoke" in first


def test_create_run_dir_refuses_to_reuse_existing_directory(tmp_path):
    create_run_dir("same", root=tmp_path)
    with pytest.raises(FileExistsError):
        create_run_dir("same", root=tmp_path)


def test_resolved_config_saved(tmp_path):
    run_dir = create_run_dir("r2", root=tmp_path)
    save_resolved_config(_config(), run_dir)
    saved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    assert saved["logging"]["mode"] == "disabled"
    assert saved["optim"]["epochs"] == _config().optim.epochs


def test_finalize_completed_records_summary(tmp_path):
    run_dir = create_run_dir("r3", root=tmp_path)
    create_manifest(_config(), run_dir)
    finalize_manifest(
        run_dir,
        status="completed",
        summary={"completed_steps": 100, "best_metric": 0.9},
        final_checkpoint="runs/r3/checkpoints/final.pt",
    )
    manifest = read_manifest(run_dir)
    assert manifest["status"] == "completed"
    assert manifest["completed_at"] is not None
    assert manifest["summary"]["completed_steps"] == 100
    assert manifest["artifacts"]["final_checkpoint"].endswith("final.pt")


def test_finalize_failed_records_error(tmp_path):
    run_dir = create_run_dir("r4", root=tmp_path)
    create_manifest(_config(), run_dir)
    finalize_manifest(run_dir, status="failed", error="ValueError: boom")
    manifest = read_manifest(run_dir)
    assert manifest["status"] == "failed"
    assert manifest["error"] == "ValueError: boom"


def test_record_tracking_updates_in_place(tmp_path):
    run_dir = create_run_dir("r5", root=tmp_path)
    create_manifest(_config(), run_dir)
    record_tracking(run_dir, wandb_run_id="abc123", wandb_url="https://wandb.ai/x/y/runs/abc123")
    manifest = read_manifest(run_dir)
    assert manifest["tracking"]["wandb_run_id"] == "abc123"
    # other fields preserved by the deep merge.
    assert manifest["tracking"]["backend"] == "wandb"
    assert manifest["status"] == "running"


def test_group_hash_ignores_seed_but_config_hash_does_not():
    c0 = ProjectConfig(seed=0)
    c1 = ProjectConfig(seed=1)
    # Group hash collapses seeds together; provenance hash keeps them distinct.
    assert compute_group_hash(c0) == compute_group_hash(c1)
    assert compute_config_hash(c0) != compute_config_hash(c1)


@pytest.mark.parametrize(
    "section,changed",
    [
        ("model", {"model": {"d_model": 256}}),
        ("data", {"data": {"train_frac": 0.75}}),
        ("optim", {"optim": {"lr": 0.5}}),
    ],
)
def test_group_hash_changes_with_the_scientific_config(section, changed):
    """model / data / optim define the experiment: each must move the hash."""
    base = ProjectConfig(seed=0)
    assert compute_group_hash(base) != compute_group_hash(ProjectConfig(seed=0, **changed))


@pytest.mark.parametrize(
    "operational",
    [
        {"logging": {"mode": "offline"}},
        {"logging": {"tags": ["ablation"]}},
        {"logging": {"notes": "rerun after the fix"}},
        {"device": "cpu"},
        {"deterministic": True},
        {"experiment": {"name": "renamed"}},
        {"snapshot": {"interval": 250}},
        {"eval": {"max_steps_warn": 50_000}},
    ],
)
def test_group_hash_ignores_operational_config(operational):
    """The group hash keys W&B run-grouping: a tag or device name changes nothing
    about the science, so it must not move the group hash."""
    base = ProjectConfig(seed=0)
    assert compute_group_hash(base) == compute_group_hash(ProjectConfig(seed=0, **operational))
    # The full-config provenance hash still records the difference.
    assert compute_config_hash(base) != compute_config_hash(ProjectConfig(seed=0, **operational))


def test_manifest_records_both_hashes(tmp_path):
    run_dir = create_run_dir("rgh", root=tmp_path)
    create_manifest(_config(), run_dir)
    provenance = read_manifest(run_dir)["provenance"]
    assert provenance["config_hash"] == compute_config_hash(_config())
    assert provenance["config_group_hash"] == compute_group_hash(_config())


def test_manifest_includes_dataset_provenance(tmp_path):
    run_dir = create_run_dir("rd", root=tmp_path)
    create_manifest(_config(), run_dir)
    ds = read_manifest(run_dir)["dataset"]
    assert ds["name"] == "SmallGroup(8,1)"
    assert ds["spec_hash"] == compute_dataset_hash(_config())


def test_dataset_hash_changes_with_the_data_spec():
    from group_algorithm_interp.config import DataConfig

    base = ProjectConfig(seed=0)
    assert compute_dataset_hash(base) != compute_dataset_hash(
        ProjectConfig(seed=0, data=DataConfig(train_frac=0.75))
    )
    assert compute_dataset_hash(base) != compute_dataset_hash(
        ProjectConfig(seed=0, data=DataConfig(group="S3"))
    )
    # The split seed changes which pairs are held out.
    assert compute_dataset_hash(base) != compute_dataset_hash(
        ProjectConfig(seed=0, data=DataConfig(split_seed=1))
    )


def test_dataset_hash_ignores_the_run_seed_when_the_split_seed_is_pinned():
    """`data.split_seed` is pinned by default, so `config.seed` must not touch
    the dataset hash -- otherwise a 5-seed sweep over one byte-identical split
    gets five different `spec_hash` values, and grouping runs by `spec_hash`
    to confirm a shared split fails."""
    from group_algorithm_interp.config import DataConfig

    hashes = {
        compute_dataset_hash(ProjectConfig(seed=seed, data=DataConfig(split_seed=0)))
        for seed in range(5)
    }
    assert len(hashes) == 1


def test_dataset_hash_uses_the_effective_split_seed():
    """An unpinned split_seed falls back to config.seed, so the hash must follow
    the seed that actually produced the split, not the raw field."""
    from group_algorithm_interp.config import DataConfig

    unpinned_3 = ProjectConfig(seed=3, data=DataConfig(split_seed=None))
    unpinned_4 = ProjectConfig(seed=4, data=DataConfig(split_seed=None))
    pinned_3 = ProjectConfig(seed=99, data=DataConfig(split_seed=3))
    assert compute_dataset_hash(unpinned_3) != compute_dataset_hash(unpinned_4)
    assert compute_dataset_hash(unpinned_3) == compute_dataset_hash(pinned_3)


def test_manifest_records_the_effective_split_seed(tmp_path):
    """With split_seed unset, the manifest must record the effective seed
    (config.seed), not the raw `null` -- otherwise which seed made the split
    is recorded nowhere."""
    from group_algorithm_interp.config import DataConfig

    config = ProjectConfig(seed=3, data=DataConfig(split_seed=None))
    run_dir = create_run_dir("rss", root=tmp_path)
    create_manifest(config, run_dir)
    assert read_manifest(run_dir)["dataset"]["split_seed"] == 3


def test_manifest_records_lockfile_hash(tmp_path):
    run_dir = create_run_dir("rl", root=tmp_path)
    create_manifest(_config(), run_dir)
    manifest = read_manifest(run_dir)
    assert manifest["provenance"]["lockfile_hash"] == compute_lockfile_hash()


def test_compute_lockfile_hash_present_and_missing(tmp_path):
    present = tmp_path / "uv.lock"
    present.write_text("locked-deps")
    h = compute_lockfile_hash(present)
    assert h is not None and len(h) == 64  # sha256 hex
    assert compute_lockfile_hash(tmp_path / "absent.lock") is None


def test_finalize_records_warnings(tmp_path):
    run_dir = create_run_dir("rw", root=tmp_path)
    create_manifest(_config(), run_dir)
    assert read_manifest(run_dir)["warnings"] == []
    finalize_manifest(run_dir, status="completed", warnings=["UserWarning: heads up"])
    assert read_manifest(run_dir)["warnings"] == ["UserWarning: heads up"]


def test_manifest_seeds_validation_block(tmp_path):
    from group_algorithm_interp.config import ProjectConfig, ValidationConfig
    from group_algorithm_interp.manifest import create_manifest, create_run_dir, read_manifest

    cfg = ProjectConfig(
        validation=ValidationConfig(
            stance="confirmatory",
            prediction={"metric": "accuracy", "direction": "higher", "value": 0.7},
        )
    )
    run_dir = create_run_dir("rv", root=tmp_path)
    create_manifest(cfg, run_dir)
    v = read_manifest(run_dir)["validation"]
    assert v["stance"] == "confirmatory"
    assert v["prediction"] == {"metric": "accuracy", "direction": "higher", "value": 0.7}
    assert v["holdout"] is None and v["holdout_result"] is None


def test_record_holdout_and_result(tmp_path):
    from group_algorithm_interp.manifest import (
        create_manifest,
        create_run_dir,
        read_manifest,
        record_holdout,
        record_holdout_result,
    )

    run_dir = create_run_dir("rh", root=tmp_path)
    create_manifest(_config(), run_dir)
    record_holdout(run_dir, fingerprint="abc123", size=128, spec={"frac": 0.1, "seed": 0})
    record_holdout_result(run_dir, metrics={"accuracy": 0.81}, outcome="predicted")

    v = read_manifest(run_dir)["validation"]
    assert v["holdout"] == {"fingerprint": "abc123", "size": 128, "spec": {"frac": 0.1, "seed": 0}}
    assert v["holdout_result"]["metrics"] == {"accuracy": 0.81}
    assert v["holdout_result"]["outcome"] == "predicted"
    assert v["holdout_result"]["evaluated_at"]
