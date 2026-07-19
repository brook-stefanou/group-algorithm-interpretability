import logging

from group_algorithm_interp.logging_utils import setup_logging


def test_setup_logging_writes_run_log(tmp_path):
    logger = setup_logging(run_dir=tmp_path, use_rich=False)
    logger.info("hello-from-test")
    for h in logger.handlers:
        h.flush()
    log_text = (tmp_path / "run.log").read_text()
    assert "hello-from-test" in log_text


def test_setup_logging_is_idempotent(tmp_path):
    logger = setup_logging(run_dir=tmp_path, use_rich=False)
    n = len(logger.handlers)
    logger = setup_logging(run_dir=tmp_path, use_rich=False)
    assert len(logger.handlers) == n  # no handler accumulation on repeat calls


def test_setup_logging_console_only(tmp_path):
    logger = setup_logging(run_dir=None, use_rich=False)
    assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    assert logger.handlers  # at least a console handler
