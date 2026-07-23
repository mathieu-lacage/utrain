import os
import pathlib
import subprocess

import pytest

_CRAM_DIR = pathlib.Path(__file__).parent / "cram"
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def _run_cram(path: pathlib.Path) -> None:
    env = dict(os.environ)
    if "COVERAGE_PROCESS_START" in env:
        env["COVERAGE_PROCESS_START"] = str(
            (_PROJECT_ROOT / env["COVERAGE_PROCESS_START"]).resolve()
        )
    result = subprocess.run(
        ["uv", "run", "cram", str(path)], capture_output=True, text=True, env=env
    )
    if result.returncode == 80:
        pytest.skip("cram setup skipped (image missing)")
    if result.returncode != 0:
        pytest.fail(result.stdout + result.stderr)


def test_compute(fake_image: None) -> None:
    _run_cram(_CRAM_DIR / "compute.t")


def test_id_prefix(fake_image: None) -> None:
    _run_cram(_CRAM_DIR / "id-prefix.t")


def test_image(fake_image: None) -> None:
    _run_cram(_CRAM_DIR / "image.t")


def test_phase_read(fake_image: None) -> None:
    _run_cram(_CRAM_DIR / "phase-read.t")


def test_run_config(fake_image: None) -> None:
    _run_cram(_CRAM_DIR / "run-config.t")


def test_run_create(fake_image: None) -> None:
    _run_cram(_CRAM_DIR / "run-create.t")


def test_run_delete(fake_image: None) -> None:
    _run_cram(_CRAM_DIR / "run-delete.t")


def test_run_lifecycle(fake_image: None) -> None:
    _run_cram(_CRAM_DIR / "run-lifecycle.t")


def test_run_restart(fake_image: None) -> None:
    _run_cram(_CRAM_DIR / "run-restart.t")


def test_run_stop(fake_image: None) -> None:
    _run_cram(_CRAM_DIR / "run-stop.t")
