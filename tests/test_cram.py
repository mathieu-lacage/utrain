import os
import pathlib
import subprocess

import pytest

_CRAM_DIR = pathlib.Path(__file__).parent / "cram"
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def _run_cram(path: pathlib.Path, image_env: dict[str, str]) -> None:
    env = dict(os.environ)
    env.update(image_env)
    if "COVERAGE_PROCESS_START" in env:
        env["COVERAGE_PROCESS_START"] = str(
            (_PROJECT_ROOT / env["COVERAGE_PROCESS_START"]).resolve()
        )
        env["COVERAGE_FILE"] = str(_PROJECT_ROOT / ".coverage")
    result = subprocess.run(
        ["uv", "run", "cram", str(path)], capture_output=True, text=True, env=env
    )
    if result.returncode == 80:
        pytest.skip("cram setup skipped (image missing)")
    if result.returncode != 0:
        pytest.fail(result.stdout + result.stderr)


def test_compute() -> None:
    _run_cram(_CRAM_DIR / "compute.t", {})


def test_version() -> None:
    _run_cram(_CRAM_DIR / "version.t", {})


def test_id_prefix(fake_image: dict[str, str]) -> None:
    _run_cram(_CRAM_DIR / "id-prefix.t", fake_image)


def test_gpu(gpu_passthrough: None, fake_gpu_image: dict[str, str]) -> None:
    _run_cram(_CRAM_DIR / "gpu.t", fake_gpu_image)


def test_image(fake_image_url: str, fake_gpu_image_url: str) -> None:
    # Two distinct image URLs so image.t can exercise removing several images
    # at once (`utrain image remove a b`).
    _run_cram(
        _CRAM_DIR / "image.t",
        {"UTRAIN_TEST_IMAGE_URL": fake_image_url, "UTRAIN_TEST_IMAGE_URL2": fake_gpu_image_url},
    )


def test_orchestrator_crash(fake_image: dict[str, str]) -> None:
    _run_cram(_CRAM_DIR / "orchestrator-crash.t", fake_image)


def test_serve(fake_image_url: str, fake_gpu_image_url: str) -> None:
    # Two image URLs: fake implements serve, fake-gpu does not, which is how
    # serve.t exercises utrain's refusal on `can_serve: false`.
    _run_cram(
        _CRAM_DIR / "serve.t",
        {"UTRAIN_TEST_IMAGE_URL": fake_image_url, "UTRAIN_TEST_IMAGE_URL2": fake_gpu_image_url},
    )


def test_phase_show(fake_image: dict[str, str]) -> None:
    _run_cram(_CRAM_DIR / "phase-show.t", fake_image)


def test_phase_restart(fake_image: dict[str, str]) -> None:
    _run_cram(_CRAM_DIR / "phase-restart.t", fake_image)


def test_run_config(fake_image: dict[str, str]) -> None:
    _run_cram(_CRAM_DIR / "run-config.t", fake_image)


def test_run_config_protocol(fake_image: dict[str, str]) -> None:
    _run_cram(_CRAM_DIR / "run-config-protocol.t", fake_image)


def test_run_create(fake_image: dict[str, str]) -> None:
    _run_cram(_CRAM_DIR / "run-create.t", fake_image)


def test_run_delete(fake_image: dict[str, str]) -> None:
    _run_cram(_CRAM_DIR / "run-delete.t", fake_image)


def test_run_lifecycle(fake_image: dict[str, str]) -> None:
    _run_cram(_CRAM_DIR / "run-lifecycle.t", fake_image)


def test_run_restart(fake_image: dict[str, str]) -> None:
    _run_cram(_CRAM_DIR / "run-restart.t", fake_image)


def test_attempt_show(fake_image: dict[str, str]) -> None:
    _run_cram(_CRAM_DIR / "attempt-show.t", fake_image)


def test_run_stop(fake_image: dict[str, str]) -> None:
    _run_cram(_CRAM_DIR / "run-stop.t", fake_image)


def test_data_store(fake_image: dict[str, str]) -> None:
    _run_cram(_CRAM_DIR / "data-store.t", fake_image)


def test_wandb_compat(fake_image: dict[str, str]) -> None:
    _run_cram(_CRAM_DIR / "wandb-compat.t", fake_image)


def test_building_a_container(tutorial_image: None) -> None:
    # No image fixture: the .t builds the tutorial's container from the
    # Containerfile in the doc, which is half of what it is testing.
    _run_cram(_CRAM_DIR / "building-a-container.t", {})


def test_check_cache(fake_image: dict[str, str]) -> None:
    _run_cram(_CRAM_DIR / "check-cache.t", fake_image)
