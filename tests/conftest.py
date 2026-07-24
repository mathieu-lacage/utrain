import pathlib
import shutil
import subprocess

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent


@pytest.fixture()
def fake_image() -> None:
    result = subprocess.run(["enroot", "list"], capture_output=True, text=True)
    existing = set(result.stdout.strip().splitlines())
    if "utrain-fake+utrain" not in existing:
        result = subprocess.run(["make", "containers/fake"], cwd=_PROJECT_ROOT)
        if result.returncode != 0:
            pytest.skip("could not build utrain-fake+utrain; run: make containers/fake")


@pytest.fixture()
def fake_gpu_image() -> None:
    result = subprocess.run(["enroot", "list"], capture_output=True, text=True)
    existing = set(result.stdout.strip().splitlines())
    if "utrain-fake-gpu+utrain" not in existing:
        result = subprocess.run(["make", "containers/fake-gpu"], cwd=_PROJECT_ROOT)
        if result.returncode != 0:
            pytest.skip("could not build utrain-fake-gpu+utrain; run: make containers/fake-gpu")


@pytest.fixture()
def gpu_passthrough() -> None:
    if shutil.which("nvidia-container-cli") is None:
        pytest.skip("nvidia-container-cli not installed; enroot GPU passthrough unavailable")
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip("no GPU on host (nvidia-smi -L failed)")
