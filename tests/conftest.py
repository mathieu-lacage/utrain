import pathlib
import shutil
import subprocess

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def _image(name: str) -> None:
    result = subprocess.run(["enroot", "list"], capture_output=True, text=True)
    existing = set(result.stdout.strip().splitlines())
    target = name.rstrip("")
    image_name = f"utrain-{name}+utrain"
    if image_name not in existing:
        result = subprocess.run(["make", f"containers/{name}"], cwd=_PROJECT_ROOT)
        if result.returncode != 0:
            pytest.skip(f"could not build {image_name} run: make containers/{name}")

@pytest.fixture()
def fake_image() -> None:
    return _image("fake")


@pytest.fixture()
def fake_gpu_image() -> None:
    return _image("fake-gpu")


@pytest.fixture()
def gpu_passthrough() -> None:
    if shutil.which("nvidia-container-cli") is None:
        pytest.skip("nvidia-container-cli not installed; enroot GPU passthrough unavailable")
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip("no GPU on host (nvidia-smi -L failed)")
