import pathlib
import shutil
import subprocess

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_CONTAINERS_DIR = pathlib.Path(__file__).parent / "containers"


def _build_image(name: str) -> str | None:
    """Build the test container `name` with podman and return a URL usable with
    `utrain image add`.

    Returns None if podman/enroot is missing or the build fails (callers turn
    None into a skip). The build context is the repo root (not the container
    dir) because the Containerfile `COPY`s repo-root-relative paths.
    """
    if shutil.which("podman") is None or shutil.which("enroot") is None:
        return None
    # Tag with the bare name (not `utrain-<name>`): `utrain image add` namespaces
    # the URL basename to `utrain-<basename>`, so `podman://fake:utrain` becomes
    # the `utrain-fake` preset the cram tests reference via `--image utrain-fake`.
    image = f"{name}:utrain"
    containerfile = _CONTAINERS_DIR / name / "Containerfile"
    build = subprocess.run(
        ["podman", "build", "-t", image, "-f", str(containerfile), "."],
        cwd=_PROJECT_ROOT,
    )
    if build.returncode != 0:
        return None
    return f"podman://{image}"


def _image_url(name: str) -> str:
    url = _build_image(name)
    if url is None:
        pytest.skip(f"could not build utrain-{name} image")
    return url


@pytest.fixture(scope="session")
def fake_image_url() -> str:
    return _image_url("fake")


@pytest.fixture(scope="session")
def fake_gpu_image_url() -> str:
    return _image_url("fake-gpu")


@pytest.fixture()
def fake_image(fake_image_url: str) -> dict[str, str]:
    # Cram tests run `utrain image add "$UTRAIN_TEST_IMAGE_URL"` themselves into the
    # per-test isolated enroot store set up by tests/cram/setup.sh.
    return {"UTRAIN_TEST_IMAGE_URL": fake_image_url}


@pytest.fixture()
def fake_gpu_image(fake_gpu_image_url: str) -> dict[str, str]:
    return {"UTRAIN_TEST_IMAGE_URL": fake_gpu_image_url}


@pytest.fixture()
def gpu_passthrough() -> None:
    if shutil.which("nvidia-container-cli") is None:
        pytest.skip("nvidia-container-cli not installed; enroot GPU passthrough unavailable")
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip("no GPU on host (nvidia-smi -L failed)")
