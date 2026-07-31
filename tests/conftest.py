import pathlib
import shutil
import subprocess
import typing

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_CONTAINERS_DIR = pathlib.Path(__file__).parent / "containers"


def _build_image(name: str) -> str | None:
    """Build the test container `name` with podman and return a URL usable with
    `utrain image add`.

    Returns None if podman is missing or the build fails (callers turn None into
    a skip). The build context is the repo root (not the container dir) because
    the Containerfile `COPY`s repo-root-relative paths.
    """
    if shutil.which("podman") is None:
        return None
    # Tag with the bare name (not `utrain-<name>`): `utrain image add` namespaces
    # the URL basename to `utrain-<basename>`, so `podman://fake:utrain` becomes
    # the `utrain-fake` preset the cram tests reference via `--image utrain-fake`.
    # The bare `fake:utrain` tag is not itself a preset -- list_presets only
    # matches the `utrain-` prefix -- so it stays out of `utrain image list`.
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


@pytest.fixture(scope="session", autouse=True)
def _clean_preset_tags() -> typing.Iterator[None]:
    """Drop `utrain-fake*` preset tags before and after the session.

    Cram tests share the developer's podman store (podman has no equivalent of
    enroot's cheap per-store env vars, and `podman tag` is a zero-copy pointer,
    so `image add` costs nothing). They remove their own images, but a failed
    test can leave a tag behind and perturb the next `utrain image list`.
    """
    _remove_preset_tags()
    yield
    _remove_preset_tags()


def _remove_preset_tags() -> None:
    if shutil.which("podman") is None:
        return
    listed = subprocess.run(
        ["podman", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
    )
    for ref in listed.stdout.split():
        if ref.startswith("localhost/utrain-fake") and ref.endswith(":utrain"):
            subprocess.run(["podman", "rmi", ref], capture_output=True)


@pytest.fixture()
def fake_image(fake_image_url: str) -> dict[str, str]:
    # Cram tests run `utrain image add "$UTRAIN_TEST_IMAGE_URL"` themselves; the
    # `podman://` URL resolves against the local store the build above populated.
    return {"UTRAIN_TEST_IMAGE_URL": fake_image_url}


@pytest.fixture()
def fake_gpu_image(fake_gpu_image_url: str) -> dict[str, str]:
    return {"UTRAIN_TEST_IMAGE_URL": fake_gpu_image_url}


@pytest.fixture()
def gpu_passthrough() -> None:
    if shutil.which("nvidia-ctk") is None:
        pytest.skip("nvidia-ctk not installed; cannot generate a CDI spec for GPU passthrough")
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip("no GPU on host (nvidia-smi -L failed)")
