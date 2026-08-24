import collections.abc
import fcntl
import os
import pathlib
import shutil
import subprocess
import sys
import typing
import uuid

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_CONTAINERS_DIR = pathlib.Path(__file__).parent / "containers"


def _build_image(name: str, lock_dir: pathlib.Path) -> str | None:
    """Build the test container `name` with podman and return a URL usable with
    `utrain image add`.

    Returns None if podman is missing or the build fails (callers turn None into
    a skip). The build context is the repo root (not the container dir) because
    the Containerfile `COPY`s repo-root-relative paths.

    Serialized on a lockfile in `lock_dir`: this is session-scoped, and under
    pytest-xdist every worker is its own process, so without the lock N workers
    race to build the same tag. The first worker does the real build and the
    rest fall through on podman's layer cache.
    """
    if shutil.which("podman") is None:
        return None
    # Tag with the bare name (not `utrain-<name>`): `utrain image add` namespaces
    # the URL basename to `utrain-<basename>`, so `podman://fake:utrain` becomes
    # the `utrain-fake` preset the cram tests reference via `--image utrain-fake`.
    # This build tag is deliberately fixed rather than following UTRAIN_IMAGE_TAG:
    # it is the shared source image that `image add` tags *from*, and a per-test
    # UTRAIN_IMAGE_TAG keeps it out of `utrain image list` on its own.
    image = f"{name}:utrain"
    containerfile = _CONTAINERS_DIR / name / "Containerfile"
    with open(lock_dir / f"utrain-build-{name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        build = subprocess.run(
            ["podman", "build", "-t", image, "-f", str(containerfile), "."],
            cwd=_PROJECT_ROOT,
        )
    if build.returncode != 0:
        return None
    return f"podman://{image}"


def _image_url(name: str, tmp_path_factory: pytest.TempPathFactory) -> str:
    # getbasetemp() is per-worker under xdist; its parent is shared by all of
    # them, which is what the build lock needs.
    url = _build_image(name, tmp_path_factory.getbasetemp().parent)
    if url is None:
        pytest.skip(f"could not build utrain-{name} image")
    return url


@pytest.fixture(scope="session")
def fake_image_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    return _image_url("fake", tmp_path_factory)


@pytest.fixture(scope="session")
def fake_gpu_image_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    return _image_url("fake-gpu", tmp_path_factory)


@pytest.fixture(autouse=True)
def _image_namespace(monkeypatch: pytest.MonkeyPatch) -> typing.Iterator[None]:
    """Give each test a private preset tag namespace in the shared podman store.

    Cram tests share the developer's store (podman has no equivalent of enroot's
    cheap per-store env vars, and `podman tag` is a zero-copy pointer, so
    `image add` costs nothing). So isolate by tag instead of by store: with
    UTRAIN_IMAGE_TAG set, every preset this test creates is
    `localhost/utrain-<name>:<tag>`, invisible both to tests running in parallel
    and to the developer's own presets. The preset *key* is still `utrain-fake`,
    so the .t files need no knowledge of any of this.

    Swept here rather than in setup.sh's EXIT trap so that a cram script which
    dies mid-run still gets cleaned up.
    """
    tag = f"utrain-t-{uuid.uuid4().hex[:12]}"
    monkeypatch.setenv("UTRAIN_IMAGE_TAG", tag)
    yield
    _remove_tags(tag)


def _remove_tags(tag: str) -> None:
    """Untag every image carrying `tag`.

    `podman rmi` on a name an image shares with others only drops that name, and
    every preset here is a tag of a build image that keeps its own `:utrain`
    name, so this reclaims the namespace without touching image data.
    """
    if shutil.which("podman") is None:
        return
    listed = subprocess.run(
        ["podman", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
    )
    for ref in listed.stdout.split():
        if ref.endswith(f":{tag}"):
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
def tutorial_image() -> typing.Iterator[None]:
    """Sweep the image docs/building-a-container.md tells the reader to build.

    That tutorial is exercised end to end by building it exactly as written, so
    unlike the images above it is not built here -- the cram script does it. It
    does need cleaning up though: the doc's tag is `localhost/utrain-demo:utrain`,
    which `utrain image list` would pick up as a preset in the developer's own
    store. Removing it drops the tag only; podman's build cache survives, so the
    next run rebuilds in under a second.
    """
    if shutil.which("podman") is None:
        pytest.skip("podman not installed")
    yield
    subprocess.run(["podman", "rmi", "localhost/utrain-demo:utrain"], capture_output=True)


@pytest.fixture()
def gpu_passthrough() -> None:
    if shutil.which("nvidia-ctk") is None:
        pytest.skip("nvidia-ctk not installed; cannot generate a CDI spec for GPU passthrough")
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip("no GPU on host (nvidia-smi -L failed)")


# The reference container's serve command, run from the checkout rather than
# from an image. The contract says a container must be runnable that way, and
# it is what lets the serve tests -- and the TUI's chat screen tests -- run on
# CI, which has no podman at all.
_FAKE = _CONTAINERS_DIR / "fake" / "fake.py"
# The wandb shim, so fake.py's module-level `import wandb` resolves without the
# real package installed -- the same substitution utrain does inside a container.
_SHIM = _PROJECT_ROOT / "src" / "utrain" / "container" / "wandb_shim"


@pytest.fixture()
def spawn_serve() -> typing.Iterator[
    collections.abc.Callable[[pathlib.Path], subprocess.Popen[bytes]]
]:
    """Start the fake container's `serve` under a utrain root, and clean it up.

    stdout and stderr are left alone: the port arrives through
    `<root>/serve/port.json`, so the container's output is not a protocol and
    nothing here has to consume it.
    """
    started: list[subprocess.Popen[bytes]] = []

    def spawn(root: pathlib.Path) -> subprocess.Popen[bytes]:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(_SHIM), env.get("PYTHONPATH", "")])
        proc = subprocess.Popen(
            [sys.executable, str(_FAKE), "--utrain-root", str(root), "serve", "--port", "0"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        started.append(proc)
        return proc

    yield spawn
    for proc in started:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)
