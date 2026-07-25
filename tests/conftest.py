import os
import pathlib
import shutil
import subprocess

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def _build_sqsh(name: str, work: pathlib.Path) -> pathlib.Path | None:
    """Build the podman image for `name` and import it to a private `.sqsh`.

    The import uses an isolated enroot cache/temp so it never touches the
    developer's global enroot store. Returns the sqsh path, or None if a
    required tool is missing or a step fails (callers turn None into a skip).
    """
    if shutil.which("podman") is None or shutil.which("enroot") is None:
        return None
    image = f"utrain-{name}:utrain"
    build = subprocess.run(
        ["podman", "build", "-t", image, "-f", f"containers/{name}/Containerfile", "."],
        cwd=_PROJECT_ROOT,
    )
    if build.returncode != 0:
        return None
    env = dict(os.environ)
    env["ENROOT_CACHE_PATH"] = str(work / "cache")
    env["ENROOT_TEMP_PATH"] = str(work / "temp")
    pathlib.Path(env["ENROOT_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)
    pathlib.Path(env["ENROOT_TEMP_PATH"]).mkdir(parents=True, exist_ok=True)
    sqsh = work / f"utrain-{name}+utrain.sqsh"
    imported = subprocess.run(["enroot", "import", "-o", str(sqsh), f"podman://{image}"], env=env)
    if imported.returncode != 0:
        return None
    return sqsh


def _sqsh(name: str, tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    result = _build_sqsh(name, tmp_path_factory.mktemp(f"sqsh-{name}"))
    if result is None:
        pytest.skip(f"could not build utrain-{name} image")
    return result


def _isolated_enroot(name: str, sqsh: pathlib.Path, base: pathlib.Path) -> dict[str, str]:
    """Unpack `name`'s image into a private enroot store and return the env that
    points enroot (and thus utrain) at it.

    Each test gets its own data/runtime/cache/temp paths, so image mutations
    (e.g. `utrain image remove`) and running containers stay isolated from other
    tests and from the developer's global enroot store.
    """
    env = {
        "ENROOT_DATA_PATH": str(base / "data"),
        "ENROOT_RUNTIME_PATH": str(base / "runtime"),
        "ENROOT_CACHE_PATH": str(base / "cache"),
        "ENROOT_TEMP_PATH": str(base / "temp"),
    }
    for path in env.values():
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["enroot", "create", "-n", f"utrain-{name}+utrain", str(sqsh)],
        env={**os.environ, **env},
        check=True,
    )
    return env


@pytest.fixture(scope="session")
def _fake_sqsh(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    return _sqsh("fake", tmp_path_factory)


@pytest.fixture(scope="session")
def _fake_gpu_sqsh(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    return _sqsh("fake-gpu", tmp_path_factory)


@pytest.fixture()
def fake_image(_fake_sqsh: pathlib.Path, tmp_path: pathlib.Path) -> dict[str, str]:
    return _isolated_enroot("fake", _fake_sqsh, tmp_path)


@pytest.fixture()
def fake_gpu_image(_fake_gpu_sqsh: pathlib.Path, tmp_path: pathlib.Path) -> dict[str, str]:
    return _isolated_enroot("fake-gpu", _fake_gpu_sqsh, tmp_path)


@pytest.fixture()
def gpu_passthrough() -> None:
    if shutil.which("nvidia-container-cli") is None:
        pytest.skip("nvidia-container-cli not installed; enroot GPU passthrough unavailable")
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip("no GPU on host (nvidia-smi -L failed)")
