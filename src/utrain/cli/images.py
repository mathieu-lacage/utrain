import os
import pathlib
import subprocess

import sqlalchemy
import sqlalchemy.orm

from .. import container
from . import db as dbmod
from . import exceptions


class ImageInfo:
    name: str
    size_str: str
    run_count: int

    def __init__(self, name: str, size_str: str, run_count: int) -> None:
        self.name = name
        self.size_str = size_str
        self.run_count = run_count


def _enroot_data_path() -> pathlib.Path:
    data_path = os.environ.get("ENROOT_DATA_PATH")
    if data_path:
        return pathlib.Path(data_path)
    return pathlib.Path.home() / ".local" / "share" / "enroot"


def _image_size(name: str) -> str:
    enroot_dir = _enroot_data_path() / f"{name}+utrain"
    try:
        result = subprocess.run(
            ["du", "-sh", str(enroot_dir)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.split()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "?"


def list_images(session: sqlalchemy.orm.Session) -> list[ImageInfo]:
    presets = container.enroot.list_presets()
    images: list[ImageInfo] = []
    for name in sorted(presets):
        run_count = session.execute(
            sqlalchemy.select(sqlalchemy.func.count()).where(
                (dbmod.runs.c.image == name) & (dbmod.runs.c.status != "deleted")
            )
        ).scalar_one()
        images.append(ImageInfo(name=name, size_str=_image_size(name), run_count=int(run_count)))
    return images


def add_image(url: str) -> str:
    # Derive <name> from last path component of the URL (strip scheme and tag)
    path_part = url.split("://", 1)[-1]
    base = path_part.split("/")[-1].split(":")[0]
    name = base

    sqsh = pathlib.Path(f"/tmp/utrain-{name}+utrain.sqsh")
    # enroot import exits non-zero on some hosts (xattr warnings) even when it
    # produces a valid image, so remove any stale file and trust the output file,
    # not the exit code (mirrors the Makefile's `|| true` and tests/conftest.py).
    sqsh.unlink(missing_ok=True)
    subprocess.run(["enroot", "import", "-o", str(sqsh), url])
    if not sqsh.exists() or sqsh.stat().st_size == 0:
        raise exceptions.UI("abort: enroot import failed")

    result = subprocess.run(["enroot", "create", str(sqsh)])
    if result.returncode != 0:
        raise exceptions.UI(f"abort: enroot create failed (exit {result.returncode})")

    sqsh.unlink(missing_ok=True)
    return name


def remove_image(name: str, session: sqlalchemy.orm.Session, force: bool = False) -> None:
    if name not in container.enroot.list_presets():
        raise exceptions.UI(f"abort: image '{name}' not found")

    run_count = session.execute(
        sqlalchemy.select(sqlalchemy.func.count()).where(
            (dbmod.runs.c.image == name) & (dbmod.runs.c.status != "deleted")
        )
    ).scalar_one()

    if int(run_count) > 0 and not force:
        raise exceptions.UI(
            f"abort: image '{name}' is used by {run_count} run(s); use --force to remove anyway"
        )

    result = subprocess.run(["enroot", "remove", f"{name}+utrain"], input="y\n", text=True)
    if result.returncode != 0:
        raise exceptions.UI(f"abort: enroot remove failed (exit {result.returncode})")
