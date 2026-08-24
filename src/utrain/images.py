import subprocess

import sqlalchemy
import sqlalchemy.orm

from . import container, exceptions
from . import db as dbmod


class ImageInfo:
    name: str
    size_str: str
    run_count: int

    def __init__(self, name: str, size_str: str, run_count: int) -> None:
        self.name = name
        self.size_str = size_str
        self.run_count = run_count


def _image_exists(ref: str) -> bool:
    result = subprocess.run(["podman", "image", "exists", ref], capture_output=True)
    return result.returncode == 0


def _format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "K", "M", "G"):
        if size < 1024 or unit == "G":
            return f"{size:.0f}{unit}"
        size /= 1024
    return "?"


def _image_size(name: str) -> str:
    try:
        result = subprocess.run(
            [
                "podman",
                "image",
                "inspect",
                container.podman.image_ref(name),
                "--format",
                "{{.Size}}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return _format_size(int(result.stdout.strip()))
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass
    return "?"


def list_images(session: sqlalchemy.orm.Session) -> list[ImageInfo]:
    presets = container.podman.list_presets()
    images: list[ImageInfo] = []
    for name in sorted(presets):
        run_count = session.execute(
            sqlalchemy.select(sqlalchemy.func.count()).where(
                (dbmod.runs.c.image == name) & (dbmod.runs.c.status != "deleted")
            )
        ).scalar_one()
        images.append(ImageInfo(name=name, size_str=_image_size(name), run_count=int(run_count)))
    return images


def add_image(url: str, quiet: bool = False) -> str:
    """Pull an image into the local store and tag it as a utrain preset.

    `quiet` decides where podman's pull progress goes. The CLI lets it through
    to the terminal, which is the whole feedback there is during a pull of
    several gigabytes. A TUI cannot: it is in the alternate screen buffer, and
    a subprocess writing there draws over the display. So the caller that has a
    screen asks for the output to be captured, and gets it back on the error
    instead.
    """
    # Derive <name> from the last path component of the URL (strip scheme and tag).
    path_part = url.split("://", 1)[-1]
    base = path_part.split("/")[-1].split(":")[0]
    name = container.podman.preset_key(base)

    # `podman://` is an enroot transport, not a podman one; it means "already in
    # the local store", so there is nothing to pull. Everything else goes to
    # `podman pull` with its scheme intact (docker://, or a bare registry ref).
    if not url.startswith("podman://") and not _image_exists(path_part):
        result = subprocess.run(["podman", "pull", url], capture_output=quiet, text=True)
        if result.returncode != 0:
            # The stderr is there to be quoted only when it was captured;
            # otherwise the viewer has already watched podman print it.
            detail = (result.stderr or "").strip().splitlines()
            reason = f": {detail[-1]}" if quiet and detail else ""
            raise exceptions.UI(f"podman pull failed (exit {result.returncode}){reason}")

    # Tag from the scheme-stripped ref: that is the name a pull stores locally,
    # and `podman tag` rejects a transport prefix.
    result = subprocess.run(
        ["podman", "tag", path_part, container.podman.image_ref(name)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise exceptions.UI(f"podman tag failed: {result.stderr.strip()}")

    return name


def remove_image(name: str, session: sqlalchemy.orm.Session, force: bool = False) -> None:
    if name not in container.podman.list_presets():
        raise exceptions.UI(f"image '{name}' not found")

    run_count = session.execute(
        sqlalchemy.select(sqlalchemy.func.count()).where(
            (dbmod.runs.c.image == name) & (dbmod.runs.c.status != "deleted")
        )
    ).scalar_one()

    if int(run_count) > 0 and not force:
        raise exceptions.UI(
            f"image '{name}' is used by {run_count} run(s); use --force to remove anyway"
        )

    # Quiet: `podman rmi` reports every tag it drops ("Untagged: ..."), which is
    # podman's bookkeeping, not utrain's output. Removal is silent on success.
    result = subprocess.run(
        ["podman", "rmi", container.podman.image_ref(name)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise exceptions.UI(f"podman rmi failed: {result.stderr.strip()}")
