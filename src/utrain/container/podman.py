import json
import os
import subprocess
import typing

from . import schema

# Every utrain-managed image is tagged `localhost/<key>:utrain`, where <key> is
# the `utrain-` prefixed name stored in the `runs.image` DB column. The tag is
# what marks an image as a utrain preset; the prefix keeps unrelated images
# tagged `:utrain` (e.g. the bare `fake:utrain` the test suite builds before
# `image add` namespaces it) out of `utrain image list`.
#
# UTRAIN_IMAGE_TAG overrides the tag so a caller can claim a private preset
# namespace inside a shared podman store -- the image-store counterpart to
# UTRAIN_DATA_DIR isolating run state. The test suite gives every test its own,
# so parallel tests cannot untag each other's images. `podman tag` is zero-copy,
# so the namespaces all share one set of layers.
_TAG = os.environ.get("UTRAIN_IMAGE_TAG", "utrain")
_PREFIX = "utrain-"

# Every container utrain starts carries this label, naming the preset-tag
# namespace the image came from. A container started by image *id* is reported
# by `podman ps` under whichever tag of that image podman picks, and several
# namespaces can tag the same content -- so the tag a ps listing shows proves
# nothing about who started it, while the label does.
CONTAINER_LABEL = "utrain.image-tag"


def container_label_args() -> list[str]:
    """`--label` args branding a container with this namespace."""
    return ["--label", f"{CONTAINER_LABEL}={_TAG}"]


class _Image(typing.TypedDict, total=False):
    Names: list[str]


def image_ref(key: str) -> str:
    """Full podman reference for a preset key, e.g. `localhost/utrain-fake:utrain`."""
    return f"localhost/{key}:{_TAG}"


def image_id(ref: str) -> str:
    """The content id of an image reference, as bare hex.

    Stored on a run when it is created (or first touched after the upgrade), so
    the run keeps pointing at exactly that content even if the preset tag is
    later moved. Bare hex rather than `sha256:...` because `podman run <hex-id>`
    works everywhere `podman run <ref>` does -- describe, phases, serve.
    """
    result = subprocess.run(
        ["podman", "image", "inspect", ref, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"podman image inspect failed: {result.stderr.strip()}")
    return result.stdout.strip().removeprefix("sha256:")


def image_exists(ref: str) -> bool:
    result = subprocess.run(
        ["podman", "image", "exists", ref],
        capture_output=True,
        timeout=30,
    )
    return result.returncode == 0


def preset_key(name: str) -> str:
    """Namespaced preset key for a bare image name, avoiding a doubled prefix."""
    return name if name.startswith(_PREFIX) else f"{_PREFIX}{name}"


def list_presets() -> dict[str, str]:
    result = subprocess.run(
        ["podman", "images", "--format", "json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"podman images failed: {result.stderr.strip()}")

    entries: list[_Image] = json.loads(result.stdout)
    presets: dict[str, str] = {}
    for entry in entries:
        for name in entry.get("Names") or []:
            if not name.endswith(f":{_TAG}"):
                continue
            key = name.removesuffix(f":{_TAG}").removeprefix("localhost/")
            if key.startswith(_PREFIX):
                presets[key] = name
    return presets


def _run_cmd(image: str, args: list[str], timeout: int = 60) -> str:
    result = subprocess.run(
        ["podman", "run", "--rm", *container_label_args(), image] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"podman command failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def describe(image: str) -> schema.DescribeOutput:
    out = _run_cmd(image, ["describe"])
    return schema.DescribeOutput.model_validate(json.loads(out))
