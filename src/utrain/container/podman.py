import json
import subprocess
import typing


class _Image(typing.TypedDict, total=False):
    Names: list[str]


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
            if name.endswith(":utrain"):
                repository = name[: -len(":utrain")]
                preset_name = repository.removeprefix("localhost/")
                presets[preset_name] = name
    return presets
