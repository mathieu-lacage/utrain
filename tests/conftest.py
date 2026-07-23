import pathlib
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
