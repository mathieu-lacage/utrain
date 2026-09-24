"""Orchestrator behaviour that needs a real container rather than a fake one.

Everything here drives podman, so like the cram suite it skips where podman is
missing -- which is CI. Run it locally.
"""

import os
import pathlib
import subprocess

import pytest

import utrain.container.podman
import utrain.orchestrator


def _running_images() -> list[str]:
    listed = subprocess.run(
        ["podman", "ps", "--format", "{{.Image}}"], capture_output=True, text=True
    )
    return listed.stdout.split()


def test_check_cache_leaves_no_container_when_it_times_out(
    fake_image_url: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A check-cache that never answers is given up on -- and so is its container.

    `subprocess.run`'s timeout kills the podman client, which by itself stops
    nothing: the container keeps running, pinning the image and the read-only
    data dir it was handed. The timeout has to remove it by id.
    """
    # Under the private tag `_image_namespace` gave this test, so the assertion
    # below can name the image exactly rather than filtering by ancestry, which
    # would also match the cram tests' containers on the same image id. It also
    # means that fixture sweeps up if this test dies mid-run.
    image = f"localhost/fake:{os.environ['UTRAIN_IMAGE_TAG']}"
    subprocess.run(["podman", "tag", "fake:utrain", image], check=True)
    monkeypatch.setattr(utrain.orchestrator, "_MANIFEST_TIMEOUT_S", 3)

    config = tmp_path / "config.yaml"
    config.write_text("check_cache_hang: true\n")
    attempt_dir = tmp_path / "attempt" / "1"
    attempt_dir.mkdir(parents=True)
    utrain.orchestrator.init_mount_dir(attempt_dir, config)
    data_dir = attempt_dir / "data" / "tokenizer"
    data_dir.mkdir(parents=True)

    manifest = utrain.orchestrator._check_cache(image, attempt_dir, "tokenizer", data_dir)

    # A timeout is a cache miss as far as the caller is concerned.
    assert manifest is None
    assert image not in _running_images()
