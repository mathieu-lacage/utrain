"""The `download` phase: fetch the corpus into the data dir.

Split out of `tokenizer` so its output can be cached. The corpus is the same
bytes for every run, so declaring its hash up front lets utrain hand later runs
the copy already in its content-addressed store instead of fetching it again --
see `check-cache` in the container contract.
"""

import hashlib
import pathlib
import urllib.request

import wandb

from . import paths

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

CORPUS_FILENAME = "input.txt"

# sha256 of the file at SHAKESPEARE_URL, pinned like a lockfile entry.
# `check-cache` has to say what this phase would produce *without* producing it,
# which for a download is only possible if the bytes are known in advance. `run`
# verifies what it fetched against this, so an upstream change fails loudly here
# rather than quietly never hitting the cache again -- a file whose real hash
# differs from the declared one is stored under the real one, which no future
# manifest will ever ask for.
CORPUS_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def manifest() -> dict[str, object]:
    """What `run` would leave in the data dir, for `check-cache` to declare."""
    return {"files": [{"path": CORPUS_FILENAME, "sha256": CORPUS_SHA256}]}


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(p: paths.Paths) -> bool:
    run = wandb.init(dir=str(p.run_dir))
    target = p.data_dir / CORPUS_FILENAME

    print(f"Downloading {SHAKESPEARE_URL}...", flush=True)
    try:
        urllib.request.urlretrieve(SHAKESPEARE_URL, str(target))
    except Exception as e:
        print(f"Download failed: {e}", flush=True)
        run.finish(exit_code=1)
        return False

    digest = _sha256(target)
    if digest != CORPUS_SHA256:
        print(
            f"Corpus hash mismatch: expected {CORPUS_SHA256}, got {digest}. "
            "Upstream has changed -- update CORPUS_SHA256 in download.py.",
            flush=True,
        )
        run.finish(exit_code=1)
        return False

    corpus_bytes = target.stat().st_size
    print(f"Downloaded {corpus_bytes:,} bytes", flush=True)
    run.log({"corpus_bytes": float(corpus_bytes)})
    run.finish(exit_code=0)
    return True
