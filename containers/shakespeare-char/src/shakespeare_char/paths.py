"""The utrain filesystem contract.

A single root whose layout is fixed: the root holds the read-only config.yaml
and control.json, while `data/`, `wandb/` and `serve/` beneath it are writable,
and `data/` is the only channel phases hand state through.

The root itself is *not* fixed -- utrain passes it as `--utrain-root` (it mounts
everything at /utrain), and it defaults to `$CWD/run` so this container's phases
can be run straight from a checkout, without building an image first.
"""

import dataclasses
import json
import pathlib

# Relative to the working directory; the CLI resolves it against CWD at startup.
DEFAULT_ROOT = "run"


@dataclasses.dataclass(frozen=True)
class Paths:
    """The contract paths, hanging off whichever root the caller supplied."""

    run_dir: pathlib.Path

    @property
    def data_dir(self) -> pathlib.Path:
        return self.run_dir / "data"

    @property
    def serve_dir(self) -> pathlib.Path:
        """Writable, and mounted only for `serve`. Holds `port.json`."""
        return self.run_dir / "serve"

    def ensure(self) -> None:
        """Create the dirs a phase writes into.

        A no-op under utrain, which mounts them already; what makes a bare
        `shakespeare-char run --phase ...` work outside a container.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "wandb").mkdir(parents=True, exist_ok=True)


def write_port_file(p: Paths, port: int) -> None:
    """Publish the bound port for whoever launched this container.

    Written to a temporary name and renamed, because rename is atomic within a
    filesystem
    """
    if not p.serve_dir.exists():
        # when run under podman via utrain, this directory already exists
        p.serve_dir.mkdir(parents=True, exist_ok=True)
    target = p.serve_dir / "port.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"port": port}))
    tmp.replace(target)
