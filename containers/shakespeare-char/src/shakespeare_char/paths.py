"""The utrain filesystem contract.

A single mount root, whose layout is fixed, so a container hardcodes these
rather than receiving them. RUN_DIR is read-only (config.yaml, control.json);
RUN_DIR/data and RUN_DIR/wandb are writable, and RUN_DIR/data is the only
channel phases hand state through.
"""

import pathlib

RUN_DIR = pathlib.Path("/utrain")
DATA_DIR = RUN_DIR / "data"
