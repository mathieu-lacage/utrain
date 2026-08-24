"""Re-point nanochat's wandb calls at the phase utrain is running.

nanochat logs its metrics itself, with good names -- `train/loss`, `val/bpb`,
`reward`. utrain's shim already makes `import wandb` mean naw, so those metrics
are almost in the right place already; what is wrong is the address. naw files a
run at ``<dir>/wandb/<project>/<id>.rtsdb``, and utrain reads
``<root>/wandb/<phase>/<run_id>.rtsdb``, while nanochat asks for
``project="nanochat-sft"`` and lets ``dir`` default to the working directory.

So rather than parse nanochat's stdout for numbers it is already logging
properly, this module wraps ``wandb.init`` and overrides those three arguments
from the environment. It is named ``sitecustomize`` because that is imported
automatically by every interpreter that finds it on ``sys.path``, which means
the override applies without patching a line of nanochat.

Only ``project``, ``id`` and ``dir`` are touched; ``name``, ``config`` and the
rest pass through, so the run still carries nanochat's own description of itself.
"""

import os

_PROJECT = os.environ.get("UTRAIN_WANDB_PROJECT")
_ID = os.environ.get("UTRAIN_WANDB_ID")
_DIR = os.environ.get("UTRAIN_WANDB_DIR")


def _install() -> None:
    import wandb

    original = wandb.init

    def init(*args: object, **kwargs: object) -> object:
        kwargs["project"] = _PROJECT
        kwargs["id"] = _ID
        kwargs["dir"] = _DIR
        return original(*args, **kwargs)

    wandb.init = init


if _PROJECT and _ID and _DIR:
    try:
        _install()
    except Exception as exc:  # pragma: no cover - a phase must not die over this
        # A phase whose metrics go missing is worth a line in the log; a phase
        # that refuses to start because of them is not.
        print(f"utrain wandb bridge inactive: {exc}", flush=True)
