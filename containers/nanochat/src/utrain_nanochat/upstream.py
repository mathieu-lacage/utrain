"""Driving nanochat's own scripts as child processes.

nanochat is a checkout, not a library: every stage is a `python -m scripts.X`
entry point that reads its state from a base directory. So this container wraps
rather than reimplements -- it points that base directory at `<root>/data`, runs
the script, and relays the stop flag.

Two environment details carry the whole integration:

* ``NANOCHAT_BASE_DIR`` lands inside the data dir, which is the only channel
  utrain hands state through. Each phase therefore starts from the previous
  phase's shards, tokenizer and checkpoints without any copying of our own.
* ``PYTHONPATH`` gains the ``wandb_bridge`` directory ahead of everything else,
  so the child imports its ``sitecustomize`` and nanochat's own ``wandb.init``
  calls are re-pointed at this phase's rtsdb file. See that module.

Everything runs single-process: utrain hands a container one GPU (``compute`` is
``cpu`` or ``gpu<N>``), so there is nothing for ``torchrun`` to distribute over.
"""

import os
import pathlib
import signal
import subprocess
import sys
import time

from . import config, paths

# Where the Containerfile clones nanochat. Overridable so the phases can be
# driven against a checkout on a developer's machine, the way `--utrain-root`
# lets everything else run outside a container.
NANOCHAT_DIR = pathlib.Path(os.environ.get("NANOCHAT_DIR", "/app/nanochat"))

# Shipped beside the package rather than inside it: it has to be a directory on
# PYTHONPATH holding a top-level `sitecustomize`, not an importable submodule.
# Installed, that puts it next to the package in site-packages; in a checkout it
# sits one level further out, beside `src/`.
_BRIDGE_CANDIDATES = (
    pathlib.Path(__file__).resolve().parent.parent / "wandb_bridge",
    pathlib.Path(__file__).resolve().parent.parent.parent / "wandb_bridge",
)
_BRIDGE_DIR = next(
    (c for c in _BRIDGE_CANDIDATES if c.is_dir()),
    _BRIDGE_CANDIDATES[0],
)

# How long a phase gets between the stop flag being set and SIGKILL. Long enough
# for torch to tear a CUDA context down, short enough that `utrain run stop`
# feels immediate.
_TERM_GRACE_S = 20.0

_POLL_INTERVAL_S = 1.0


def base_dir(p: paths.Paths) -> pathlib.Path:
    """nanochat's own artifact root: shards, tokenizer, every checkpoint dir."""
    return p.data_dir / "nanochat"


def child_env(p: paths.Paths, phase: str, run_id: str) -> dict[str, str]:
    env = dict(os.environ)
    env["NANOCHAT_BASE_DIR"] = str(base_dir(p))
    # nanochat fetches its task datasets itself, under NANOCHAT_BASE_DIR, so this
    # only covers what its dependencies pull from the hub on their own. Pointing
    # it at the data dir keeps that out of the container's ephemeral filesystem.
    env["HF_HOME"] = str(p.data_dir / "hf")
    env["OMP_NUM_THREADS"] = "1"
    # Never let a stray host credential turn into a real upload: what `wandb`
    # resolves to here is utrain's shim, but the phases are also runnable from a
    # checkout, where it is the genuine package.
    env["WANDB_MODE"] = "offline"
    env["UTRAIN_WANDB_PROJECT"] = phase
    env["UTRAIN_WANDB_ID"] = run_id
    env["UTRAIN_WANDB_DIR"] = str(p.run_dir)
    inherited = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_BRIDGE_DIR}{os.pathsep}{inherited}" if inherited else str(_BRIDGE_DIR)
    return env


def run_script(
    p: paths.Paths,
    phase: str,
    run_id: str,
    module: str,
    args: list[str],
) -> bool:
    """Run one nanochat entry point to completion, honouring the stop flag.

    Returns True when the child exited zero. The child's stdout and stderr are
    inherited rather than piped: utrain already redirects ours to the phase's
    log files, and passing the file descriptors straight through keeps its
    progress bars intact and costs no relay thread.
    """
    # No `--` before the flags. nanochat's documented invocations carry one, but
    # it belongs to `torchrun` -- it separates the launcher's arguments from the
    # script's. Passed straight to `python -m`, argparse would read everything
    # after it as positional and reject the lot.
    argv = [sys.executable, "-m", module, *args]
    print(f"+ {' '.join(argv)}", flush=True)

    child = subprocess.Popen(
        argv,
        cwd=str(NANOCHAT_DIR),
        env=child_env(p, phase, run_id),
        # Its own process group, so the terminate below reaches any worker the
        # script spawned rather than just the interpreter that spawned them.
        start_new_session=True,
    )
    return _wait(p, child) == 0


def run_code(p: paths.Paths, phase: str, run_id: str, code: str) -> bool:
    """Run a snippet inside nanochat's checkout, with the same env as a script.

    For the odd job nanochat has no entry point for -- warming a dataset cache,
    reading a checkpoint's metadata -- where a `-c` snippet is smaller than a
    file shipped alongside the package.
    """
    child = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=str(NANOCHAT_DIR),
        env=child_env(p, phase, run_id),
        start_new_session=True,
    )
    return _wait(p, child) == 0


def _wait(p: paths.Paths, child: "subprocess.Popen[bytes]") -> int:
    while True:
        code = child.poll()
        if code is not None:
            return code
        if config.read_control(p) == "stop":
            print("Stop requested; terminating nanochat.", flush=True)
            return _stop(child)
        time.sleep(_POLL_INTERVAL_S)


def _stop(child: "subprocess.Popen[bytes]") -> int:
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=_TERM_GRACE_S)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    # A stopped phase is a failed phase as far as utrain is concerned, and the
    # child's own status after a signal is not meaningfully non-zero on every
    # platform, so say so plainly.
    return 1
