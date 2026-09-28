"""Liveness of the orchestrator that owns an attempt.

The orchestrator holds an exclusive `flock` on a file inside the attempt dir
for its whole life. The kernel releases that lock when the process goes away,
however it goes away -- clean exit, SIGTERM, SIGKILL, OOM kill, or a reboot.
So a reader that finds the lock free has proven the holder is gone, which is a
stronger statement than any pid check can make: `os.kill(pid, 0)` answers
"some process has this id", and pids are recycled.

That distinction matters because the orchestrator writes every status
transition itself, so the only thing left for reconciliation is the case where
it died without getting to write. A false "still alive" leaves a run stuck at
`running` forever; a false "dead" finalizes a run that is still going.

Two details keep the false-dead case out:

- Readers probe with a *shared* lock, so they never exclude each other, and
  they hold it for the length of two syscalls.
- A lock file has to be created before it can be locked, and it is written
  only once held. A reader that finds it empty is looking at that gap, and
  falls back to the pid rather than calling the orchestrator dead.

`hold` retries briefly for the mirror-image reason: a reader's shared lock is
momentary and should not be mistaken for a second orchestrator.

The lock lives directly in the attempt dir, which is host-only: the container
is given `mnt`, `data`, `wandb` and `serve` beneath it (see
`orchestrator._mount_args`), never the dir itself.
"""

import fcntl
import json
import os
import pathlib
import time
import typing

from . import exceptions

# How long `hold` keeps retrying before calling it a duplicate orchestrator.
# Readers hold their shared lock across two syscalls, so anything above a few
# milliseconds is already generous; the ceiling only bounds the pathological
# case of a machine so loaded that a reader is descheduled mid-probe.
_ACQUIRE_TIMEOUT = 2.0
_ACQUIRE_INTERVAL = 0.01


# The lock file's name. An orchestrator owns an attempt dir; a sweep's
# dispatcher owns the sweep's dir and takes the same kind of lock under its
# own name, so that one reading of "is it alive" serves both.
ORCHESTRATOR = "orchestrator.lock"
DISPATCHER = "dispatcher.lock"


def path(attempt_dir: pathlib.Path, name: str = ORCHESTRATOR) -> pathlib.Path:
    """Host path of the attempt's orchestrator lock."""
    return attempt_dir / name


def hold(attempt_dir: pathlib.Path, name: str = ORCHESTRATOR) -> typing.BinaryIO:
    """Take the attempt's lock, or raise if another orchestrator holds it.

    The caller must keep the returned file open for as long as it wants the
    lock: closing it -- or dropping the last reference and letting the garbage
    collector close it -- releases the lock immediately.

    The contents are for whoever is reading the attempt dir by hand; nothing
    reads them back, because the lock itself carries the answer. They are
    written after the lock is taken, which is what makes an empty file mean
    "being created" to a reader.
    """
    # O_CREAT|O_RDWR rather than "wb": opening for write would truncate the
    # incumbent's file before we even learn the lock is taken.
    fd = os.open(path(attempt_dir, name), os.O_CREAT | os.O_RDWR, 0o644)
    f = os.fdopen(fd, "r+b")

    deadline = time.time() + _ACQUIRE_TIMEOUT
    while True:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if time.time() >= deadline:
                f.close()
                raise exceptions.UI(f"an orchestrator is already running for {attempt_dir}")
            time.sleep(_ACQUIRE_INTERVAL)

    f.truncate(0)
    f.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}).encode())
    f.flush()
    return f


def is_held(attempt_dir: pathlib.Path, fallback_pid: int | None, name: str = ORCHESTRATOR) -> bool:
    """Whether an orchestrator is alive and owns this attempt.

    `fallback_pid` is the pid the parent recorded when it spawned the
    orchestrator, for the two cases the lock cannot answer: an attempt started
    before the lock file existed, and the moment between the file being created
    and the lock being taken. Reporting either as dead would finalize a run
    that is still going. Pass None when there is nothing to fall back to.
    """
    try:
        f = open(path(attempt_dir, name), "rb")
    except OSError:
        # No lock file: either an attempt predating it, or one whose
        # orchestrator has not started yet.
        return _pid_is_alive(fallback_pid)

    try:
        if os.fstat(f.fileno()).st_size == 0:
            # Created but not yet written, so not yet locked either.
            return _pid_is_alive(fallback_pid)
        try:
            # Shared, so concurrent readers do not exclude one another and a
            # probe cannot make the orchestrator's own acquire fail.
            fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        f.close()


def _pid_is_alive(pid: int | None) -> bool:
    """Answers "a process with this id exists", which a recycled pid satisfies.

    Only ever a fallback -- see the module docstring for why it is not good
    enough on its own.
    """
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
