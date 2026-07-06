import json
import os
import pathlib
import signal
import subprocess

_active: dict[str, int] = {}
_serve_active: dict[str, int] = {}


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def register_run(run_id: str, proc: subprocess.Popen[bytes]) -> None:
    _active[run_id] = proc.pid


def register_run_pid(run_id: str, pid: int) -> None:
    _active[run_id] = pid


def register_serve(run_id: str, proc: subprocess.Popen[bytes]) -> None:
    _serve_active[run_id] = proc.pid


def get_pid(run_id: str) -> int | None:
    return _active.get(run_id)


def finish_run(run_id: str) -> None:
    _active.pop(run_id, None)


def write_control(run_dir: pathlib.Path, action: str) -> None:
    control_path = run_dir / "control.json"
    control_path.write_text(json.dumps({"action": action}))


def stop_run(run_id: str, run_dir: pathlib.Path) -> None:
    write_control(run_dir, "stop")
    pid = _active.get(run_id)
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def stop_serve(run_id: str) -> None:
    pid = _serve_active.get(run_id)
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        del _serve_active[run_id]
