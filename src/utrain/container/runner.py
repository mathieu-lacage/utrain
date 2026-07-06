import json
import pathlib
import subprocess

_active: dict[str, subprocess.Popen[bytes]] = {}
_serve_active: dict[str, subprocess.Popen[bytes]] = {}


def register_run(run_id: str, proc: subprocess.Popen[bytes]) -> None:
    _active[run_id] = proc


def register_serve(run_id: str, proc: subprocess.Popen[bytes]) -> None:
    _serve_active[run_id] = proc


def poll_run(run_id: str) -> subprocess.Popen[bytes] | None:
    return _active.get(run_id)


def finish_run(run_id: str) -> None:
    _active.pop(run_id, None)


def write_control(run_dir: pathlib.Path, action: str) -> None:
    control_path = run_dir / "control.json"
    control_path.write_text(json.dumps({"action": action}))


def stop_run(run_id: str, run_dir: pathlib.Path) -> None:
    write_control(run_dir, "stop")
    proc = _active.get(run_id)
    if proc is not None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass


def stop_serve(run_id: str) -> None:
    proc = _serve_active.get(run_id)
    if proc is not None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        del _serve_active[run_id]
