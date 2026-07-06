import json
import pathlib
import subprocess

from . import schema


def _run_cmd(image: str, args: list[str], timeout: int = 60) -> str:
    result = subprocess.run(
        ["enroot", "start", image, "--"] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"enroot command failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def describe(image: str) -> schema.DescribeOutput:
    out = _run_cmd(image, ["describe"])
    return schema.DescribeOutput.model_validate(json.loads(out))


def check_compat(image: str) -> schema.CompatResult:
    out = _run_cmd(image, ["check-compat"], timeout=120)
    return schema.CompatResult.model_validate(json.loads(out))


def start_run(image: str, run_dir: pathlib.Path) -> subprocess.Popen[bytes]:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = open(log_dir / "stdout.log", "wb")
    stderr = open(log_dir / "stderr.log", "wb")
    return subprocess.Popen(
        ["enroot", "start", "--mount", f"{run_dir}:/run", image, "--", "run", "/run"],
        stdout=stdout,
        stderr=stderr,
    )


def start_serve(image: str, run_dir: pathlib.Path, port: int) -> subprocess.Popen[bytes]:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = open(log_dir / "serve_stdout.log", "wb")
    stderr = open(log_dir / "serve_stderr.log", "wb")
    return subprocess.Popen(
        [
            "enroot",
            "start",
            "--mount",
            f"{run_dir}:/run",
            image,
            "--",
            "serve",
            "/run",
            "--port",
            str(port),
        ],
        stdout=stdout,
        stderr=stderr,
    )
