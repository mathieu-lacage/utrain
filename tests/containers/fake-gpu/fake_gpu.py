#!/usr/bin/env python3
"""Trivial container whose single phase proves the host GPU is visible inside it."""

import argparse
import json
import pathlib
import subprocess
import sys

DESCRIBE = {
    "name": "fake-gpu",
    "version": "1.0.0",
    "phases": [{"name": "gpu-check", "label": "GPU Check"}],
    "phase_order": ["gpu-check"],
}


def cmd_describe() -> None:
    print(json.dumps(DESCRIBE))


def cmd_run(phase: str) -> None:
    if phase != "gpu-check":
        print(f"unknown phase: {phase}", file=sys.stderr)
        sys.exit(1)
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("describe")
    run_p = sub.add_parser("run")
    run_p.add_argument("run_dir", type=pathlib.Path)
    run_p.add_argument("--phase", required=True)

    args = parser.parse_args()
    if args.cmd == "describe":
        cmd_describe()
    elif args.cmd == "run":
        cmd_run(args.phase)


if __name__ == "__main__":
    main()
