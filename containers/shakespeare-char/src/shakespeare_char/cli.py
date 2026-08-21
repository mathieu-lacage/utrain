"""The utrain container CLI: argument parsing, dispatch, and exit codes.

Everything else in this package is importable and callable without going
through argv; this module is the only place that knows about subcommands.
"""

import argparse
import collections.abc
import json
import pathlib
import sys

import torch

from . import config, describe, paths, pretrain, serve, tokenizer

# The phases utrain may ask for, normalized to one signature so dispatch is a
# lookup rather than an if-chain. Each returns True when the phase succeeded.
_PHASES: dict[str, collections.abc.Callable[[paths.Paths, str, dict[str, object]], bool]] = {
    "tokenizer": lambda p, run_id, cfg: tokenizer.run(p, run_id),
    "pretrain": pretrain.run,
}


def cmd_describe() -> None:
    print(json.dumps(describe.DESCRIBE))


def cmd_check_compat() -> None:
    cuda_ok = torch.cuda.is_available()
    details = f"CUDA available: {cuda_ok}. Preset runs on CPU or GPU."
    print(json.dumps({"compatible": True, "details": details}))


def cmd_run(p: paths.Paths, phase: str) -> None:
    cfg = config.flatten(config.load(p), phase)
    run_id = config.get_str(cfg, "run_id", "")
    entry = _PHASES.get(phase)
    if entry is None:
        print(f"unknown phase: {phase}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0 if entry(p, run_id, cfg) else 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    # Where the utrain contract tree lives. utrain always passes /utrain; the
    # default lets the phases be run straight from a checkout, no image needed.
    parser.add_argument(
        "--utrain-root",
        type=pathlib.Path,
        default=pathlib.Path.cwd() / paths.DEFAULT_ROOT,
        help=f"root of the utrain filesystem contract (default: ./{paths.DEFAULT_ROOT})",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("describe")
    sub.add_parser("check-compat")
    run_p = sub.add_parser("run")
    run_p.add_argument("--phase", required=True)
    serve_p = sub.add_parser("serve")
    serve_p.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()
    if args.cmd == "describe":
        cmd_describe()
    elif args.cmd == "check-compat":
        cmd_check_compat()
    else:
        p = paths.Paths(args.utrain_root)
        p.ensure()
        if args.cmd == "run":
            cmd_run(p, args.phase)
        elif args.cmd == "serve":
            serve.serve(p, args.port)


if __name__ == "__main__":
    main()
