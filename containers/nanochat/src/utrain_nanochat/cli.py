"""The utrain container CLI: argument parsing, dispatch, and exit codes.

Everything else in this package is importable and callable without going
through argv; this module is the only place that knows about subcommands.

There is no `_MANIFESTS` table here, unlike shakespeare-char's: no nanochat
phase is cacheable, so `check-cache` refuses every phase and utrain reads that
as a plain cache miss. See `describe.py`.
"""

import argparse
import collections.abc
import json
import pathlib
import sys

from . import config, describe, download, paths, pretrain, rl, serve, sft, tokenizer

# The phases utrain may ask for, normalized to one signature so dispatch is a
# lookup rather than an if-chain. Each returns True when the phase succeeded.
_PHASES: dict[str, collections.abc.Callable[[paths.Paths, str, dict[str, object]], bool]] = {
    "download": download.run,
    "tokenizer": tokenizer.run,
    "pretrain": pretrain.run,
    "sft": sft.run,
    "rl": rl.run,
}


def cmd_describe() -> None:
    print(json.dumps(describe.DESCRIBE))


def cmd_check_compat() -> None:
    # Imported here rather than at module scope: `describe` is on utrain's
    # critical path (it runs it under a timeout right after `image add`) and has
    # no use for torch, which takes seconds to import.
    import torch

    cuda_ok = torch.cuda.is_available()
    details = (
        f"CUDA available: {cuda_ok}. nanochat trains on a single GPU here -- utrain grants "
        "one device per run, so the 8xGPU speedrun settings are out of reach and the "
        "defaults are sized accordingly. A CPU run works but is impractically slow."
    )
    print(json.dumps({"compatible": True, "details": details}))


def cmd_check_cache(phase: str) -> None:
    print(f"phase not cacheable: {phase}", file=sys.stderr)
    sys.exit(1)


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
    parser.add_argument(
        "--utrain-root",
        type=pathlib.Path,
        default=pathlib.Path.cwd() / paths.DEFAULT_ROOT,
        help=f"root of the utrain filesystem contract (default: ./{paths.DEFAULT_ROOT})",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("describe")
    sub.add_parser("check-compat")
    check_cache_p = sub.add_parser("check-cache")
    check_cache_p.add_argument("--phase", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--phase", required=True)
    serve_p = sub.add_parser("serve")
    serve_p.add_argument("--port", type=int, default=0)
    # Required: three phases here leave a model behind, so there is no phase
    # this image could sensibly default to.
    serve_p.add_argument("--phase", required=True)

    args = parser.parse_args()
    if args.cmd == "describe":
        cmd_describe()
    elif args.cmd == "check-compat":
        cmd_check_compat()
    elif args.cmd == "check-cache":
        cmd_check_cache(args.phase)
    else:
        p = paths.Paths(args.utrain_root)
        if args.cmd == "run":
            p.ensure()
            cmd_run(p, args.phase)
        elif args.cmd == "serve":
            serve.serve(p, args.port, args.phase)


if __name__ == "__main__":
    main()
