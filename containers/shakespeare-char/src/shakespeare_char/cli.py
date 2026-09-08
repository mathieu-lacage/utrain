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

from . import config, describe, download, paths, pretrain, serve, tokenizer

# The phases utrain may ask for, normalized to one signature so dispatch is a
# lookup rather than an if-chain. Each returns True when the phase succeeded.
_PHASES: dict[str, collections.abc.Callable[[paths.Paths, str, dict[str, object]], bool]] = {
    "download": lambda p, run_id, cfg: download.run(p, run_id),
    "tokenizer": lambda p, run_id, cfg: tokenizer.run(p, run_id),
    "pretrain": pretrain.run,
}

# Phases that can answer `check-cache`, mapped to the manifest they would
# produce. Keep in sync with `cacheable` in describe.DESCRIBE.
_MANIFESTS: dict[str, collections.abc.Callable[[], dict[str, object]]] = {
    "download": download.manifest,
}


def cmd_describe() -> None:
    print(json.dumps(describe.DESCRIBE))


def cmd_check_compat() -> None:
    cuda_ok = torch.cuda.is_available()
    details = f"CUDA available: {cuda_ok}. Preset runs on CPU or GPU."
    print(json.dumps({"compatible": True, "details": details}))


def cmd_check_cache(phase: str) -> None:
    """Declare a cacheable phase's output without running it.

    utrain treats any non-zero exit as a plain cache miss, so refusing an
    uncacheable phase here costs nothing but says what happened.
    """
    entry = _MANIFESTS.get(phase)
    if entry is None:
        print(f"phase not cacheable: {phase}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(entry()))


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
    # utrain always passes --phase. `pretrain` is the only phase here that
    # leaves a model behind, so it is also the default for a hand-run serve.
    serve_p.add_argument("--phase", default=serve.SERVABLE_PHASE)

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
