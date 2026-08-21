#!/usr/bin/env python3
"""Fake training container implementing the utrain container contract."""

import argparse
import hashlib
import http.server
import json
import math
import pathlib
import random
import sys
import time

import wandb
import yaml

# The utrain filesystem contract: a single mount root, whose layout is fixed, so
# a container hardcodes these rather than receiving them. RUN_DIR is read-only
# (config.yaml, control.json); RUN_DIR/data and RUN_DIR/wandb are writable.
RUN_DIR = pathlib.Path("/utrain")
DATA_DIR = RUN_DIR / "data"

# Kept in sync with what _run_tokenizer actually writes -- manifest declares
# these without running the phase, so a real container would compute this
# deterministically from its config/inputs instead of hardcoding it.
_TOKENIZER_FILES = {
    "tokenizer.txt": "tokenizer output\n",
    "common.txt": "shared payload\n",
}

DESCRIBE = {
    "name": "nanochat-d12-english",
    "version": "1.0.0",
    "phases": [
        {"name": "tokenizer", "label": "Tokenizer Training", "cacheable": True},
        {"name": "pretrain", "label": "Pre-Training"},
    ],
    "phase_order": ["tokenizer", "pretrain"],
    "config_schema": {
        "globals": {
            "groups": [
                {
                    "name": "model",
                    "label": "Model Architecture",
                    "fields": [
                        {
                            "key": "num_layers",
                            "label": "Layers",
                            "type": "int",
                            "default": 12,
                            "min": 1,
                            "max": 48,
                            "description": "Transformer depth",
                        }
                    ],
                }
            ]
        },
        "phases": {
            "pretrain": {
                "groups": [
                    {
                        "name": "training",
                        "label": "Training",
                        "fields": [
                            {
                                "key": "batch_size",
                                "label": "Batch Size",
                                "type": "int",
                                "default": 32,
                            },
                            {
                                "key": "learning_rate",
                                "label": "Learning Rate",
                                "type": "float",
                                "default": 3e-4,
                            },
                        ],
                    }
                ]
            }
        },
    },
    "can_serve": True,
}


def cmd_describe() -> None:
    print(json.dumps(DESCRIBE))


def cmd_check_compat() -> None:
    print(json.dumps({"compatible": True, "details": "fake GPU ok (always compatible)"}))


def _write_phase_data(files: dict[str, str]) -> None:
    """Write phase output files into the data mount (if mounted)."""
    if not DATA_DIR.exists():
        return
    for name, content in files.items():
        (DATA_DIR / name).write_text(content)


def _read_control() -> str:
    try:
        data = json.loads((RUN_DIR / "control.json").read_text())
        return str(data.get("action", "continue"))
    except Exception:
        return "continue"


def _flatten_config(cfg: dict[str, object], phase: str) -> dict[str, object]:
    """Merge the utrain config into the flat namespace this phase reads.

    utrain writes config.yaml with keys grouped under a ``globals`` section and
    a per-phase ``phases.<phase>`` section. Within a section a value may be a
    field group (a dict, e.g. ``globals.model``) whose members are the real
    keys, or a bare field. A container must merge, for the phase it runs:
    top-level scalars + ``globals`` + ``phases.<phase>``, expanding any group
    dict one level (phase values win over globals on collision).
    """
    flat: dict[str, object] = {k: v for k, v in cfg.items() if k not in ("globals", "phases")}

    def _merge(section: object) -> None:
        if not isinstance(section, dict):
            return
        for key, value in section.items():
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value

    _merge(cfg.get("globals"))
    phases_section = cfg.get("phases")
    if isinstance(phases_section, dict):
        _merge(phases_section.get(phase))

    return flat


def _wait_for_gate(cfg: dict[str, object], timeout: float = 120.0) -> None:
    """Block after the started event until the host creates `RUN_DIR/gate`.

    Opt-in via a `gate: true` config key. It lets a test observe a phase in
    `running` (and its successors in `pending`) deterministically, instead of
    racing the phase's own duration -- on a loaded CI runner the 5s tokenizer
    can finish before the next `utrain` invocation gets to look at it. The
    timeout only exists so a test that dies before releasing the gate leaves no
    container behind.
    """
    if not cfg.get("gate"):
        return
    gate = RUN_DIR / "gate"
    deadline = time.monotonic() + timeout
    while not gate.exists() and time.monotonic() < deadline:
        time.sleep(0.05)


def _run_tokenizer(run_id: str, cfg: dict[str, object], total_steps: int = 50) -> bool:
    run = wandb.init(project="tokenizer", id=run_id, dir=str(RUN_DIR))
    run.log({"_phase_event": "tokenizer/started"})
    _wait_for_gate(cfg)
    ok = True
    for step in range(total_steps):
        if _read_control() == "stop":
            run.log({"_phase_event": "tokenizer/failed"})
            ok = False
            break
        vocab_coverage = 0.5 + 0.5 * (1 - math.exp(-step / 20))
        run.log({"vocab_coverage": vocab_coverage}, step=step, commit=True)
        time.sleep(0.1)
    if ok:
        _write_phase_data(_TOKENIZER_FILES)
        run.log({"_phase_event": "tokenizer/completed"})
    run.finish(exit_code=0 if ok else 1)
    return ok


def _run_pretrain(run_id: str, cfg: dict[str, object], total_steps: int = 200) -> bool:
    run = wandb.init(project="pretrain", id=run_id, dir=str(RUN_DIR))
    run.log({"_phase_event": "pretrain/started"})
    # Echo the effective config so the e2e suite can verify the utrain config
    # protocol: `num_layers` comes from `globals`, `batch_size` from this phase.
    num_layers = int(cfg.get("num_layers", 12))
    batch_size = int(cfg.get("batch_size", 32))
    print(f"config: num_layers={num_layers} batch_size={batch_size}", flush=True)
    ok = True
    for step in range(total_steps):
        if _read_control() == "stop":
            run.log({"_phase_event": "pretrain/failed"})
            ok = False
            break
        t = step / total_steps
        loss = 3.5 * math.exp(-2.5 * t) + 1.2 + random.gauss(0, 0.05)
        bpb = loss / math.log(2)
        mfu = 0.35 * (1 - math.exp(-5 * t)) + random.gauss(0, 0.005)
        gpu_power = 280 + random.gauss(0, 5)
        run.log(
            {"loss": loss, "bpb": bpb, "mfu": mfu, "gpu_power_w": gpu_power},
            step=step,
            commit=True,
        )
        time.sleep(0.1)
    if ok:
        _write_phase_data({"pretrain.txt": "pretrain output\n", "common2.txt": "shared payload\n"})
        run.log({"_phase_event": "pretrain/completed"})
    run.finish(exit_code=0 if ok else 1)
    return ok


def cmd_check_cache(phase: str) -> None:
    if phase != "tokenizer":
        print(f"phase not cacheable: {phase}", file=sys.stderr)
        sys.exit(1)
    files = [
        {"path": name, "sha256": hashlib.sha256(content.encode()).hexdigest()}
        for name, content in _TOKENIZER_FILES.items()
    ]
    print(json.dumps({"files": files}))


def cmd_run(phase: str) -> None:
    raw_cfg: dict[str, object] = yaml.safe_load((RUN_DIR / "config.yaml").read_text()) or {}
    cfg = _flatten_config(raw_cfg, phase)
    run_id = str(cfg.get("run_id", ""))
    if phase == "tokenizer":
        ok = _run_tokenizer(run_id, cfg)
    elif phase == "pretrain":
        ok = _run_pretrain(run_id, cfg)
    else:
        print(f"unknown phase: {phase}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0 if ok else 1)


class _ChatHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        if self.path != "/chat":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()
        try:
            msg = json.loads(body).get("message", "")
        except Exception:
            msg = ""
        reply = json.dumps({"reply": f"[fake model] You said: {msg!r}"})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply.encode())


def cmd_serve(port: int) -> None:
    server = http.server.HTTPServer(("", port), _ChatHandler)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("describe")
    sub.add_parser("check-compat")
    check_cache_p = sub.add_parser("check-cache")
    check_cache_p.add_argument("--phase", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--phase", required=True)
    serve_p = sub.add_parser("serve")
    serve_p.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()
    if args.cmd == "describe":
        cmd_describe()
    elif args.cmd == "check-compat":
        cmd_check_compat()
    elif args.cmd == "check-cache":
        cmd_check_cache(args.phase)
    elif args.cmd == "run":
        cmd_run(args.phase)
    elif args.cmd == "serve":
        cmd_serve(args.port)


if __name__ == "__main__":
    main()
