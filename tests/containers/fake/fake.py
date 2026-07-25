#!/usr/bin/env python3
"""Fake training container implementing the utrain container contract."""

import argparse
import http.server
import json
import math
import pathlib
import random
import sys
import time

import baw.wandb
import yaml

DESCRIBE = {
    "name": "nanochat-d12-english",
    "version": "1.0.0",
    "phases": [
        {"name": "tokenizer", "label": "Tokenizer Training"},
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


def _read_control(control_path: pathlib.Path) -> str:
    try:
        data = json.loads(control_path.read_text())
        return str(data.get("action", "continue"))
    except Exception:
        return "continue"


def _run_tokenizer(run_dir: pathlib.Path, run_id: str, total_steps: int = 50) -> bool:
    run = baw.wandb.init(project="tokenizer", id=run_id, dir=str(run_dir))
    run.log({"_phase_event": "tokenizer/started"})
    ok = True
    for step in range(total_steps):
        if _read_control(run_dir / "control.json") == "stop":
            run.log({"_phase_event": "tokenizer/failed"})
            ok = False
            break
        vocab_coverage = 0.5 + 0.5 * (1 - math.exp(-step / 20))
        run.log({"vocab_coverage": vocab_coverage}, step=step, commit=True)
        time.sleep(0.1)
    if ok:
        run.log({"_phase_event": "tokenizer/completed"})
    run.finish(exit_code=0 if ok else 1)
    return ok


def _run_pretrain(run_dir: pathlib.Path, run_id: str, total_steps: int = 200) -> bool:
    run = baw.wandb.init(project="pretrain", id=run_id, dir=str(run_dir))
    run.log({"_phase_event": "pretrain/started"})
    ok = True
    for step in range(total_steps):
        if _read_control(run_dir / "control.json") == "stop":
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
        run.log({"_phase_event": "pretrain/completed"})
    run.finish(exit_code=0 if ok else 1)
    return ok


def cmd_run(run_dir: pathlib.Path, phase: str) -> None:
    cfg: dict[str, object] = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
    run_id = str(cfg.get("run_id", ""))
    if phase == "tokenizer":
        ok = _run_tokenizer(run_dir, run_id)
    elif phase == "pretrain":
        ok = _run_pretrain(run_dir, run_id)
    else:
        print(f"unknown phase: {phase}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0 if ok else 1)


def cmd_run_all(run_dir: pathlib.Path) -> None:
    cfg: dict[str, object] = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
    run_id = str(cfg.get("run_id", ""))
    for phase in DESCRIBE["phase_order"]:
        if _read_control(run_dir / "control.json") == "stop":
            sys.exit(1)
        if phase == "tokenizer":
            ok = _run_tokenizer(run_dir, run_id)
        elif phase == "pretrain":
            ok = _run_pretrain(run_dir, run_id)
        else:
            print(f"unknown phase: {phase}", file=sys.stderr)
            sys.exit(1)
        if not ok:
            sys.exit(1)
    sys.exit(0)


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


def cmd_serve(run_dir: pathlib.Path, port: int) -> None:
    server = http.server.HTTPServer(("", port), _ChatHandler)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("describe")
    sub.add_parser("check-compat")
    run_p = sub.add_parser("run")
    run_p.add_argument("run_dir", type=pathlib.Path)
    run_p.add_argument("--phase", required=True)
    run_all_p = sub.add_parser("run-all")
    run_all_p.add_argument("run_dir", type=pathlib.Path)
    serve_p = sub.add_parser("serve")
    serve_p.add_argument("run_dir", type=pathlib.Path)
    serve_p.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()
    if args.cmd == "describe":
        cmd_describe()
    elif args.cmd == "check-compat":
        cmd_check_compat()
    elif args.cmd == "run":
        cmd_run(args.run_dir, args.phase)
    elif args.cmd == "run-all":
        cmd_run_all(args.run_dir)
    elif args.cmd == "serve":
        cmd_serve(args.run_dir, args.port)


if __name__ == "__main__":
    main()
