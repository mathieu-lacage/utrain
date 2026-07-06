#!/usr/bin/env python3
"""Fake training container implementing the utrain container contract."""

import argparse
import http.server
import json
import math
import pathlib
import random
import sqlite3
import sys
import time

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


def _init_db(db_path: pathlib.Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE IF NOT EXISTS phase_events (phase TEXT, event TEXT, timestamp REAL)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS metrics "
        "(step INTEGER, timestamp REAL, phase TEXT, name TEXT, value REAL)"
    )
    conn.commit()
    return conn


def _read_control(control_path: pathlib.Path) -> str:
    try:
        data = json.loads(control_path.read_text())
        return str(data.get("action", "continue"))
    except Exception:
        return "continue"


def _run_tokenizer(
    conn: sqlite3.Connection,
    run_dir: pathlib.Path,
    total_steps: int = 50,
) -> bool:
    conn.execute("INSERT INTO phase_events VALUES (?, ?, ?)", ("tokenizer", "started", time.time()))
    conn.commit()
    for step in range(total_steps):
        if _read_control(run_dir / "control.json") == "stop":
            conn.execute(
                "INSERT INTO phase_events VALUES (?, ?, ?)",
                ("tokenizer", "failed", time.time()),
            )
            conn.commit()
            return False
        vocab_coverage = 0.5 + 0.5 * (1 - math.exp(-step / 20))
        conn.execute(
            "INSERT INTO metrics VALUES (?, ?, ?, ?, ?)",
            (step, time.time(), "tokenizer", "vocab_coverage", vocab_coverage),
        )
        conn.commit()
        time.sleep(0.1)
    conn.execute(
        "INSERT INTO phase_events VALUES (?, ?, ?)", ("tokenizer", "completed", time.time())
    )
    conn.commit()
    return True


def _run_pretrain(
    conn: sqlite3.Connection,
    run_dir: pathlib.Path,
    total_steps: int = 200,
) -> bool:
    conn.execute("INSERT INTO phase_events VALUES (?, ?, ?)", ("pretrain", "started", time.time()))
    conn.commit()
    for step in range(total_steps):
        if _read_control(run_dir / "control.json") == "stop":
            conn.execute(
                "INSERT INTO phase_events VALUES (?, ?, ?)",
                ("pretrain", "failed", time.time()),
            )
            conn.commit()
            return False
        # Fake metrics: loss decays, bpb decays, mfu ramps, gpu power steady
        t = step / total_steps
        loss = 3.5 * math.exp(-2.5 * t) + 1.2 + random.gauss(0, 0.05)
        bpb = loss / math.log(2)
        mfu = 0.35 * (1 - math.exp(-5 * t)) + random.gauss(0, 0.005)
        gpu_power = 280 + random.gauss(0, 5)
        for name, value in [("loss", loss), ("bpb", bpb), ("mfu", mfu), ("gpu_power_w", gpu_power)]:
            conn.execute(
                "INSERT INTO metrics VALUES (?, ?, ?, ?, ?)",
                (step, time.time(), "pretrain", name, value),
            )
        conn.commit()
        time.sleep(0.1)
    conn.execute(
        "INSERT INTO phase_events VALUES (?, ?, ?)", ("pretrain", "completed", time.time())
    )
    conn.commit()
    return True


def cmd_run(run_dir: pathlib.Path) -> None:
    db_path = run_dir / "metrics.db"
    conn = _init_db(db_path)
    ok = _run_tokenizer(conn, run_dir)
    if ok:
        ok = _run_pretrain(conn, run_dir)
    conn.close()
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
    serve_p = sub.add_parser("serve")
    serve_p.add_argument("run_dir", type=pathlib.Path)
    serve_p.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()
    if args.cmd == "describe":
        cmd_describe()
    elif args.cmd == "check-compat":
        cmd_check_compat()
    elif args.cmd == "run":
        cmd_run(args.run_dir)
    elif args.cmd == "serve":
        cmd_serve(args.run_dir, args.port)


if __name__ == "__main__":
    main()
