#!/usr/bin/env python3
"""Fake training container implementing the utrain container contract."""

import argparse
import collections.abc
import dataclasses
import hashlib
import json
import math
import pathlib
import random
import socket
import sys
import time

import fastapi
import fastapi.exceptions
import fastapi.responses
import pydantic
import uvicorn
import wandb
import yaml

# The utrain filesystem contract: a single root whose layout is fixed. The root
# holds the read-only config.yaml and control.json; `data/` and `wandb/` beneath
# it are writable. The root itself comes in as `--utrain-root` (utrain mounts
# everything at /utrain) and defaults to $CWD/run so the phases can be run
# without building an image.
DEFAULT_ROOT = "run"


@dataclasses.dataclass(frozen=True)
class Paths:
    run_dir: pathlib.Path

    @property
    def data_dir(self) -> pathlib.Path:
        return self.run_dir / "data"

    @property
    def serve_dir(self) -> pathlib.Path:
        """Writable, and mounted only for `serve`. Holds `port.json`."""
        return self.run_dir / "serve"

    def ensure(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "wandb").mkdir(parents=True, exist_ok=True)


def write_port_file(p: Paths, port: int) -> None:
    if not p.serve_dir.exists():
        p.serve_dir.mkdir(parents=True, exist_ok=True)
    target = p.serve_dir / "port.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"port": port}))
    tmp.replace(target)


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
        # Named plots: the TUI shows these two instead of one plot per metric.
        # `tokenizer` deliberately names none, so both paths are exercised.
        #
        # `can_serve` likewise: only `pretrain` leaves a model behind, so it is
        # the one phase chat is offered on. `fake-gpu`, which names none, is
        # the image utrain refuses to serve at all.
        {
            "name": "pretrain",
            "label": "Pre-Training",
            "can_serve": True,
            "plots": [{"x": "step", "y": "loss"}, {"x": "step", "y": "mfu"}],
        },
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
}


def cmd_describe() -> None:
    print(json.dumps(DESCRIBE))


def cmd_check_compat() -> None:
    print(json.dumps({"compatible": True, "details": "fake GPU ok (always compatible)"}))


def _write_phase_data(p: Paths, files: dict[str, str]) -> None:
    """Write phase output files into the data dir."""
    for name, content in files.items():
        (p.data_dir / name).write_text(content)


def _read_control(p: Paths) -> str:
    try:
        data = json.loads((p.run_dir / "control.json").read_text())
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


def _wait_for_gate(p: Paths, cfg: dict[str, object], timeout: float = 120.0) -> None:
    """Block after the started event until the host creates `<root>/gate`.

    Opt-in via a `gate: true` config key. It lets a test observe a phase in
    `running` (and its successors in `pending`) deterministically, instead of
    racing the phase's own duration -- on a loaded CI runner the 5s tokenizer
    can finish before the next `utrain` invocation gets to look at it. The
    timeout only exists so a test that dies before releasing the gate leaves no
    container behind.
    """
    if not cfg.get("gate"):
        return
    gate = p.run_dir / "gate"
    deadline = time.monotonic() + timeout
    while not gate.exists() and time.monotonic() < deadline:
        time.sleep(0.05)


def _run_tokenizer(p: Paths, cfg: dict[str, object], total_steps: int = 50) -> bool:
    run = wandb.init()
    _wait_for_gate(p, cfg)
    ok = True
    for step in range(total_steps):
        if _read_control(p) == "stop":
            ok = False
            break
        vocab_coverage = 0.5 + 0.5 * (1 - math.exp(-step / 20))
        run.log({"vocab_coverage": vocab_coverage}, step=step, commit=True)
        time.sleep(0.1)
    if ok:
        _write_phase_data(p, _TOKENIZER_FILES)
    run.finish(exit_code=0 if ok else 1)
    return ok


def _run_pretrain(p: Paths, cfg: dict[str, object], total_steps: int = 200) -> bool:
    run = wandb.init()
    # Echo the effective config so the e2e suite can verify the utrain config
    # protocol: `num_layers` comes from `globals`, `batch_size` from this phase.
    num_layers = int(cfg.get("num_layers", 12))
    batch_size = int(cfg.get("batch_size", 32))
    print(f"config: num_layers={num_layers} batch_size={batch_size}", flush=True)
    # One line on stderr, so the merged output log has both streams in it and
    # the e2e suite can see them interleaved in `run logs`.
    print("pretrain: note on stderr", file=sys.stderr)
    ok = True
    for step in range(total_steps):
        if _read_control(p) == "stop":
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
        _write_phase_data(
            p, {"pretrain.txt": "pretrain output\n", "common2.txt": "shared payload\n"}
        )
    run.finish(exit_code=0 if ok else 1)
    return ok


def cmd_check_cache(p: Paths, phase: str) -> None:
    if phase != "tokenizer":
        print(f"phase not cacheable: {phase}", file=sys.stderr)
        sys.exit(1)
    # `check_cache_hang: true` in the run config makes this never answer, which
    # is what utrain's manifest timeout is for. A test uses it to prove the
    # container does not outlive the timeout that killed the podman client.
    cfg_path = p.run_dir / "config.yaml"
    if cfg_path.exists() and (yaml.safe_load(cfg_path.read_text()) or {}).get("check_cache_hang"):
        time.sleep(3600)
    files = [
        {"path": name, "sha256": hashlib.sha256(content.encode()).hexdigest()}
        for name, content in _TOKENIZER_FILES.items()
    ]
    print(json.dumps({"files": files}))


def cmd_run(p: Paths, phase: str) -> None:
    # Missing config.yaml -> the phase's own defaults, so a bare local run works.
    cfg_path = p.run_dir / "config.yaml"
    raw_cfg: dict[str, object] = (
        (yaml.safe_load(cfg_path.read_text()) or {}) if cfg_path.exists() else {}
    )
    cfg = _flatten_config(raw_cfg, phase)
    if phase == "tokenizer":
        ok = _run_tokenizer(p, cfg)
    elif phase == "pretrain":
        ok = _run_pretrain(p, cfg)
    else:
        print(f"unknown phase: {phase}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0 if ok else 1)


class ChatMessage(pydantic.BaseModel):
    role: str
    content: str = ""


class ChatCompletionRequest(pydantic.BaseModel):
    # Extra fields are accepted and ignored: real clients send plenty this
    # container has no use for, and refusing them would break callers.
    model_config = pydantic.ConfigDict(extra="allow")

    messages: list[ChatMessage] = pydantic.Field(min_length=1)
    model: str | None = None
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None


class Choice(pydantic.BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(pydantic.BaseModel):
    id: str = "chatcmpl-fake"
    object: str = "chat.completion"
    created: int = 0
    model: str = "fake"
    choices: list[Choice]


class Delta(pydantic.BaseModel):
    role: str | None = None
    content: str | None = None


class ChunkChoice(pydantic.BaseModel):
    index: int = 0
    delta: Delta
    finish_reason: str | None = None


class ChatCompletionChunk(pydantic.BaseModel):
    id: str = "chatcmpl-fake"
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = "fake"
    choices: list[ChunkChoice]

    def to_sse(self) -> str:
        return f"data: {self.model_dump_json()}\n\n"


class ModelCard(pydantic.BaseModel):
    id: str = "fake"
    object: str = "model"
    created: int = 0
    owned_by: str = "utrain"
    # Neither is part of the OpenAI schema; the tests read them back to prove
    # utrain's filesystem contract and its `--phase` reached `serve`.
    data_dir: str
    phase: str


class ModelList(pydantic.BaseModel):
    object: str = "list"
    data: list[ModelCard]


class ErrorBody(pydantic.BaseModel):
    message: str
    type: str = "invalid_request_error"


class ErrorResponse(pydantic.BaseModel):
    error: ErrorBody


router = fastapi.APIRouter()


def _validation_error(request: fastapi.Request, exc: Exception) -> fastapi.responses.JSONResponse:
    """FastAPI answers 422 {"detail": ...}; the contract wants 400 + this envelope."""
    errors = exc.errors() if isinstance(exc, fastapi.exceptions.RequestValidationError) else []
    detail = "; ".join(
        f"{'.'.join(str(part) for part in e['loc'][1:])}: {e['msg']}" for e in errors
    )
    return fastapi.responses.JSONResponse(
        status_code=400,
        content=ErrorResponse(error=ErrorBody(message=detail or "invalid request")).model_dump(),
    )


@router.get("/v1/models", response_model=ModelList)
def list_models(request: fastapi.Request) -> ModelList:
    return ModelList(
        data=[ModelCard(data_dir=request.app.state.data_dir, phase=request.app.state.phase)]
    )


@router.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> fastapi.Response:
    # The API is stateless, so the turn count comes from the history the client
    # resent rather than from anything held here.
    turns = sum(1 for m in req.messages if m.role == "user")
    reply = f"[fake model] turn {turns}: {req.messages[-1].content!r}"

    if not req.stream:
        return fastapi.responses.JSONResponse(
            ChatCompletionResponse(
                choices=[Choice(message=ChatMessage(role="assistant", content=reply))]
            ).model_dump()
        )

    def events() -> collections.abc.Iterator[str]:
        def chunk(delta: Delta, finish: str | None) -> str:
            return ChatCompletionChunk(
                choices=[ChunkChoice(delta=delta, finish_reason=finish)]
            ).to_sse()

        yield chunk(Delta(role="assistant"), None)
        # Split across several chunks so a test can see this streams rather than
        # arriving in one piece.
        for i in range(0, len(reply), 8):
            yield chunk(Delta(content=reply[i : i + 8]), None)
        yield chunk(Delta(), "stop")
        yield "data: [DONE]\n\n"

    return fastapi.responses.StreamingResponse(events(), media_type="text/event-stream")


def cmd_serve(p: Paths, port: int, phase: str) -> None:
    # Whatever utrain says, unchecked: a real container maps the phase to a
    # checkpoint, but this one has no model, and the tests serve phases the
    # image does not declare to prove the flag is passed through verbatim.
    app = fastapi.FastAPI(title="fake")
    app.state.data_dir = str(p.data_dir)
    app.state.phase = phase
    app.add_exception_handler(fastapi.exceptions.RequestValidationError, _validation_error)
    app.include_router(router)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen()
    write_port_file(p, sock.getsockname()[1])

    server = uvicorn.Server(uvicorn.Config(app, access_log=True, log_level="info"))
    try:
        server.run(sockets=[sock])
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--utrain-root",
        type=pathlib.Path,
        default=pathlib.Path.cwd() / DEFAULT_ROOT,
        help=f"root of the utrain filesystem contract (default: ./{DEFAULT_ROOT})",
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
    # Defaulted rather than required: only `pretrain` is servable here, so a
    # hand-run `serve` has nothing else it could mean.
    serve_p.add_argument("--phase", default="pretrain")

    args = parser.parse_args()
    if args.cmd == "describe":
        cmd_describe()
    elif args.cmd == "check-compat":
        cmd_check_compat()
    elif args.cmd == "check-cache":
        # No ensure(): check-cache gets <root>/data read-only, and writing to
        # it is exactly what the contract forbids here.
        cmd_check_cache(Paths(args.utrain_root), args.phase)
    elif args.cmd == "run":
        p = Paths(args.utrain_root)
        p.ensure()
        cmd_run(p, args.phase)
    elif args.cmd == "serve":
        # No ensure(): <root>/data is mounted read-only for serve.
        cmd_serve(Paths(args.utrain_root), args.port, args.phase)


if __name__ == "__main__":
    main()
