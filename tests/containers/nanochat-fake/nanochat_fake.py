#!/usr/bin/env python3
"""A nanochat-shaped container that fakes the training.

Same contract as `tests/containers/fake`, but wearing the real nanochat image's
face: it serves that container's own `describe.py`, vendored next to this file
from gitlab.inria.fr/mlacage/utrain-nanochat, so its five phases, their plots
and the whole config form are nanochat's own. What it does not have is nanochat -- each phase sleeps
its way through a plausible metric curve instead of training anything, so the
run that takes hours on eight GPUs takes about twenty seconds on any laptop.

This is what `scripts/tour.py` records the TUI against.
"""

import argparse
import collections.abc
import dataclasses
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
import nanochat_describe
import pydantic
import uvicorn
import wandb
import yaml

# The utrain filesystem contract: one root whose layout is fixed. utrain mounts
# everything at /utrain; the default lets the phases be run from a checkout.
DEFAULT_ROOT = "run"

# What `/v1/models` advertises. The context window is nanochat's `max_seq_len`
# default, so the chat banner reads the way the real image's does.
MODEL_ID = "nanochat"
CONTEXT_WINDOW = 2048


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


def cmd_describe() -> None:
    print(json.dumps(nanochat_describe.DESCRIBE))


def cmd_check_compat() -> None:
    print(json.dumps({"compatible": True, "details": "fake nanochat; no GPU required"}))


def cmd_check_cache(phase: str) -> None:
    """Refuse every phase, which is what the real image does.

    nanochat declares nothing `cacheable`: `check-cache` would have to name the
    sha256 of every file a phase writes without doing the work, and its shards
    are not hash-pinned upstream while everything after them is stochastic.
    utrain reads the refusal as a plain cache miss.
    """
    print(f"phase not cacheable: {phase}", file=sys.stderr)
    sys.exit(1)


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


# -- the phases -----------------------------------------------------------
#
# Each is a metric function over `t`, the fraction of the phase that has
# elapsed, plus a line to print every so often and the files it leaves behind.
# `steps` sets both the length of the curve and, at 0.1s a step, how long the
# phase takes -- long enough that the TUI's plots visibly fill, short enough
# that the whole run is over in the time a tour can show it.


@dataclasses.dataclass(frozen=True)
class Phase:
    steps: int
    metrics: collections.abc.Callable[[int, float], dict[str, float]]
    log: collections.abc.Callable[[int, float], str | None]
    outputs: dict[str, str]


def _download_metrics(step: int, t: float) -> dict[str, float]:
    return {"shards": step + 1.0, "chars_millions": 250.0 * (step + 1)}


def _download_log(step: int, t: float) -> str | None:
    if step % 6:
        return None
    return f"fetched shard {step + 1:03d}/030  climbmix  ~250M chars"


def _tokenizer_metrics(step: int, t: float) -> dict[str, float]:
    return {
        "merges": 32768.0 * t,
        "compression_ratio": 2.0 + 2.4 * (1 - math.exp(-3 * t)),
    }


def _tokenizer_log(step: int, t: float) -> str | None:
    if step % 6:
        return None
    return f"bpe merges {int(32768 * t):5d}/32768  compression {2.0 + 2.4 * (1 - t):.3f}"


def _pretrain_metrics(step: int, t: float) -> dict[str, float]:
    loss = 2.4 * math.exp(-3.0 * t) + 1.35 + random.gauss(0, 0.03)
    return {
        "train/loss": loss,
        "val/bpb": loss / math.log(2) * 0.55 + random.gauss(0, 0.01),
        "lrm": max(0.0, 1.0 - t),
        "mfu": 0.42 * (1 - math.exp(-6 * t)) + random.gauss(0, 0.004),
    }


def _pretrain_log(step: int, t: float) -> str | None:
    if step % 8:
        return None
    loss = 2.4 * math.exp(-3.0 * t) + 1.35
    return f"step {step:04d}/0080  loss {loss:.4f}  lrm {max(0.0, 1 - t):.3f}  mfu {0.42 * t:.3f}"


def _sft_metrics(step: int, t: float) -> dict[str, float]:
    return {
        "train/loss": 1.6 * math.exp(-2.2 * t) + 0.85 + random.gauss(0, 0.02),
        "chatcore_metric": 0.07 + 0.11 * (1 - math.exp(-4 * t)) + random.gauss(0, 0.002),
    }


def _sft_log(step: int, t: float) -> str | None:
    if step % 8:
        return None
    loss = 1.6 * math.exp(-2.2 * t) + 0.85
    return f"step {step:04d}/0040  loss {loss:.4f}  mixture smoltalk+mmlu+gsm8k"


def _rl_metrics(step: int, t: float) -> dict[str, float]:
    return {
        "reward": 0.18 + 0.34 * (1 - math.exp(-2.5 * t)) + random.gauss(0, 0.01),
        "pass@1": 0.05 + 0.17 * (1 - math.exp(-2.5 * t)) + random.gauss(0, 0.005),
    }


def _rl_log(step: int, t: float) -> str | None:
    if step % 8:
        return None
    reward, pass1 = 0.18 + 0.34 * t, 0.05 + 0.17 * t
    return f"grpo step {step:04d}/0040  reward {reward:.4f}  gsm8k pass@1 {pass1:.4f}"


PHASES = {
    "download": Phase(30, _download_metrics, _download_log, {"shards.txt": "climbmix\n"}),
    "tokenizer": Phase(30, _tokenizer_metrics, _tokenizer_log, {"tokenizer.pkl": "bpe\n"}),
    "pretrain": Phase(80, _pretrain_metrics, _pretrain_log, {"base_model.pt": "weights\n"}),
    "sft": Phase(40, _sft_metrics, _sft_log, {"sft_model.pt": "weights\n"}),
    "rl": Phase(40, _rl_metrics, _rl_log, {"rl_model.pt": "weights\n"}),
}


def _run_phase(p: Paths, name: str, run_id: str, cfg: dict[str, object]) -> bool:
    phase = PHASES[name]
    run = wandb.init(project=name, id=run_id, dir=str(p.run_dir))
    # Echo what the config form actually decided, so the log pane shows the
    # edit the viewer just made reaching the container.
    print(f"depth={cfg.get('depth')} max_seq_len={cfg.get('max_seq_len')}", flush=True)
    ok = True
    for step in range(phase.steps):
        if _read_control(p) == "stop":
            ok = False
            break
        t = step / phase.steps
        run.log(phase.metrics(step, t), step=step, commit=True)
        line = phase.log(step, t)
        if line is not None:
            print(line, flush=True)
        time.sleep(0.1)
    if ok:
        for filename, content in phase.outputs.items():
            (p.data_dir / filename).write_text(content)
    run.finish(exit_code=0 if ok else 1)
    return ok


def cmd_run(p: Paths, phase: str) -> None:
    if phase not in PHASES:
        print(f"unknown phase: {phase}", file=sys.stderr)
        sys.exit(1)
    # Missing config.yaml -> the phase's own defaults, so a bare local run works.
    cfg_path = p.run_dir / "config.yaml"
    raw_cfg: dict[str, object] = (
        (yaml.safe_load(cfg_path.read_text()) or {}) if cfg_path.exists() else {}
    )
    cfg = _flatten_config(raw_cfg, phase)
    sys.exit(0 if _run_phase(p, phase, str(cfg.get("run_id", "")), cfg) else 1)


# -- serve ----------------------------------------------------------------


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
    model: str = MODEL_ID
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
    model: str = MODEL_ID
    choices: list[ChunkChoice]

    def to_sse(self) -> str:
        return f"data: {self.model_dump_json()}\n\n"


class ModelCard(pydantic.BaseModel):
    id: str = MODEL_ID
    object: str = "model"
    created: int = 0
    owned_by: str = "utrain"
    # Not part of OpenAI's schema; utrain shows it in the chat banner.
    context_window: int = CONTEXT_WINDOW


class ModelList(pydantic.BaseModel):
    object: str = "list"
    data: list[ModelCard]


class ErrorBody(pydantic.BaseModel):
    message: str
    type: str = "invalid_request_error"


class ErrorResponse(pydantic.BaseModel):
    error: ErrorBody


# Canned replies, keyed by a word in the question. A real 8-layer nanochat is
# not eloquent either, so these are written to be about as good as one: short,
# mostly on topic, and visibly a small model.
_REPLIES = {
    "capital": (
        "The capital of France is Paris. It sits on the river Seine in the north "
        "of the country and has been the seat of government since the Middle Ages."
    ),
    "gravity": (
        "Gravity is the attraction between things that have mass. The more mass "
        "something has, the harder it pulls, which is why we fall towards the "
        "Earth and not the other way around."
    ),
    "why": (
        "That is a good question. I am a small model trained on a single GPU, so "
        "I will give you the short version: mostly it comes down to what the "
        "training data happened to contain."
    ),
}
_DEFAULT_REPLY = (
    "I am a very small nanochat model, so I am better at sounding fluent than at "
    "being right. Ask me something simple and I will do my best."
)

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


def _reply_to(question: str) -> str:
    lowered = question.lower()
    for keyword, reply in _REPLIES.items():
        if keyword in lowered:
            return reply
    return _DEFAULT_REPLY


@router.get("/v1/models", response_model=ModelList)
def list_models() -> ModelList:
    return ModelList(data=[ModelCard()])


@router.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> fastapi.Response:
    reply = _reply_to(req.messages[-1].content)

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
        # A token at a time, paced: the tour is a recording of a model
        # answering, and a reply that arrives in one frame does not read as one.
        for word in reply.split(" "):
            yield chunk(Delta(content=word + " "), None)
            time.sleep(0.04)
        yield chunk(Delta(), "stop")
        yield "data: [DONE]\n\n"

    return fastapi.responses.StreamingResponse(events(), media_type="text/event-stream")


def cmd_serve(p: Paths, port: int, phase: str) -> None:
    app = fastapi.FastAPI(title="nanochat-fake")
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
    serve_p.add_argument("--phase", default="sft")

    args = parser.parse_args()
    if args.cmd == "describe":
        cmd_describe()
    elif args.cmd == "check-compat":
        cmd_check_compat()
    elif args.cmd == "check-cache":
        cmd_check_cache(args.phase)
    elif args.cmd == "run":
        p = Paths(args.utrain_root)
        p.ensure()
        cmd_run(p, args.phase)
    elif args.cmd == "serve":
        # No ensure(): <root>/data is mounted read-only for serve.
        cmd_serve(Paths(args.utrain_root), args.port, args.phase)


if __name__ == "__main__":
    main()
