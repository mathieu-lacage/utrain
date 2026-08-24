"""The `serve` command: an OpenAI-compatible chat endpoint over the trained model.

The wire half of this file is shakespeare-char's, unchanged in shape. What
differs is the model half, and the seam that container's docstring points at:
nanochat is instruction-tuned and has a real chat template, so `messages` are
rendered into nanochat's conversation special tokens rather than concatenated.

Which checkpoint gets loaded depends on how far the run got. `rl` is preferred,
then `sft`, then `base`, so chat works after phase 3 as well as after phase 5 --
a base model will answer poorly, but it will answer, and `GET /v1/models` says
which one is loaded.
"""

import collections.abc
import os
import socket
import sys
import time
import typing

import fastapi
import fastapi.exceptions
import fastapi.responses
import uvicorn

from . import config, paths, schemas, upstream

# Guards against a divide-by-zero in the sampler when a client asks for greedy
# decoding by sending temperature 0.
_MIN_TEMPERATURE = 1e-3

# Best first: each is a fine-tune of the one after it.
_SOURCES = ("rl", "sft", "base")


class _Model:
    """The loaded checkpoint, hung off `app.state` rather than a global."""

    def __init__(
        self,
        engine: object,
        tokenizer: object,
        source: str,
        context_window: int,
        temperature: float,
        top_k: int,
        max_new_tokens: int,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.source = source
        self.context_window = context_window
        self.temperature = temperature
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

    @property
    def model_id(self) -> str:
        return f"nanochat-{self.source}"

    def special(self, name: str) -> int:
        return int(self.tokenizer.encode_special(name))  # type: ignore[attr-defined]

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text))  # type: ignore[attr-defined]

    def decode(self, token: int) -> str:
        return str(self.tokenizer.decode([token]))  # type: ignore[attr-defined]

    def prompt_tokens(self, messages: list[schemas.ChatMessage]) -> list[int]:
        """Render `messages` the way nanochat was fine-tuned to read them.

        A system message is folded into the message after it, which is what
        nanochat's own `render_conversation` does: the vocabulary has no system
        role. The token stream ends on an open `<|assistant_start|>`, which is
        what primes the model to reply rather than to continue the user.
        """
        tokens = [int(self.tokenizer.get_bos_token_id())]  # type: ignore[attr-defined]
        pending = ""
        for message in messages:
            if message.role == "system":
                pending = f"{pending}{message.content}\n\n"
                continue
            content = f"{pending}{message.content}"
            pending = ""
            if message.role == "assistant":
                tokens.append(self.special("<|assistant_start|>"))
                tokens.extend(self.encode(content))
                tokens.append(self.special("<|assistant_end|>"))
            else:
                tokens.append(self.special("<|user_start|>"))
                tokens.extend(self.encode(content))
                tokens.append(self.special("<|user_end|>"))
        if pending:
            tokens.append(self.special("<|user_start|>"))
            tokens.extend(self.encode(pending))
            tokens.append(self.special("<|user_end|>"))
        tokens.append(self.special("<|assistant_start|>"))
        return tokens

    def stream(
        self, tokens: list[int], max_tokens: int, temperature: float
    ) -> collections.abc.Iterator[str]:
        """Decoded text, one generated token at a time, stopping at end of turn."""
        stop = {self.special("<|assistant_end|>"), int(self.tokenizer.get_bos_token_id())}  # type: ignore[attr-defined]
        generate = self.engine.generate  # type: ignore[attr-defined]
        for token_column, _masks in generate(
            tokens,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=self.top_k,
        ):
            token = int(token_column[0])
            if token in stop:
                return
            yield self.decode(token)


def _load(p: paths.Paths, cfg: dict[str, object]) -> _Model:
    # nanochat resolves every path off this, and `serve` gets the data dir
    # read-only, which is all a checkpoint read needs.
    os.environ["NANOCHAT_BASE_DIR"] = str(upstream.base_dir(p))
    # nanochat is a checkout, not an installed distribution, so it is importable
    # only from its own directory. The phases get that for free by running as
    # subprocesses with that cwd; `serve` imports it in-process, and a console
    # script puts its own bin dir on sys.path rather than the working directory.
    if str(upstream.NANOCHAT_DIR) not in sys.path:
        sys.path.insert(0, str(upstream.NANOCHAT_DIR))
    import nanochat.checkpoint_manager
    import nanochat.common
    import nanochat.engine

    device_type = nanochat.common.autodetect_device_type()
    _ddp, _rank, _local_rank, _world, device = nanochat.common.compute_init(device_type)

    last_error: Exception | None = None
    for source in _SOURCES:
        try:
            model, tokenizer, _meta = nanochat.checkpoint_manager.load_model(
                source, device, phase="eval"
            )
        except Exception as exc:
            last_error = exc
            continue
        return _Model(
            nanochat.engine.Engine(model, tokenizer),
            tokenizer,
            source,
            int(model.config.sequence_len),
            config.get_float(cfg, "temperature", 0.6),
            config.get_int(cfg, "top_k", 50),
            config.get_int(cfg, "max_new_tokens", 512),
        )
    raise RuntimeError(f"no {'/'.join(_SOURCES)} checkpoint in the data dir: {last_error}")


router = fastapi.APIRouter()


def _loaded(request: fastapi.Request) -> _Model:
    """The checkpoint for this app. Routes are module-level, so state travels
    on the app rather than in a closure."""
    return typing.cast(_Model, request.app.state.loaded)


def validation_error_handler(
    request: fastapi.Request, exc: Exception
) -> fastapi.responses.JSONResponse:
    errors = exc.errors() if isinstance(exc, fastapi.exceptions.RequestValidationError) else []
    detail = "; ".join(
        f"{'.'.join(str(part) for part in e['loc'][1:])}: {e['msg']}" for e in errors
    )
    return fastapi.responses.JSONResponse(
        status_code=400,
        content=schemas.ErrorResponse(
            error=schemas.ErrorBody(message=detail or "invalid request")
        ).model_dump(),
    )


@router.get("/v1/models", response_model=schemas.ModelList)
def list_models(request: fastapi.Request) -> schemas.ModelList:
    loaded = _loaded(request)
    return schemas.ModelList(
        data=[schemas.ModelCard(id=loaded.model_id, context_window=loaded.context_window)]
    )


@router.post("/v1/chat/completions")
def chat_completions(
    request: fastapi.Request, req: schemas.ChatCompletionRequest
) -> fastapi.Response:
    loaded = _loaded(request)
    temperature = max(
        req.temperature if req.temperature is not None else loaded.temperature, _MIN_TEMPERATURE
    )
    prompt = loaded.prompt_tokens(req.messages)
    budget = req.token_budget(loaded.max_new_tokens)
    pieces = loaded.stream(prompt, budget, temperature)

    if req.stream:
        return fastapi.responses.StreamingResponse(
            _stream(pieces, loaded.model_id), media_type="text/event-stream"
        )

    text = "".join(pieces)
    completion_tokens = len(loaded.encode(text))
    return fastapi.responses.JSONResponse(
        schemas.ChatCompletionResponse(
            model=loaded.model_id,
            choices=[schemas.Choice(message=schemas.ChatMessage(role="assistant", content=text))],
            usage=schemas.Usage(
                prompt_tokens=len(prompt),
                completion_tokens=completion_tokens,
                total_tokens=len(prompt) + completion_tokens,
            ),
        ).model_dump()
    )


def create_app(loaded: _Model) -> fastapi.FastAPI:
    app = fastapi.FastAPI(
        title="nanochat",
        summary="An OpenAI-compatible endpoint over a nanochat model.",
    )
    app.state.loaded = loaded
    app.add_exception_handler(fastapi.exceptions.RequestValidationError, validation_error_handler)
    app.include_router(router)
    return app


def _stream(pieces: collections.abc.Iterator[str], model_id: str) -> collections.abc.Iterator[str]:
    # One id and one timestamp for the whole stream: every chunk of a completion
    # carries the same pair, which is how a client groups them.
    completion_id = schemas.new_completion_id()
    created = int(time.time())

    def chunk(delta: schemas.Delta, finish: str | None) -> str:
        return schemas.ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model_id,
            choices=[schemas.ChunkChoice(delta=delta, finish_reason=finish)],
        ).to_sse()

    yield chunk(schemas.Delta(role="assistant"), None)
    for piece in pieces:
        yield chunk(schemas.Delta(content=piece), None)
    yield chunk(schemas.Delta(), "stop")
    yield "data: [DONE]\n\n"


def serve(p: paths.Paths, port: int) -> None:
    # The chat knobs live on the `sft` phase, which is where a user configuring
    # a run looks for them.
    cfg = config.flatten(config.load(p), "sft")
    app = create_app(_load(p, cfg))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen()
    paths.write_port_file(p, sock.getsockname()[1])

    server = uvicorn.Server(uvicorn.Config(app, access_log=True, log_level="info"))
    try:
        server.run(sockets=[sock])
    except KeyboardInterrupt:
        pass
