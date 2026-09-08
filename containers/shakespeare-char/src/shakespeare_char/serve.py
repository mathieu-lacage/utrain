"""The `serve` command: an OpenAI-compatible chat endpoint over the checkpoint."""

import collections.abc
import json
import socket
import time
import typing

import fastapi
import fastapi.exceptions
import fastapi.responses
import torch
import uvicorn

from . import config, paths, schemas
from . import model as model_mod

# Guards against a divide-by-zero in the sampler when a client asks for greedy
# decoding by sending temperature 0.
_MIN_TEMPERATURE = 1e-3

_MODEL_ID = "shakespeare-char"

# The one phase whose data dir holds a model.pt, and so the only phase this
# image can be asked to serve -- see `describe.py`.
SERVABLE_PHASE = "pretrain"


class _Model:
    """The loaded checkpoint, hung off `app.state` rather than a global."""

    def __init__(
        self,
        model: model_mod.CharLM,
        stoi: dict[str, int],
        itos: dict[str, str],
        block_size: int,
        default_max_tokens: int,
    ) -> None:
        self.model = model
        self.stoi = stoi
        # JSON turned the int keys of the vocab into strings.
        self.itos = itos
        self.block_size = block_size
        self.default_max_tokens = default_max_tokens

    def encode(self, text: str) -> list[int]:
        # Unknown characters fall back to token 0 rather than failing the
        # request: this vocabulary only covers what tiny-shakespeare contained.
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, token: int) -> str:
        return self.itos.get(str(token), "?")


def _load_model(p: paths.Paths) -> tuple[model_mod.CharLM, dict[str, int], dict[str, str], int]:
    ckpt = torch.load(str(p.data_dir / "model.pt"), map_location="cpu", weights_only=True)
    model = model_mod.CharLM(
        ckpt["vocab_size"], ckpt["n_embd"], ckpt["n_layer"], ckpt["n_head"], ckpt["block_size"]
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    vocab = json.loads((p.data_dir / "vocab.json").read_text())
    return model, vocab["stoi"], vocab["itos"], ckpt["block_size"]


def _prompt_from_messages(messages: list[schemas.ChatMessage]) -> str:
    """Flatten OpenAI `messages` into the flat text this model understands.

    A character-level Shakespeare model has no chat template and no concept of a
    role -- its entire world is a running stream of characters. So the contents
    are concatenated in order and the roles dropped. Because the API is
    stateless the client resends every turn, which makes this the whole
    conversation as one continuous text, exactly what the model was trained on.

    A container wrapping an instruction-tuned model would apply its real chat
    template here instead; that is the point of the seam.
    """
    return "".join(m.content for m in messages)


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
    return schemas.ModelList(
        data=[schemas.ModelCard(id=_MODEL_ID, context_window=_loaded(request).block_size)]
    )


@router.post("/v1/chat/completions")
def chat_completions(
    request: fastapi.Request, req: schemas.ChatCompletionRequest
) -> fastapi.Response:
    loaded = _loaded(request)
    prompt = _prompt_from_messages(req.messages)
    if not prompt:
        # The model has to condition on something; a newline is the most
        # neutral prompt this vocabulary has.
        prompt = "\n"

    temperature = max(req.temperature if req.temperature is not None else 0.8, _MIN_TEMPERATURE)

    encoded = loaded.encode(prompt)
    idx = torch.tensor([encoded], dtype=torch.long)
    tokens = loaded.model.stream_generate(
        idx, req.token_budget(loaded.default_max_tokens), temperature
    )

    if req.stream:
        return fastapi.responses.StreamingResponse(
            _stream(tokens, loaded), media_type="text/event-stream"
        )

    text = "".join(loaded.decode(t) for t in tokens)
    return fastapi.responses.JSONResponse(
        schemas.ChatCompletionResponse(
            model=_MODEL_ID,
            choices=[schemas.Choice(message=schemas.ChatMessage(role="assistant", content=text))],
            usage=schemas.Usage(
                prompt_tokens=len(encoded),
                completion_tokens=len(text),
                total_tokens=len(encoded) + len(text),
            ),
        ).model_dump()
    )


def create_app(loaded: _Model) -> fastapi.FastAPI:
    app = fastapi.FastAPI(
        title="shakespeare-char",
        summary="An OpenAI-compatible endpoint over a character-level Shakespeare model.",
    )
    app.state.loaded = loaded
    app.add_exception_handler(fastapi.exceptions.RequestValidationError, validation_error_handler)
    app.include_router(router)
    return app


def _stream(tokens: collections.abc.Iterator[int], loaded: _Model) -> collections.abc.Iterator[str]:
    # One id and one timestamp for the whole stream: every chunk of a completion
    # carries the same pair, which is how a client groups them.
    completion_id = schemas.new_completion_id()
    created = int(time.time())

    def chunk(delta: schemas.Delta, finish: str | None) -> str:
        return schemas.ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=_MODEL_ID,
            choices=[schemas.ChunkChoice(delta=delta, finish_reason=finish)],
        ).to_sse()

    yield chunk(schemas.Delta(role="assistant"), None)
    for token in tokens:
        yield chunk(schemas.Delta(content=loaded.decode(token)), None)
    yield chunk(schemas.Delta(), "stop")
    yield "data: [DONE]\n\n"


def serve(p: paths.Paths, port: int, phase: str) -> None:
    if phase != SERVABLE_PHASE:
        raise RuntimeError(f"phase '{phase}' leaves no model to serve, only '{SERVABLE_PHASE}'")
    cfg = config.flatten(config.load(p), "pretrain")
    model, stoi, itos, block_size = _load_model(p)
    app = create_app(
        _Model(model, stoi, itos, block_size, config.get_int(cfg, "generate_len", 200))
    )

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
