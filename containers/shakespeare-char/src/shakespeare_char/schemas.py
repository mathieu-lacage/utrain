"""The OpenAI chat-completions wire contract, as Pydantic models.

This is the subset of `/v1/chat/completions` the utrain container contract
requires (see `docs/container-contract.md`). It is deliberately a module of its
own: it *is* the contract, so it should be readable in one sitting and copyable
into another container without dragging any model code along.

Nothing here is utrain-specific. The one addition to OpenAI's schema is
`ModelCard.context_window`, which clients ignore and utrain shows in its banner.
"""

import time
import typing
import uuid

import pydantic


def new_completion_id() -> str:
    """Ids are per completion, and every chunk of a stream repeats the same one."""
    return f"chatcmpl-{uuid.uuid4().hex}"


class ChatMessage(pydantic.BaseModel):
    role: str
    content: str = ""


class ChatCompletionRequest(pydantic.BaseModel):
    # Extra fields are accepted and ignored rather than rejected: real clients
    # send plenty this container has no use for (top_p, presence_penalty, user,
    # ...) and refusing them would break callers for no benefit.
    model_config = pydantic.ConfigDict(extra="allow")

    messages: list[ChatMessage] = pydantic.Field(min_length=1)
    model: str | None = None
    stream: bool = False
    temperature: float | None = None
    # `max_completion_tokens` is the current spelling; `max_tokens` is what
    # existing clients still send. Accept both and let the endpoint prefer the
    # newer one.
    max_tokens: int | None = pydantic.Field(default=None, gt=0)
    max_completion_tokens: int | None = pydantic.Field(default=None, gt=0)

    def token_budget(self, default: int) -> int:
        return self.max_completion_tokens or self.max_tokens or default


class Choice(pydantic.BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(pydantic.BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(pydantic.BaseModel):
    id: str = pydantic.Field(default_factory=new_completion_id)
    object: typing.Literal["chat.completion"] = "chat.completion"
    created: int = pydantic.Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage


class Delta(pydantic.BaseModel):
    """The incremental half of a streamed choice.

    Both fields are optional because a stream's first chunk carries only a role
    and its last carries neither. They serialise as explicit nulls, which every
    OpenAI client tolerates -- the SDK types both as optional -- and which keeps
    chunk serialisation a single uniform dump.
    """

    role: str | None = None
    content: str | None = None


class ChunkChoice(pydantic.BaseModel):
    index: int = 0
    delta: Delta
    # Must stay in the payload as `null` on non-final chunks rather than being
    # omitted: clients read it to decide whether a stream has ended.
    finish_reason: str | None = None


class ChatCompletionChunk(pydantic.BaseModel):
    id: str
    object: typing.Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]

    def to_sse(self) -> str:
        """Serialise as one server-sent event, the framing SSE requires."""
        return f"data: {self.model_dump_json()}\n\n"


class ModelCard(pydantic.BaseModel):
    id: str
    object: typing.Literal["model"] = "model"
    created: int = 0
    owned_by: str = "utrain"
    # Not part of OpenAI's schema; harmless to clients, and utrain surfaces it.
    context_window: int | None = None


class ModelList(pydantic.BaseModel):
    object: typing.Literal["list"] = "list"
    data: list[ModelCard]


class ErrorBody(pydantic.BaseModel):
    message: str
    type: str = "invalid_request_error"


class ErrorResponse(pydantic.BaseModel):
    """OpenAI's error envelope, so a client's normal error handling applies."""

    error: ErrorBody
