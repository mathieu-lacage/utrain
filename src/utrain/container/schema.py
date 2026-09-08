import typing

import pydantic


class FieldSchema(pydantic.BaseModel):
    key: str
    label: str
    type: typing.Literal["int", "float", "str", "bool", "enum"]
    default: int | float | str | bool | None = None
    description: str = ""
    required: bool = False
    min: int | float | None = None
    max: int | float | None = None
    options: list[str] = []


class FieldGroup(pydantic.BaseModel):
    name: str
    label: str
    fields: list[FieldSchema]


class PhaseConfigSchema(pydantic.BaseModel):
    groups: list[FieldGroup] = []


class GlobalConfigSchema(pydantic.BaseModel):
    groups: list[FieldGroup] = []


class ConfigSchema(pydantic.BaseModel):
    globals: GlobalConfigSchema = GlobalConfigSchema()
    phases: dict[str, PhaseConfigSchema] = {}


class PlotSpec(pydantic.BaseModel):
    """One plot a phase wants shown by default: `y` against `x`.

    `x` is `"step"`, `"elapsed"`, or the name of another logged metric -- the
    same three things the TUI's `x` key cycles through.
    """

    x: str = "step"
    y: str


class PhaseInfo(pydantic.BaseModel):
    name: str
    label: str
    cacheable: bool = False
    # Empty means the viewer gets every metric the phase logs, each against
    # step. A phase that names its plots gets those instead, until the viewer
    # says otherwise.
    plots: list[PlotSpec] = []
    # Whether this phase's snapshot is worth chatting with. It is the only
    # thing that decides: an image none of whose phases set it serves nothing.
    can_serve: bool = False


class DescribeOutput(pydantic.BaseModel):
    name: str
    version: str = "1.0.0"
    phases: list[PhaseInfo]
    phase_order: list[str]
    config_schema: ConfigSchema = ConfigSchema()


class CompatResult(pydantic.BaseModel):
    compatible: bool
    details: str = ""


class CacheFile(pydantic.BaseModel):
    path: str
    sha256: str


class CacheManifest(pydantic.BaseModel):
    files: list[CacheFile] = []
