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


class PhaseInfo(pydantic.BaseModel):
    name: str
    label: str
    cacheable: bool = False


class DescribeOutput(pydantic.BaseModel):
    name: str
    version: str = "1.0.0"
    phases: list[PhaseInfo]
    phase_order: list[str]
    config_schema: ConfigSchema = ConfigSchema()
    can_serve: bool = False


class CompatResult(pydantic.BaseModel):
    compatible: bool
    details: str = ""


class CacheFile(pydantic.BaseModel):
    path: str
    sha256: str


class CacheManifest(pydantic.BaseModel):
    files: list[CacheFile] = []
