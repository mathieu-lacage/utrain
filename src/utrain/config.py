import pathlib
import typing

import pydantic
import pydantic_settings
import yaml


class Settings(pydantic_settings.BaseSettings):
    host: str = "127.0.0.1"
    port: int = 7613
    data_dir: pathlib.Path = pathlib.Path("utrain")
    # How the TUI draws a curve. Braille packs 2x4 points into a cell against
    # half-blocks' 2x2, so it is the finer of the two wherever the font has the
    # glyphs -- and whether it does is the one thing a terminal program cannot
    # ask. `auto` guesses from the things it *can* see; see
    # `tui.render.default_charset`. This is the escape hatch for a viewer whose
    # font it guessed wrong about, so that they pin it once rather than
    # pressing `b` every launch.
    tui_charset: typing.Literal["auto", "block", "braille"] = "auto"
    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="UTRAIN_",
        env_nested_delimiter="__",
    )

    @pydantic.field_validator("data_dir")
    @classmethod
    def _resolve_data_dir(cls, value: pathlib.Path) -> pathlib.Path:
        # `data_dir` seeds `run_dir`, which podman later mounts with `-v`. A
        # relative host path there isn't parsed as a bind mount -- podman
        # instead tries to create a *named* volume with that string, which
        # fails since volume names can't contain `/`. Resolving here also
        # protects against the orchestrator subprocess having a different cwd
        # than wherever the run was created.
        return value.resolve()

    @pydantic.model_validator(mode="before")
    @classmethod
    def load_yaml(cls, data: object) -> dict[str, object]:
        raw: dict[str, object] = (
            {str(k): v for k, v in data.items()} if isinstance(data, dict) else {}
        )
        yaml_path = pathlib.Path("utrain.yaml")
        if yaml_path.exists():
            with open(yaml_path) as f:
                file_data: dict[str, object] = yaml.safe_load(f) or {}
            return {**file_data, **raw}
        return raw

    @property
    def db_path(self) -> pathlib.Path:
        return self.data_dir / "utrain.db"

    @property
    def runs_dir(self) -> pathlib.Path:
        return self.data_dir / "runs"

    @property
    def dispatcher_dir(self) -> pathlib.Path:
        """Where the dispatcher keeps its lock and its log."""
        return self.data_dir / "dispatcher"
