import pathlib

import pydantic
import pydantic_settings
import yaml


class Settings(pydantic_settings.BaseSettings):
    host: str = "127.0.0.1"
    port: int = 7612
    data_dir: pathlib.Path = pathlib.Path(".")
    presets: dict[str, str] = {}

    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="UTRAIN_",
        env_nested_delimiter="__",
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def load_yaml(cls, data: object) -> object:
        if not isinstance(data, dict):
            data = {}
        yaml_path = pathlib.Path("utrain.yaml")
        if yaml_path.exists():
            with open(yaml_path) as f:
                file_data = yaml.safe_load(f) or {}
            merged = {**file_data, **data}
            return merged
        return data

    @property
    def db_path(self) -> pathlib.Path:
        return self.data_dir / "utrain.db"

    @property
    def runs_dir(self) -> pathlib.Path:
        return self.data_dir / "runs"
