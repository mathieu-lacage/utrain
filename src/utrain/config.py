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
