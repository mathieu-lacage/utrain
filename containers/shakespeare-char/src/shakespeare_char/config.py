"""Reading the two files utrain writes into the utrain root: config.yaml and control.json."""

import json
import typing

import yaml

from . import paths


def read_control(p: paths.Paths) -> str:
    try:
        data = json.loads((p.run_dir / "control.json").read_text())
        return str(data.get("action", "continue"))
    except Exception:
        return "continue"


def load(p: paths.Paths) -> dict[str, object]:
    cfg_path = p.run_dir / "config.yaml"
    if cfg_path.exists():
        return dict(yaml.safe_load(cfg_path.read_text()) or {})
    return {}


def flatten(cfg: dict[str, object], phase: str) -> dict[str, object]:
    """Merge the utrain config into the flat namespace this phase reads.

    utrain writes config.yaml with keys grouped under a ``globals`` section and
    a per-phase ``phases.<phase>`` section. Within a section a value may be a
    field group (a dict, e.g. ``globals.model``) whose members are the real
    keys, or a bare field. The container reads flat keys, so for the phase being
    run we merge top-level scalars + ``globals`` + ``phases.<phase>``, expanding
    any group dict one level. Phase values win over globals on collision.
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
        _merge(typing.cast(dict[str, object], phases_section).get(phase))

    return flat


def get_int(cfg: dict[str, object], key: str, default: int) -> int:
    val = cfg.get(key, default)
    return int(val)  # type: ignore[arg-type]


def get_float(cfg: dict[str, object], key: str, default: float) -> float:
    val = cfg.get(key, default)
    return float(val)  # type: ignore[arg-type]


def get_str(cfg: dict[str, object], key: str, default: str) -> str:
    val = cfg.get(key, default)
    return str(val)
