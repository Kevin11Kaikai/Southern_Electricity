"""Configuration loading and repository-relative path resolution for Phase B."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "phase_b.toml"


@dataclass(frozen=True)
class PhaseBConfig:
    """Thin typed wrapper around the TOML configuration."""

    raw: dict[str, Any]
    source: Path

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["seed"])

    def path(self, key: str) -> Path:
        value = Path(self.raw["paths"][key])
        return value if value.is_absolute() else REPO_ROOT / value

    def section(self, key: str) -> dict[str, Any]:
        return dict(self.raw[key])


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PhaseBConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)
    return PhaseBConfig(raw=raw, source=config_path)

