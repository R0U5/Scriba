"""Settings: sane defaults, overridable by TOML file or environment.

Resolution order (last wins):
  1. built-in defaults
  2. ``scriba.toml`` in the current directory, or ``$SCRIBA_CONFIG``
  3. ``<user config dir>/scriba/config.toml``
  4. ``SCRIBA_*`` environment variables
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir, user_data_dir
from pydantic import BaseModel, Field

from .errors import ConfigError

APP_NAME = "scriba"
ENV_PREFIX = "SCRIBA_"


class Settings(BaseModel):
    """Runtime configuration for both the CLI and the server."""

    # --- Output -----------------------------------------------------------
    output_dir: Path = Field(default_factory=lambda: Path.cwd() / "transcripts")
    keep_audio: bool = False

    # --- Storage ----------------------------------------------------------
    work_dir: Path = Field(default_factory=lambda: Path(user_cache_dir(APP_NAME)) / "work")
    data_dir: Path = Field(default_factory=lambda: Path(user_data_dir(APP_NAME)))

    # --- Server -----------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8420
    open_browser: bool = True

    # --- Engine defaults --------------------------------------------------
    engine: str = "faster-whisper"
    model: str | None = None
    device: str | None = None  # None -> auto
    compute_type: str | None = None  # None -> engine default for the device
    language: str | None = None  # None -> autodetect

    # --- Networking -------------------------------------------------------
    download_timeout: float = 900.0
    max_download_bytes: int = 4 * 1024 * 1024 * 1024  # 4 GiB
    allow_private_hosts: bool = False  # SSRF guard; set True for LAN/NAS sources
    user_agent: str = "Scriba/1.0 (+https://github.com/R0U5/Scriba)"

    # --- Runtime ----------------------------------------------------------
    max_concurrent_jobs: int = 1  # one GPU, one model in VRAM at a time
    history_limit: int = 200
    log_level: str = "INFO"

    model_config = {"arbitrary_types_allowed": True}

    def ensure_dirs(self) -> None:
        for path in (self.output_dir, self.work_dir, self.data_dir):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "history.db"


def _config_candidates() -> list[Path]:
    explicit = os.environ.get(f"{ENV_PREFIX}CONFIG")
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(Path.cwd() / "scriba.toml")
    candidates.append(Path(user_data_dir(APP_NAME)).parent / APP_NAME / "config.toml")
    return candidates


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:  # pragma: no cover - unreadable file
        raise ConfigError(f"Could not read config file {path}: {exc}") from exc
    except Exception as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc
    # Allow either a flat table or a [scriba] section.
    section = data.get("scriba")
    return section if isinstance(section, dict) else data


_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


def _coerce(field_name: str, raw: str) -> Any:
    annotation = Settings.model_fields[field_name].annotation
    text = raw.strip()
    if annotation is bool or annotation == (bool | None):
        low = text.lower()
        if low in _BOOL_TRUE:
            return True
        if low in _BOOL_FALSE:
            return False
        raise ConfigError(f"{ENV_PREFIX}{field_name.upper()} must be a boolean, got {raw!r}")
    return text


def _env_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for field_name in Settings.model_fields:
        raw = os.environ.get(f"{ENV_PREFIX}{field_name.upper()}")
        if raw is None or raw == "":
            continue
        overrides[field_name] = _coerce(field_name, raw)
    return overrides


def load_settings(**overrides: Any) -> Settings:
    """Build a :class:`Settings`, layering file config then env then kwargs."""

    data: dict[str, Any] = {}
    for candidate in _config_candidates():
        if candidate.is_file():
            data.update(_load_toml(candidate))
            break
    data.update(_env_overrides())
    data.update({k: v for k, v in overrides.items() if v is not None})

    try:
        settings = Settings(**data)
    except Exception as exc:
        raise ConfigError(f"Invalid configuration: {exc}") from exc

    settings.output_dir = settings.output_dir.expanduser().resolve()
    settings.work_dir = settings.work_dir.expanduser().resolve()
    settings.data_dir = settings.data_dir.expanduser().resolve()
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (used by the server)."""
    settings = load_settings()
    settings.ensure_dirs()
    return settings
