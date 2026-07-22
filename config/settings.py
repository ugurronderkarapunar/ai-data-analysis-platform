"""Application settings and path configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for the platform.

    Attributes:
        project_root: Absolute path to the project root.
        data_dir: Directory for uploaded / sample datasets.
        logs_dir: Directory for application log files.
        outputs_dir: Directory for generated reports and artifacts.
        log_level: Default logging level name.
        max_upload_mb: Maximum upload size in megabytes.
        default_encoding: Fallback text encoding for CSV files.
        random_seed: Seed for reproducible ML / stats operations.
        dark_mode_default: Whether the Streamlit UI starts in dark mode.
    """

    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")
    outputs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    log_level: str = "INFO"
    max_upload_mb: int = 200
    default_encoding: str = "utf-8"
    random_seed: int = 42
    dark_mode_default: bool = True


def get_settings() -> Settings:
    """Return application settings and ensure required directories exist.

    Returns:
        Settings: Configured settings instance.

    Raises:
        OSError: If required directories cannot be created.
    """
    settings = Settings()
    try:
        for path in (settings.data_dir, settings.logs_dir, settings.outputs_dir):
            path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Failed to create required directories: {exc}") from exc
    return settings
