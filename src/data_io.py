"""Input/output helpers that preserve the immutable source data."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "configs" / "fast.yaml") -> dict[str, Any]:
    """Load a project YAML configuration relative to the repository root."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(path: str | Path) -> Path:
    """Resolve a configured project-relative path."""
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def normalize_identifier(value: object) -> str:
    """Normalize study/instance identifiers without discarding alphanumeric suffixes."""
    if pd.isna(value):
        raise ValueError("Identifier cannot be missing")
    text = str(value).strip().lower()
    text = re.sub(r"\.0+$", "", text)
    return re.sub(r"\s+", "", text)


def load_cleaned_csv(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load the source cleaned modeling table without modifying it."""
    config = config or load_config()
    return pd.read_csv(project_path(config["paths"]["cleaned_csv"]))


def load_raw_workbook(config: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """Load every worksheet from the immutable raw workbook."""
    config = config or load_config()
    workbook = project_path(config["paths"]["raw_workbook"])
    excel = pd.ExcelFile(workbook)
    return {name: pd.read_excel(workbook, sheet_name=name) for name in excel.sheet_names}


def sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a file SHA-256 digest without loading the full file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

