"""Save and load the scored pipeline and the JSON model card."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

BUNDLE_KEYS = ("pipeline", "threshold", "reference", "feature_columns", "model_name")


def save_bundle(path: Path, bundle: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)


def load_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, dict):
        raise TypeError("Model bundle must be a dictionary.")
    missing = [key for key in BUNDLE_KEYS if key not in bundle]
    if missing:
        raise KeyError(f"Model bundle is missing keys: {missing}")
    return bundle


def save_model_card(path: Path, card: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(json_safe(card), indent=2) + "\n", encoding="utf-8")


def load_model_card(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Recursively convert numpy and pandas objects into JSON values."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.DataFrame):
        return json_safe(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return json_safe(value.to_dict())
    if value is pd.NA:
        return None
    return value
