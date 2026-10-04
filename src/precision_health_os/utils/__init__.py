"""Utility functions for Precision Health OS."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import UTC, datetime
from typing import Any


def generate_id() -> str:
    """Generate a unique identifier."""
    return str(uuid.uuid4())


def hash_sensitive(value: str) -> str:
    """One-way hash for sensitive data (HIPAA safe harbor)."""
    return hashlib.sha256(value.encode()).hexdigest()


def generate_token(length: int = 32) -> str:
    """Generate a cryptographically secure token."""
    return secrets.token_urlsafe(length)


def utcnow() -> datetime:
    """Get current UTC datetime (timezone-aware)."""
    from datetime import timezone
    return datetime.now(UTC)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    """Split a list into chunks of given size."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def z_score(value: float, mean: float, std: float) -> float:
    """Compute z-score for anomaly detection."""
    if std == 0:
        return 0.0
    return (value - mean) / std


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two vectors."""
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=False)) ** 0.5


def normalize(values: list[float]) -> list[float]:
    """Min-max normalize a list of values to [0, 1]."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]
