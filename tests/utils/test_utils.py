"""Tests for utility functions."""

import pytest

from precision_health_os.utils import (
    chunked,
    deep_merge,
    euclidean_distance,
    generate_id,
    hash_sensitive,
    normalize,
    z_score,
)


class TestGenerateId:
    """Tests for generate_id."""

    def test_unique_ids(self) -> None:
        ids = [generate_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_id_format(self) -> None:
        id_val = generate_id()
        assert isinstance(id_val, str)
        assert len(id_val) == 36  # UUID4 length


class TestHashSensitive:
    """Tests for hash_sensitive."""

    def test_deterministic(self) -> None:
        h1 = hash_sensitive("test_value")
        h2 = hash_sensitive("test_value")
        assert h1 == h2

    def test_different_inputs(self) -> None:
        h1 = hash_sensitive("value1")
        h2 = hash_sensitive("value2")
        assert h1 != h2

    def test_sha256_format(self) -> None:
        h = hash_sensitive("test")
        assert len(h) == 64  # SHA-256 hex length


class TestDeepMerge:
    """Tests for deep_merge."""

    def test_simple_merge(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3, "z": 4}}
        result = deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_does_not_mutate_base(self) -> None:
        base = {"a": 1}
        deep_merge(base, {"b": 2})
        assert base == {"a": 1}


class TestChunked:
    """Tests for chunked."""

    def test_even_split(self) -> None:
        result = chunked([1, 2, 3, 4], 2)
        assert result == [[1, 2], [3, 4]]

    def test_uneven_split(self) -> None:
        result = chunked([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_empty_list(self) -> None:
        assert chunked([], 3) == []


class TestZScore:
    """Tests for z_score."""

    def test_zero_std(self) -> None:
        assert z_score(5, 5, 0) == 0.0

    def test_positive_z(self) -> None:
        assert z_score(10, 5, 2.5) == 2.0

    def test_negative_z(self) -> None:
        assert z_score(0, 5, 2.5) == -2.0


class TestEuclideanDistance:
    """Tests for euclidean_distance."""

    def test_same_point(self) -> None:
        assert euclidean_distance([0, 0], [0, 0]) == 0.0

    def test_simple_distance(self) -> None:
        assert euclidean_distance([0, 0], [3, 4]) == 5.0

    def test_3d_distance(self) -> None:
        assert euclidean_distance([0, 0, 0], [1, 1, 1]) == pytest.approx(1.732, rel=1e-3)


class TestNormalize:
    """Tests for normalize."""

    def test_basic_normalize(self) -> None:
        result = normalize([0, 50, 100])
        assert result == [0.0, 0.5, 1.0]

    def test_empty_list(self) -> None:
        assert normalize([]) == []

    def test_constant_values(self) -> None:
        result = normalize([5, 5, 5])
        assert result == [0.5, 0.5, 0.5]
