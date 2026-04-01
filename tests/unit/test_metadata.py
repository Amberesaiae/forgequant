"""Tests for forgequant.blocks.metadata."""

from __future__ import annotations

import pytest

from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.core.types import BlockCategory


class TestParameterSpec:
    """Tests for ParameterSpec dataclass."""

    def test_valid_creation(self) -> None:
        p = ParameterSpec(
            name="period",
            param_type="int",
            default=14,
            min_value=2,
            max_value=500,
            description="Lookback period",
        )
        assert p.name == "period"
        assert p.param_type == "int"
        assert p.default == 14
        assert p.min_value == 2
        assert p.max_value == 500

    def test_invalid_name_not_identifier(self) -> None:
        with pytest.raises(ValueError, match="not a valid Python identifier"):
            ParameterSpec(name="123bad", param_type="int", default=1)

    def test_invalid_param_type(self) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            ParameterSpec(name="x", param_type="list", default=[])

    def test_min_exceeds_max(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            ParameterSpec(name="x", param_type="int", default=5, min_value=10, max_value=5)

    def test_empty_choices(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ParameterSpec(name="x", param_type="str", default="a", choices=())

    def test_validate_value_int(self) -> None:
        p = ParameterSpec(name="period", param_type="int", default=14, min_value=2, max_value=500)
        assert p.validate_value(20) == 20
        assert p.validate_value("30") == 30  # Coercion from string

    def test_validate_value_below_min(self) -> None:
        p = ParameterSpec(name="period", param_type="int", default=14, min_value=2, max_value=500)
        with pytest.raises(ValueError, match="below minimum"):
            p.validate_value(1)

    def test_validate_value_above_max(self) -> None:
        p = ParameterSpec(name="period", param_type="int", default=14, min_value=2, max_value=500)
        with pytest.raises(ValueError, match="exceeds maximum"):
            p.validate_value(501)

    def test_validate_value_choices(self) -> None:
        p = ParameterSpec(
            name="method",
            param_type="str",
            default="ema",
            choices=("sma", "ema"),
        )
        assert p.validate_value("sma") == "sma"
        with pytest.raises(ValueError, match="not in allowed choices"):
            p.validate_value("wma")

    def test_validate_value_type_error(self) -> None:
        p = ParameterSpec(name="count", param_type="int", default=5)
        with pytest.raises(ValueError, match="cannot convert"):
            p.validate_value("not_a_number")

    def test_validate_float(self) -> None:
        p = ParameterSpec(
            name="multiplier", param_type="float", default=2.0, min_value=0.5, max_value=5.0
        )
        assert p.validate_value(3.5) == 3.5
        assert p.validate_value(1) == 1.0  # int -> float

    def test_validate_bool(self) -> None:
        p = ParameterSpec(name="use_ema", param_type="bool", default=True)
        assert p.validate_value(True) is True
        assert p.validate_value(False) is False


class TestBlockMetadata:
    """Tests for BlockMetadata dataclass."""

    def _make_metadata(self, **kwargs: object) -> BlockMetadata:
        """Helper to create BlockMetadata with sensible defaults."""
        defaults = dict(
            name="test_block",
            display_name="Test Block",
            category=BlockCategory.INDICATOR,
            description="A test block for testing",
        )
        defaults.update(kwargs)
        return BlockMetadata(**defaults)  # type: ignore[arg-type]

    def test_valid_creation(self) -> None:
        meta = self._make_metadata()
        assert meta.name == "test_block"
        assert meta.display_name == "Test Block"
        assert meta.category == BlockCategory.INDICATOR
        assert meta.version == "1.0.0"
        assert meta.author == "forgequant"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            self._make_metadata(name="")

    def test_non_lowercase_name_raises(self) -> None:
        with pytest.raises(ValueError, match="must be lowercase"):
            self._make_metadata(name="TestBlock")

    def test_invalid_chars_in_name_raises(self) -> None:
        with pytest.raises(ValueError, match="must contain only"):
            self._make_metadata(name="test-block")

    def test_empty_display_name_raises(self) -> None:
        with pytest.raises(ValueError, match="display_name cannot be empty"):
            self._make_metadata(display_name="")

    def test_empty_description_raises(self) -> None:
        with pytest.raises(ValueError, match="description cannot be empty"):
            self._make_metadata(description="")

    def test_duplicate_parameter_names_raises(self) -> None:
        params = (
            ParameterSpec(name="period", param_type="int", default=14),
            ParameterSpec(name="period", param_type="int", default=20),
        )
        with pytest.raises(ValueError, match="duplicate parameter names"):
            self._make_metadata(parameters=params)

    def test_get_parameter_found(self) -> None:
        params = (ParameterSpec(name="period", param_type="int", default=14),)
        meta = self._make_metadata(parameters=params)
        p = meta.get_parameter("period")
        assert p is not None
        assert p.name == "period"

    def test_get_parameter_not_found(self) -> None:
        meta = self._make_metadata()
        assert meta.get_parameter("nonexistent") is None

    def test_get_defaults(self) -> None:
        params = (
            ParameterSpec(name="period", param_type="int", default=14),
            ParameterSpec(name="multiplier", param_type="float", default=2.0),
        )
        meta = self._make_metadata(parameters=params)
        defaults = meta.get_defaults()
        assert defaults == {"period": 14, "multiplier": 2.0}

    def test_validate_params_defaults(self) -> None:
        params = (
            ParameterSpec(name="period", param_type="int", default=14),
        )
        meta = self._make_metadata(parameters=params)
        result = meta.validate_params({})
        assert result == {"period": 14}

    def test_validate_params_override(self) -> None:
        params = (
            ParameterSpec(name="period", param_type="int", default=14, min_value=2, max_value=500),
        )
        meta = self._make_metadata(parameters=params)
        result = meta.validate_params({"period": 20})
        assert result == {"period": 20}

    def test_validate_params_unknown_raises(self) -> None:
        meta = self._make_metadata()
        with pytest.raises(ValueError, match="unknown parameters"):
            meta.validate_params({"bogus": 42})

    def test_validate_params_invalid_value_raises(self) -> None:
        params = (
            ParameterSpec(name="period", param_type="int", default=14, min_value=2, max_value=500),
        )
        meta = self._make_metadata(parameters=params)
        with pytest.raises(ValueError, match="below minimum"):
            meta.validate_params({"period": 1})
