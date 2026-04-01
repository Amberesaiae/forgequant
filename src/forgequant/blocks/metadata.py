"""
Block metadata and parameter specification.

Every block declares a BlockMetadata instance that describes its identity,
category, parameters, and usage characteristics. This metadata is used by
the registry for search/filtering, by the AI Forge for prompt grounding,
and by the frontend for dynamic UI generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forgequant.core.types import BlockCategory


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """
    Specification for a single block parameter.

    Attributes:
        name: Parameter identifier (must be a valid Python identifier).
        param_type: Expected Python type as a string ("int", "float", "str", "bool").
        default: Default value if the parameter is not provided.
        min_value: Minimum allowed value (for numeric types). None means unbounded.
        max_value: Maximum allowed value (for numeric types). None means unbounded.
        description: Human-readable explanation of what this parameter controls.
        choices: If set, the parameter must be one of these values.
    """

    name: str
    param_type: str
    default: Any
    min_value: float | int | None = None
    max_value: float | int | None = None
    description: str = ""
    choices: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the parameter spec at creation time."""
        if not self.name.isidentifier():
            raise ValueError(
                f"Parameter name '{self.name}' is not a valid Python identifier"
            )

        valid_types = {"int", "float", "str", "bool"}
        if self.param_type not in valid_types:
            raise ValueError(
                f"Parameter type '{self.param_type}' must be one of {valid_types}"
            )

        if self.min_value is not None and self.max_value is not None:
            if self.min_value > self.max_value:
                raise ValueError(
                    f"Parameter '{self.name}': min_value ({self.min_value}) "
                    f"cannot exceed max_value ({self.max_value})"
                )

        if self.choices is not None and len(self.choices) == 0:
            raise ValueError(
                f"Parameter '{self.name}': choices must be non-empty if provided"
            )

    def validate_value(self, value: Any) -> Any:
        """
        Validate and coerce a value against this parameter's constraints.

        Args:
            value: The value to validate.

        Returns:
            The validated (and possibly coerced) value.

        Raises:
            ValueError: If the value violates any constraint.
        """
        # Type coercion
        type_map: dict[str, type] = {
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
        }
        target_type = type_map[self.param_type]

        try:
            coerced = target_type(value)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Parameter '{self.name}': cannot convert {value!r} to {self.param_type}"
            ) from e

        # Range validation for numeric types
        if self.param_type in ("int", "float"):
            if self.min_value is not None and coerced < self.min_value:
                raise ValueError(
                    f"Parameter '{self.name}': value {coerced} is below "
                    f"minimum {self.min_value}"
                )
            if self.max_value is not None and coerced > self.max_value:
                raise ValueError(
                    f"Parameter '{self.name}': value {coerced} exceeds "
                    f"maximum {self.max_value}"
                )

        # Choices validation
        if self.choices is not None and coerced not in self.choices:
            raise ValueError(
                f"Parameter '{self.name}': value {coerced!r} is not in "
                f"allowed choices {self.choices}"
            )

        return coerced


@dataclass(frozen=True, slots=True)
class BlockMetadata:
    """
    Complete metadata for a strategy building block.

    Attributes:
        name: Unique block identifier (lowercase, underscore-separated).
        display_name: Human-friendly display name.
        category: Which category this block belongs to.
        description: Detailed description of block behavior.
        parameters: Ordered list of parameter specifications.
        tags: Searchable tags for discovery.
        typical_use: Prose description of when/how this block is typically used.
        version: Semantic version of this block's implementation.
        author: Block author or "forgequant" for built-in blocks.
    """

    name: str
    display_name: str
    category: BlockCategory
    description: str
    parameters: tuple[ParameterSpec, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
    typical_use: str = ""
    version: str = "1.0.0"
    author: str = "forgequant"

    def __post_init__(self) -> None:
        """Validate metadata at creation time."""
        if not self.name:
            raise ValueError("Block name cannot be empty")

        if not self.name.replace("_", "").isalnum():
            raise ValueError(
                f"Block name '{self.name}' must contain only lowercase letters, "
                f"digits, and underscores"
            )

        if self.name != self.name.lower():
            raise ValueError(
                f"Block name '{self.name}' must be lowercase"
            )

        if not self.display_name:
            raise ValueError("Block display_name cannot be empty")

        if not self.description:
            raise ValueError("Block description cannot be empty")

        # Ensure parameter names are unique
        param_names = [p.name for p in self.parameters]
        if len(param_names) != len(set(param_names)):
            duplicates = [n for n in param_names if param_names.count(n) > 1]
            raise ValueError(
                f"Block '{self.name}' has duplicate parameter names: {set(duplicates)}"
            )

    def get_parameter(self, name: str) -> ParameterSpec | None:
        """Look up a parameter spec by name. Returns None if not found."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None

    def get_defaults(self) -> dict[str, Any]:
        """Return a dict of all parameter defaults."""
        return {p.name: p.default for p in self.parameters}

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Validate a full set of parameters against this metadata's specs.

        Missing parameters are filled with defaults. Unknown parameters
        are rejected.

        Args:
            params: User-provided parameter values.

        Returns:
            A complete, validated parameter dictionary.

        Raises:
            ValueError: If any parameter is invalid or unknown.
        """
        validated: dict[str, Any] = {}

        # Check for unknown parameters
        known_names = {p.name for p in self.parameters}
        unknown = set(params.keys()) - known_names
        if unknown:
            raise ValueError(
                f"Block '{self.name}': unknown parameters {sorted(unknown)}. "
                f"Known: {sorted(known_names)}"
            )

        # Validate or fill defaults for each spec
        for spec in self.parameters:
            if spec.name in params:
                validated[spec.name] = spec.validate_value(params[spec.name])
            else:
                validated[spec.name] = spec.default

        return validated
