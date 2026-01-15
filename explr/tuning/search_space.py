"""
Unified search space definition for hyperparameter tuning.

Provides a fluent builder interface for defining search spaces that
work with both Optuna and Ray Tune.
"""

from __future__ import annotations

from typing import Any, Dict, List


class SearchSpaceBuilder:
    """
    Fluent builder for hyperparameter search spaces.

    Provides a convenient API for defining search spaces that can be
    converted to either Optuna or Ray Tune format.

    Example:
        search_space = (
            SearchSpaceBuilder()
            .loguniform("learning_rate", 1e-5, 1e-1)
            .categorical("optimizer", ["adam", "sgd", "adamw"])
            .integer("batch_size", 16, 128, step=16)
            .integer("hidden_layers", 1, 5)
            .uniform("dropout", 0.0, 0.5)
            .build()
        )

        config.tuning.search_space.parameters = search_space
    """

    def __init__(self):
        """Initialize an empty search space builder."""
        self._parameters: Dict[str, Dict[str, Any]] = {}

    def categorical(self, name: str, choices: List[Any]) -> "SearchSpaceBuilder":
        """
        Add a categorical parameter.

        Args:
            name: Parameter name
            choices: List of possible values

        Returns:
            Self for chaining
        """
        self._parameters[name] = {"type": "categorical", "choices": choices}
        return self

    def integer(
        self,
        name: str,
        low: int,
        high: int,
        step: int = 1,
        log: bool = False,
    ) -> "SearchSpaceBuilder":
        """
        Add an integer parameter.

        Args:
            name: Parameter name
            low: Minimum value (inclusive)
            high: Maximum value (inclusive)
            step: Step size for discrete values
            log: Whether to use log-scale sampling

        Returns:
            Self for chaining
        """
        self._parameters[name] = {
            "type": "int",
            "low": low,
            "high": high,
            "step": step,
            "log": log,
        }
        return self

    def uniform(
        self,
        name: str,
        low: float,
        high: float,
        step: float | None = None,
    ) -> "SearchSpaceBuilder":
        """
        Add a uniform float parameter.

        Args:
            name: Parameter name
            low: Minimum value
            high: Maximum value
            step: Optional step size for discrete values

        Returns:
            Self for chaining
        """
        self._parameters[name] = {
            "type": "float",
            "low": low,
            "high": high,
            "step": step,
            "log": False,
        }
        return self

    def loguniform(
        self,
        name: str,
        low: float,
        high: float,
    ) -> "SearchSpaceBuilder":
        """
        Add a log-uniform float parameter.

        Samples are uniformly distributed in log space, which is ideal
        for parameters like learning rates.

        Args:
            name: Parameter name
            low: Minimum value (must be positive)
            high: Maximum value

        Returns:
            Self for chaining
        """
        self._parameters[name] = {
            "type": "loguniform",
            "low": low,
            "high": high,
        }
        return self

    def build(self) -> Dict[str, Dict[str, Any]]:
        """
        Build the search space dictionary.

        Returns:
            Dictionary mapping parameter names to their space definitions
        """
        return self._parameters.copy()

    def __repr__(self) -> str:
        """String representation of the search space."""
        params = ", ".join(
            f"{name}: {spec['type']}" for name, spec in self._parameters.items()
        )
        return f"SearchSpace({params})"
