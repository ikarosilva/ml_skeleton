"""Tests for core module."""

import pytest
from ml_skeleton.core.protocols import TrainingContext, TrainingResult
from ml_skeleton.core.config import ExperimentConfig, TuningConfig, TunerType, MLflowConfig


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_minimal_result(self):
        """Test creating result with only required fields."""
        result = TrainingResult(primary_metric=0.5)
        assert result.primary_metric == 0.5
        assert result.primary_metric_name == "val_loss"
        assert result.minimize is True
        assert result.metrics == {}
        assert result.best_model_path is None

    def test_full_result(self):
        """Test creating result with all fields."""
        result = TrainingResult(
            primary_metric=0.3,
            primary_metric_name="accuracy",
            minimize=False,
            metrics={"loss": 0.5, "f1": 0.8},
            best_model_path="/path/to/model",
            epochs_completed=50,
            early_stopped=True,
        )
        assert result.primary_metric == 0.3
        assert result.primary_metric_name == "accuracy"
        assert result.minimize is False
        assert result.metrics == {"loss": 0.5, "f1": 0.8}
        assert result.epochs_completed == 50
        assert result.early_stopped is True


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExperimentConfig()
        assert config.name == "experiment"
        assert config.framework == "pytorch"
        assert config.seed == 42
        assert config.tuning.tuner_type == TunerType.NONE

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExperimentConfig(
            name="my_experiment",
            framework="tensorflow",
            seed=123,
            hyperparameters={"lr": 0.01},
        )
        assert config.name == "my_experiment"
        assert config.framework == "tensorflow"
        assert config.seed == 123
        assert config.hyperparameters["lr"] == 0.01

    def test_tuning_config(self):
        """Test tuning configuration."""
        config = ExperimentConfig(
            tuning=TuningConfig(
                tuner_type=TunerType.OPTUNA,
                n_trials=100,
            )
        )
        assert config.tuning.tuner_type == TunerType.OPTUNA
        assert config.tuning.n_trials == 100


class TestTunerType:
    """Tests for TunerType enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert TunerType.NONE.value == "none"
        assert TunerType.OPTUNA.value == "optuna"
        assert TunerType.RAY_TUNE.value == "ray_tune"

    def test_enum_from_string(self):
        """Test creating enum from string."""
        assert TunerType("optuna") == TunerType.OPTUNA
        assert TunerType("ray_tune") == TunerType.RAY_TUNE
