"""
TensorFlow/Keras helper utilities for use within train_model().

Provides common functionality for TensorFlow training including
GPU setup, reproducibility, and MLflow callbacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from explr.core.protocols import TrainingContext


class TensorFlowHelper:
    """
    Helper utilities for TensorFlow training within train_model().

    Example:
        def train_model(ctx: TrainingContext) -> TrainingResult:
            from explr.frameworks.tensorflow import TensorFlowHelper

            # Set up GPU and reproducibility
            TensorFlowHelper.setup_gpu()
            TensorFlowHelper.setup_seed(ctx.seed)

            # Create callbacks
            mlflow_callback = TensorFlowHelper.create_mlflow_callback(ctx)
            checkpoint_callback = TensorFlowHelper.create_checkpoint_callback(
                filepath="best_model.keras"
            )

            # Train model
            model.fit(
                train_data,
                validation_data=val_data,
                callbacks=[mlflow_callback, checkpoint_callback]
            )

            return TrainingResult(...)
    """

    @staticmethod
    def setup_gpu() -> None:
        """
        Configure GPU settings for TensorFlow.

        Enables memory growth to avoid allocating all GPU memory.
        """
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU setup error: {e}")

    @staticmethod
    def setup_seed(seed: Optional[int]) -> None:
        """
        Set random seeds for reproducibility.

        Args:
            seed: Random seed (None to skip)
        """
        if seed is None:
            return

        import tensorflow as tf

        tf.random.set_seed(seed)

    @staticmethod
    def create_mlflow_callback(ctx: "TrainingContext") -> Any:
        """
        Create a Keras callback for MLflow logging.

        Args:
            ctx: Training context

        Returns:
            Keras callback instance
        """
        import tensorflow as tf

        trial = getattr(ctx, "_optuna_trial", None)
        return TensorFlowMLflowCallback(ctx.tracker, trial)

    @staticmethod
    def create_checkpoint_callback(
        filepath: str,
        monitor: str = "val_loss",
        save_best_only: bool = True,
        mode: str = "min",
    ) -> Any:
        """
        Create a Keras ModelCheckpoint callback.

        Args:
            filepath: Path to save the model
            monitor: Metric to monitor
            save_best_only: Only save when monitored metric improves
            mode: "min" or "max" for the monitored metric

        Returns:
            ModelCheckpoint callback
        """
        import tensorflow as tf

        return tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=False,
            mode=mode,
        )

    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """
        Get information about available TensorFlow devices.

        Returns:
            Dictionary with device information
        """
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")

        info = {
            "tensorflow_version": tf.__version__,
            "gpu_available": len(gpus) > 0,
            "device_count": len(gpus),
            "devices": [],
        }

        for i, gpu in enumerate(gpus):
            info["devices"].append(
                {
                    "name": gpu.name,
                    "device_type": gpu.device_type,
                }
            )

        return info


class TensorFlowMLflowCallback:
    """
    Keras callback for logging training metrics to MLflow.

    Supports intermediate metric reporting for Optuna pruning.
    """

    def __init__(self, tracker: Any, trial: Optional[Any] = None):
        """
        Initialize the callback.

        Args:
            tracker: ExplrTracker instance
            trial: Optional Optuna trial for pruning
        """
        self.tracker = tracker
        self.trial = trial

    def __call__(self) -> "TensorFlowMLflowKerasCallback":
        """
        Return a Keras-compatible callback.

        Returns:
            Keras callback instance
        """
        import tensorflow as tf

        class KerasCallback(tf.keras.callbacks.Callback):
            def __init__(inner_self):
                super().__init__()
                inner_self.tracker = self.tracker
                inner_self.trial = self.trial

            def on_epoch_end(inner_self, epoch, logs=None):
                logs = logs or {}

                # Log all metrics
                inner_self.tracker.log_metrics(logs, step=epoch)

                # Check for pruning
                val_loss = logs.get("val_loss", logs.get("loss", 0))
                if inner_self.trial is not None:
                    should_continue = inner_self.tracker.report_intermediate(
                        val_loss, epoch, inner_self.trial
                    )
                    if not should_continue:
                        inner_self.model.stop_training = True

        return KerasCallback()


# Alias for convenience
TensorFlowMLflowKerasCallback = TensorFlowMLflowCallback
