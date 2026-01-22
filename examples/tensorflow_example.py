"""
Minimal TensorFlow/Keras example for the explr framework.

This is a skeleton implementation showing the interface.
Fill in the TODO sections with your actual training logic.
"""

from ml_skeleton import TrainingContext, TrainingResult, ExperimentConfig, run_experiment
from ml_skeleton.core.config import TunerType
from ml_skeleton.tuning import SearchSpaceBuilder


def train_model(ctx: TrainingContext) -> TrainingResult:
    """
    User-provided training function using TensorFlow/Keras.

    This is where you implement your training logic:
    - Data loading and preprocessing
    - Model creation
    - Training with Keras fit()
    - Evaluation
    - Model saving

    Args:
        ctx: TrainingContext with hyperparameters and MLflow tracker

    Returns:
        TrainingResult with the primary optimization metric
    """
    # Import TensorFlow (only when function is called)
    import tensorflow as tf
    import numpy as np

    # Extract hyperparameters from context
    hp = ctx.hyperparameters
    learning_rate = hp.get("learning_rate", 0.001)
    batch_size = hp.get("batch_size", 32)
    hidden_size = hp.get("hidden_size", 256)
    dropout = hp.get("dropout", 0.2)
    epochs = hp.get("epochs", 10)

    # Set random seed for reproducibility
    if ctx.seed:
        tf.random.set_seed(ctx.seed)
        np.random.seed(ctx.seed)

    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    # Log the hyperparameters being used
    ctx.tracker.log_params(hp)

    # =================================================================
    # TODO: Replace this section with your actual data loading
    # =================================================================
    # Example placeholder - create dummy data
    # In real usage:
    #   train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    #   val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    input_size = 784  # Example: MNIST-like
    num_classes = 10
    num_train_samples = 1000
    num_val_samples = 200

    # Dummy data for demonstration
    x_train = np.random.randn(num_train_samples, input_size).astype(np.float32)
    y_train = np.random.randint(0, num_classes, (num_train_samples,))
    x_val = np.random.randn(num_val_samples, input_size).astype(np.float32)
    y_val = np.random.randint(0, num_classes, (num_val_samples,))

    # =================================================================
    # TODO: Replace this section with your actual model
    # =================================================================
    # Example simple model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_size,)),
            tf.keras.layers.Dense(hidden_size, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(hidden_size // 2, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # =================================================================
    # Custom callback for MLflow logging
    # =================================================================
    class MLflowCallback(tf.keras.callbacks.Callback):
        def __init__(self, tracker, trial=None):
            super().__init__()
            self.tracker = tracker
            self.trial = trial
            self.best_val_loss = float("inf")

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}

            # Log metrics to MLflow
            self.tracker.log_metrics(
                {
                    "train_loss": logs.get("loss", 0),
                    "train_accuracy": logs.get("accuracy", 0),
                    "val_loss": logs.get("val_loss", 0),
                    "val_accuracy": logs.get("val_accuracy", 0),
                },
                step=epoch,
            )

            # Track best validation loss
            val_loss = logs.get("val_loss", float("inf"))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

            # Check for pruning (Optuna integration)
            if self.trial is not None:
                should_continue = self.tracker.report_intermediate(
                    val_loss, epoch, self.trial
                )
                if not should_continue:
                    self.model.stop_training = True

    # Create callbacks
    trial = getattr(ctx, "_optuna_trial", None)
    mlflow_callback = MLflowCallback(ctx.tracker, trial)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{ctx.checkpoint_dir}/best_model.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )

    # =================================================================
    # Training
    # =================================================================
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[mlflow_callback, checkpoint_callback],
        verbose=1,
    )

    # Get final metrics
    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    final_val_accuracy = history.history["val_accuracy"][-1]
    epochs_completed = len(history.history["loss"])

    # =================================================================
    # Return results
    # =================================================================
    return TrainingResult(
        primary_metric=mlflow_callback.best_val_loss,
        primary_metric_name="val_loss",
        minimize=True,
        metrics={
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "final_val_accuracy": final_val_accuracy,
        },
        best_model_path=f"{ctx.checkpoint_dir}/best_model.keras",
        epochs_completed=epochs_completed,
    )


def main():
    """Run the example."""
    import os

    # Create checkpoint directory
    os.makedirs("./checkpoints", exist_ok=True)

    # Option 1: Single training run
    print("=" * 50)
    print("Running single training...")
    print("=" * 50)

    config = ExperimentConfig(
        name="tensorflow_example",
        framework="tensorflow",
        seed=42,
        hyperparameters={
            "epochs": 5,
            "learning_rate": 0.001,
            "batch_size": 32,
            "hidden_size": 256,
            "dropout": 0.2,
        },
    )

    result = run_experiment(train_model, config)
    print(f"\nTraining completed!")
    print(f"  Best val_loss: {result.primary_metric:.4f}")
    print(f"  Epochs: {result.epochs_completed}")

    # Option 2: Hyperparameter tuning (uncomment to run)
    # print("\n" + "=" * 50)
    # print("Running hyperparameter tuning...")
    # print("=" * 50)
    #
    # # Define search space
    # search_space = (
    #     SearchSpaceBuilder()
    #     .loguniform("learning_rate", 1e-4, 1e-1)
    #     .categorical("batch_size", [16, 32, 64])
    #     .categorical("hidden_size", [128, 256, 512])
    #     .uniform("dropout", 0.0, 0.5)
    #     .build()
    # )
    #
    # config.tuning.tuner_type = TunerType.OPTUNA
    # config.tuning.n_trials = 10
    # config.tuning.search_space.parameters = search_space
    # config.hyperparameters["epochs"] = 5  # Shorter for tuning
    #
    # results = run_experiment(train_model, config, tune=True)
    # print(f"\nTuning completed!")
    # print(f"  Best value: {results['best_value']:.4f}")
    # print(f"  Best params: {results['best_params']}")


if __name__ == "__main__":
    main()
