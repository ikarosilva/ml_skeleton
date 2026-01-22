"""
Minimal PyTorch example for the explr framework.

This is a skeleton implementation showing the interface.
Fill in the TODO sections with your actual training logic.
"""

from ml_skeleton import TrainingContext, TrainingResult, ExperimentConfig, run_experiment
from ml_skeleton.core.config import TunerType
from ml_skeleton.tuning import SearchSpaceBuilder


def train_model(ctx: TrainingContext) -> TrainingResult:
    """
    User-provided training function.

    This is where you implement your training logic:
    - Data loading and preprocessing
    - Model creation
    - Training loop
    - Evaluation
    - Model saving

    Args:
        ctx: TrainingContext with hyperparameters and MLflow tracker

    Returns:
        TrainingResult with the primary optimization metric
    """
    # Import PyTorch (only when function is called)
    import torch
    import torch.nn as nn

    # Extract hyperparameters from context
    hp = ctx.hyperparameters
    learning_rate = hp.get("learning_rate", 0.001)
    batch_size = hp.get("batch_size", 32)
    hidden_size = hp.get("hidden_size", 256)
    dropout = hp.get("dropout", 0.2)
    epochs = hp.get("epochs", 10)

    # Set up device
    device = torch.device(ctx.device if torch.cuda.is_available() else "cpu")

    # Set random seed for reproducibility
    if ctx.seed:
        torch.manual_seed(ctx.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(ctx.seed)

    # Log the hyperparameters being used
    ctx.tracker.log_params(hp)

    # =================================================================
    # TODO: Replace this section with your actual data loading
    # =================================================================
    # Example placeholder - create dummy data
    # In real usage:
    #   train_loader = DataLoader(YourDataset(...), batch_size=batch_size, ...)
    #   val_loader = DataLoader(YourDataset(...), batch_size=batch_size, ...)

    input_size = 784  # Example: MNIST-like
    num_classes = 10
    num_train_samples = 1000
    num_val_samples = 200

    # Dummy data for demonstration
    train_x = torch.randn(num_train_samples, input_size)
    train_y = torch.randint(0, num_classes, (num_train_samples,))
    val_x = torch.randn(num_val_samples, input_size)
    val_y = torch.randint(0, num_classes, (num_val_samples,))

    # =================================================================
    # TODO: Replace this section with your actual model
    # =================================================================
    # Example simple model
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size // 2, num_classes),
    )
    model = model.to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # =================================================================
    # Training loop
    # =================================================================
    best_val_loss = float("inf")
    best_model_path = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_x_device = train_x.to(device)
        train_y_device = train_y.to(device)

        optimizer.zero_grad()
        outputs = model(train_x_device)
        loss = criterion(outputs, train_y_device)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_x_device = val_x.to(device)
            val_y_device = val_y.to(device)
            val_outputs = model(val_x_device)
            val_loss = criterion(val_outputs, val_y_device).item()

            # Calculate accuracy
            _, predicted = val_outputs.max(1)
            correct = predicted.eq(val_y_device).sum().item()
            val_accuracy = correct / len(val_y)

        # Log metrics to MLflow
        ctx.tracker.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            },
            step=epoch,
        )

        # Check for pruning (Optuna integration)
        trial = getattr(ctx, "_optuna_trial", None)
        if trial is not None:
            should_continue = ctx.tracker.report_intermediate(val_loss, epoch, trial)
            if not should_continue:
                # Trial was pruned
                break

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = f"{ctx.checkpoint_dir}/best_model.pt"
            torch.save(model.state_dict(), best_model_path)

        # Print progress
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_accuracy:.4f}"
        )

    # =================================================================
    # Return results
    # =================================================================
    return TrainingResult(
        primary_metric=best_val_loss,
        primary_metric_name="val_loss",
        minimize=True,
        metrics={
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "final_val_accuracy": val_accuracy,
        },
        best_model_path=best_model_path,
        epochs_completed=epoch + 1,
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
        name="pytorch_example",
        framework="pytorch",
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
