"""
Minimal Two Moons Classification Example.

A simple test case using sklearn's two moons dataset with a
single hidden layer PyTorch MLP. Use this to test and iterate
on the explr framework.

Usage:
    cd /home/ikaro/git/explr
    pip install -e .
    python examples/two_moons.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from explr import TrainingContext, TrainingResult, ExperimentConfig, run_experiment
from explr.core.config import TunerType
from explr.tuning import SearchSpaceBuilder
from explr.utils.memory import limit_gpu_memory


class SimpleMLP(nn.Module):
    """Single hidden layer MLP for binary classification."""

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_model(ctx: TrainingContext) -> TrainingResult:
    """
    Training function for two moons classification.

    Args:
        ctx: TrainingContext with hyperparameters and tracker

    Returns:
        TrainingResult with validation loss
    """
    # Apply memory limit
    limit_gpu_memory()

    # Extract hyperparameters
    hp = ctx.hyperparameters
    hidden_size = hp.get("hidden_size", 32)
    learning_rate = hp.get("learning_rate", 0.01)
    batch_size = hp.get("batch_size", 32)
    epochs = hp.get("epochs", 100)
    noise = hp.get("noise", 0.2)

    # Set seed
    if ctx.seed:
        torch.manual_seed(ctx.seed)

    # Device
    device = torch.device(ctx.device if torch.cuda.is_available() else "cpu")

    # Generate data
    X, y = make_moons(n_samples=1000, noise=noise, random_state=ctx.seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=ctx.seed
    )

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val).unsqueeze(1)

    # Data loaders
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    model = SimpleMLP(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # Log hyperparameters
    ctx.tracker.log_params(hp)

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()

                predicted = (output > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct / total

        # Log metrics
        ctx.tracker.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            },
            step=epoch,
        )

        # Check for pruning (Optuna)
        trial = getattr(ctx, "_optuna_trial", None)
        if trial is not None:
            should_continue = ctx.tracker.report_intermediate(val_loss, epoch, trial)
            if not should_continue:
                break

        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, Acc: {val_accuracy:.2%}"
            )

    return TrainingResult(
        primary_metric=best_val_loss,
        primary_metric_name="val_loss",
        minimize=True,
        metrics={
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "final_val_accuracy": val_accuracy,
        },
        epochs_completed=epoch + 1,
    )


def main():
    """Run the two moons example."""
    import os

    os.makedirs("./checkpoints", exist_ok=True)

    # ===========================================
    # Option 1: Single training run
    # ===========================================
    print("=" * 50)
    print("Two Moons Classification - Single Run")
    print("=" * 50)

    config = ExperimentConfig(
        name="two_moons",
        framework="pytorch",
        seed=42,
        hyperparameters={
            "hidden_size": 32,
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 100,
            "noise": 0.2,
        },
    )

    result = run_experiment(train_model, config)

    print(f"\nTraining completed!")
    print(f"  Best val_loss: {result.primary_metric:.4f}")
    print(f"  Final accuracy: {result.metrics['final_val_accuracy']:.2%}")

    # ===========================================
    # Option 2: Hyperparameter tuning (uncomment)
    # ===========================================
    # print("\n" + "=" * 50)
    # print("Two Moons Classification - Hyperparameter Tuning")
    # print("=" * 50)
    #
    # search_space = (
    #     SearchSpaceBuilder()
    #     .categorical("hidden_size", [16, 32, 64, 128])
    #     .loguniform("learning_rate", 1e-4, 1e-1)
    #     .categorical("batch_size", [16, 32, 64])
    #     .build()
    # )
    #
    # config.tuning.tuner_type = TunerType.OPTUNA
    # config.tuning.n_trials = 20
    # config.tuning.search_space.parameters = search_space
    # config.hyperparameters["epochs"] = 50  # Shorter for tuning
    #
    # results = run_experiment(train_model, config, tune=True)
    #
    # print(f"\nTuning completed!")
    # print(f"  Best val_loss: {results['best_value']:.4f}")
    # print(f"  Best params: {results['best_params']}")


if __name__ == "__main__":
    main()
