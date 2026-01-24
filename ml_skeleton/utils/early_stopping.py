"""Early stopping utility for training loops.

Stops training when validation metric stops improving after a patience period.
Useful for hyperparameter tuning to avoid overfitting and save compute.
"""

import numpy as np
from typing import Optional


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving.

    Args:
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in monitored metric to qualify as improvement
        mode: 'min' for minimization (loss), 'max' for maximization (accuracy)
        verbose: If True, prints messages when improvement occurs or patience runs out
        restore_best_weights: If True, model will be restored to best weights on stop

    Example:
        >>> early_stop = EarlyStopping(patience=5, min_delta=0.001, mode='min')
        >>> for epoch in range(num_epochs):
        >>>     train_loss = train_epoch()
        >>>     val_loss = validate()
        >>>
        >>>     if early_stop(val_loss, epoch):
        >>>         print(f"Early stopping triggered at epoch {epoch}")
        >>>         break
        >>>
        >>>     if early_stop.should_save_checkpoint():
        >>>         save_checkpoint(model, f"best_model.pt")
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True,
        restore_best_weights: bool = False
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights

        # State
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self._improved_this_epoch = False

        # Comparison function
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
            self.best_score = np.inf
        elif mode == 'max':
            self.is_better = lambda current, best: current > best + min_delta
            self.best_score = -np.inf
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

    def __call__(self, metric: float, epoch: int) -> bool:
        """Check if training should stop.

        Args:
            metric: Current validation metric value
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        self._improved_this_epoch = False

        if self.is_better(metric, self.best_score):
            # Improvement found
            self.best_score = metric
            self.best_epoch = epoch
            self.counter = 0
            self._improved_this_epoch = True

            if self.verbose:
                print(f"  [EarlyStopping] Validation metric improved to {metric:.6f}")

        else:
            # No improvement
            self.counter += 1

            if self.verbose:
                print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                if self.verbose:
                    print(f"  [EarlyStopping] Early stopping triggered!")
                    print(f"  [EarlyStopping] Best score: {self.best_score:.6f} at epoch {self.best_epoch + 1}")
                self.early_stop = True
                return True

        return False

    def should_save_checkpoint(self) -> bool:
        """Check if current epoch has the best metric so far.

        Returns:
            True if this epoch improved the metric, False otherwise
        """
        return self._improved_this_epoch

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = np.inf if self.mode == 'min' else -np.inf
        self.early_stop = False
        self.best_epoch = 0
        self._improved_this_epoch = False

    def get_best_score(self) -> float:
        """Get the best metric score achieved.

        Returns:
            Best metric value
        """
        return self.best_score

    def get_best_epoch(self) -> int:
        """Get the epoch with the best metric.

        Returns:
            Epoch number (0-indexed)
        """
        return self.best_epoch

    def stopped_early(self) -> bool:
        """Check if early stopping was triggered.

        Returns:
            True if training was stopped early, False otherwise
        """
        return self.early_stop
