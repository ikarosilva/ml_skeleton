"""A/B testing utilities for comparing classifier models.

Provides statistical tests to compare new model vs production model
on held-out new ratings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from .baseline_classifier import SimpleRatingClassifier
from .dataset import EmbeddingDataset
from .losses import BinaryRatingLoss, RatingLoss


def load_classifier_from_checkpoint(
    checkpoint_path: str,
    embedding_dim: int,
    device: str = "cuda"
) -> tuple[SimpleRatingClassifier, dict]:
    """Load classifier from checkpoint, inferring architecture.

    Args:
        checkpoint_path: Path to checkpoint file
        embedding_dim: Embedding dimension
        device: Device to load model to

    Returns:
        Tuple of (classifier, checkpoint_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Infer hidden_dims from state dict
    hidden_dims = []
    layer_idx = 0
    while f"mlp.{layer_idx}.weight" in state_dict:
        weight = state_dict[f"mlp.{layer_idx}.weight"]
        out_features = weight.shape[0]
        if out_features > 1:
            hidden_dims.append(out_features)
        layer_idx += 3

    classifier = SimpleRatingClassifier(
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        dropout=0.0  # Dropout doesn't matter for inference
    )
    classifier.load_state_dict(state_dict)
    classifier = classifier.to(device)
    classifier.eval()

    return classifier, checkpoint


def run_ab_test(
    new_classifier_path: str,
    prod_classifier_path: str,
    test_dataset: EmbeddingDataset,
    classification_mode: str = "binary",
    device: str = "cuda",
    verbose: bool = True
) -> dict:
    """Run A/B test comparing new classifier vs production classifier.

    Uses McNemar's test for paired binary classification, or paired t-test
    for regression mode.

    Args:
        new_classifier_path: Path to new classifier checkpoint
        prod_classifier_path: Path to production classifier checkpoint
        test_dataset: Dataset of new ratings (held-out from training)
        classification_mode: 'binary' or 'regression'
        device: Device for inference
        verbose: Print detailed results

    Returns:
        Dictionary with A/B test results including:
        - new_accuracy/mae: New model performance
        - prod_accuracy/mae: Production model performance
        - improvement: Absolute improvement
        - p_value: Statistical significance
        - significant: Whether improvement is significant (p < 0.05)
    """
    from scipy import stats

    if len(test_dataset) == 0:
        return {
            'error': 'No test samples available',
            'n_samples': 0
        }

    # Get embedding dimension from dataset
    sample = test_dataset[0]
    embedding_dim = sample['embedding'].shape[0]

    # Load both classifiers
    new_classifier, _ = load_classifier_from_checkpoint(
        new_classifier_path, embedding_dim, device
    )
    prod_classifier, _ = load_classifier_from_checkpoint(
        prod_classifier_path, embedding_dim, device
    )

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0
    )

    # Get predictions from both models
    new_preds = []
    prod_preds = []
    targets = []

    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch['embedding'].to(device)
            ratings = batch['rating'].cpu().numpy()

            new_out = new_classifier(embeddings).cpu().numpy()
            prod_out = prod_classifier(embeddings).cpu().numpy()

            new_preds.extend(new_out.flatten())
            prod_preds.extend(prod_out.flatten())
            targets.extend(ratings.flatten())

    new_preds = np.array(new_preds)
    prod_preds = np.array(prod_preds)
    targets = np.array(targets)
    n_samples = len(targets)

    if classification_mode == "binary":
        # Binary classification: use McNemar's test
        new_probs = 1 / (1 + np.exp(-new_preds))  # Sigmoid
        prod_probs = 1 / (1 + np.exp(-prod_preds))

        new_binary = (new_probs > 0.5).astype(int)
        prod_binary = (prod_probs > 0.5).astype(int)
        targets_binary = targets.astype(int)

        # Accuracy
        new_correct = (new_binary == targets_binary)
        prod_correct = (prod_binary == targets_binary)

        new_accuracy = new_correct.mean()
        prod_accuracy = prod_correct.mean()

        # McNemar's test contingency table
        # b = new correct, prod wrong
        # c = new wrong, prod correct
        b = ((new_correct) & (~prod_correct)).sum()
        c = ((~new_correct) & (prod_correct)).sum()

        # McNemar's test (with continuity correction)
        if b + c > 0:
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)
            p_value = 1 - stats.chi2.cdf(chi2, df=1)
        else:
            p_value = 1.0

        # Precision, Recall, F1 for both
        def compute_metrics(preds, targets):
            tp = ((preds == 1) & (targets == 1)).sum()
            fp = ((preds == 1) & (targets == 0)).sum()
            fn = ((preds == 0) & (targets == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            return {'precision': precision, 'recall': recall, 'f1': f1}

        new_metrics = compute_metrics(new_binary, targets_binary)
        prod_metrics = compute_metrics(prod_binary, targets_binary)

        result = {
            'n_samples': n_samples,
            'classification_mode': 'binary',
            'new_accuracy': new_accuracy,
            'prod_accuracy': prod_accuracy,
            'improvement': new_accuracy - prod_accuracy,
            'improvement_pct': (new_accuracy - prod_accuracy) / prod_accuracy * 100 if prod_accuracy > 0 else 0,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'test_used': "McNemar's test",
            'new_metrics': new_metrics,
            'prod_metrics': prod_metrics,
            'mcnemar_b': int(b),  # new correct, prod wrong
            'mcnemar_c': int(c),  # new wrong, prod correct
        }

    else:
        # Regression: use paired t-test on absolute errors
        new_errors = np.abs(new_preds - targets)
        prod_errors = np.abs(prod_preds - targets)

        new_mae = new_errors.mean()
        prod_mae = prod_errors.mean()

        # Paired t-test (is new model's error significantly different?)
        t_stat, p_value = stats.ttest_rel(prod_errors, new_errors)

        # Correlation
        new_corr = np.corrcoef(new_preds, targets)[0, 1] if np.std(new_preds) > 1e-8 else 0
        prod_corr = np.corrcoef(prod_preds, targets)[0, 1] if np.std(prod_preds) > 1e-8 else 0

        result = {
            'n_samples': n_samples,
            'classification_mode': 'regression',
            'new_mae': new_mae,
            'prod_mae': prod_mae,
            'improvement': prod_mae - new_mae,  # Lower is better
            'improvement_pct': (prod_mae - new_mae) / prod_mae * 100 if prod_mae > 0 else 0,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'test_used': "Paired t-test",
            'new_correlation': new_corr,
            'prod_correlation': prod_corr,
            't_statistic': t_stat,
        }

    if verbose:
        print("\n" + "=" * 60)
        print("A/B TEST RESULTS (New Model vs Production)")
        print("=" * 60)
        print(f"  Test samples: {n_samples} (new ratings since last training)")
        print(f"  Mode: {classification_mode}")
        print()

        if classification_mode == "binary":
            print(f"  NEW MODEL:")
            print(f"    Accuracy:  {result['new_accuracy']:.4f}")
            print(f"    Precision: {new_metrics['precision']:.4f}")
            print(f"    Recall:    {new_metrics['recall']:.4f}")
            print(f"    F1:        {new_metrics['f1']:.4f}")
            print()
            print(f"  PRODUCTION MODEL:")
            print(f"    Accuracy:  {result['prod_accuracy']:.4f}")
            print(f"    Precision: {prod_metrics['precision']:.4f}")
            print(f"    Recall:    {prod_metrics['recall']:.4f}")
            print(f"    F1:        {prod_metrics['f1']:.4f}")
            print()
            print(f"  COMPARISON:")
            print(f"    Accuracy improvement: {result['improvement']:+.4f} ({result['improvement_pct']:+.1f}%)")
        else:
            print(f"  NEW MODEL:")
            print(f"    MAE:         {result['new_mae']:.4f}")
            print(f"    Correlation: {result['new_correlation']:.4f}")
            print()
            print(f"  PRODUCTION MODEL:")
            print(f"    MAE:         {result['prod_mae']:.4f}")
            print(f"    Correlation: {result['prod_correlation']:.4f}")
            print()
            print(f"  COMPARISON:")
            print(f"    MAE improvement: {result['improvement']:+.4f} ({result['improvement_pct']:+.1f}%)")

        print()
        print(f"  STATISTICAL SIGNIFICANCE ({result['test_used']}):")
        print(f"    p-value: {result['p_value']:.4f}")
        if result['significant']:
            print(f"    Result: ✓ SIGNIFICANT (p < 0.05)")
            if classification_mode == "binary" and result['improvement'] > 0:
                print(f"    → New model is significantly BETTER")
            elif classification_mode == "binary" and result['improvement'] < 0:
                print(f"    → New model is significantly WORSE")
            elif result['improvement'] > 0:
                print(f"    → New model is significantly BETTER (lower MAE)")
            else:
                print(f"    → New model is significantly WORSE (higher MAE)")
        else:
            print(f"    Result: Not significant (p >= 0.05)")
            print(f"    → No conclusive difference between models")

        print("=" * 60)

    return result


def format_ab_test_summary(result: dict) -> str:
    """Format A/B test result as a summary string.

    Args:
        result: Result dict from run_ab_test

    Returns:
        Formatted summary string
    """
    if 'error' in result:
        return f"A/B Test: {result['error']}"

    if result['classification_mode'] == 'binary':
        return (
            f"A/B Test: n={result['n_samples']}, "
            f"new={result['new_accuracy']:.3f}, "
            f"prod={result['prod_accuracy']:.3f}, "
            f"Δ={result['improvement']:+.3f}, "
            f"p={result['p_value']:.3f}"
            f"{' ✓' if result['significant'] else ''}"
        )
    else:
        return (
            f"A/B Test: n={result['n_samples']}, "
            f"new_MAE={result['new_mae']:.4f}, "
            f"prod_MAE={result['prod_mae']:.4f}, "
            f"Δ={result['improvement']:+.4f}, "
            f"p={result['p_value']:.3f}"
            f"{' ✓' if result['significant'] else ''}"
        )
