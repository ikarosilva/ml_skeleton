"""A/B testing utilities for comparing classifier models.

Provides statistical tests to compare new model vs production model
on held-out new ratings.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from .baseline_classifier import (
    AttentionRatingClassifier,
    LegacySimpleRatingClassifier,
    SimpleRatingClassifier,
)
from .dataset import EmbeddingDataset
from .losses import BinaryRatingLoss, RatingLoss


def load_classifier_from_checkpoint(
    checkpoint_path: str,
    embedding_dim: int,
    device: str = "cuda"
) -> tuple[SimpleRatingClassifier | LegacySimpleRatingClassifier | AttentionRatingClassifier, dict]:
    """Load classifier from checkpoint, inferring architecture.

    Supports current format (blocks/skips/output) and legacy format (mlp.*).
    Old production checkpoints may use the legacy 'mlp' layout.

    Args:
        checkpoint_path: Path to checkpoint file
        embedding_dim: Embedding dimension (from encoder; genre adds NUM_GENRES if use_genre)
        device: Device to load model to

    Returns:
        Tuple of (classifier, checkpoint_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    use_genre = checkpoint.get('use_genre', False)
    use_batch_norm = checkpoint.get('use_batch_norm', False)
    use_residual = checkpoint.get('use_residual', False)

    # Get hidden_dims from checkpoint or infer from state dict
    hidden_dims = checkpoint.get('hidden_dims')
    if hidden_dims is None:
        hidden_dims = []
        if "blocks.0.0.weight" in state_dict:
            i = 0
            while f"blocks.{i}.0.weight" in state_dict:
                hidden_dims.append(int(state_dict[f"blocks.{i}.0.weight"].shape[0]))
                i += 1
        else:
            layer_idx = 0
            while f"mlp.{layer_idx}.weight" in state_dict:
                weight = state_dict[f"mlp.{layer_idx}.weight"]
                out_features = weight.shape[0]
                if out_features > 1:
                    hidden_dims.append(int(out_features))
                layer_idx += 3
    else:
        if isinstance(hidden_dims, str):
            hidden_dims = ast.literal_eval(hidden_dims)
        hidden_dims = list(hidden_dims)

    # Attention checkpoints (chunk_proj, pos_embed, query) vs MLP (blocks/skips from embedding)
    is_attention = (
        checkpoint.get("classifier_type") == "AttentionRatingClassifier"
        or "chunk_proj.weight" in state_dict
    )

    if "mlp.0.weight" in state_dict:
        classifier = LegacySimpleRatingClassifier(
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            use_genre=use_genre,
        )
    elif is_attention:
        d_model = checkpoint.get("d_model", 512)
        num_heads = checkpoint.get("num_heads", 4)
        max_chunks = checkpoint.get("max_chunks", 16)
        use_pos_encoding = checkpoint.get("use_pos_encoding", True)
        classifier = AttentionRatingClassifier(
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=0.0,
            use_genre=use_genre,
            use_batch_norm=use_batch_norm,
            use_residual=use_residual,
            d_model=d_model,
            num_heads=num_heads,
            max_chunks=max_chunks,
            use_pos_encoding=use_pos_encoding,
        )
    else:
        classifier = SimpleRatingClassifier(
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=0.0,  # Dropout doesn't matter for inference
            use_genre=use_genre,
            use_batch_norm=use_batch_norm,
            use_residual=use_residual,
        )
    classifier.load_state_dict(state_dict, strict=True)
    classifier = classifier.to(device)
    classifier.eval()

    return classifier, checkpoint


def run_ab_test(
    new_classifier_path: str,
    prod_classifier_path: str,
    test_dataset: EmbeddingDataset,
    classification_mode: str = "binary",
    device: str = "cuda",
    verbose: bool = True,
    prod_test_dataset: Optional[EmbeddingDataset] = None,
) -> dict:
    """Run A/B test comparing new classifier vs production classifier.

    Uses McNemar's test for paired binary classification, or paired t-test
    for regression mode.

    Args:
        new_classifier_path: Path to new classifier checkpoint
        prod_classifier_path: Path to production classifier checkpoint
        test_dataset: Dataset for NEW model (e.g. new encoder's embeddings)
        classification_mode: 'binary' or 'regression'
        device: Device for inference
        verbose: Print detailed results
        prod_test_dataset: If provided, production model is run on this dataset
            (e.g. original embeddings) so prod pipeline never changes. Must have
            same length and sample order as test_dataset. When None, both models
            run on test_dataset.

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
    if prod_test_dataset is not None and len(prod_test_dataset) != len(test_dataset):
        return {
            'error': f'prod_test_dataset length {len(prod_test_dataset)} != test_dataset length {len(test_dataset)}',
            'n_samples': 0
        }

    # Get embedding dimension from dataset (support per-song (D,) or (C, D) chunks)
    sample = test_dataset[0]
    emb = sample['embedding']
    if hasattr(emb, 'ndim') and emb.ndim == 2:
        embedding_dim = int(emb.shape[-1])
    else:
        embedding_dim = int(emb.shape[0])

    # Load both classifiers (each may have use_genre from its checkpoint)
    new_classifier, new_ckpt = load_classifier_from_checkpoint(
        new_classifier_path, embedding_dim, device
    )
    try:
        prod_classifier, prod_ckpt = load_classifier_from_checkpoint(
            prod_classifier_path, embedding_dim, device
        )
    except RuntimeError as e:
        # Common case: production classifier was trained on a different embedding_dim
        # (e.g. 2048) than the current pipeline (e.g. 4096). In that case, we skip
        # the A/B test rather than failing the whole training run.
        return {
            "error": "Production classifier incompatible with current embeddings (skipping A/B test)",
            "details": str(e),
            "n_samples": len(test_dataset),
        }
    new_use_genre = new_ckpt.get('use_genre', False)
    prod_use_genre = prod_ckpt.get('use_genre', False)
    chunk_agg_new = new_ckpt.get('chunk_aggregation', 'mean')
    chunk_agg_prod = prod_ckpt.get('chunk_aggregation', 'mean')

    def aggregate(view):  # (B, C) -> (B,)
        return view.max(dim=1)[0] if (view.shape[1] > 1 and chunk_agg_new == 'max') else view.mean(dim=1)
    def aggregate_prod(view):
        return view.max(dim=1)[0] if (view.shape[1] > 1 and chunk_agg_prod == 'max') else view.mean(dim=1)

    # Data loaders: new model on test_dataset; prod on prod_test_dataset or same
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0
    )
    prod_loader: DataLoader
    if prod_test_dataset is not None:
        prod_loader = DataLoader(
            prod_test_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=0
        )
    else:
        prod_loader = test_loader

    # Get predictions: new on test_dataset, prod on prod_test_dataset (or same)
    new_preds = []
    prod_preds = []
    targets = []

    with torch.no_grad():
        for batch_new, batch_prod in zip(test_loader, prod_loader):
            emb_new = batch_new['embedding'].to(device)
            emb_prod = batch_prod['embedding'].to(device)
            ratings = batch_new['rating'].cpu().numpy()
            genre_new = batch_new.get('genre')
            genre_prod = batch_prod.get('genre')
            if genre_new is not None:
                genre_new = genre_new.to(device)
            if genre_prod is not None:
                genre_prod = genre_prod.to(device)

            new_genre = genre_new if new_use_genre else None
            prod_genre = genre_prod if prod_use_genre else None
            # Chunked (B, C, D): attention classifier uses (B,C,D)->(B,); MLP uses flatten then aggregate
            new_handles_chunks = getattr(new_classifier, "handles_chunk_sequence", False)
            prod_handles_chunks = getattr(prod_classifier, "handles_chunk_sequence", False)
            if emb_new.dim() == 3:
                B, C, D = emb_new.shape
                if new_handles_chunks:
                    new_out = new_classifier(emb_new, new_genre).cpu().numpy()
                else:
                    emb_new_flat = emb_new.view(B * C, D)
                    if new_genre is not None:
                        new_genre = new_genre.unsqueeze(1).expand(-1, C, -1).reshape(B * C, new_genre.size(-1))
                    new_out = aggregate(new_classifier(emb_new_flat, new_genre).view(B, C)).cpu().numpy()
            else:
                new_out = new_classifier(emb_new, new_genre).cpu().numpy()
            if emb_prod.dim() == 3:
                B, C, D = emb_prod.shape
                if prod_handles_chunks:
                    prod_out = prod_classifier(emb_prod, prod_genre).cpu().numpy()
                else:
                    emb_prod_flat = emb_prod.view(B * C, D)
                    if prod_genre is not None:
                        prod_genre = prod_genre.unsqueeze(1).expand(-1, C, -1).reshape(B * C, prod_genre.size(-1))
                    prod_out = aggregate_prod(prod_classifier(emb_prod_flat, prod_genre).view(B, C)).cpu().numpy()
            else:
                prod_out = prod_classifier(emb_prod, prod_genre).cpu().numpy()

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


def ab_result_to_mlflow_metrics(result: dict) -> dict:
    """Build a flat dict of A/B test metrics for MLflow (all numeric).

    Use with MlflowClient().log_metric(run_id, k, v) or log_metrics().

    Args:
        result: Result dict from run_ab_test (must not be error dict).

    Returns:
        Dict of metric name -> float (keys prefixed with ab_ for clarity).
    """
    if result.get("n_samples", 0) == 0 or "classification_mode" not in result:
        return {}
    out = {
        "ab_n_samples": float(result["n_samples"]),
        "ab_new_accuracy": float(result["new_accuracy"]),
        "ab_prod_accuracy": float(result["prod_accuracy"]),
        "ab_improvement": float(result["improvement"]),
        "ab_improvement_pct": float(result.get("improvement_pct", 0)),
        "ab_p_value": float(result["p_value"]),
        "ab_significant": float(result["significant"]),
    }
    if result["classification_mode"] == "binary":
        nm = result.get("new_metrics") or {}
        pm = result.get("prod_metrics") or {}
        out["ab_new_precision"] = float(nm.get("precision", 0))
        out["ab_new_recall"] = float(nm.get("recall", 0))
        out["ab_new_f1"] = float(nm.get("f1", 0))
        out["ab_prod_precision"] = float(pm.get("precision", 0))
        out["ab_prod_recall"] = float(pm.get("recall", 0))
        out["ab_prod_f1"] = float(pm.get("f1", 0))
    else:
        out["ab_new_mae"] = float(result["new_mae"])
        out["ab_prod_mae"] = float(result["prod_mae"])
        out["ab_new_correlation"] = float(result.get("new_correlation", 0))
        out["ab_prod_correlation"] = float(result.get("prod_correlation", 0))
    return out


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
