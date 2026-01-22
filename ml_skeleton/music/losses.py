"""Loss functions for music recommendation training.

Includes:
- Rating prediction loss (MSE)
- Multi-task loss with album classification
- Contrastive loss for self-supervised encoder training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RatingLoss(nn.Module):
    """Simple MSE loss for rating prediction.

    Predicts continuous ratings in [0, 1] range.

    Args:
        None
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE loss.

        Args:
            predictions: Predicted ratings, shape (batch_size, 1)
            targets: Target ratings, shape (batch_size, 1) or (batch_size,)
                    Unrated songs have rating < 0 and are excluded from loss

        Returns:
            loss: Scalar loss value
        """
        # Ensure shapes match
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        # Mask out unrated songs (rating < 0)
        valid_mask = (targets >= 0).squeeze()

        if valid_mask.sum() == 0:
            # No rated songs in batch - return zero loss with gradient
            return torch.zeros(1, device=predictions.device, requires_grad=True).mean()

        # Only compute loss on rated songs
        valid_predictions = predictions[valid_mask]
        valid_targets = targets[valid_mask]

        return self.mse(valid_predictions, valid_targets)


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining rating prediction and album classification.

    Supports songs appearing on multiple albums. For songs with multiple albums,
    averages the classification loss across all valid albums.

    Args:
        rating_weight: Weight for rating loss
        album_weight: Weight for album classification loss
    """

    def __init__(
        self,
        rating_weight: float = 1.0,
        album_weight: float = 0.5
    ):
        super().__init__()
        self.rating_weight = rating_weight
        self.album_weight = album_weight
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(
        self,
        rating_preds: torch.Tensor,
        rating_targets: torch.Tensor,
        album_logits: torch.Tensor,
        album_labels: list[list[int]]
    ) -> tuple[torch.Tensor, dict]:
        """Compute combined loss with multi-album support.

        Args:
            rating_preds: Predicted ratings, shape (batch_size, 1)
            rating_targets: Target ratings, shape (batch_size,) or (batch_size, 1)
            album_logits: Album classification logits, shape (batch_size, num_albums)
            album_labels: List of album indices per song
                         Length: batch_size
                         Each element: list[int] of album indices
                         Example: [[0, 5], [2], [], [1, 3, 7]]
                         Empty list = song not in any album

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Rating loss (only on rated songs)
        if rating_targets.dim() == 1:
            rating_targets = rating_targets.unsqueeze(1)

        # Mask out unrated songs (rating < 0)
        valid_mask = (rating_targets >= 0).squeeze()

        if valid_mask.sum() == 0:
            # No rated songs - rating loss is zero
            rating_loss = torch.zeros(1, device=rating_preds.device)
        else:
            valid_preds = rating_preds[valid_mask]
            valid_targets = rating_targets[valid_mask]
            rating_loss = self.mse(valid_preds, valid_targets)

        # Album classification loss (averaged over multiple albums per song)
        album_loss = self._compute_multi_album_loss(album_logits, album_labels)

        # Combined loss
        total_loss = (
            self.rating_weight * rating_loss +
            self.album_weight * album_loss
        )

        loss_dict = {
            "total": total_loss.item(),
            "rating": rating_loss.item(),
            "album": album_loss.item()
        }

        return total_loss, loss_dict

    def _compute_multi_album_loss(
        self,
        logits: torch.Tensor,
        album_labels: list[list[int]]
    ) -> torch.Tensor:
        """Compute album classification loss with multi-album support.

        For each song, averages cross-entropy loss over all its albums.
        Songs with no albums contribute zero loss.

        Args:
            logits: Shape (batch_size, num_albums)
            album_labels: List of lists of album indices

        Returns:
            loss: Scalar loss value
        """
        batch_size = logits.size(0)
        total_loss = 0.0
        num_valid = 0

        for i in range(batch_size):
            song_albums = album_labels[i]

            # Skip songs with no album labels
            if len(song_albums) == 0:
                continue

            # Compute loss for each album, then average
            song_loss = 0.0
            for album_idx in song_albums:
                target = torch.tensor([album_idx], device=logits.device)
                song_loss += self.ce(logits[i:i+1], target)

            # Average over all albums for this song
            song_loss = song_loss / len(song_albums)
            total_loss += song_loss
            num_valid += 1

        # Average over all songs with valid albums
        if num_valid > 0:
            total_loss = total_loss / num_valid
        else:
            total_loss = torch.tensor(0.0, device=logits.device)

        return total_loss


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy Loss (SimCLR).

    Contrastive loss for self-supervised encoder training.
    Useful for pre-training encoders before rating prediction.

    Treats augmented versions of the same song as positive pairs,
    all other songs in batch as negatives.

    Args:
        temperature: Temperature scaling parameter
        use_cosine_similarity: Use cosine similarity (True) or dot product (False)
    """

    def __init__(
        self,
        temperature: float = 0.5,
        use_cosine_similarity: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            embeddings: Embedding vectors, shape (2 * batch_size, embedding_dim)
                       First half: original songs
                       Second half: augmented versions (positive pairs)
            labels: Optional labels for supervised contrastive learning
                   If None, uses augmentation-based pairs

        Returns:
            loss: Scalar loss value
        """
        device = embeddings.device
        batch_size = embeddings.size(0) // 2

        # Normalize embeddings if using cosine similarity
        if self.use_cosine_similarity:
            embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T)

        # Create mask for positive pairs
        # Positive pairs: (i, i + batch_size) for i in [0, batch_size)
        mask = torch.zeros((2 * batch_size, 2 * batch_size), device=device)
        for i in range(batch_size):
            mask[i, i + batch_size] = 1
            mask[i + batch_size, i] = 1

        # Mask out self-similarity
        mask_diag = torch.eye(2 * batch_size, device=device)
        mask = mask * (1 - mask_diag)

        # Compute loss
        similarity_matrix = similarity_matrix / self.temperature

        # For each anchor, compute log probability of positive pair
        exp_sim = torch.exp(similarity_matrix)

        # Sum over all similarities (excluding self)
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True) - torch.diag(exp_sim).unsqueeze(1)

        # Positive pair similarities
        pos_sim = (exp_sim * mask).sum(dim=1)

        # Loss: -log(pos / sum)
        loss = -torch.log(pos_sim / (sum_exp_sim.squeeze() + 1e-8))
        loss = loss.mean()

        return loss


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss using rating labels.

    Similar songs (similar ratings) should have similar embeddings.
    Pulls together songs with similar ratings, pushes apart dissimilar ones.

    Args:
        temperature: Temperature scaling parameter
        rating_threshold: Max rating difference for positive pairs
    """

    def __init__(
        self,
        temperature: float = 0.5,
        rating_threshold: float = 0.2
    ):
        super().__init__()
        self.temperature = temperature
        self.rating_threshold = rating_threshold

    def forward(
        self,
        embeddings: torch.Tensor,
        ratings: torch.Tensor
    ) -> torch.Tensor:
        """Compute supervised contrastive loss.

        Args:
            embeddings: Shape (batch_size, embedding_dim)
            ratings: Shape (batch_size,) in [0, 1]

        Returns:
            loss: Scalar loss value
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        # Filter out unrated songs (rating < 0)
        valid_mask = ratings >= 0

        if valid_mask.sum() < 2:
            # Need at least 2 rated songs for contrastive learning
            return torch.zeros(1, device=device, requires_grad=True).mean()

        # Only use rated songs
        valid_embeddings = embeddings[valid_mask]
        valid_ratings = ratings[valid_mask]
        valid_batch_size = valid_embeddings.size(0)

        # Normalize embeddings
        valid_embeddings = F.normalize(valid_embeddings, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(valid_embeddings, valid_embeddings.T)
        similarity_matrix = similarity_matrix / self.temperature

        # Create positive pair mask based on rating similarity
        rating_diff = torch.abs(valid_ratings.unsqueeze(0) - valid_ratings.unsqueeze(1))
        pos_mask = (rating_diff < self.rating_threshold).float()

        # Remove self-similarity
        pos_mask = pos_mask * (1 - torch.eye(valid_batch_size, device=device))

        # Compute loss for each anchor
        exp_sim = torch.exp(similarity_matrix)

        # Sum over all similarities (excluding self)
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True) - torch.diag(exp_sim).unsqueeze(1)

        # For each anchor, sum over all positive pairs
        losses = []

        for i in range(valid_batch_size):
            num_pos = pos_mask[i].sum()
            if num_pos > 0:
                # Sum positive similarities
                pos_sim = (exp_sim[i] * pos_mask[i]).sum()
                # Loss: -log(pos / sum)
                anchor_loss = -torch.log(pos_sim / (sum_exp_sim[i] + 1e-8))
                losses.append(anchor_loss)

        if len(losses) > 0:
            loss = torch.stack(losses).mean()
        else:
            # No valid positive pairs - return zero loss with gradient
            loss = torch.zeros(1, device=device, requires_grad=True).mean()

        return loss


def build_album_mapping(songs: list) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Build album mappings for multi-album support.

    Args:
        songs: List of Song objects

    Returns:
        album_to_idx: Mapping from album key to integer index
        filename_to_albums: Mapping from filename to list of album keys

    Note:
        Album key format: "artist|||album" (handles same album name by different artists)
    """
    # Collect all unique albums
    albums_set = set()
    filename_to_albums_list = {}

    for song in songs:
        # Create album key (artist|||album)
        album_key = f"{song.artist}|||{song.album}"
        albums_set.add(album_key)

        # Add to filename mapping
        if song.filename not in filename_to_albums_list:
            filename_to_albums_list[song.filename] = []
        if album_key not in filename_to_albums_list[song.filename]:
            filename_to_albums_list[song.filename].append(album_key)

    # Create album to index mapping (sorted for consistency)
    albums_sorted = sorted(albums_set)
    album_to_idx = {album: idx for idx, album in enumerate(albums_sorted)}

    return album_to_idx, filename_to_albums_list
