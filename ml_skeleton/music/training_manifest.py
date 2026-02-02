"""Training manifest for tracking which songs were used in classifier training.

Enables proper A/B testing by tracking:
- Which songs were in training set
- Which songs were in validation set
- Which songs are in the A/B test vault (never used for training)
- When training occurred
- Model version info

The vault contains a fixed set of ratings (default 200) that are NEVER used for
training, only for A/B testing. This allows multiple A/B test runs without
needing to rate new songs each time.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

# Default number of ratings to keep in the A/B test vault
DEFAULT_VAULT_SIZE = 200


class TrainingManifest:
    """Tracks training/validation split for classifier training.

    Enables:
    1. Recording which files were used for training vs validation
    2. Maintaining a "vault" of ratings reserved for A/B testing only
    3. A/B testing on truly held-out vault ratings

    The vault is populated once and never changes (unless manually reset).
    New ratings beyond the vault size go to training/validation.

    Example:
        # During training
        manifest = TrainingManifest.load_or_create("prod/training_manifest.json")
        train_files, val_files, vault_files = manifest.split_with_vault(
            all_rated_files, train_ratio=0.8, vault_size=200
        )

        # After training
        manifest.save()

        # For A/B testing - always use vault files
        vault_files = manifest.get_vault_files()
    """

    def __init__(self, path: str = "training_manifest.json"):
        """Initialize training manifest.

        Args:
            path: Path to manifest JSON file
        """
        self.path = Path(path)
        self.training_files: set[str] = set()
        self.validation_files: set[str] = set()
        self.vault_files: set[str] = set()  # Reserved for A/B testing only
        self.training_timestamp: Optional[str] = None
        self.encoder_version: Optional[str] = None
        self.classifier_version: Optional[str] = None
        self.classification_mode: Optional[str] = None
        self.metadata: dict = {}

    @classmethod
    def load_or_create(cls, path: str) -> "TrainingManifest":
        """Load existing manifest or create new one.

        Args:
            path: Path to manifest file

        Returns:
            TrainingManifest instance
        """
        manifest = cls(path)
        if manifest.path.exists():
            manifest.load()
        return manifest

    def load(self) -> None:
        """Load manifest from JSON file."""
        with open(self.path, 'r') as f:
            data = json.load(f)

        self.training_files = set(data.get('training_files', []))
        self.validation_files = set(data.get('validation_files', []))
        self.vault_files = set(data.get('vault_files', []))
        self.training_timestamp = data.get('training_timestamp')
        self.encoder_version = data.get('encoder_version')
        self.classifier_version = data.get('classifier_version')
        self.classification_mode = data.get('classification_mode')
        self.metadata = data.get('metadata', {})

    def save(self) -> None:
        """Save manifest to JSON file."""
        data = {
            'training_files': sorted(list(self.training_files)),
            'validation_files': sorted(list(self.validation_files)),
            'vault_files': sorted(list(self.vault_files)),
            'training_timestamp': self.training_timestamp,
            'encoder_version': self.encoder_version,
            'classifier_version': self.classifier_version,
            'classification_mode': self.classification_mode,
            'metadata': self.metadata
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2)

    def split_with_vault(
        self,
        all_files: list[str],
        train_ratio: float = 0.8,
        vault_size: int = DEFAULT_VAULT_SIZE,
        seed: int = 42,
        file_ratings: Optional[dict[str, int]] = None
    ) -> tuple[list[str], list[str], list[str]]:
        """Split files, reserving vault_size files for A/B testing only.

        The vault uses a ROLLING strategy:
        - New ratings (files not in train/val/vault) go INTO the vault
        - When vault exceeds vault_size, oldest files move to training
        - If file_ratings provided, maintains ~50% class balance in vault

        Args:
            all_files: List of all rated filenames
            train_ratio: Ratio for training set (of non-vault files)
            vault_size: Number of files to reserve for A/B testing (default 200)
            seed: Random seed for initial split
            file_ratings: Optional dict of {filename: binary_rating} for class balancing

        Returns:
            Tuple of (training_files, validation_files, vault_files)
            - training_files: Files to use for training
            - validation_files: Files to use for validation
            - vault_files: Files reserved for A/B testing (never used for training)
        """
        import random

        all_files_set = set(all_files)
        known_files = self.training_files | self.validation_files | self.vault_files

        # Find truly new files (not in any previous set)
        new_files = [f for f in all_files if f not in known_files]

        if self.vault_files:
            # Existing vault - use rolling strategy
            vault_files_before = set(f for f in self.vault_files if f in all_files_set)
            vault_files = list(vault_files_before)

            if new_files:
                # Add new files to vault (combined list, may exceed vault_size)
                if file_ratings:
                    # Class-balanced: combine and trim, tracking what was removed
                    combined = new_files + vault_files
                    vault_files = self._add_balanced_to_vault(
                        vault_files, new_files, vault_size, file_ratings
                    )
                else:
                    # Simple FIFO: new files go in, old files get pushed out
                    vault_files = new_files + vault_files

                # Trim vault to size and identify pushed-out files
                if len(vault_files) > vault_size:
                    vault_files = vault_files[:vault_size]

                # Find files that were in vault before but aren't now
                vault_files_set = set(vault_files)
                pushed_out = [f for f in vault_files_before if f not in vault_files_set]

                print(f"  Rolling vault update:")
                print(f"    New ratings added to vault: {len(new_files)}")
                if pushed_out:
                    # Split pushed files 80/20 between training and validation
                    random.shuffle(pushed_out)
                    split_idx = int(len(pushed_out) * train_ratio)
                    pushed_to_train = pushed_out[:split_idx]
                    pushed_to_val = pushed_out[split_idx:]
                    print(f"    Old vault files → training: {len(pushed_to_train)}")
                    print(f"    Old vault files → validation: {len(pushed_to_val)}")
                    self.training_files.update(pushed_to_train)
                    self.validation_files.update(pushed_to_val)
                else:
                    print(f"    Vault not yet full, no files pushed out")
            else:
                print(f"  Using existing vault: {len(vault_files)} files for A/B testing")
        else:
            # Create new vault from random sample (or all new files if first time)
            random.seed(seed)
            shuffled = all_files.copy()
            random.shuffle(shuffled)

            if file_ratings:
                # Class-balanced initial vault
                vault_files = self._create_balanced_vault(shuffled, vault_size, file_ratings)
            else:
                vault_files = shuffled[:min(vault_size, len(shuffled))]

            print(f"  Created new vault: {len(vault_files)} files reserved for A/B testing")

        vault_set = set(vault_files)

        # Remaining files (not in vault) go to training/validation
        available_files = [f for f in all_files if f not in vault_set]

        # If we have previous training files, use them (excluding vault)
        if self.training_files:
            train_files = [f for f in self.training_files if f in all_files_set and f not in vault_set]
            val_files = [f for f in available_files if f not in self.training_files]

            print(f"  Using previous training manifest:")
            print(f"    Training files: {len(train_files)}")
            print(f"    Validation files: {len(val_files)}")
        else:
            # First training - do random split of non-vault files
            random.seed(seed + 1)  # Different seed for train/val split
            shuffled = available_files.copy()
            random.shuffle(shuffled)

            split_idx = int(len(shuffled) * train_ratio)
            train_files = shuffled[:split_idx]
            val_files = shuffled[split_idx:]

            print(f"  Initial training split:")
            print(f"    Training: {len(train_files)} files")
            print(f"    Validation: {len(val_files)} files")

        # Show class balance if ratings available
        if file_ratings:
            vault_likes = sum(1 for f in vault_files if file_ratings.get(f, 0) == 1)
            vault_dislikes = len(vault_files) - vault_likes
            print(f"    Vault (A/B test only): {len(vault_files)} files ({vault_likes} likes, {vault_dislikes} dislikes)")
        else:
            print(f"    Vault (A/B test only): {len(vault_files)} files")

        # Update manifest
        self.training_files = set(train_files)
        self.validation_files = set(val_files)
        self.vault_files = set(vault_files)
        self.training_timestamp = datetime.now().isoformat()

        return train_files, val_files, vault_files

    def _create_balanced_vault(
        self,
        files: list[str],
        vault_size: int,
        file_ratings: dict[str, int]
    ) -> list[str]:
        """Create a class-balanced vault from files.

        Args:
            files: List of files to sample from (already shuffled)
            vault_size: Target vault size
            file_ratings: Dict of {filename: binary_rating (0 or 1)}

        Returns:
            List of vault files with ~50% class balance
        """
        likes = [f for f in files if file_ratings.get(f, 0) == 1]
        dislikes = [f for f in files if file_ratings.get(f, 0) == 0]

        # Take equal amounts from each class
        half_size = vault_size // 2
        vault_likes = likes[:min(half_size, len(likes))]
        vault_dislikes = dislikes[:min(half_size, len(dislikes))]

        # If one class is short, take more from the other
        remaining = vault_size - len(vault_likes) - len(vault_dislikes)
        if remaining > 0:
            if len(vault_likes) < half_size:
                vault_dislikes.extend(dislikes[len(vault_dislikes):len(vault_dislikes) + remaining])
            else:
                vault_likes.extend(likes[len(vault_likes):len(vault_likes) + remaining])

        return vault_likes + vault_dislikes

    def _add_balanced_to_vault(
        self,
        current_vault: list[str],
        new_files: list[str],
        vault_size: int,
        file_ratings: dict[str, int]
    ) -> list[str]:
        """Add new files to vault while maintaining class balance.

        Strategy:
        - New files go to front of vault (most recent)
        - When removing to stay at vault_size, prefer removing from over-represented class

        Args:
            current_vault: Current vault files (oldest at end)
            new_files: New files to add
            vault_size: Target vault size
            file_ratings: Dict of {filename: binary_rating (0 or 1)}

        Returns:
            Updated vault list with new files at front
        """
        # Add new files to front
        combined = new_files + current_vault

        if len(combined) <= vault_size:
            return combined

        # Need to remove some - prefer removing from over-represented class
        likes = [f for f in combined if file_ratings.get(f, 0) == 1]
        dislikes = [f for f in combined if file_ratings.get(f, 0) == 0]

        target_per_class = vault_size // 2

        # Keep up to target_per_class from each, preferring newer (earlier in list)
        kept_likes = likes[:min(target_per_class, len(likes))]
        kept_dislikes = dislikes[:min(target_per_class, len(dislikes))]

        # Fill remaining slots
        remaining = vault_size - len(kept_likes) - len(kept_dislikes)
        if remaining > 0:
            if len(kept_likes) < target_per_class:
                kept_dislikes.extend(dislikes[len(kept_dislikes):len(kept_dislikes) + remaining])
            else:
                kept_likes.extend(likes[len(kept_likes):len(kept_likes) + remaining])

        # Maintain order: newer files first
        result = []
        kept_set = set(kept_likes + kept_dislikes)
        for f in combined:
            if f in kept_set:
                result.append(f)
                if len(result) >= vault_size:
                    break

        return result

    def get_vault_files(self) -> list[str]:
        """Get files reserved for A/B testing.

        Returns:
            List of vault files (never used for training)
        """
        return list(self.vault_files)

    def split_with_new_to_validation(
        self,
        all_files: list[str],
        train_ratio: float = 0.8,
        seed: int = 42
    ) -> tuple[list[str], list[str], list[str]]:
        """Split files, putting new files (not in previous training) to validation.

        If this is the first training (no previous manifest), does normal random split.
        If there's a previous training, keeps old training files and puts all new files
        into validation set for unbiased evaluation.

        Args:
            all_files: List of all rated filenames
            train_ratio: Ratio for training set (only used for initial split)
            seed: Random seed for initial split

        Returns:
            Tuple of (training_files, validation_files, new_files)
            - training_files: Files to use for training
            - validation_files: Files to use for validation
            - new_files: Subset of validation that are new since last training
        """
        import random

        all_files_set = set(all_files)

        # Check if we have a previous training
        if self.training_files:
            # Use previous training files that still exist
            train_files = [f for f in self.training_files if f in all_files_set]

            # All other files go to validation (includes new ratings)
            val_files = [f for f in all_files if f not in self.training_files]

            # Track which are truly new (not in previous training OR validation)
            previously_known = self.training_files | self.validation_files
            new_files = [f for f in all_files if f not in previously_known]

            print(f"  Using previous training manifest:")
            print(f"    Previous training files: {len(self.training_files)}")
            print(f"    Still available: {len(train_files)}")
            print(f"    Validation files: {len(val_files)}")
            print(f"    New ratings (held out): {len(new_files)}")
        else:
            # First training - do normal random split
            random.seed(seed)
            shuffled = all_files.copy()
            random.shuffle(shuffled)

            split_idx = int(len(shuffled) * train_ratio)
            train_files = shuffled[:split_idx]
            val_files = shuffled[split_idx:]
            new_files = []  # No new files on first training

            print(f"  Initial training split (seed={seed}):")
            print(f"    Training: {len(train_files)} files")
            print(f"    Validation: {len(val_files)} files")

        # Update manifest with current split
        self.training_files = set(train_files)
        self.validation_files = set(val_files)
        self.training_timestamp = datetime.now().isoformat()

        return train_files, val_files, new_files

    def get_new_files(self, current_files: list[str]) -> list[str]:
        """Get files that are new since last training.

        Args:
            current_files: List of currently available rated files

        Returns:
            List of files not in previous training or validation sets
        """
        previously_known = self.training_files | self.validation_files
        return [f for f in current_files if f not in previously_known]

    def set_version_info(
        self,
        encoder_version: str,
        classifier_version: str,
        classification_mode: str = "regression"
    ) -> None:
        """Set version information for the training.

        Args:
            encoder_version: Encoder version used
            classifier_version: Classifier version
            classification_mode: 'regression' or 'binary'
        """
        self.encoder_version = encoder_version
        self.classifier_version = classifier_version
        self.classification_mode = classification_mode

    def set_metadata(self, key: str, value) -> None:
        """Set arbitrary metadata.

        Args:
            key: Metadata key
            value: Metadata value (must be JSON serializable)
        """
        self.metadata[key] = value

    def __repr__(self) -> str:
        return (
            f"TrainingManifest("
            f"train={len(self.training_files)}, "
            f"val={len(self.validation_files)}, "
            f"timestamp={self.training_timestamp})"
        )
