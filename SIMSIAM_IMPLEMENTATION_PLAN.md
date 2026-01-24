# SimSiam Encoder Implementation Plan

## Overview

Replace the current metadata-based contrastive learning approach with **SimSiam** (Simple Siamese Networks). This eliminates the need for positive pairs based on metadata (artist/album/year) and instead learns representations through **augmentation-based self-supervision**.

**Key Advantage**: No need for metadata matching - every song provides its own positive pair via augmentation.

---

## 1. Architecture Changes

### Current Architecture
```
Audio → SimpleAudioEncoder (1D CNN) → Embedding (512/2048 dim)
                                          ↓
                              MetadataContrastiveLoss
                              (requires artist/album/year pairs)
```

### Proposed SimSiam Architecture
```
Audio → Augment₁ → Spectrogram → ResNet50 Backbone → Projection MLP → z₁
    ↘                                                                   ↓
     Augment₂ → Spectrogram → ResNet50 Backbone → Projection MLP → z₂  ↓
                                                        ↓               ↓
                                              Predictor MLP → p₂       ↓
                                                        ↓               ↓
                                           SimSiamLoss(p₂, stop_grad(z₁))
```

---

## 2. Component Specifications

### 2.1 Audio-to-Spectrogram Transform
Convert raw audio waveform to mel-spectrogram for 2D processing:

```
Input:  (batch, 960000)           # 60s @ 16kHz
        ↓ MelSpectrogram
Output: (batch, 1, 128, T)        # 128 mel bins, T time frames
        ↓ Repeat to 3 channels (for ResNet compatibility)
Output: (batch, 3, 128, T)        # "RGB-like" spectrogram
```

**Parameters:**
- `n_mels`: 128
- `n_fft`: 2048
- `hop_length`: 512
- `sample_rate`: 16000

### 2.2 ResNet50 Backbone (Modified for Spectrograms)

Use `torchvision.models.resnet50` with modifications:
- Remove final FC layer (use as feature extractor)
- Modify first conv layer if needed for spectrogram dimensions
- Output: 2048-dimensional feature vector

```python
backbone = resnet50(pretrained=False)  # or pretrained=True for transfer learning
backbone.fc = nn.Identity()  # Remove classifier, output 2048-dim
```

**Alternative**: Use `torchvision.models.resnet18` for faster training (512-dim output).

### 2.3 Projection MLP (Encoder Head)

Maps backbone features to projection space:

```
Input:  2048 (from ResNet50)
        ↓ Linear(2048, 2048, bias=False)
        ↓ BatchNorm1d(2048)
        ↓ ReLU
        ↓ Linear(2048, 2048, bias=False)
        ↓ BatchNorm1d(2048)
Output: 2048 (projection z)
```

**Key**: No ReLU after final layer, just BatchNorm.

### 2.4 Predictor MLP (Critical for SimSiam)

Asymmetric component that prevents collapse:

```
Input:  2048 (projection z)
        ↓ Linear(2048, 512, bias=False)  # Bottleneck!
        ↓ BatchNorm1d(512)
        ↓ ReLU
        ↓ Linear(512, 2048)              # Back to projection dim
Output: 2048 (prediction p)
```

**Key**: Bottleneck architecture (2048 → 512 → 2048) is important.

---

## 3. Loss Function: SimSiamLoss

### Formula
```
L = D(p₁, stopgrad(z₂)) / 2 + D(p₂, stopgrad(z₁)) / 2

where D(p, z) = -cosine_similarity(p, z) = -(p·z) / (||p|| ||z||)
```

### Implementation
```python
class SimSiamLoss(nn.Module):
    def forward(self, p1, p2, z1, z2):
        # CRITICAL: stop gradient on z
        z1 = z1.detach()
        z2 = z2.detach()

        # L2 normalize
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Negative cosine similarity (symmetric)
        loss = -(p1 * z2).sum(dim=1).mean() / 2 \
               -(p2 * z1).sum(dim=1).mean() / 2

        return loss
```

**Critical**: The `z.detach()` (stop-gradient) prevents representation collapse.

---

## 4. Audio Augmentation Pipeline

Since SimSiam relies on augmentation for positive pairs, this is crucial.

### Recommended Audio Augmentations

| Augmentation | Description | Probability |
|-------------|-------------|-------------|
| **Time Shift** | Random circular shift in time | 100% |
| **Time Stretch** | Speed up/slow down (0.8x-1.2x) | 50% |
| **Pitch Shift** | Shift pitch ±2 semitones | 50% |
| **Add Noise** | Gaussian noise (SNR 15-30 dB) | 30% |
| **Frequency Mask** | Mask random frequency bands | 50% |
| **Time Mask** | Mask random time segments | 50% |
| **Volume Change** | Random gain ±6 dB | 80% |
| **Reverb** | Add room reverb | 20% |

### Spectrogram-Level Augmentations (SpecAugment)
- **FrequencyMask**: Mask F consecutive frequency bins
- **TimeMask**: Mask T consecutive time frames
- These are applied AFTER mel-spectrogram conversion

### Implementation Approach
```python
class AudioAugmentor:
    def __call__(self, waveform):
        # Waveform augmentations
        if random.random() < 0.5:
            waveform = time_stretch(waveform, rate=uniform(0.8, 1.2))
        if random.random() < 0.5:
            waveform = pitch_shift(waveform, n_steps=uniform(-2, 2))
        if random.random() < 0.3:
            waveform = add_noise(waveform, snr_db=uniform(15, 30))

        # Convert to spectrogram
        spec = mel_spectrogram(waveform)

        # SpecAugment
        spec = frequency_mask(spec, F=27)
        spec = time_mask(spec, T=100)

        return spec
```

---

## 5. Dataset Changes

### Current MusicDataset
Returns: `{audio, rating, albums, filename, artist, album, year}`

### New SimSiamDataset
Returns two augmented views of the same audio:

```python
class SimSiamMusicDataset(Dataset):
    def __getitem__(self, idx):
        audio = load_audio(self.songs[idx])

        # Create two different augmented views
        view1 = self.augmentor(audio)  # Spectrogram after augmentation
        view2 = self.augmentor(audio)  # Different augmentation

        return {
            "view1": view1,
            "view2": view2,
            "filename": self.songs[idx].filename
        }
```

**Key**: Same audio, different augmentations = positive pair.

---

## 6. Training Loop Changes

### Current Flow
```python
embeddings = encoder(audio)
loss = metadata_contrastive_loss(embeddings, artists, albums, years)
```

### New SimSiam Flow
```python
# Forward pass for both views
z1 = encoder(view1)  # projection
z2 = encoder(view2)  # projection
p1 = predictor(z1)   # prediction
p2 = predictor(z2)   # prediction

# SimSiam loss (stop-gradient inside)
loss = simsiam_loss(p1, p2, z1, z2)
```

---

## 7. File Changes Required

| File | Change |
|------|--------|
| `ml_skeleton/music/encoder_factory.py` | **NEW**: Factory pattern for encoder/loss/dataset creation |
| `ml_skeleton/music/simsiam_encoder.py` | **NEW**: SimSiam encoder with ResNet50 + projection + predictor |
| `ml_skeleton/music/losses.py` | **ADD**: SimSiamLoss class |
| `ml_skeleton/music/augmentations.py` | **NEW**: Audio augmentation pipeline |
| `ml_skeleton/music/dataset.py` | **ADD**: SimSiamMusicDataset class |
| `ml_skeleton/training/encoder_trainer.py` | **MODIFY**: Support SimSiam training loop, MLflow tagging |
| `configs/music_recommendation.yaml` | **MODIFY**: Add encoder_type selector + nested config structure |
| `ml_skeleton/protocols/encoder.py` | **MODIFY**: Update protocol for SimSiam |
| `examples/music_recommendation.py` | **MODIFY**: Add --encoder-type CLI arg, use factory |
| `run_music_pipeline.sh` | **MODIFY**: Pass --encoder-type to scripts |

---

## 8. Configuration Changes

### 8.1 Encoder Type Selection (Single Config Switch)

```yaml
encoder:
  # ============================================================
  # ENCODER TYPE SELECTOR (A/B Testing)
  # ============================================================
  # Change this single value to switch encoder strategies:
  #   - "simple"      : Current 1D CNN on raw waveform (baseline)
  #   - "simsiam"     : SimSiam with ResNet50 on spectrograms (NEW)
  # ============================================================
  encoder_type: "simple"  # ← Change to "simsiam" for new approach

  # ------------------------------------------------------------
  # Common settings (used by all encoder types)
  # ------------------------------------------------------------
  embedding_dim: 2048
  epochs: 20
  final_training_epochs: 50
  batch_size: 64
  learning_rate: 0.001

  # ------------------------------------------------------------
  # Simple encoder settings (encoder_type: "simple")
  # ------------------------------------------------------------
  simple:
    base_channels: 32
    loss_type: "metadata_contrastive"
    # Metadata contrastive settings
    use_artist: true
    use_album: true
    use_year: true
    year_threshold: 2
    contrastive_temperature: 0.5

  # ------------------------------------------------------------
  # SimSiam encoder settings (encoder_type: "simsiam")
  # ------------------------------------------------------------
  simsiam:
    # Backbone
    backbone: "resnet50"        # "resnet18", "resnet34", "resnet50"
    pretrained_backbone: false  # Use ImageNet weights?

    # Architecture
    projection_dim: 2048
    predictor_hidden_dim: 512   # Bottleneck dimension

    # Spectrogram settings
    n_mels: 128
    n_fft: 2048
    hop_length: 512

    # Augmentation strengths
    augmentation:
      time_stretch_range: [0.8, 1.2]
      pitch_shift_range: [-2, 2]      # semitones
      noise_snr_range: [15, 30]       # dB
      spec_augment: true
      frequency_mask_param: 27
      time_mask_param: 100

    # Training (SimSiam works better with SGD)
    optimizer: "sgd"
    sgd_momentum: 0.9
    weight_decay: 0.0005
```

### 8.2 MLflow A/B Testing Configuration

```yaml
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "music_recommendation"

  # A/B Testing: Auto-tag runs with encoder type
  auto_tags:
    - encoder_type      # Automatically logged from config
    - loss_type         # Automatically logged from config
    - backbone          # For simsiam runs

  # Compare runs by filtering on encoder_type tag
  # In MLflow UI: filter by `tags.encoder_type = "simsiam"`
```

---

## 9. A/B Testing Strategy in MLflow

### 9.1 Automatic Run Tagging

Every training run will automatically log:

```python
# In encoder_trainer.py or experiment runner
mlflow.set_tag("encoder_type", config.encoder.encoder_type)
mlflow.set_tag("loss_type", loss_type)
mlflow.set_tag("experiment_variant", f"{encoder_type}_{loss_type}")

# SimSiam-specific tags
if config.encoder.encoder_type == "simsiam":
    mlflow.set_tag("backbone", config.encoder.simsiam.backbone)
    mlflow.set_tag("pretrained", config.encoder.simsiam.pretrained_backbone)
```

### 9.2 Metrics to Compare

| Metric | Description | Lower is Better |
|--------|-------------|-----------------|
| `encoder/train_loss` | Training loss | Yes |
| `encoder/val_loss` | Validation loss | Yes |
| `classifier/val_mse` | Rating prediction MSE | Yes |
| `classifier/val_mae` | Rating prediction MAE | Yes |
| `classifier/val_correlation` | Embedding-rating correlation | No (higher better) |
| `training_time_hours` | Total training time | Yes |

### 9.3 MLflow Comparison Workflow

```bash
# Run baseline (simple encoder)
python examples/music_recommendation.py --encoder-type simple

# Run SimSiam variant
python examples/music_recommendation.py --encoder-type simsiam

# Compare in MLflow UI:
# 1. Go to http://localhost:5000
# 2. Select experiment "music_recommendation"
# 3. Filter: tags.encoder_type = "simple" OR tags.encoder_type = "simsiam"
# 4. Select runs → Compare → View charts
```

### 9.4 Programmatic Comparison

```python
import mlflow

# Query runs by encoder type
simple_runs = mlflow.search_runs(
    experiment_names=["music_recommendation"],
    filter_string="tags.encoder_type = 'simple'",
    order_by=["metrics.classifier/val_mse ASC"]
)

simsiam_runs = mlflow.search_runs(
    experiment_names=["music_recommendation"],
    filter_string="tags.encoder_type = 'simsiam'",
    order_by=["metrics.classifier/val_mse ASC"]
)

# Compare best runs
print(f"Best Simple MSE:  {simple_runs.iloc[0]['metrics.classifier/val_mse']:.4f}")
print(f"Best SimSiam MSE: {simsiam_runs.iloc[0]['metrics.classifier/val_mse']:.4f}")
```

---

## 10. Encoder Factory Pattern

### Implementation for Easy Switching

```python
# ml_skeleton/music/encoder_factory.py

def create_encoder(config) -> nn.Module:
    """Factory function to create encoder based on config.

    Args:
        config: Configuration with encoder.encoder_type

    Returns:
        Encoder module (SimpleAudioEncoder or SimSiamEncoder)
    """
    encoder_type = config.encoder.encoder_type

    if encoder_type == "simple":
        from .baseline_encoder import SimpleAudioEncoder
        return SimpleAudioEncoder(
            sample_rate=config.music.sample_rate,
            duration=config.music.audio_duration,
            embedding_dim=config.encoder.embedding_dim,
            base_channels=config.encoder.simple.base_channels
        )

    elif encoder_type == "simsiam":
        from .simsiam_encoder import SimSiamEncoder
        return SimSiamEncoder(
            sample_rate=config.music.sample_rate,
            duration=config.music.audio_duration,
            embedding_dim=config.encoder.embedding_dim,
            backbone=config.encoder.simsiam.backbone,
            pretrained=config.encoder.simsiam.pretrained_backbone,
            projection_dim=config.encoder.simsiam.projection_dim,
            predictor_hidden_dim=config.encoder.simsiam.predictor_hidden_dim,
            n_mels=config.encoder.simsiam.n_mels,
            n_fft=config.encoder.simsiam.n_fft,
            hop_length=config.encoder.simsiam.hop_length
        )

    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")


def create_loss_fn(config) -> nn.Module:
    """Factory function to create loss based on encoder type."""
    encoder_type = config.encoder.encoder_type

    if encoder_type == "simple":
        loss_type = config.encoder.simple.loss_type
        if loss_type == "metadata_contrastive":
            from .losses import MetadataContrastiveLoss
            return MetadataContrastiveLoss(
                temperature=config.encoder.simple.contrastive_temperature,
                year_threshold=config.encoder.simple.year_threshold,
                use_artist=config.encoder.simple.use_artist,
                use_album=config.encoder.simple.use_album,
                use_year=config.encoder.simple.use_year
            )
        # ... other loss types

    elif encoder_type == "simsiam":
        from .losses import SimSiamLoss
        return SimSiamLoss()

    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")


def create_dataset(config, songs, is_training=True):
    """Factory function to create dataset based on encoder type."""
    encoder_type = config.encoder.encoder_type

    if encoder_type == "simple":
        from .dataset import MusicDataset
        return MusicDataset(songs, ...)

    elif encoder_type == "simsiam":
        from .dataset import SimSiamMusicDataset
        augmentor = create_augmentor(config) if is_training else None
        return SimSiamMusicDataset(songs, augmentor=augmentor, ...)
```

---

## 11. CLI Support for A/B Testing

```bash
# Override encoder type via CLI (without editing config)
python examples/music_recommendation.py --encoder-type simsiam

# Or use environment variable
ENCODER_TYPE=simsiam python examples/music_recommendation.py

# HPO with specific encoder type
./run_music_pipeline.sh hpo --encoder-type simsiam
```

### Implementation in music_recommendation.py

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--encoder-type', choices=['simple', 'simsiam'],
                    help='Override encoder type from config')
args = parser.parse_args()

# Load config and override if specified
config = load_config('configs/music_recommendation.yaml')
if args.encoder_type:
    config.encoder.encoder_type = args.encoder_type
    print(f"Encoder type overridden to: {args.encoder_type}")
```

---

## 12. Training Hyperparameters (Recommended)

Based on SimSiam paper and Keras implementation:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | SGD | Works better than Adam for SimSiam |
| Momentum | 0.9 | Standard |
| Learning Rate | 0.03 | With cosine decay |
| Weight Decay | 0.0005 | L2 regularization |
| Batch Size | 128-256 | Larger helps but not required |
| Epochs | 100-200 | SimSiam needs more epochs |

**Note**: SimSiam is more stable than SimCLR with smaller batch sizes.

---

## 13. Evaluation Strategy

After training, evaluate embeddings:

1. **Linear Evaluation**: Freeze encoder, train linear classifier on embeddings → ratings
2. **k-NN Evaluation**: Use k-NN on embeddings to predict ratings
3. **Visualization**: t-SNE/UMAP of embeddings colored by rating/genre

---

## 14. Implementation Order

### Phase 1: Infrastructure (A/B Testing Support)
1. [ ] Update config schema with `encoder_type` selector and nested structure
2. [ ] Create `encoder_factory.py` with factory functions
3. [ ] Add `--encoder-type` CLI argument to `music_recommendation.py`
4. [ ] Add MLflow auto-tagging for encoder_type, loss_type, backbone

### Phase 2: Core SimSiam Components
5. [ ] Implement `SimSiamLoss` in losses.py
6. [ ] Implement audio augmentation pipeline (`augmentations.py`)
7. [ ] Create `SimSiamMusicDataset` with dual-view output

### Phase 3: Encoder Architecture
8. [ ] Implement mel-spectrogram transform
9. [ ] Create `SimSiamEncoder` with ResNet50 backbone
10. [ ] Add projection MLP head (2048→2048→2048)
11. [ ] Add predictor MLP (2048→512→2048)

### Phase 4: Training Integration
12. [ ] Modify `EncoderTrainer` for SimSiam training loop
13. [ ] Update `run_music_pipeline.sh` to accept encoder type
14. [ ] Ensure embedding extraction works with both encoders

### Phase 5: Testing & A/B Comparison
15. [ ] Test SimSiam on subset of data (verify loss decreases)
16. [ ] Run full training with both encoder types
17. [ ] Compare in MLflow UI (filter by tags.encoder_type)
18. [ ] Document results and tune augmentation if needed

---

## 15. Advantages of SimSiam over Current Approach

| Aspect | Current (Metadata Contrastive) | SimSiam |
|--------|-------------------------------|---------|
| Positive pairs | Requires metadata matching | Self-supervised (augmentation) |
| Batch size dependency | Needs large batches for collisions | Works with small batches |
| Metadata quality | Affected by unknown/incorrect metadata | No metadata needed |
| Learning signal | Sparse (few pairs per batch) | Dense (every sample is a pair) |
| Scalability | Limited by metadata availability | Works on any audio |

---

## 16. References

- [SimSiam Paper (arXiv:2011.10566)](https://arxiv.org/abs/2011.10566)
- [Keras SimSiam Tutorial](https://keras.io/examples/vision/simsiam/)
- [LearnOpenCV SimSiam Guide](https://learnopencv.com/simsiam/)

---

## 17. Questions to Resolve

1. **Pretrained ResNet50?** - Use ImageNet weights or train from scratch?
2. **Spectrogram dimensions?** - 128 mel bins standard, but time dimension varies with audio length
3. **Augmentation strength?** - May need tuning for music domain
4. **Embedding dimension for classifier?** - Use projection (2048) or backbone output (2048)?
