# Encoder & Classifier Customization Guide

This guide explains how to quickly customize the stub models in [music_hello_world.py](music_hello_world.py) for your specific needs.

## Quick Start

```bash
# Test the stub models
python examples/music_hello_world.py

# Edit the file to add your architecture
# Then test again to verify it works
```

---

## Customizing the Encoder

The encoder converts raw audio waveforms into embeddings (latent representations).

### Input/Output Specification

```python
Input:  torch.Tensor of shape (batch_size, num_samples)
        - num_samples = sample_rate × duration
        - Example: 22050 Hz × 30 seconds = 661,500 samples

Output: torch.Tensor of shape (batch_size, embedding_dim)
        - embedding_dim is your choice (Z dimension)
        - Common values: 128, 256, 512, 1024
```

### Architecture Options

#### Option 1: Keep the Simple CNN (Good for Testing)

The stub already has a minimal 1D CNN. Just adjust the embedding dimension:

```python
encoder = HelloWorldEncoder(embedding_dim=512)  # Change 128 to your desired size
```

#### Option 2: Use a Pre-trained Model (Recommended)

Replace the stub with a pre-trained audio model:

```python
import torchaudio
from torchaudio.models import wav2vec2_base

class MyEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Load pre-trained Wav2Vec2
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec = bundle.get_model()

        # Project to your embedding dimension
        self.projection = nn.Linear(768, embedding_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # Wav2Vec2 expects (batch, num_samples)
        features, _ = self.wav2vec.extract_features(audio)

        # Average over time dimension
        pooled = features[-1].mean(dim=1)  # (batch, 768)

        # Project to embedding_dim
        embeddings = self.projection(pooled)  # (batch, embedding_dim)
        return embeddings

    def get_embedding_dim(self) -> int:
        return self.embedding_dim
```

#### Option 3: Custom Architecture

Build your own architecture from scratch:

```python
class MyCustomEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim

        # YOUR ARCHITECTURE HERE
        # Examples:
        # - Multi-layer 1D CNN
        # - Transformer encoder
        # - ResNet-style blocks
        # - Attention mechanisms

        self.my_model = nn.Sequential(
            # ... your layers ...
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # Process audio through your model
        embeddings = self.my_model(audio)

        # Make sure output shape is (batch, embedding_dim)
        return embeddings

    def get_embedding_dim(self) -> int:
        return self.embedding_dim
```

### Common Audio Processing Patterns

#### Add Mel-Spectrogram Conversion

```python
import torchaudio.transforms as T

class SpectrogramEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 512, sample_rate: int = 22050):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Convert waveform to mel-spectrogram
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )

        # 2D CNN for spectrogram processing
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # Convert to mel-spectrogram
        mel = self.mel_transform(audio)  # (batch, n_mels, time)
        mel = mel.unsqueeze(1)  # (batch, 1, n_mels, time)

        # Process with CNN
        features = self.cnn(mel).squeeze(-1).squeeze(-1)  # (batch, 64)

        # Project to embedding
        embeddings = self.fc(features)
        return embeddings

    def get_embedding_dim(self) -> int:
        return self.embedding_dim
```

---

## Customizing the Classifier

The classifier predicts ratings from embeddings.

### Input/Output Specification

```python
Input:  torch.Tensor of shape (batch_size, embedding_dim)
        - embedding_dim matches your encoder's output

Output: torch.Tensor of shape (batch_size, 1)
        - Values in range [0, 1]
        - 0.0 = terrible/unrated
        - 1.0 = 5 stars
```

### Architecture Options

#### Option 1: Keep the Simple MLP (Good for Testing)

The stub is already a 2-layer MLP. Just adjust the hidden dimension:

```python
classifier = HelloWorldClassifier(embedding_dim=512, hidden_dim=256)
```

#### Option 2: Deeper Network

Add more layers for better capacity:

```python
class DeepClassifier(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(embeddings)
```

#### Option 3: Residual Connections

Add skip connections for gradient flow:

```python
class ResidualClassifier(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super().__init__()

        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # First layer
        x = torch.relu(self.fc1(embeddings))  # (batch, 256)
        x = self.dropout(x)

        # Residual block
        residual = x
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = x + residual  # Skip connection

        # Output layer
        x = torch.sigmoid(self.fc3(x))  # (batch, 1)
        return x
```

---

## Testing Your Changes

After modifying the models, always test them:

```python
# At the bottom of music_hello_world.py, change the test to use your models:

if __name__ == "__main__":
    # Test with your custom models
    print("Testing custom models...")

    # Replace HelloWorldEncoder with your encoder
    encoder = MyCustomEncoder(embedding_dim=512)

    # Replace HelloWorldClassifier with your classifier
    classifier = DeepClassifier(embedding_dim=512)

    # Test them
    batch_size = 4
    num_samples = 661500  # 30 seconds at 22050 Hz

    fake_audio = torch.randn(batch_size, num_samples)
    embeddings = encoder(fake_audio)
    ratings = classifier(embeddings)

    print(f"✓ Embeddings shape: {embeddings.shape}")
    print(f"✓ Ratings shape: {ratings.shape}")
    print(f"✓ Ratings range: [{ratings.min():.4f}, {ratings.max():.4f}]")
```

---

## Common Pitfalls

### 1. Wrong Output Shape

**Error**: `Expected shape (batch, embedding_dim), got (batch, embedding_dim, 1)`

**Fix**: Use `squeeze()` to remove extra dimensions:
```python
embeddings = self.model(audio).squeeze(-1)
```

### 2. Ratings Outside [0, 1]

**Error**: Ratings are negative or > 1

**Fix**: Add `nn.Sigmoid()` as final layer:
```python
self.network = nn.Sequential(
    # ... your layers ...
    nn.Sigmoid()  # MUST be last layer
)
```

### 3. Dimension Mismatch

**Error**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

**Fix**: Check your dimensions at each layer. Add print statements:
```python
def forward(self, x):
    print(f"Input: {x.shape}")
    x = self.layer1(x)
    print(f"After layer1: {x.shape}")
    x = self.layer2(x)
    print(f"After layer2: {x.shape}")
    # ... etc
```

### 4. Out of Memory

**Error**: `CUDA out of memory`

**Fix**:
- Reduce embedding dimension
- Use gradient checkpointing
- Process audio in smaller chunks

---

## Integration with Training Pipeline

Once the full framework is ready, your models will be used like this:

```python
# In your config file (configs/music_example.yaml):
hyperparameters:
  embedding_dim: 512
  encoder_class: "examples.music_hello_world:MyCustomEncoder"
  classifier_class: "examples.music_hello_world:DeepClassifier"

# The framework will automatically:
# 1. Import your classes
# 2. Instantiate them with the embedding_dim
# 3. Train them on your music data
# 4. Save checkpoints
# 5. Generate recommendations
```

---

## Example: Complete Custom Implementation

Here's a complete example replacing both models:

```python
import torch
import torch.nn as nn
import torchaudio.transforms as T


class MyAudioEncoder(nn.Module):
    """My custom encoder using mel-spectrograms."""

    def __init__(self, embedding_dim: int = 512, sample_rate: int = 22050):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Convert to mel-spectrogram
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )

        # CNN backbone
        self.encoder = nn.Sequential(
            # (batch, 1, 128, time)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # To spectrogram
        mel = self.mel(audio).unsqueeze(1)  # (batch, 1, 128, time)

        # Encode
        features = self.encoder(mel).squeeze(-1).squeeze(-1)  # (batch, 256)

        # Project
        embeddings = self.fc(features)
        return embeddings

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


class MyRatingClassifier(nn.Module):
    """My custom classifier with residual connections."""

    def __init__(self, embedding_dim: int = 512):
        super().__init__()

        self.input_layer = nn.Linear(embedding_dim, 256)

        # Residual blocks
        self.res1 = nn.Linear(256, 256)
        self.res2 = nn.Linear(256, 256)

        self.output_layer = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_layer(embeddings))
        x = self.dropout(x)

        # Residual block 1
        residual = x
        x = torch.relu(self.res1(x))
        x = self.dropout(x)
        x = x + residual

        # Residual block 2
        residual = x
        x = torch.relu(self.res2(x))
        x = self.dropout(x)
        x = x + residual

        # Output
        x = torch.sigmoid(self.output_layer(x))
        return x


# Test your models
if __name__ == "__main__":
    encoder = MyAudioEncoder(embedding_dim=512)
    classifier = MyRatingClassifier(embedding_dim=512)

    # Test with 30 seconds of audio at 22050 Hz
    audio = torch.randn(4, 661500)
    embeddings = encoder(audio)
    ratings = classifier(embeddings)

    print(f"✓ Audio: {audio.shape}")
    print(f"✓ Embeddings: {embeddings.shape}")
    print(f"✓ Ratings: {ratings.shape}, range [{ratings.min():.4f}, {ratings.max():.4f}]")
```

---

## Next Steps

1. **Customize the models** in [music_hello_world.py](music_hello_world.py)
2. **Test your changes** by running the file
3. **Wait for framework implementation** (the ml_skeleton.music modules)
4. **Update the training functions** once the framework is ready
5. **Run actual training** on your music database

Good luck! 🎵
