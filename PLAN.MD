# PLAN.md: Music Acoustic Encoder + Recommender (CQT FIXED)

## 🎯 Goal
Train **acoustic-only** encoder (CQT → 2048-dim embeddings) for personalized rating prediction on Clementine DB.

## 🏗️ Architecture
Input: MP3 → 16kHz .npy CACHE (4 chunks/song) → AUGMENT → nnAudio CQT →
[ResNet-50 2D] → 2048-dim embedding
↓
├── MoCo v2 contrastive head (queue=4096)
└── Genre multi-label BCE head

text

## 🔧 **CQT: nnAudio (GPU-accelerated)** ⭐
nnAudio.features.CQT2010v2 (PyTorch native, GPU):

python
from nnAudio.features import CQT2010v2
cqt_transform = CQT2010v2(
    sr=16000,
    n_bins=84,
    fmin=32.7,           # C1
    hop_length=512,
    output_format='Magnitude'
)
# Input: (B, T) → Output: (B, 84, T//512) → Ready for ResNet-2D
Why nnAudio > librosa:

GPU-accelerated (critical for batch=128)

PyTorch native tensors (no CPU bottleneck)

Real-time during training (post-augmentation)

text

## 📋 **Training Strategy (RTX 5090: batch=128)**

### **Phase 0: Build Cache (~30GB)**
```bash
pip install nnAudio  # Add to requirements
./run_music_pipeline.sh build-cache  # 16kHz .npy, 4 chunks/song
Phase 1: MoCo v2 + Genre BCE (100 epochs)
text
Loss: **0.6×MoCo_v2 + 0.4×Genre_BCE**
Pipeline: .npy → augment → **nnAudio.CQT** → ResNet50-2D → heads
moco_v2:
  - momentum: 0.999
  - temperature: 0.07  
  - queue_size: 4096
Batch size: 128 ✓ RTX 5090 friendly
🔧 Augmentations → nnAudio CQT
text
.npy chunk → FRESH → nnAudio CQT → ResNet:
Always: 
- time-crop (5-15s from 30s): 100%
- gain ±2dB: 100%

50%:
- Gaussian noise (SNR 25-35dB)
- channel select (L/R/avg stereo) 
- audio mixup α=0.1 (same-genre chunks)
🗂️ ml_skeleton Config
text
music:
  sample_rate: 16000
  cqt:
    package: "nnAudio"              # ← FIXED
    n_bins: 84
    fmin: 32.7
    hop_length: 512
    sr: 16000
  max_chunks_per_song: 4
  waveform_cache_format: "npy"

encoder:
  backbone: "resnet50_2d"
  batch_size: 128
  moco_v2:
    momentum: 0.999
    temperature: 0.07
    queue_size: 4096
  loss_weights:
    moco: 0.6
    genre_bce: 0.4
Key code snippet
python
import nnAudio.features as nnaudio
cqt_transform = nnaudio.CQT2010v2(
    sr=16000, n_bins=84, fmin=32.7, hop_length=512
)

def forward(audio):  # (B, T)
    cqt = cqt_transform(audio)  # (B, 84, T//512) GPU!
    return resnet50_2d(cqt)     # (B, 2048)
📊 Success Metrics
text
Phase 1: Genre mAP >0.7
Phase 2: Rating correlation >0.4
CQT: GPU-accelerated, no CPU bottleneck
Cache: ~30GB, 15x speedup
🚀 Updated Next Steps
text
1. [ ] pip install nnAudio
2. [ ] .npy cache builder (4 chunks/song)  
3. [ ] nnAudio CQT + ResNet50-2D + MoCo v2
4. [ ] Genre multi-label parser (split "/")
5. [ ] RTX 5090 batch=128 test
💾 ~30GB Storage ✅
text
60K × 4 chunks × 30s × 16kHz = 29GB
nnAudio CQT = perfect solution. GPU-native, post-augmentation, PyTorch tensors.