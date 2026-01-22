# Music Recommendation System - Examples & Documentation

This directory contains examples and documentation for the music recommendation system extension to `ml_skeleton`.

---

## 📚 Documentation Index

### Getting Started

1. **[QUICK_START.md](QUICK_START.md)** ⭐ START HERE
   - Quick reference for file locations
   - Customization workflow
   - Safety reminders
   - What you can do now while framework is being built

2. **[AUDIO_FILES_README.md](AUDIO_FILES_README.md)** 🔒 SAFETY & FILE LOCATIONS
   - Where your audio files are located
   - Read-only guarantee (your files are safe!)
   - File access patterns
   - Troubleshooting

3. **[CUSTOMIZATION_GUIDE.md](CUSTOMIZATION_GUIDE.md)** 🎨 CUSTOMIZE YOUR MODELS
   - How to modify the encoder
   - How to modify the classifier
   - Common audio processing patterns
   - Complete examples

---

## 🎵 Code Examples

### Hello World Stub (Ready to Use)

**[music_hello_world.py](music_hello_world.py)** - Minimal encoder/classifier stubs

```bash
# Test the stubs
python examples/music_hello_world.py
```

**What's included:**
- `HelloWorldEncoder` - Simple 1D CNN (~57K parameters)
- `HelloWorldClassifier` - 2-layer MLP (~8K parameters)
- Test suite that verifies models work
- Placeholder training functions

**Customize it:** See [CUSTOMIZATION_GUIDE.md](CUSTOMIZATION_GUIDE.md)

### Full Example (Coming Soon)

**[music_recommendation.py](music_recommendation.py)** - Complete working example (not yet implemented)

Will include:
- Real encoder and classifier implementations
- Training functions integrated with framework
- Embedding extraction pipeline
- Recommendation generation

---

## 📋 Implementation Plan

**[../LISTEN.MD](../LISTEN.MD)** - Complete implementation plan

Detailed plan covering:
- Architecture overview
- All modules to be created (15 files, ~3500 lines)
- Database schemas
- Training workflows
- Performance optimizations
- Testing strategy
- Timeline (3-4 weeks)

---

## 🔐 Safety Guarantees

### Your Files Are Safe

✅ **Music files** (`/home/ikaro/Music/`) - **READ-ONLY**
✅ **Clementine database** (`clementine.db`) - **READ-ONLY**
✅ **No modifications** to your existing files
✅ **New data** goes to separate locations

### What Gets Created

| File | Location | Size |
|------|----------|------|
| Embeddings Database | `./embeddings.db` | ~30-240 MB |
| Model Checkpoints | `./checkpoints/` | ~500 MB |
| Recommendation Playlists | `~/Music/recommendations/*.xspf` | <1 MB each |

**See [AUDIO_FILES_README.md](AUDIO_FILES_README.md) for details.**

---

## 🚀 Quick Start

### 1. Test the Hello World Stubs

```bash
cd /git/ml_skeleton
python examples/music_hello_world.py
```

Expected: `✓ All tests passed!`

### 2. Customize Your Models

Edit [music_hello_world.py](music_hello_world.py):
- Replace `HelloWorldEncoder` with your encoder architecture
- Replace `HelloWorldClassifier` with your classifier architecture

See [CUSTOMIZATION_GUIDE.md](CUSTOMIZATION_GUIDE.md) for examples.

### 3. Test Your Changes

```bash
python examples/music_hello_world.py
```

### 4. Wait for Framework (Then Train)

Once the `ml_skeleton/music/` modules are implemented:

```bash
# Stage 1: Train encoder
mlskel run configs/music_example.yaml --train-fn examples.music_hello_world:train_encoder_stage

# Stage 2: Train classifier
mlskel run configs/music_example.yaml --train-fn examples.music_hello_world:train_classifier_stage

# Generate recommendations
mlskel recommend checkpoints/encoder_best.pt checkpoints/classifier_best.pt \
    --output ~/Music/recommendations/test.xspf \
    --top-n 100
```

---

## 📖 Additional Resources

### Configuration

**[../configs/music_example.yaml](../configs/music_example.yaml)** (not yet created)

Will contain:
- Audio processing settings (sample rate, duration, center crop)
- Model hyperparameters (embedding dimension, learning rates)
- Training configuration (epochs, batch size)
- MLflow tracking settings

### Implementation Details

From the main plan ([LISTEN.MD](../LISTEN.MD)):

**Phase 1: Core Infrastructure** (3-5 days)
- Protocols for encoder/classifier
- Clementine DB interface (read-only)
- Embedding storage (SQLite)

**Phase 2: Audio Processing** (3-5 days)
- Multiprocessing audio loader (center crop)
- PyTorch datasets

**Phase 3: Training** (4-6 days)
- Encoder trainer
- Classifier trainer
- Joint trainer

**Phase 4: Recommendations** (3-4 days)
- Recommendation engine
- XSPF export

**Phase 5: Examples & Docs** (2-3 days)
- Complete examples
- Documentation updates

**Phase 6: Testing** (3-5 days)
- Unit tests
- Integration tests
- Performance tuning

---

## 🎯 Current Status

| Component | Status |
|-----------|--------|
| Planning | ✅ Complete |
| Hello World Stubs | ✅ Complete |
| Documentation | ✅ Complete |
| Core Framework | ⏳ Pending (Phase 1-6) |

---

## 💡 What You Can Do Now

While waiting for framework implementation:

1. ✅ **Experiment with model architectures** in [music_hello_world.py](music_hello_world.py)
2. ✅ **Read the documentation** to understand the system
3. ✅ **Verify your audio files** are accessible
4. ✅ **Plan your encoder design** (CNN, Transformer, pre-trained, etc.)
5. ✅ **Prepare for training** by thinking about hyperparameters

---

## 📞 Need Help?

1. Check [QUICK_START.md](QUICK_START.md) for common questions
2. Review [AUDIO_FILES_README.md](AUDIO_FILES_README.md) for file location issues
3. See [CUSTOMIZATION_GUIDE.md](CUSTOMIZATION_GUIDE.md) for model customization
4. Read the full plan in [LISTEN.MD](../LISTEN.MD)

---

## 🎵 Happy Music Recommendation Building!

Your music files are safe, the stubs are ready, and you can start customizing your models right now!
