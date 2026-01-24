# Claude Code Best Practices

Guidelines for efficient Claude Code sessions with this repository.

## Session Workflow

### One Task Per Session
- Complete one feature/fix, commit, then start new session
- Avoids context bloat from accumulated file reads
- Commit often as checkpoints

### Be Specific in Requests
```
# Good - specific file and line
"Fix the error in encoder_factory.py:45"

# Bad - requires exploration
"Find where the encoder is created and fix it"
```

### Batch Related Changes
```
# Good - single request
"Update the learning rate to 0.001 and batch size to 64 in music_recommendation.yaml"

# Bad - multiple requests
"Change learning rate" ... "Now change batch size"
```

## File References

Use `file:line` format for precise navigation:
- `ml_skeleton/music/encoder_factory.py:37` - specific line
- `configs/music_recommendation.yaml` - whole file

## TODOS.txt Format

Keep actionable items with file references:
```
*Fix batch size issue in encoder_trainer.py:156
*Add validation for config in music_recommendation.py:340
```

## What NOT to Commit

These are in `.gitignore` - don't ask Claude to read them:
- `cache/` - waveform cache (large, regenerated)
- `checkpoints/` - model files
- `*.db` - embeddings, MLflow databases
- `recommendations.txt`, `recommender_*.xspf` - generated outputs

## Reducing Claude's Work

### Avoid Exploration Requests
```
# Triggers file exploration (slow, uses context)
"What files handle audio loading?"

# Direct (fast)
"Read ml_skeleton/music/audio_loader.py"
```

### Use grep/glob hints
```
# If you know a pattern exists
"The function is called `train_encoder` in examples/"
```

## Key Files Quick Reference

| Purpose | File |
|---------|------|
| Main pipeline | `examples/music_recommendation.py` |
| Config | `configs/music_recommendation.yaml` |
| Encoder | `ml_skeleton/music/simsiam_encoder.py` |
| Dataset | `ml_skeleton/music/dataset.py` |
| Training | `ml_skeleton/training/encoder_trainer.py` |
| Classifier | `ml_skeleton/training/classifier_trainer.py` |

## Running the Pipeline

```bash
./run_music_pipeline.sh encoder    # Stage 1
./run_music_pipeline.sh classifier # Stage 2
./run_music_pipeline.sh recommend  # Stage 3
./run_music_pipeline.sh all        # Full pipeline
```

## When to Start New Session

- After completing a feature and committing
- If context feels slow/large
- After major refactoring
- Before starting unrelated task
