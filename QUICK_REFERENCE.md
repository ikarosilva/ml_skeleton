# Quick Reference Card - System Monitoring

**Print this for quick access during training**

---

## 🔍 Essential Monitoring Commands

```bash
# GPU status (run every few minutes)
nvidia-smi

# Complete system snapshot
nvidia-smi && echo "---" && uptime && echo "---" && free -h

# Training process check
ps aux | grep music_recommendation
```

---

## ✅ Healthy System Values

| Metric | Your Current | Safe Range |
|--------|-------------|------------|
| **GPU Power** | 122W | < 400W |
| **GPU Temp** | 41°C | < 75°C |
| **GPU Memory** | 4.3GB | < 24GB |
| **CPU Load** | 7.4 | < 15 |
| **RAM Used** | 23GB | < 80GB |
| **Total Power** | 300W | < 900W |
| **Circuit Usage** | 17% | < 60% |

---

## ⚠️ Warning Thresholds

**Stop training and investigate if:**
- GPU temp > 80°C
- RAM usage > 100GB
- Disk space < 100GB free
- CPU load > 20 sustained
- GPU power draw > 450W

---

## 🚨 If System Goes Down

### 1. Check Logs (after reboot)
```bash
sudo journalctl -b -1 | tail -200 > ~/shutdown_log.txt
cat ~/shutdown_log.txt | grep -i "error\|critical\|shutdown"
```

### 2. Check What Happened
```bash
# Power/thermal issue?
sudo dmesg -T | grep -i "thermal\|power\|shutdown"

# Out of memory?
sudo dmesg | grep -i "oom\|out of memory"

# Training crash?
tail -100 encoder_train.log
```

### 3. Verify System Health
```bash
nvidia-smi              # GPU OK?
free -h                 # RAM OK?
df -h                   # Disk OK?
uptime                  # Load OK?
```

### 4. Check Training State
```bash
ls -lh embeddings.db                    # Embeddings saved?
ls -lht checkpoints/ | head             # Checkpoints?
python diagnose_variance_simple.py      # Embedding quality?
```

---

## �� Power Safety Facts

- **Your usage:** 300W (~2.5A @ 120V)
- **Circuit limit:** 1,800W (15A) or 2,400W (20A)
- **Safety margin:** 1,500W remaining (5x your usage)
- **Breaker trip risk:** Near zero

**You could run 4-5 systems like yours on one circuit.**

---

## 🔄 Resume Training

```bash
# Check if still running
ps aux | grep music_recommendation.py

# If stopped, restart (auto-resumes from checkpoint)
./run_music_pipeline.sh encoder 2>&1 | tee -a encoder_train.log
```

---

## 📞 Emergency Commands

```bash
# Kill training (if needed)
pkill -f music_recommendation.py

# Check GPU is responsive
nvidia-smi

# Reboot if system frozen
sudo reboot
```

---

## 🎯 Expected Training Duration

- **With 50 epochs:** ~2-4 hours
- **With 100 epochs:** ~4-8 hours
- **Check progress:** Look at checkpoint timestamps

```bash
ls -lt checkpoints/encoder_*.pth | head -5
```

---

## ✨ After Training Completes

```bash
# 1. Verify embeddings are good
python diagnose_variance_simple.py

# Should see:
# ✓ Unique embeddings: >90% (not 0.02%)
# ✓ Mean std: >0.1 (not 0.000002)
# ✓ Zero variance dims: <10% (not 71%)

# 2. If good, train classifier
./run_music_pipeline.sh classifier

# 3. If bad, check training logs for issues
grep -i "loss\|accuracy" encoder_train.log | tail -20
```

---

**Created:** 2026-01-29
**System:** Intel Ultra 9 285K + RTX 5090
**Config:** batch_size=32, workers=4, epochs=50
