# System Monitoring & Troubleshooting Guide

**Date Created:** 2026-01-29
**System:** Intel Core Ultra 9 285K + NVIDIA RTX 5090 (32GB)
**Purpose:** Reference guide for monitoring ML training and diagnosing system shutdowns

---

## Quick Reference: Power Safety

### Your Current System Power Draw
- **GPU:** 122W (during training) / 575W max
- **CPU:** ~100W (during training) / 250W max
- **Total System:** ~300W typical / ~950W absolute max
- **Circuit Safety:** Using only 17-21% of standard 15A circuit

### Circuit Limits
| Circuit Type | Max Power | Safe Continuous | Your Usage | Headroom |
|-------------|-----------|-----------------|------------|----------|
| Standard 15A @ 120V | 1,800W | 1,440W (80%) | 300W | 1,140W (79% free) |
| Dedicated 20A @ 120V | 2,400W | 1,920W (80%) | 300W | 1,620W (84% free) |

**Verdict:** Your training is extremely safe - using only 21% of circuit capacity.

---

## Real-Time Monitoring Commands

### GPU Monitoring
```bash
# Quick GPU status
nvidia-smi

# Continuous monitoring (updates every 1 second)
watch -n 1 nvidia-smi

# Power-focused view
watch -n 1 'nvidia-smi --query-gpu=power.draw,power.limit,utilization.gpu,temperature.gpu,memory.used,memory.total --format=csv,noheader,nounits'
```

**Expected values during training:**
- Power draw: 100-300W (yours: 122W is very low)
- GPU utilization: 30-90%
- Temperature: 40-75°C (yours: 41°C is excellent)
- Memory: 4-16GB / 32GB (yours: 4.3GB is conservative)

### CPU Monitoring
```bash
# CPU load average
uptime

# Top CPU processes
ps aux --sort=-%cpu | head -n 10

# Detailed CPU usage
htop
# Or: top
```

**Expected values during training:**
- Load average: 4-8 (yours: 7.4 with 24 cores)
- CPU processes: Main training + 4 dataloader workers
- Combined CPU: ~200-300% (multiple cores)

### Memory Monitoring
```bash
# RAM usage
free -h

# Monitor RAM continuously
watch -n 1 free -h
```

**Expected values:**
- Used: 20-40 GB / 125 GB
- Available: >80 GB
- Swap: Should remain at 0 or very low

### Disk Space Monitoring
```bash
# Check disk space
df -h

# Check cache size
du -sh /git/ml_skeleton/cache

# Monitor disk usage
watch -n 5 'df -h | grep -E "Filesystem|/git"'
```

**Expected values:**
- Free space: >200 GB (yours: 408 GB)
- Cache size: ~427 GB (stable)

### Combined Dashboard
```bash
# Create a simple monitoring script
cat > /tmp/monitor.sh << 'EOF'
#!/bin/bash
clear
echo "=== SYSTEM MONITORING ==="
echo
echo "GPU Status:"
nvidia-smi --query-gpu=power.draw,utilization.gpu,temperature.gpu,memory.used --format=csv,noheader,nounits | awk -F', ' '{printf "  Power: %sW | Util: %s%% | Temp: %s°C | Mem: %sMB\n", $1, $2, $3, $4}'
echo
echo "CPU Load:"
uptime | awk -F'load average:' '{print "  " $2}'
echo
echo "Memory:"
free -h | grep Mem | awk '{printf "  Used: %s / %s (Available: %s)\n", $3, $2, $7}'
echo
echo "Training Process:"
ps aux | grep "music_recommendation.py" | grep -v grep | head -1 | awk '{printf "  PID: %s | CPU: %s%% | Runtime: \n", $2, $3}'
ps -p $(pgrep -f music_recommendation.py | head -1) -o pid,etime 2>/dev/null | tail -1
echo
echo "Press Ctrl+C to exit"
EOF
chmod +x /tmp/monitor.sh

# Run it
watch -n 2 /tmp/monitor.sh
```

---

## Warning Signs (Check These If System Becomes Unstable)

### Critical Warning Signs
1. **GPU Temperature > 85°C**
   - Current: 41°C (excellent)
   - Check: `nvidia-smi` temperature column
   - Action: Improve cooling, reduce batch size

2. **Power Draw Spikes to 500W+**
   - Current: 122W (very safe)
   - Check: `nvidia-smi` power column
   - Action: Not expected with your settings

3. **RAM Usage > 100 GB**
   - Current: 23 GB (safe)
   - Check: `free -h`
   - Action: Reduce num_workers or batch size

4. **Disk < 50 GB Free**
   - Current: 408 GB (plenty)
   - Check: `df -h`
   - Action: Clear cache or logs

5. **CPU Load > 20 (sustained)**
   - Current: 7.4 (normal)
   - Check: `uptime`
   - Action: Reduce num_workers

### Early Warning Indicators
```bash
# Check system logs for errors
sudo dmesg | tail -50

# Check for kernel errors
sudo journalctl -p err -n 50

# Check for thermal throttling
sudo dmesg | grep -i "thermal\|throttle"
```

---

## If System Goes Down: Diagnostic Checklist

### 1. Power-Related Shutdown

**Symptoms:**
- Sudden shutdown with no warning
- System completely off
- No logs or error messages

**Check:**
```bash
# After reboot, check system logs
sudo journalctl -b -1 -n 100  # Last boot logs

# Look for power events
sudo journalctl -b -1 | grep -i "power\|shutdown\|critical"

# Check hardware logs
sudo dmesg -T | grep -i "hardware\|critical"
```

**Likely causes:**
- ❌ Circuit breaker tripped (unlikely at 300W)
- ❌ PSU overload (unlikely)
- ✓ Thermal shutdown (check temps)
- ✓ Hardware failure

### 2. Thermal Shutdown

**Symptoms:**
- System shuts down during training
- High temperatures before shutdown
- System restarts automatically

**Check:**
```bash
# Check CPU/GPU temperatures from logs
sudo journalctl -b -1 | grep -i "temperature\|thermal\|overheat"

# Check if thermal throttling occurred
sudo dmesg -T | grep -i "throttle"
```

**Prevention:**
- Monitor temps: `watch -n 1 nvidia-smi`
- Improve cooling/airflow
- Reduce batch size or learning rate

### 3. Memory Issues (OOM)

**Symptoms:**
- Process killed during training
- "Out of memory" errors
- System becomes unresponsive

**Check:**
```bash
# Check for OOM killer
sudo dmesg | grep -i "out of memory\|oom"

# Check which process was killed
sudo journalctl -b -1 | grep -i "killed\|oom"
```

**Prevention:**
- Reduce batch size in config
- Reduce num_workers
- Check: `free -h` during training

### 4. Software/Training Crash

**Symptoms:**
- Python process exits with error
- Training stops but system stays on
- Error message in terminal/logs

**Check:**
```bash
# Check training logs
tail -100 encoder_train.log  # Or whatever your log file is

# Check Python errors
grep -i "error\|exception\|traceback" encoder_train.log
```

---

## Training Status Checks

### Is Training Still Running?
```bash
# Check if process is running
ps aux | grep music_recommendation.py

# Check process details
ps -p $(pgrep -f music_recommendation.py) -o pid,etime,cmd

# Check GPU is being used
nvidia-smi | grep python
```

### How Long Has It Been Running?
```bash
ps -p $(pgrep -f music_recommendation.py | head -1) -o pid,etime,cmd
```

### Check Training Progress
```bash
# If using MLflow
# Open browser: http://localhost:5000

# Check checkpoint files
ls -lht /git/ml_skeleton/checkpoints/ | head

# Check latest checkpoint timestamp
stat /git/ml_skeleton/checkpoints/*.pth 2>/dev/null | grep Modify | tail -1
```

### Verify Embeddings After Training
```bash
# After encoder completes, check that embeddings were written
sqlite3 embeddings.db "SELECT COUNT(*) FROM embeddings;"
ls -lt checkpoints/encoder_*.pth | head -3
```

---

## Recovery After System Shutdown

### 1. Check What Happened
```bash
# View logs from previous boot
sudo journalctl -b -1 | tail -200 > /tmp/shutdown_logs.txt
less /tmp/shutdown_logs.txt

# Look for keywords
grep -i "shutdown\|power\|critical\|error\|thermal" /tmp/shutdown_logs.txt
```

### 2. Verify System Health
```bash
# GPU health check
nvidia-smi

# CPU check
lscpu
uptime

# Memory check
free -h

# Disk check
df -h
```

### 3. Check Training State
```bash
# Check if embeddings database was created
ls -lh /git/ml_skeleton/embeddings.db

# Check checkpoint files
ls -lht /git/ml_skeleton/checkpoints/

# Check MLflow runs
ls -lht /git/ml_skeleton/mlruns/
```

### 4. Resume or Restart Training

**If training was interrupted:**
```bash
# Check last checkpoint
ls -lt checkpoints/encoder_*.pth | head -1

# The training script should auto-resume from checkpoints
./run_music_pipeline.sh encoder 2>&1 | tee -a encoder_train_resume.log
```

**If starting fresh:**
```bash
# Backup old embeddings if needed
mv embeddings.db embeddings_old_$(date +%Y%m%d).db

# Start training
./run_music_pipeline.sh encoder 2>&1 | tee encoder_train.log
```

---

## Resource Configuration Reference

### Current Optimized Settings (from configs/music_moco.yaml)

```yaml
# CPU/Workers
num_workers: 4                    # Data loading workers
dataloader_workers: 4             # PyTorch dataloader workers
chunk_cache_workers: 4            # Cache building workers

# GPU
batch_size: 32                    # Per-batch GPU memory
gpu_memory_limit_gb: 20           # Max GPU memory allowed
precision: "bf16-mixed"           # Memory-efficient precision

# Training
epochs: 50                        # Reduced from 100 for testing
```

### If You Need to Reduce Resources Further:

```yaml
# More conservative settings
num_workers: 2                    # Half the workers
batch_size: 16                    # Half the batch size
gpu_memory_limit_gb: 16           # Even more conservative
```

---

## Emergency Contacts & Resources

### System Specs
- **CPU:** Intel Core Ultra 9 285K (24 cores)
- **GPU:** NVIDIA RTX 5090 (32GB VRAM, 575W TDP)
- **RAM:** 125 GB
- **Disk:** 1.9 TB (408 GB free)
- **OS:** Linux 6.14.0-37-generic

### Key File Locations
- **Config:** `/git/ml_skeleton/configs/music_moco.yaml`
- **Embeddings:** `/git/ml_skeleton/embeddings.db`
- **Checkpoints:** `/git/ml_skeleton/checkpoints/`
- **Cache:** `/git/ml_skeleton/cache/chunks/` (427 GB)
- **Logs:** Training logs in current directory

### Verify State
```bash
# Embedding count and latest checkpoints
sqlite3 embeddings.db "SELECT COUNT(*) FROM embeddings;"
ls -lt checkpoints/*.pth | head -5
```

---

## Quick Decision Tree

```
System went down?
│
├─ Was it sudden (instant off)?
│  └─ Check power logs, breaker, PSU
│
├─ Did it gradually slow down?
│  └─ Check temps, RAM, swap usage
│
├─ Did training process crash?
│  └─ Check training logs, Python errors
│
└─ Did it restart automatically?
   └─ Check thermal logs, OOM killer logs
```

**Most Likely Causes (in order):**
1. Software bug/training issue (check logs)
2. Thermal shutdown (check temps > 85°C)
3. OOM killer (check RAM usage > 100GB)
4. Hardware failure (rare)
5. Power issues (very unlikely at 300W)

---

## Normal Operating Ranges (Your System)

| Metric | Normal | Warning | Critical |
|--------|--------|---------|----------|
| GPU Power | 100-250W | 300-400W | >450W |
| GPU Temp | 35-65°C | 70-80°C | >85°C |
| GPU Memory | 4-16GB | 20-28GB | >30GB |
| CPU Load | 4-10 | 12-18 | >20 |
| RAM Usage | 20-50GB | 60-90GB | >100GB |
| Disk Free | >200GB | 100-200GB | <50GB |
| System Power | 250-400W | 500-700W | >900W |

Your current values are all in the "Normal" range.

---

## Final Notes

**Your system is running very conservatively:**
- GPU: 21% of max power (122W / 575W)
- CPU: 31% utilization (7.4 / 24 cores)
- RAM: 18% usage (23GB / 125GB)
- Power: 17% of circuit capacity (300W / 1800W)

**Likelihood of power-related shutdown: EXTREMELY LOW**

If your system does go down, it's most likely:
1. Software crash (check training logs)
2. Hardware issue unrelated to power
3. Network/storage issue

Power consumption is NOT a concern with these settings.
