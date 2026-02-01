#!/bin/bash
# Training Monitoring Dashboard
# Usage: ./monitor_training.sh

clear
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          ML TRAINING SYSTEM MONITOR                            ║"
echo "║          Press Ctrl+C to exit                                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo

while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║          ML TRAINING SYSTEM MONITOR                            ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo

    # Timestamp
    echo "📅 $(date '+%Y-%m-%d %H:%M:%S')"
    echo

    # GPU Status
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎮 GPU STATUS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=power.draw,power.limit,utilization.gpu,temperature.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{
            printf "  Power:  %6.1fW / %6.1fW", $1, $2
            if ($1 > 400) printf " ⚠️  HIGH"
            else if ($1 > 250) printf " ⚡ ACTIVE"
            else printf " ✓ Normal"
            printf "\n"

            printf "  Usage:  %5.1f%%", $3
            if ($3 > 90) printf "            🔥 MAXED"
            else if ($3 > 50) printf "            ⚙️  WORKING"
            else if ($3 > 10) printf "            💤 LIGHT"
            else printf "            ⏸️  IDLE"
            printf "\n"

            printf "  Temp:   %5.1f°C", $4
            if ($4 > 80) printf "            🌡️  HOT!"
            else if ($4 > 65) printf "            🌡️  Warm"
            else printf "            ❄️  Cool"
            printf "\n"

            printf "  Memory: %6.0fMB / %6.0fMB", $5, $6
            pct = ($5 / $6) * 100
            if (pct > 90) printf "  ⚠️  FULL"
            else if (pct > 70) printf "  📊 HIGH"
            else printf "  ✓ OK"
            printf "\n"
        }'
    else
        echo "  ❌ nvidia-smi not available"
    fi
    echo

    # CPU Status
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "💻 CPU STATUS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')
    LOAD_INT=$(echo $LOAD | awk '{print int($1)}')
    printf "  Load:   %5s (24 cores)", "$LOAD"
    if [ "$LOAD_INT" -gt 18 ]; then
        echo "        ⚠️  HIGH"
    elif [ "$LOAD_INT" -gt 10 ]; then
        echo "        ⚙️  ACTIVE"
    else
        echo "        ✓ Normal"
    fi
    echo

    # RAM Status
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🧠 MEMORY STATUS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    free -h | grep Mem | awk '{
        printf "  Used:   %6s / %6s", $3, $2
        # Extract numeric values for comparison
        used_val = $3
        total_val = $2
    }'
    USED_GB=$(free -g | grep Mem | awk '{print $3}')
    if [ "$USED_GB" -gt 100 ]; then
        echo "      ⚠️  HIGH"
    elif [ "$USED_GB" -gt 60 ]; then
        echo "      📊 MODERATE"
    else
        echo "      ✓ OK"
    fi
    free -h | grep Mem | awk '{printf "  Free:   %6s\n", $7}'
    echo

    # Disk Status
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "💾 DISK STATUS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    df -h /git/ml_skeleton | grep -v Filesystem | awk '{
        printf "  Used:   %6s / %6s (%s used)\n", $3, $2, $5
        printf "  Free:   %6s", $4
        free_val = $4
    }'
    FREE_GB=$(df -BG /git/ml_skeleton | grep -v Filesystem | awk '{print $4}' | tr -d 'G')
    if [ "$FREE_GB" -lt 100 ]; then
        echo "                  ⚠️  LOW"
    elif [ "$FREE_GB" -lt 200 ]; then
        echo "                  ⚡ MODERATE"
    else
        echo "                  ✓ PLENTY"
    fi
    echo

    # Training Process Status
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🏃 TRAINING PROCESS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    TRAIN_PID=$(pgrep -f "music_recommendation.py.*encoder" | head -1)
    if [ -n "$TRAIN_PID" ]; then
        echo "  Status: ✅ RUNNING"
        ps -p $TRAIN_PID -o pid,etime,%cpu,%mem | tail -1 | awk '{printf "  PID:    %s\n  Runtime: %s\n  CPU:    %s%%\n  Memory: %s%%\n", $1, $2, $3, $4}'

        # Check latest checkpoint
        if [ -d "checkpoints" ]; then
            LATEST=$(ls -t checkpoints/encoder_*.pth 2>/dev/null | head -1)
            if [ -n "$LATEST" ]; then
                MOD_TIME=$(stat -c %y "$LATEST" 2>/dev/null | cut -d. -f1)
                echo "  Latest checkpoint: $(basename $LATEST)"
                echo "  Updated: $MOD_TIME"
            fi
        fi
    else
        echo "  Status: ⏸️  NOT RUNNING"
        echo "  (No active training process found)"
    fi
    echo

    # Power Estimate
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⚡ POWER ESTIMATE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if command -v nvidia-smi &> /dev/null; then
        GPU_POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits)
        CPU_EST=100  # Estimated CPU power
        OTHER_EST=80  # Other components
        TOTAL=$(echo "$GPU_POWER + $CPU_EST + $OTHER_EST" | bc)
        CIRCUIT_LIMIT=1800
        SAFE_LIMIT=1440
        PCT=$(echo "scale=1; ($TOTAL / $SAFE_LIMIT) * 100" | bc)

        printf "  GPU:      %6.1fW\n" $GPU_POWER
        printf "  CPU:      ~%3dW (estimated)\n" $CPU_EST
        printf "  Other:    ~%3dW\n" $OTHER_EST
        printf "  ─────────────────────\n"
        printf "  Total:    ~%6.1fW", $TOTAL

        if (( $(echo "$TOTAL > 900" | bc -l) )); then
            echo "            ⚠️  VERY HIGH"
        elif (( $(echo "$TOTAL > 600" | bc -l) )); then
            echo "            ⚡ HIGH"
        elif (( $(echo "$TOTAL > 400" | bc -l) )); then
            echo "            📊 MODERATE"
        else
            echo "            ✓ LOW"
        fi

        printf "  Circuit:  %.0f%% of 15A circuit\n" $PCT
        printf "  Headroom: %.0fW remaining\n" $(echo "$SAFE_LIMIT - $TOTAL" | bc)
    fi
    echo

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Updating in 5 seconds... (Ctrl+C to exit)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    sleep 5
done
