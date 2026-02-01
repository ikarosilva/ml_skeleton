# System Recovery Plan

This document provides a checklist for recovering the ML pipeline after an unexpected system shutdown (e.g., power loss, system crash).

---

## 1. Immediate System Health Check

Before checking the pipeline, ensure the underlying system is stable.

1.  **Check System Logs:** Open a terminal and look for critical errors during the last boot.
    ```bash
    # Look for any kernel errors from the current boot
    journalctl -p err -b
    ```
    Look for disk I/O errors or file system corruption warnings.

2.  **Check Disk Usage:** Ensure you have adequate disk space.
    ```bash
    df -h
    ```

---

## 2. ML Pipeline Integrity Checklist

Follow these steps to check each component of the music recommendation pipeline.

### ✅ Step 2.1: Check MLflow Server & Database

The MLflow server tracks your experiment history. The database (`mlflow.db`) can sometimes be corrupted by a sudden shutdown.

1.  **Start the MLflow UI:**
    ```bash
    ml_skeleton mlflow-ui
    ```
    Alternatively, use the direct command: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

2.  **Verify Experiments:** Open your browser to `http://localhost:5000` and check if your previous experiments and runs are visible.

3.  **What if it's corrupt?** If the UI fails to start or shows errors, the `mlflow.db` file may be corrupt.
    *   **Recovery:** The MLflow database is for tracking and is not critical for resuming training. Your models are safe in `./checkpoints`. You can safely delete `mlflow.db` and start fresh, though you will lose your experiment history.
    *   **Prevention:** For long-term projects, consider configuring MLflow to use a more robust database backend like PostgreSQL.

### ✅ Step 2.2: Check Data Caches & Databases

The pipeline uses a waveform cache and an embedding database.

1.  **Waveform Cache (`./cache`):**
    *   **Purpose:** Stores resampled audio to speed up training. It is non-critical.
    *   **Action:** It's safest to clear the cache after a crash to prevent loading potentially corrupt files.
    ```bash
    # Clear the cache
    ./run_music_pipeline.sh clear-cache
    ```
    The cache will be rebuilt automatically on the next training run, or you can pre-build it: `./run_music_pipeline.sh build-cache`.

2.  **Embedding Database (`./embeddings.db`):**
    *   **Purpose:** Stores the output of the encoder (Stage 1). This is a critical artifact.
    *   **Action:** Check the SQLite database integrity.
    ```bash
    sqlite3 ./embeddings.db "PRAGMA integrity_check;"
    ```
    *   **Recovery:** If the command returns anything other than `ok`, the database is corrupt. You must regenerate it by re-running the encoder stage.
    ```bash
    ./run_music_pipeline.sh encoder
    ```

### ✅ Step 2.3: Check Model Checkpoints

Your trained models are the most important asset. They are saved in `./checkpoints`.

1.  **Find the Latest Checkpoint:**
    ```bash
    ls -lt ./checkpoints
    ```
    This lists files by modification time, showing the most recent ones first. You should see `encoder_best.pt` and `classifier_best.pt`.

2.  **Integrity:** The `torch.save` operation is generally atomic, so checkpoint files are unlikely to be corrupt. If you encounter loading errors, try using the `_final.pt` version instead of the `_best.pt` one.

---

## 3. Resuming Training

Once you've verified the components, you can resume the pipeline.

1.  **If `embeddings.db` was OK:** You don't need to re-train the encoder. You can proceed directly to training the classifier or generating recommendations.
    ```bash
    # Re-run only the classifier stage
    ./run_music_pipeline.sh classifier
    ```

2.  **If `embeddings.db` was corrupt:** You must run the full pipeline, which will use your existing best encoder checkpoint to regenerate the embeddings faster than training from scratch.
    ```bash
    # Run the full pipeline
    ./run_music_pipeline.sh all
    ```

---

## 4. Future Prevention: UPS Monitoring

Since you have a UPS, you can configure the system to shut down gracefully before the battery runs out.

1.  **Check Connectivity:**
    Use the provided helper script to see if your system detects the UPS.
    **Note:** Run this on the **HOST** system, not inside the Docker container.
    ```bash
    chmod +x ./check_power_status.sh
    ./check_power_status.sh
    ```

2.  **Install Daemon (Ubuntu/Debian):**
    *   **APC Devices:** `sudo apt install apcupsd` (Edit `/etc/apcupsd/apcupsd.conf` to set `ISCONFIGURED=yes`)
    *   **Generic/Other:** `sudo apt install nut`

3.  **Configuration:**
    Most UPS daemons are configured by default to trigger `/sbin/shutdown -h now` when battery is critical (e.g., < 5%). This prevents file corruption.