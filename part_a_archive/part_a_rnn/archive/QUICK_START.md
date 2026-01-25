# Quick Start - Run Your Notebook on GPU

## Why You See the Warning

```
WARNING: Could not query GPU 3 with nvidia-smi
```

**This is NORMAL!** You're currently on the **login node** which has no GPUs.

You need to request a **GPU node** from SLURM to run the notebook.

---

## How to Run

### Step 1: SSH to H100 Node

```bash
# SSH to the Cortex H100 node
ssh ctxh100-01
```

### Step 2: Check GPU

```bash
# Verify GPU 3 is available
nvidia-smi
```

You should see GPU 0, 1, 2, 3 listed.

### Step 3: Run Notebook

```bash
# Go to project
cd /home/lab/rabanof/projects/Emotion_Detection_DL

# Download GloVe (first time only)
./download_glove.sh

# Run notebook on GPU 3
./run_notebook_gpu3.sh
```

### Alternative: Run in Background with Screen/Tmux

If you want to run in background (survives disconnects):

```bash
# SSH to H100 node
ssh ctxh100-01

# Start screen session
screen -S emotion_training

# Go to project
cd /home/lab/rabanof/projects/Emotion_Detection_DL

# Run notebook
./run_notebook_gpu3.sh

# Detach: Press Ctrl+A then D
# Reattach later: screen -r emotion_training
```

---

## What Each Method Does

### Interactive (Option A):
- ✓ You get a shell on GPU node
- ✓ Can see output in real-time
- ✓ Good for debugging
- ✗ Disconnects if SSH drops

### Batch (Option B):
- ✓ Runs in background
- ✓ Survives disconnects
- ✓ Good for long jobs
- ✗ Can't see output in real-time (but can tail log)

---

## Timeline

1. **Request GPU:** 0-10 minutes (depends on availability)
2. **Download GloVe:** 2-5 minutes (first time only)
3. **Notebook execution:** 20-40 minutes
4. **Total:** 25-60 minutes

---

## After Execution

Check results:
```bash
# Metrics
cat results/ultimate_emotion_detection_metrics.json

# Executed notebook (with all outputs)
# Open: notebooks/complete_pipeline_executed.ipynb

# Training log
cat logs/ultimate_emotion_detection_training.csv
```

---

## Files You Have

| File | Purpose |
|------|---------|
| `run_notebook_gpu3.sh` | Main execution script |
| `run_notebook_batch.sh` | SLURM batch job script |
| `download_glove.sh` | Download GloVe embeddings |
| `notebooks/complete_pipeline.ipynb` | Your notebook (66 cells) |

---

## Expected Results

- **Accuracy:** 85-90%
- **Macro F1:** 0.83-0.87
- **Time:** 20-40 minutes

---

## Need Help?

- **[CORTEX_CLUSTER_GUIDE.md](CORTEX_CLUSTER_GUIDE.md)** - Complete cluster guide
- **[GPU3_EXECUTION_GUIDE.md](GPU3_EXECUTION_GUIDE.md)** - Detailed execution guide
- **[RUN_NOTEBOOK_README.md](RUN_NOTEBOOK_README.md)** - Quick reference

---

## Summary

**You're on the login node (no GPUs).**

**To run your notebook:**

1. SSH to H100 node: `ssh ctxh100-01`
2. Check GPU: `nvidia-smi`
3. Go to project: `cd /home/lab/rabanof/projects/Emotion_Detection_DL`
4. Download GloVe (first time): `./download_glove.sh`
5. Run notebook: `./run_notebook_gpu3.sh`
6. Wait 20-40 minutes
7. Check results

**The warning is normal and will disappear once you SSH to ctxh100-01!**
