# Running on Cortex Cluster - Complete Guide

## Important: You Need to Be on a GPU Node

Currently you're on a **login node** (no GPUs available). You need to access a **GPU node** to run the notebook.

---

## Option 1: Interactive GPU Session (Recommended for Testing)

### Step 1: Request GPU Node

```bash
# SSH to Cortex
ssh your_username@cortex.cse.bgu.ac.il

# Request interactive session with GPU
srun --gres=gpu:1 --time=2:00:00 --pty bash
```

This gives you:
- 1 GPU (any available)
- 2 hours
- Interactive shell

### Step 2: Check Which GPU You Got

```bash
nvidia-smi
```

You'll see which GPU is assigned (e.g., GPU 0, 1, 2, or 3).

### Step 3: Run the Notebook

```bash
cd /home/lab/rabanof/projects/Emotion_Detection_DL

# If you want to use the assigned GPU (whatever it is):
./run_notebook_gpu3.sh  # Will use GPU via CUDA_VISIBLE_DEVICES

# OR if you want to specify which GPU:
# Edit the script first to use the correct GPU number
```

---

## Option 2: Batch Job (Recommended for Long Training)

Create a SLURM batch script to run the notebook in the background.

### Step 1: Create Batch Script

```bash
cat > run_notebook_batch.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=emotion_notebook
#SBATCH --output=notebook_job_%j.log
#SBATCH --error=notebook_job_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

echo "Job started on $(hostname)"
echo "GPU assigned:"
nvidia-smi

cd /home/lab/rabanof/projects/Emotion_Detection_DL

# Run notebook
./run_notebook_gpu3.sh

echo "Job completed"
EOF

chmod +x run_notebook_batch.sh
```

### Step 2: Submit Job

```bash
sbatch run_notebook_batch.sh
```

### Step 3: Monitor Job

```bash
# Check job status
squeue -u your_username

# View output (replace JOBID with actual job ID)
tail -f notebook_job_JOBID.log

# Or view execution log
tail -f notebook_execution.log
```

---

## Option 3: Specific GPU Request

If you specifically need GPU 3:

### Interactive:
```bash
srun --gres=gpu:1 --constraint="gpu3" --time=2:00:00 --pty bash
```

### Batch:
```bash
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu3"
```

**Note:** GPU constraints depend on your cluster configuration. Check with `sinfo` or cluster docs.

---

## Checking GPU Availability

### List all GPUs on the cluster:
```bash
sinfo -o "%20N %10c %10m %25f %10G"
```

### Check which GPUs are free:
```bash
squeue -o "%10i %15j %10u %10T %10M %10l %5D %N %b"
```

### See GPU usage:
```bash
# On a GPU node:
nvidia-smi
```

---

## Current Situation

You're seeing:
```
WARNING: Could not query GPU 3 with nvidia-smi
```

This is **normal** because you're on the **login node** which has no GPUs.

**What to do:**
1. Request a GPU node (see Option 1 or 2 above)
2. Then run the script
3. It will work on the GPU node

---

## Complete Workflow Example

### For Interactive Session:

```bash
# 1. SSH to Cortex
ssh your_username@cortex.cse.bgu.ac.il

# 2. Request GPU node (2 hours)
srun --gres=gpu:1 --time=2:00:00 --pty bash

# 3. You're now on a GPU node! Check:
nvidia-smi

# 4. Go to project
cd /home/lab/rabanof/projects/Emotion_Detection_DL

# 5. Download GloVe (if needed, first time only)
./download_glove.sh

# 6. Run notebook
./run_notebook_gpu3.sh

# 7. Wait 20-40 minutes

# 8. Check results
cat results/ultimate_emotion_detection_metrics.json
```

### For Batch Job:

```bash
# 1. SSH to Cortex
ssh your_username@cortex.cse.bgu.ac.il

# 2. Go to project
cd /home/lab/rabanof/projects/Emotion_Detection_DL

# 3. Download GloVe (if needed, first time only)
./download_glove.sh

# 4. Submit batch job
sbatch run_notebook_batch.sh

# 5. Monitor (from login node)
tail -f notebook_execution.log

# 6. When done, check results
cat results/ultimate_emotion_detection_metrics.json
```

---

## GPU Configuration in the Script

The script sets:
```bash
export CUDA_VISIBLE_DEVICES=3
```

**What this means:**
- If you have multiple GPUs, it will try to use GPU 3
- If you only have 1 GPU assigned, it will use that GPU (regardless of the number)
- The SLURM scheduler handles actual GPU allocation

**To use a different GPU:**

Edit `run_notebook_gpu3.sh` and change:
```bash
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
# or
export CUDA_VISIBLE_DEVICES=1  # Use GPU 1
```

Or just let SLURM assign any available GPU (recommended):
```bash
# Comment out or remove the CUDA_VISIBLE_DEVICES line
# SLURM will automatically make the assigned GPU visible
```

---

## Troubleshooting

### "No GPUs available"

You're on the login node. Request a GPU node:
```bash
srun --gres=gpu:1 --time=2:00:00 --pty bash
```

### "Out of memory"

Reduce memory usage in the notebook (Section 2):
```python
config.batch_size = 16  # Reduce from 32
config.rnn_units = 64   # Reduce from 128
```

### "Job pending for too long"

GPUs might be busy. Check queue:
```bash
squeue -u your_username
```

Try requesting less time:
```bash
srun --gres=gpu:1 --time=1:00:00 --pty bash
```

### "Connection timeout" during training

Use batch job instead of interactive session (Option 2).

---

## Expected Timeline

### Interactive Session:
1. Request GPU: 0-10 minutes (depends on availability)
2. Setup: 1-2 minutes
3. Download GloVe (first time): 2-5 minutes
4. Notebook execution: 20-40 minutes
5. **Total:** 25-60 minutes

### Batch Job:
1. Submit job: instant
2. Job starts: 0-30 minutes (depends on queue)
3. Execution: 20-40 minutes
4. **Total:** 20-70 minutes

---

## After Execution

All results will be in:
```
/home/lab/rabanof/projects/Emotion_Detection_DL/
  ├── notebooks/complete_pipeline_executed.ipynb
  ├── results/ultimate_emotion_detection_metrics.json
  ├── saved_models/ultimate_emotion_detection_best_model.h5
  ├── logs/ultimate_emotion_detection_training.csv
  └── notebook_execution.log
```

You can view these from the login node (no GPU needed).

---

## Summary

**Current Status:** You're on login node (no GPUs)

**What to do:**
1. Request GPU node: `srun --gres=gpu:1 --time=2:00:00 --pty bash`
2. Run notebook: `./run_notebook_gpu3.sh`
3. Wait 20-40 minutes
4. Check results

**Or use batch job for background execution (recommended).**

The warning you see is normal - the script will work once you're on a GPU node!
