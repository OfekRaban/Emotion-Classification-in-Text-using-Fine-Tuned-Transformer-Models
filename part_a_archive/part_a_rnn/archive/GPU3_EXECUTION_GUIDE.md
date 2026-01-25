# GPU 3 Execution Guide - Complete Pipeline Notebook

## Overview

Your **complete_pipeline.ipynb** notebook is ready to run on GPU 3. This guide shows you how to execute it.

---

## What Will Happen

The notebook will execute all 66 cells (28 sections) automatically:

1. **Setup** - Import libraries, configure settings
2. **Data Loading** - Load train.csv and validation.csv
3. **EDA** - Exploratory data analysis and visualizations
4. **Preprocessing** - Clean and normalize text data
5. **Tokenization** - Convert text to sequences
6. **Embeddings** - Load GloVe or train Word2Vec
7. **Model Building** - Create LSTM/GRU/BiLSTM model
8. **Training** - Train model on GPU 3
9. **Evaluation** - Calculate metrics, generate plots
10. **Results** - Save all outputs

**Estimated Time:** 20-40 minutes (depends on model configuration)

---

## Before You Start

### 1. Check Data Files

```bash
cd /home/lab/rabanof/projects/Emotion_Detection_DL
ls -lh data/raw/train.csv data/raw/validation.csv
```

Should show:
- train.csv (~1.6M)
- validation.csv (~193K)

### 2. Download GloVe Embeddings (if using GloVe)

If the notebook is configured to use GloVe embeddings:

```bash
./download_glove.sh
```

This creates: `/home/lab/rabanof/Emotion_Detection_DL/glove/glove.6B.50d.txt`

**Note:** If using Word2Vec, skip this step (it trains automatically)

### 3. Check GPU 3 Availability

```bash
nvidia-smi -i 3
```

Should show GPU 3 status and memory.

---

## How to Execute

### Method 1: Simple Execution (Recommended)

```bash
cd /home/lab/rabanof/projects/Emotion_Detection_DL
./run_notebook_gpu3.sh
```

This will:
- Set GPU to 3
- Execute the entire notebook
- Save results to `notebooks/complete_pipeline_executed.ipynb`
- Log everything to `notebook_execution.log`

### Method 2: Direct Python Execution

```bash
cd /home/lab/rabanof/projects/Emotion_Detection_DL
python3 run_notebook_on_gpu.py
```

### Method 3: Interactive Jupyter (on Cortex)

If you want to run cells interactively:

```bash
# Start Jupyter on Cortex
jupyter notebook --no-browser --port=8888

# Then SSH tunnel from your local machine:
# ssh -L 8888:localhost:8888 your_username@cortex.cse.bgu.ac.il

# Open in browser: http://localhost:8888
# Navigate to notebooks/complete_pipeline.ipynb
# Before running, add this to the first cell:

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
```

---

## What Gets Generated

After execution, you'll have:

### 1. Executed Notebook
```
notebooks/complete_pipeline_executed.ipynb
```
Contains all outputs, plots, and results inline.

### 2. Model Files
```
saved_models/
  ├── ultimate_emotion_detection_best_model.h5
  └── embedding_matrix.npy
```

### 3. Results
```
results/
  ├── ultimate_emotion_detection_metrics.json
  ├── ultimate_emotion_detection_training_history.png
  ├── ultimate_emotion_detection_confusion_matrix.png
  ├── ultimate_emotion_detection_confusion_matrix_normalized.png
  ├── ultimate_emotion_detection_classification_report.png
  ├── ultimate_emotion_detection_per_class_f1.png
  ├── ultimate_emotion_detection_per_class_precision.png
  └── ultimate_emotion_detection_per_class_recall.png
```

### 4. Logs
```
logs/
  ├── ultimate_emotion_detection/  (TensorBoard logs)
  └── ultimate_emotion_detection_training.csv
```

### 5. Execution Log
```
notebook_execution.log
```

---

## Expected Results

Based on your notebook configuration:

- **Validation Accuracy:** 85-90%
- **Macro F1 Score:** 0.83-0.87
- **Training Time:** 10-30 minutes (depends on epochs/early stopping)

Per-class performance (typical):
```
Emotion     F1-Score
─────────────────────
Sadness     0.88
Joy         0.91
Love        0.87
Anger       0.85
Fear        0.82
Surprise    0.82
─────────────────────
Macro Avg   0.86
```

---

## Monitoring Execution

### Watch Progress
```bash
tail -f notebook_execution.log
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi -i 3
```

### Check Process
```bash
ps aux | grep run_notebook_on_gpu
```

---

## Customizing Before Execution

If you want to change settings before running:

### Option 1: Edit the Notebook Configuration (Section 2)

Open `notebooks/complete_pipeline.ipynb` and modify Section 2:

```python
# Model type
config.model_type = 'lstm'  # Options: 'lstm', 'gru', 'bilstm', 'bigru'

# Embedding
config.embedding_type = 'glove'  # Options: 'glove', 'word2vec'
config.embedding_dim = 50  # Options: 50, 100, 200, 300

# GloVe path (if using GloVe)
config.glove_path = "/home/lab/rabanof/Emotion_Detection_DL/glove/glove.6B.50d.txt"

# Architecture
config.rnn_units = 128  # Try: 64, 128, 256
config.num_rnn_layers = 1  # Try: 1, 2, 3

# Regularization
config.dropout = 0.2  # Try: 0.2, 0.3, 0.5
config.spatial_dropout = 0.2

# Training
config.epochs = 50
config.batch_size = 32
config.learning_rate = 0.001
config.patience = 5  # Early stopping patience

# Experiment name
config.experiment_name = 'lstm_glove50_baseline'
```

Then run the execution script.

### Option 2: Use Word2Vec Instead of GloVe

In Section 2 of the notebook, change:
```python
config.embedding_type = 'word2vec'  # Will train automatically
config.embedding_dim = 50
```

No need to download GloVe!

---

## Troubleshooting

### Error: "No module named 'nbconvert'"

Install nbconvert:
```bash
pip install --user nbconvert jupyter
```

### Error: "Notebook not found"

Make sure you're in the project directory:
```bash
cd /home/lab/rabanof/projects/Emotion_Detection_DL
```

### Error: "GPU not detected"

Check GPU 3 is available:
```bash
nvidia-smi -i 3
```

If not available, edit `run_notebook_gpu3.sh` to use a different GPU:
```bash
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0 instead
```

### Error: "Out of memory"

Edit Section 2 in the notebook to reduce memory usage:
```python
config.batch_size = 16  # Reduce from 32
config.rnn_units = 64   # Reduce from 128
```

### Error: "GloVe file not found"

If using GloVe, download it first:
```bash
./download_glove.sh
```

Or switch to Word2Vec in the notebook:
```python
config.embedding_type = 'word2vec'
```

### Execution is Taking Too Long

Check the notebook configuration (Section 2):
- Reduce `config.epochs` (e.g., from 50 to 20)
- Check `config.patience` for early stopping (default: 5)

Most models should converge in 10-15 epochs with early stopping.

---

## Viewing Results

### 1. View Executed Notebook

Open in Jupyter:
```bash
jupyter notebook notebooks/complete_pipeline_executed.ipynb
```

Or convert to HTML:
```bash
jupyter nbconvert --to html notebooks/complete_pipeline_executed.ipynb
```

### 2. View Metrics

```bash
cat results/ultimate_emotion_detection_metrics.json
```

### 3. View Training History

```bash
cat logs/ultimate_emotion_detection_training.csv
```

### 4. View Plots

All plots are saved in `results/` directory and also embedded in the executed notebook.

---

## Running Multiple Experiments

To run multiple experiments with different configurations:

1. **Run first experiment:**
   ```bash
   ./run_notebook_gpu3.sh
   ```

2. **Rename output:**
   ```bash
   mv notebooks/complete_pipeline_executed.ipynb notebooks/exp1_lstm_glove50.ipynb
   ```

3. **Edit configuration in original notebook** (Section 2)

4. **Run again:**
   ```bash
   ./run_notebook_gpu3.sh
   ```

5. **Compare results** using Section 24-25 of the notebook

---

## Summary

**To execute your complete pipeline on GPU 3:**

```bash
# 1. Go to project directory
cd /home/lab/rabanof/projects/Emotion_Detection_DL

# 2. Download GloVe (if needed, one time only)
./download_glove.sh

# 3. Run notebook on GPU 3
./run_notebook_gpu3.sh

# 4. Wait 20-40 minutes

# 5. Check results
cat results/ultimate_emotion_detection_metrics.json
```

**Expected accuracy: 85-90%**

---

## Quick Reference

| Task | Command |
|------|---------|
| Execute notebook | `./run_notebook_gpu3.sh` |
| Watch progress | `tail -f notebook_execution.log` |
| Check GPU | `nvidia-smi -i 3` |
| View results | `cat results/ultimate_emotion_detection_metrics.json` |
| View executed notebook | Open `notebooks/complete_pipeline_executed.ipynb` |

---

**You're ready to run your complete pipeline on GPU 3!**
