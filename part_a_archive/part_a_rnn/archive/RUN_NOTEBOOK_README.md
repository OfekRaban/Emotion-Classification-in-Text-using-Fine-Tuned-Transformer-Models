# Run Complete Pipeline on GPU 3 - Quick Start

## Your Notebook is Ready!

**Notebook:** [notebooks/complete_pipeline.ipynb](notebooks/complete_pipeline.ipynb)
- 66 cells, 28 sections
- All EDA, preprocessing, training, evaluation
- Emojis removed, sections fixed
- Ready to execute on GPU 3

---

## Quick Start (4 Steps)

### Step 1: SSH to H100 Node

```bash
ssh ctxh100-01
```

You need to be on the H100 node to access GPUs.

### Step 2: Download GloVe (if needed)

```bash
cd /home/lab/rabanof/projects/Emotion_Detection_DL
./download_glove.sh
```

**Skip this if:**
- You already downloaded GloVe, OR
- The notebook uses Word2Vec (check Section 2: `config.embedding_type`)

### Step 3: Run Notebook on GPU 3

```bash
./run_notebook_gpu3.sh
```

This executes all 66 cells automatically on GPU 3.

### Step 4: Check Results

```bash
# View metrics
cat results/ultimate_emotion_detection_metrics.json

# View executed notebook
# Open: notebooks/complete_pipeline_executed.ipynb
```

**Done!**

---

## What Happens

The script will:
1. Set GPU to 3 (CUDA_VISIBLE_DEVICES=3)
2. Execute all 66 cells in order
3. Train your model (LSTM/GRU/BiLSTM)
4. Generate all plots and metrics
5. Save everything to files

**Time:** 20-40 minutes

**Expected Accuracy:** 85-90%

---

## Files Created

After execution:

```
notebooks/
  └── complete_pipeline_executed.ipynb    (All results embedded)

results/
  ├── ultimate_emotion_detection_metrics.json
  ├── *_training_history.png
  ├── *_confusion_matrix.png
  ├── *_confusion_matrix_normalized.png
  ├── *_classification_report.png
  └── *_per_class_*.png

saved_models/
  └── ultimate_emotion_detection_best_model.h5

logs/
  ├── ultimate_emotion_detection_training.csv
  └── ultimate_emotion_detection/  (TensorBoard logs)

notebook_execution.log  (Execution log)
```

---

## Monitoring

Watch progress in real-time:
```bash
tail -f notebook_execution.log
```

Check GPU usage:
```bash
watch -n 1 nvidia-smi -i 3
```

---

## Configuration

The notebook reads configuration from **Section 2**.

Default settings:
- Model: LSTM
- Embedding: GloVe 50d or Word2Vec
- Units: 128
- Dropout: 0.2
- Batch size: 32
- Epochs: 50 (with early stopping)

To change settings:
1. Edit `notebooks/complete_pipeline.ipynb` Section 2
2. Run `./run_notebook_gpu3.sh` again

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `nbconvert not found` | `pip install --user nbconvert jupyter` |
| `GPU not detected` | Check: `nvidia-smi -i 3` |
| `GloVe file not found` | Run: `./download_glove.sh` |
| `Out of memory` | Edit notebook: reduce `batch_size` or `rnn_units` |
| Takes too long | Edit notebook: reduce `epochs` |

---

## Full Documentation

See [GPU3_EXECUTION_GUIDE.md](GPU3_EXECUTION_GUIDE.md) for:
- Detailed instructions
- Customization options
- Multiple experiments
- Complete troubleshooting

---

## Summary

**Your complete pipeline notebook is ready to run on GPU 3!**

**Command:**
```bash
cd /home/lab/rabanof/projects/Emotion_Detection_DL
./run_notebook_gpu3.sh
```

**Output:**
```
notebooks/complete_pipeline_executed.ipynb
results/ultimate_emotion_detection_metrics.json
saved_models/ultimate_emotion_detection_best_model.h5
```

**Expected:** 85-90% accuracy in 20-40 minutes

**Ready to execute!**
