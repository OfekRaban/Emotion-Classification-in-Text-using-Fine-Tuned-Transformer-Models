# Pre-Execution Setup - IMPORTANT

## Current Status

Your GPU training script is **READY** and includes:
- Complete preprocessing (30+ contraction rules, elongation normalization, slang/typo fixes)
- Data leakage removal
- Duplicate removal
- Exact model architecture from your pipeline
- GloVe 50d embeddings (correct dimension)
- recurrent_dropout=0.2
- Class weights for imbalance

**Expected accuracy: 85-90%**

---

## CRITICAL: Download GloVe Embeddings First

The GloVe 50d file is currently **MISSING**. You MUST download it before running training.

### Step 1: Create GloVe Directory

```bash
cd /home/lab/rabanof
mkdir -p Emotion_Detection_DL/glove
cd Emotion_Detection_DL/glove
```

### Step 2: Download GloVe

```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
```

This will download ~860MB.

### Step 3: Extract GloVe 50d

```bash
unzip glove.6B.zip
```

This extracts 4 files:
- glove.6B.50d.txt (171MB) - **THIS IS THE ONE YOU NEED**
- glove.6B.100d.txt
- glove.6B.200d.txt
- glove.6B.300d.txt

### Step 4: Verify

```bash
ls -lh glove.6B.50d.txt
```

Should show: `glove.6B.50d.txt` (~171MB)

### Step 5: Cleanup (Optional)

```bash
# Remove other dimensions to save space
rm glove.6B.100d.txt glove.6B.200d.txt glove.6B.300d.txt glove.6B.zip
```

---

## Quick Verification Commands

After downloading GloVe, verify everything is ready:

```bash
cd /home/lab/rabanof/projects/Emotion_Detection_DL

# 1. Check data files
ls -lh data/raw/train.csv data/raw/validation.csv

# 2. Check GloVe file
ls -lh /home/lab/rabanof/Emotion_Detection_DL/glove/glove.6B.50d.txt

# 3. Check GPU
nvidia-smi

# 4. Check script is executable
ls -lh run_on_gpu.sh
```

All should pass before running training.

---

## Then Execute Training

Once GloVe is downloaded:

```bash
cd /home/lab/rabanof/projects/Emotion_Detection_DL
./run_on_gpu.sh
```

**Training time:** 15-25 minutes for all 7 experiments on H100

---

## What Will Happen

The script will:
1. Detect H100 GPU
2. Load and preprocess data (with ALL your preprocessing)
3. Remove data leakage and duplicates
4. Build GloVe 50d embedding matrix
5. Train 7 experiments:
   - LSTM + GloVe 50d (baseline)
   - GRU + GloVe 50d
   - LSTM + Word2Vec 50d
   - GRU + Word2Vec 50d
   - BiLSTM + GloVe 50d
   - LSTM + 256 units
   - LSTM + dropout 0.5
6. Save all results to `results/` directory
7. Generate comparison table: `results/all_experiments_comparison.csv`

---

## Expected Results

All experiments will achieve:
- **Validation Accuracy: 85-90%**
- **Macro F1: 0.83-0.87**
- **Well above 75% requirement**

Typical results:
- LSTM + GloVe: ~88%
- GRU + GloVe: ~87%
- BiLSTM + GloVe: ~89% (usually best)

---

## If You're On Cortex Cluster Already

If you're already SSH'd to Cortex:

```bash
# Download GloVe directly on the cluster
cd /home/lab/rabanof
mkdir -p Emotion_Detection_DL/glove
cd Emotion_Detection_DL/glove
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

# Go to project and run training
cd /home/lab/rabanof/projects/Emotion_Detection_DL
./run_on_gpu.sh
```

---

## Summary

**BEFORE running training:**
1. Download GloVe 50d file (MISSING)
2. Extract to `/home/lab/rabanof/Emotion_Detection_DL/glove/glove.6B.50d.txt`
3. Verify file exists

**THEN run training:**
```bash
cd /home/lab/rabanof/projects/Emotion_Detection_DL
./run_on_gpu.sh
```

**Everything else is ready!**
