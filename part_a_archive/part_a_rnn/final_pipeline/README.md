# Final Emotion Detection Pipeline

This folder contains the final, updated version of the emotion detection pipeline.

## Files Overview

### 1. Pipeline Implementation
- **`complete_pipeline_gpu.py`** - Complete modular pipeline with all functions and classes
- **`complete_pipeline.ipynb`** - Jupyter notebook version of the pipeline

### 2. Experiment Runners
- **`run_gpu_experiments.py`** - Main script that ran the 7 baseline experiments
  - Tests: LSTM, GRU, BiLSTM with GloVe/Word2Vec embeddings
  - Fixed with patience=15 for proper training

- **`expirements_runner.py`** - Flexible experiment runner for hyperparameter tuning
  - Uses `complete_pipeline_gpu.py` as a module
  - Easy to define custom experiments

### 3. Configuration & Dependencies
- **`requirements.txt`** - Python package dependencies
- **`configs/`** - Configuration files (if any)

### 4. Results
- **`results/`** - All experiment results
  - `all_experiments_comparison.csv` - Summary of all 7 experiments
  - Individual JSON files for each experiment

## How to Run

### Run the 7 baseline experiments:
```bash
python run_gpu_experiments.py
```

### Run custom hyperparameter experiments:
```bash
python expirements_runner.py
```

## Current Best Results

1. **BiLSTM + GloVe50**: 86.84% accuracy (BEST)
2. **LSTM + GloVe50**: 83.33% accuracy
3. **LSTM + GloVe50 (dropout 0.5)**: 79.43% accuracy

## Key Findings
- GloVe embeddings significantly outperform Word2Vec
- BiLSTM architecture is the best performer
- Patience=15 is required for proper convergence
