# GPU Execution Checklist

## Before Running

- [ ] SSH connected to Cortex cluster
- [ ] In project directory: `/home/lab/rabanof/projects/Emotion_Detection_DL`
- [ ] GPU available (check: `nvidia-smi`)
- [ ] Data files present:
  - [ ] `data/raw/train.csv`
  - [ ] `data/raw/validation.csv`
  - [ ] `glove/glove.6B.100d.txt`

## Execution

- [ ] Run: `./run_on_gpu.sh` OR `python3 run_gpu_experiments.py`
- [ ] Wait for completion (~15-30 minutes)

## Verify Results

After execution, check:

- [ ] **Comparison table exists**: `results/all_experiments_comparison.csv`
- [ ] **8 result files exist**: `results/*_results.json`
- [ ] **8 log files exist**: `logs/*_training.csv`
- [ ] **8 model files exist**: `saved_models/*_best.h5`
- [ ] **All models achieved >75% accuracy** (check comparison table)

## What to Check in Results

### 1. Comparison Table
```bash
cat results/all_experiments_comparison.csv
```

Verify columns:
- experiment_name
- model_type (LSTM, GRU, BiLSTM)
- embedding_type (glove, word2vec)
- val_accuracy (should be >0.75)
- macro_f1
- training_time

### 2. Best Model Metrics
```bash
tail -20 gpu_training.log
```

Should show:
- Best model name
- Accuracy >75%
- Macro F1 score

### 3. Individual Results
```bash
# Pick any experiment:
cat results/lstm_glove_baseline_results.json
```

Should contain:
- val_accuracy
- macro_f1
- per_class_f1 (for all 6 emotions)
- training_time

## Assignment Requirements Verification

- [ ] **LSTM model trained**: Check `lstm_glove_baseline_results.json`
- [ ] **GRU model trained**: Check `gru_glove_baseline_results.json`
- [ ] **GloVe embeddings used**: Multiple experiments with `embedding_type: glove`
- [ ] **Word2Vec embeddings used**: Multiple experiments with `embedding_type: word2vec`
- [ ] **Hyperparameter variations tested**:
  - [ ] Different units (128 vs 256)
  - [ ] Different layers (1 vs 2)
  - [ ] Different dropout (0.2 vs 0.5)
- [ ] **Performance target met**: All models >75% accuracy
- [ ] **Model comparison available**: `all_experiments_comparison.csv`
- [ ] **GPU used**: Check log shows "GPUs Available: 1"

## Results Summary Template

After execution, you can report:

```
LSTM vs GRU Comparison:
- LSTM + GloVe: XX.X% accuracy
- GRU + GloVe: XX.X% accuracy
- Winner: [LSTM/GRU]

Embedding Comparison:
- LSTM + GloVe: XX.X% accuracy
- LSTM + Word2Vec: XX.X% accuracy
- Winner: [GloVe/Word2Vec]

Hyperparameter Effects:
- LSTM 128 units: XX.X%
- LSTM 256 units: XX.X%
- LSTM 1 layer: XX.X%
- LSTM 2 layers: XX.X%
- LSTM dropout 0.2: XX.X%
- LSTM dropout 0.5: XX.X%

Best Configuration:
- Model: [LSTM/GRU/BiLSTM]
- Embedding: [GloVe/Word2Vec]
- Accuracy: XX.X%
- F1-Score: X.XX
```

## Troubleshooting

If something fails:

1. **Check GPU**: `nvidia-smi`
2. **Check paths**: Verify data files exist
3. **Check logs**: `tail -100 gpu_training.log`
4. **Re-run single experiment**: Edit `run_gpu_experiments.py` to run only one

## Files to Submit/Present

1. `results/all_experiments_comparison.csv` - Main comparison
2. `results/*_results.json` - Detailed metrics
3. `gpu_training.log` - Training log
4. Any plots/visualizations you create from results

## Success Criteria

All checked means you're done:
- [ ] 8 experiments completed successfully
- [ ] All accuracy >75%
- [ ] Comparison table shows clear performance differences
- [ ] Results demonstrate hyperparameter effects
- [ ] GPU was used (verified in logs)
- [ ] All assignment requirements met
