# Ablation Study - Hyperparameter Analysis

## Overview
This ablation study tests each hyperparameter **independently** to measure its individual impact on model performance. After identifying the best value for each parameter, it runs a final experiment combining all optimal values.

## Experimental Design

### Baseline Configuration
- **Model**: LSTM
- **RNN Units**: 128
- **Dropout**: 0.2
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Epochs**: 50
- **Embedding**: GloVe 100d (fixed for all experiments)
- **Optimizer**: Adam (fixed)

### Total Experiments: 22

#### Study 1: Model Architecture (3 experiments)
Tests different RNN architectures while keeping all other parameters at baseline:
- LSTM (baseline)
- GRU
- BiLSTM

**Insight**: Which architecture works best for emotion detection?

#### Study 2: RNN Units (3 experiments)
Tests different hidden layer sizes while keeping all other parameters at baseline:
- 64 units
- 128 units (baseline)
- 256 units

**Insight**: Does model capacity (size) affect performance?

#### Study 3: Dropout Rate (4 experiments)
Tests different regularization levels while keeping all other parameters at baseline:
- 0.0 (no dropout)
- 0.2 (baseline)
- 0.4
- 0.8 (heavy dropout)

**Insight**: How much regularization is needed to prevent overfitting?

#### Study 4: Batch Size (4 experiments)
Tests different batch sizes while keeping all other parameters at baseline:
- 16 (small batches)
- 32 (baseline)
- 64
- 128 (large batches)

**Insight**: Does batch size affect convergence and final accuracy?

#### Study 5: Learning Rate (3 experiments)
Tests different optimization speeds while keeping all other parameters at baseline:
- 0.01 (fast learning)
- 0.001 (baseline)
- 0.0001 (slow learning)

**Insight**: What's the optimal learning rate for this task?

#### Study 6: Training Epochs (2 experiments)
Tests different training durations while keeping all other parameters at baseline:
- 25 epochs
- 50 epochs (baseline)

**Insight**: Is 50 epochs necessary or is 25 sufficient?

#### Study 7: Optimal Combination (1 experiment)
**After all individual studies complete**, this study automatically:
1. Identifies the best-performing value from each parameter study
2. Combines all optimal values into one configuration
3. Runs a final experiment with this optimal combination

**Example**: If the individual studies find:
- Best model: BiLSTM
- Best units: 256
- Best dropout: 0.2
- Best batch size: 32
- Best learning rate: 0.001
- Best epochs: 50

Then Study 7 will test: **BiLSTM + 256 units + dropout 0.2 + batch 32 + lr 0.001 + 50 epochs**

**Insight**: Does combining all optimal values give the best overall performance?

## How to Run

```bash
cd final_pipeline
python ablation_study.py
```

## Output Files

### Results
- `results/ablation_study_all_results.json` - Complete results for all 22 experiments
- `results/ablation_study_summary.csv` - Easy-to-read CSV table sorted by accuracy
- `results/ablation_study_by_parameter.json` - Results grouped by parameter study
- `results/ablation_study_final_summary.json` - Final summary with best model
- `ablation_study.log` - Complete execution log

### Logs
- Individual training logs in `logs/` directory
- Saved models in `saved_models/` directory

## Expected Runtime
- Each experiment: ~15-30 minutes (depending on epochs and batch size)
- Total time: ~6-11 hours for all 22 experiments

## Example Output

```
STUDY: MODEL ARCHITECTURE
  model_type = bilstm  →  Accuracy: 0.8684 ( 86.84%)  F1: 0.8428
  model_type =   lstm  →  Accuracy: 0.8333 ( 83.33%)  F1: 0.8201
  model_type =    gru  →  Accuracy: 0.1066 ( 10.66%)  F1: 0.0330

  BEST:  model_type = bilstm (86.84%)
  WORST: model_type = gru (10.66%)
  IMPACT: 76.18% difference

---

STUDY 7: OPTIMAL COMBINATION
Optimal parameters identified:
  model_type: bilstm
  rnn_units: 256
  dropout: 0.2
  batch_size: 32
  learning_rate: 0.001
  epochs: 50

Running optimal combination experiment...

OPTIMAL COMBINATION RESULTS
Validation Accuracy: 0.8892 (88.92%)
Macro F1: 0.8654

Improvement over baseline: +5.59%
```

## Key Questions Answered

1. **Which hyperparameters have the biggest impact?**
   - Compare the "IMPACT" percentage for each study

2. **What are the optimal values?**
   - Check the "BEST" value in each study

3. **Does combining all optimal values work best?**
   - Compare Study 7 (optimal combination) with individual study results

4. **Which parameters don't matter much?**
   - Look for studies with low IMPACT percentages

## Advantages Over Grid Search

- **More efficient**: 22 experiments vs 1,296 for full grid search
- **Interpretable**: Clear understanding of each parameter's individual effect
- **Scientific**: Tests one variable at a time (ablation methodology)
- **Practical**: Identifies what matters most for this specific task
