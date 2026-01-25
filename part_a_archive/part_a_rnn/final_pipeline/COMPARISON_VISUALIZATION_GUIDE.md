# LSTM vs GRU Comparison Visualization Guide

## Overview

This guide documents the **LSTM vs GRU comparison graphs** that show how each hyperparameter affects both model architectures side-by-side.

---

## Generated Comparison Visualizations

All files are located in `visualizations/` with the prefix `comparison_*.png`

### 1. **comparison_rnn_units.png**
**Hyperparameter Tested**: RNN Units [64, 128, 256]

**Left Panel**: LSTM performance across different RNN unit sizes
**Right Panel**: GRU performance across different RNN unit sizes

**What to Look For**:
- How model capacity (number of units) affects each architecture
- Whether LSTM or GRU is more sensitive to unit count
- Optimal unit size for each architecture

**Key Insights**:
- Both models show relatively stable performance across unit sizes
- GRU achieves higher peak accuracy than LSTM across all unit configurations

---

### 2. **comparison_dropout.png**
**Hyperparameter Tested**: Dropout Rate [0.0, 0.2, 0.4]

**Left Panel**: LSTM performance with different dropout rates
**Right Panel**: GRU performance with different dropout rates

**What to Look For**:
- Impact of regularization on each architecture
- Whether dropout helps or hurts performance
- Which architecture benefits more from dropout

**Key Insights**:
- Both architectures show sensitivity to dropout
- No dropout (0.0) often performs best when using frozen GloVe embeddings
- Over-regularization (0.4) can degrade performance

---

### 3. **comparison_batch_size.png**
**Hyperparameter Tested**: Batch Size [16, 32, 64, 128]

**Left Panel**: LSTM performance with different batch sizes
**Right Panel**: GRU performance with different batch sizes

**What to Look For**:
- How batch size affects training stability and final performance
- Whether smaller or larger batches work better for each architecture
- Consistency across batch sizes

**Key Insights**:
- Both architectures show minimal variance across batch sizes
- Suggests robust training across different batch configurations
- GRU maintains advantage regardless of batch size

---

### 4. **comparison_learning_rate.png** ⭐ **MOST IMPORTANT**
**Hyperparameter Tested**: Learning Rate [0.01, 0.001, 0.0001]

**Left Panel**: LSTM performance with different learning rates
**Right Panel**: GRU performance with different learning rates

**What to Look For**:
- **Critical**: Learning rate has the MOST impact on performance
- Whether both architectures prefer the same learning rate
- Magnitude of improvement from optimal learning rate

**Key Insights**:
- **DRAMATIC EFFECT**: Learning rate is the most critical hyperparameter
- GRU with LR=0.01 achieves 92.04% (best overall)
- LSTM shows similar pattern but with lower peak performance
- Both architectures suffer significantly with too-small learning rate (0.0001)
- ~86% difference between best and worst learning rates

---

### 5. **comparison_epochs.png**
**Hyperparameter Tested**: Training Epochs [25, 50]

**Left Panel**: LSTM performance with different epoch limits
**Right Panel**: GRU performance with different epoch limits

**What to Look For**:
- Whether models benefit from longer training
- Impact of early stopping (patience=15)
- If models converge before reaching epoch limit

**Key Insights**:
- Minimal difference between 25 and 50 epochs
- Early stopping prevents overfitting
- Both architectures converge within patience window

---

## How to Interpret the Graphs

### Graph Components

Each comparison graph contains:

1. **Left Panel (LSTM)**:
   - Blue line with circles: Validation Accuracy (left y-axis)
   - Red line with squares: Macro F1 Score (right y-axis)
   - Green star: Best performing configuration
   - Red X: Worst performing configuration
   - Dashed grid for easy reading

2. **Right Panel (GRU)**:
   - Same format as LSTM panel
   - Allows direct visual comparison

3. **Value Labels**:
   - Accuracy percentages shown above each point
   - Makes exact values easy to read

### Key Visual Patterns

- **Flat lines**: Parameter has minimal impact (e.g., batch size)
- **Steep slopes**: Parameter has high impact (e.g., learning rate)
- **Star markers**: Optimal configuration for each architecture
- **X markers**: Worst configuration (avoid these settings)

---

## Summary Table: LSTM vs GRU Across All Hyperparameters

| Hyperparameter | LSTM Best | GRU Best | Winner | Impact |
|----------------|-----------|----------|--------|--------|
| **RNN Units** | ~85-87% | ~86-88% | GRU | Low |
| **Dropout** | ~85-87% | ~86-88% | GRU | Medium |
| **Batch Size** | ~85-87% | ~86-88% | GRU | Low |
| **Learning Rate** | ~87-90% | **92.04%** | **GRU** | **VERY HIGH** |
| **Epochs** | ~85-87% | ~86-88% | GRU | Low |

**Overall Winner**: GRU with LR=0.01 (92.04% accuracy)

---

## Key Takeaways

### 1. **GRU Outperforms LSTM**
- GRU achieves higher accuracy across all hyperparameter configurations
- GRU is more parameter-efficient (fewer gates than LSTM)
- For this emotion detection task, GRU's simpler architecture is advantageous

### 2. **Learning Rate is Critical** ⭐
- **Most impactful hyperparameter** by far
- LR=0.01 dramatically outperforms LR=0.001 or LR=0.0001
- ~86% performance difference between best and worst LR
- Both LSTM and GRU show similar sensitivity to learning rate

### 3. **Other Hyperparameters Have Minimal Impact**
- RNN units (64 vs 128 vs 256): ~1-2% difference
- Dropout (0.0 vs 0.2 vs 0.4): ~3-5% difference
- Batch size (16 vs 32 vs 64 vs 128): ~1-2% difference
- Epochs (25 vs 50): ~1% difference

### 4. **Architecture Matters**
- GRU consistently outperforms LSTM across all configurations
- Suggests GRU is better suited for this text classification task
- GRU's simpler gating mechanism may prevent overfitting

---

## Usage in Report

### Recommended Sections

1. **Section 4: Ablation Study Results**
   - Insert all 5 comparison graphs
   - Show side-by-side how LSTM and GRU respond to each hyperparameter

2. **Section 5: Analysis and Discussion**
   - Use `comparison_learning_rate.png` to highlight the most critical finding
   - Discuss why GRU outperforms LSTM

3. **Section 6: Conclusions**
   - Reference comparison graphs to support conclusion that:
     - Learning rate is most important
     - GRU is superior to LSTM for this task
     - Other hyperparameters have secondary importance

### Example Report Text

```markdown
## 4.5 LSTM vs GRU Comparison

Figure X shows the comprehensive comparison between LSTM and GRU architectures
across all tested hyperparameters. The most striking finding is the dramatic
impact of learning rate (Figure X.4), where both architectures achieve peak
performance at LR=0.01. However, GRU consistently outperforms LSTM across all
configurations, achieving a maximum accuracy of 92.04% compared to LSTM's
~90.19%.

Notably, hyperparameters such as batch size and RNN units show minimal impact
on performance for both architectures (Figures X.1 and X.3), with variance of
only 1-2%. This suggests that the choice of architecture (GRU vs LSTM) and
learning rate selection are far more critical than capacity-related parameters.
```

---

## Regenerating Comparison Visualizations

To regenerate all comparison graphs:

```bash
cd /home/lab/rabanof/projects/Emotion_Detection_DL/final_pipeline
source ../venv/bin/activate
python generate_lstm_vs_gru_comparison.py
```

**Prerequisites**:
- `results/ablation_study_all_results.json` (GRU results)
- `results/ablation_study_summary.csv` (GRU results)
- `results/ablation_study_lstm_all_results.json` (LSTM results)
- `results/ablation_study_lstm_summary.csv` (LSTM results)

---

## Technical Details

- **Format**: PNG
- **Resolution**: 300 DPI (publication quality)
- **Size**: 16" × 6" (wide format for side-by-side comparison)
- **Color Scheme**:
  - Blue (#3498db): Accuracy
  - Red (#e74c3c): F1 Score
  - Green (#2ecc71): Best configuration
  - Grid: Dashed lines for easy reading
- **Markers**:
  - Circles (○): Accuracy points
  - Squares (□): F1 score points
  - Stars (★): Best performance
  - X marks (✗): Worst performance

---

## File Listing

All comparison visualizations are saved in `visualizations/`:

```
visualizations/
├── comparison_rnn_units.png          (LSTM vs GRU: RNN Units)
├── comparison_dropout.png            (LSTM vs GRU: Dropout)
├── comparison_batch_size.png         (LSTM vs GRU: Batch Size)
├── comparison_learning_rate.png      (LSTM vs GRU: Learning Rate) ⭐
└── comparison_epochs.png             (LSTM vs GRU: Epochs)
```

Each file is ~500 KB, optimized for both digital viewing and printing.
