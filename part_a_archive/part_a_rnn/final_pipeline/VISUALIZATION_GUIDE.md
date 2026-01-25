# Visualization Guide for Project Report

All visualizations for the project report have been generated and are located in the `visualizations/` directory.

## Generated Visualizations

### 1. Parameter Studies (Individual)

#### Study 1: Model Architecture
**File:** `study1_model_architecture.png`
- Compares LSTM vs GRU vs BiLSTM
- Shows both accuracy and macro F1 scores
- **Key Finding:** BiLSTM achieved 91.14% (best), LSTM 90.19%, GRU 26.33%

#### Study 2: RNN Units
**File:** `study2_rnn_units.png`
- Tests 64, 128, 256 units
- **Key Finding:** Minimal impact due to suboptimal baseline learning rate

#### Study 3: Dropout Rate
**File:** `study3_dropout.png`
- Tests 0.0, 0.2, 0.4 dropout
- **Key Finding:** No dropout (0.0) performed best at 86.29%

#### Study 4: Batch Size
**File:** `study4_batch_size.png`
- Tests 16, 32, 64, 128 batch sizes
- **Key Finding:** Minimal variance (~1.26% difference)

#### Study 5: Learning Rate ⭐
**File:** `study5_learning_rate.png`
- Tests 0.01, 0.001, 0.0001
- **KEY FINDING:** LR=0.01 achieved 92.04% - **MOST CRITICAL PARAMETER**
- 86.93% difference between best and worst

#### Study 6: Training Epochs
**File:** `study6_epochs.png`
- Tests 25 vs 50 epochs
- Both stopped early due to suboptimal baseline

---

### 2. Summary Visualizations

#### Parameter Impact Summary
**File:** `parameter_impact_summary.png`
- **Left panel:** Impact ranking (difference between best/worst in each study)
- **Right panel:** Best accuracy achieved in each study
- **Shows:** Learning Rate has highest impact (86.93% difference)

#### Confusion Matrix - Best Model
**File:** `confusion_matrix_best_model.png`
- **Left panel:** Raw confusion matrix (counts)
- **Right panel:** Normalized confusion matrix (percentages)
- **Model:** GRU with LR=0.01 (92.04% accuracy)
- **Shows:** Strong diagonal, minimal confusion between classes

#### Per-Class F1 Scores
**File:** `per_class_f1_scores.png`
- Shows F1 score for each of the 6 emotions
- **Best Model:** GRU with LR=0.01
- **Scores:**
  - Sadness: 95.3%
  - Joy: 93.1%
  - Anger: 92.3%
  - Love: 88.0%
  - Fear: 86.4%
  - Surprise: 86.2%
- Red dashed line shows macro F1 (90.2%)

#### Top Models Comparison
**File:** `top_models_comparison.png`
- Compares top 5 model configurations
- Shows both accuracy (blue) and macro F1 (green) side by side
- **Rankings:**
  1. GRU, LR=0.01: 92.04%
  2. BiLSTM (baseline): 91.14%
  3. LSTM (baseline): 90.19%
  4. GRU, dropout=0.0: 86.29%
  5. (next best)

#### Results Table
**File:** `results_table.png`
- Professional table showing top 10 configurations
- Columns: Experiment name, Accuracy, Macro F1, Epochs, Training Time
- Formatted for inclusion in presentations/reports

---

## How to Use These Visualizations

### For the Written Report
Add these images to the report sections:

1. **Section 4.2 (Study Results):** Insert `study1_*.png` through `study6_*.png` after each corresponding study description

2. **Section 5 (Final Results):** Insert:
   - `top_models_comparison.png`
   - `results_table.png`

3. **Section 6 (Error Analysis):** Insert:
   - `confusion_matrix_best_model.png`
   - `per_class_f1_scores.png`

4. **Appendix:** Insert:
   - `parameter_impact_summary.png`
   - All individual study plots

### For Presentations
- Use `parameter_impact_summary.png` to show which parameters matter most
- Use `study5_learning_rate.png` to highlight the key discovery
- Use `confusion_matrix_best_model.png` for error analysis
- Use `top_models_comparison.png` for final results

---

## Quick Reference: Where Each Image Goes in Report

```
PROJECT_REPORT.md Structure:

Section 4.2: Results by Parameter
├── Study 1 → Insert: study1_model_architecture.png
├── Study 2 → Insert: study2_rnn_units.png
├── Study 3 → Insert: study3_dropout.png
├── Study 4 → Insert: study4_batch_size.png
├── Study 5 → Insert: study5_learning_rate.png ⭐
└── Study 6 → Insert: study6_epochs.png

Section 5: Final Results
├── 5.2 Top Models → Insert: top_models_comparison.png
├── 5.2 Top Models → Insert: results_table.png
└── 5.3 Key Insights → Insert: parameter_impact_summary.png

Section 6: Error Analysis
├── 6.1 Confusion Matrix → Insert: confusion_matrix_best_model.png
└── 6.1 Per-Class → Insert: per_class_f1_scores.png

Appendix
└── All studies → Insert all study*.png files
```

---

## Image Specifications

All images are:
- **Format:** PNG
- **Resolution:** 300 DPI (high quality for printing)
- **Size:** Optimized for reports and presentations
- **Color Scheme:** Professional, color-blind friendly palette

## Regenerating Visualizations

To regenerate all visualizations:

```bash
cd /home/lab/rabanof/projects/Emotion_Detection_DL/final_pipeline
source ../venv/bin/activate
python generate_visualizations.py
```

This will recreate all 11 visualization files in the `visualizations/` directory.
