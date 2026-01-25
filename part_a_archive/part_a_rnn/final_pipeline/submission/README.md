# Emotion Detection - Deep Learning Project

**Authors**: Ofek Raban & Ron Gabay

---

##  Submission Structure

```
submission/
├── complete_pipeline_gpu.py       # Full training pipeline with EDA
├── test_predictions.py           # Test evaluation script (run this!)
├── tokenizer.pkl                 # Pre-fitted tokenizer
├── ablation_study/
│   ├── ablation_study_gru.py    # GRU hyperparameter ablation (19 experiments)
│   └── ablation_study_lstm.py   # LSTM hyperparameter ablation (16 experiments)
├── checkpoints/
│   ├── best_gru_model.h5        # Best GRU model (92.04% accuracy)
│   └── best_lstm_model.h5       # Best LSTM model (92.59% accuracy) ⭐
└── README.md                     # This file
```

---

##  Quick Start - Run Predictions on Test CSV

### Step 1: Prepare Your Test CSV

Create a CSV file with a `text` column:

```csv
text
"I am so happy today!"
"This is terrible and makes me sad"
"I love you so much"
```

Optional - add `label` column for automatic evaluation:
```csv
text,label
"I am so happy today!",joy
"This is terrible and makes me sad",sadness
"I love you so much",love
```

**Emotion Labels**: sadness, joy, love, anger, fear, surprise

### Step 2: Run the Prediction Script

```bash
python test_predictions.py --test_file your_test.csv
```

That's it! The script will:
-  Load both best models (GRU & LSTM)
-  Preprocess your text (same as training)
-  Generate predictions
-  Save results to `predictions.csv` and `prediction_summary.json`
-  Print sample predictions and metrics

---

##  Best Models Performance

### Best LSTM Model (Rank 1)
- **File**: `checkpoints/best_lstm_model.h5`
- **Configuration**: LSTM with 128 units
- **Validation Accuracy**: **92.59%**
- **Macro F1 Score**: 0.906
- **Learning Rate**: 0.001 (baseline)

### Best GRU Model (Rank 2)
- **File**: `checkpoints/best_gru_model.h5`
- **Configuration**: GRU with 128 units
- **Validation Accuracy**: **92.04%**
- **Macro F1 Score**: 0.902
- **Learning Rate**: 0.01 (10x higher!)

**Note**: Both models achieve excellent performance (~92%). LSTM is 0.55% better.

---

##  File Descriptions

### 1. **test_predictions.py**  MAIN EVALUATION SCRIPT

Predicts emotions on your test CSV using both best models.

**Usage**:
```bash
python test_predictions.py --test_file test.csv
```

**What it does**:
- Loads both GRU and LSTM models
- Applies EXACT same preprocessing as training:
  - Contraction expansion ("don't" → "do not")
  - Elongation normalization ("sooooo" → "so")
  - Remove URLs, emails, mentions, hashtags
  - Lowercase + remove punctuation
- Generates predictions with both models
- Compares model agreement
- Saves detailed results

**Outputs**:
- `predictions.csv` - Detailed predictions for each sample
- `prediction_summary.json` - Overall statistics

### 2. **complete_pipeline_gpu.py**

Complete training pipeline (2198 lines) including:
- Full Exploratory Data Analysis (EDA)
- Word clouds and visualizations
- Text preprocessing
- GloVe embedding loading
- Model building (LSTM/GRU/BiLSTM)
- Training with early stopping
- Evaluation and metrics

**Usage**:
```bash
python complete_pipeline_gpu.py
```

### 3. **ablation_study/** (Folder)

Contains ablation study scripts that systematically test hyperparameters.

**ablation_study_gru.py**:
- Tests 5 hyperparameters on GRU baseline
- 19 experiments total
- Run: `nohup python ablation_study/ablation_study_gru.py > gru.log 2>&1 &`

**ablation_study_lstm.py**:
- Tests 5 hyperparameters on LSTM baseline
- 16 experiments total
- Run: `nohup python ablation_study/ablation_study_lstm.py > lstm.log 2>&1 &`

**Hyperparameters Tested**:
- RNN Units: [64, 128, 256]
- Dropout: [0.0, 0.2, 0.4]
- Batch Size: [16, 32, 64, 128]
- Learning Rate: [0.01, 0.001, 0.0001]
- Epochs: [25, 50]

### 4. **checkpoints/** (Folder)

Contains the pre-trained best models.

**best_lstm_model.h5** (92.59% accuracy) 
**best_gru_model.h5** (92.04% accuracy)

### 5. **tokenizer.pkl**  REQUIRED

Pre-fitted Keras tokenizer with 15,156 vocabulary words.

**Why we need it**:
- Converts text to sequences of integers
- Maps words to indices (e.g., "happy" → 42)
- Must use the SAME tokenizer as training
- Ensures consistency: training and test use same word mappings

**Without it**: The model won't understand your text!

---

##  Key Findings from Ablation Study

### 1. Learning Rate is Most Critical 

- **86.93% performance difference** between LR=0.01 and LR=0.0001 (for GRU)
- Different architectures need different optimal learning rates:
  - **GRU optimal**: 0.01 (10x higher than baseline)
  - **LSTM optimal**: 0.001 (baseline)
- This is why LSTM outperforms GRU overall!

### 2. LSTM Slightly Better Than GRU

- **LSTM**: 92.59% (best with 128 units)
- **GRU**: 92.04% (best with LR=0.01)
- Difference: Only 0.55%
- Both are excellent choices

### 3. RNN Units Matter

LSTM results:
- 64 units: 38.54% (too small, underfitting)
- **128 units**: **92.59%**  OPTIMAL
- 256 units: 91.09% (slight overfitting)

### 4. Other Hyperparameters Have Secondary Impact

- Dropout: ~3-5% variance
- Batch Size: ~1-2% variance
- Epochs: ~1% variance (early stopping handles this)

---

##  Requirements

```bash
pip install tensorflow pandas numpy scikit-learn
```

**Versions**:
- Python 3.8+
- TensorFlow >= 2.8.0
- Pandas >= 1.3.0
- NumPy >= 1.21.0
- Scikit-learn >= 1.0.0

---

##  Output Format

### predictions.csv

Contains predictions from both models for each sample:

```csv
text,gru_prediction,gru_confidence,lstm_prediction,lstm_confidence,models_agree,...
"I am happy",joy,0.95,joy,0.92,True,...
```

Columns:
- `text` - Original text
- `gru_prediction` - GRU predicted emotion
- `gru_confidence` - GRU confidence score
- `lstm_prediction` - LSTM predicted emotion
- `lstm_confidence` - LSTM confidence score
- `models_agree` - Whether both models agree
- `gru_prob_[emotion]` - GRU probability for each emotion
- `lstm_prob_[emotion]` - LSTM probability for each emotion
- `true_label` - True label (if provided)
- `gru_correct` - Whether GRU was correct (if labels provided)
- `lstm_correct` - Whether LSTM was correct (if labels provided)

### prediction_summary.json

Overall statistics:

```json
{
  "total_samples": 1000,
  "gru_accuracy": 0.9204,
  "lstm_accuracy": 0.9259,
  "model_agreement_rate": 0.8956,
  ...
}
```

---

##  Technical Details

### Preprocessing Pipeline

The preprocessing in `test_predictions.py` is IDENTICAL to training:

1. **Contraction Expansion**: "don't" → "do not"
2. **Elongation Normalization**: "sooooo" → "so"
3. **Remove URLs**: http://example.com → ""
4. **Remove Emails**: user@domain.com → ""
5. **Remove Mentions/Hashtags**: @user #topic → ""
6. **Lowercase**: "HELLO" → "hello"
7. **Remove Numbers**: "123" → ""
8. **Remove Punctuation**: "hello!" → "hello"
9. **Normalize Whitespace**: "a    b" → "a b"

### Model Architecture

- **Embedding**: GloVe 100d (frozen, pre-trained)
- **RNN Layer**: LSTM or GRU (128 units)
- **Regularization**: Dropout + Spatial Dropout + Recurrent Dropout
- **Output**: Dense(6, softmax) for 6 emotion classes

### Training Strategy

- Optimizer: Adam
- Loss: Categorical Crossentropy
- Class Weights: Balanced (handles class imbalance)
- Early Stopping: Patience=15 epochs
- Model Checkpointing: Save best validation accuracy

### Emotion Classes (6 total)

0. Sadness - Expressions of sorrow, grief, disappointment
1. Joy - Expressions of happiness, excitement, pleasure
2. Love - Expressions of affection, care, fondness
3. Anger - Expressions of rage, frustration, irritation
4. Fear - Expressions of anxiety, worry, terror
5. Surprise - Expressions of shock, amazement, wonder

---

## 🔍 Why We Need tokenizer.pkl

**The tokenizer is CRITICAL for making predictions!**

### What the tokenizer does:

1. **Word-to-Index Mapping**:
   - "happy" → 42
   - "sad" → 156
   - "love" → 89

2. **Vocabulary**: Contains 15,156 words learned from training data

3. **Consistency**: Ensures test data uses SAME mappings as training

### Example:

**Training**: "I love you" → [10, 89, 45]
**Test** (with same tokenizer): "I love you" → [10, 89, 45] 
**Test** (without tokenizer): "I love you" → ???  FAILS!

**Without tokenizer.pkl**: The model can't understand your text because it doesn't know which word maps to which number!

---

##  Project Summary

This project demonstrates:

 Complete deep learning pipeline for text classification
 Systematic hyperparameter ablation study (35 experiments)
 Best model: LSTM achieving **92.59% accuracy** on 6-class emotion detection
 Comprehensive preprocessing pipeline
 Production-ready inference code
 Comparison of LSTM vs GRU architectures

**Best Result**: LSTM with 128 units
**Validation Accuracy**: 92.59%
**Macro F1 Score**: 0.906

---

##  For Questions

See code comments for detailed documentation. All code is well-commented at master's level.

---

**Submission Date**: December 2025
**Course**: Deep Learning for NLP
