# Unified Complete Pipeline Notebook - Guide

## 📓 Overview

**File**: [notebooks/unified_complete_pipeline.ipynb](notebooks/unified_complete_pipeline.ipynb)

This is a **completely self-contained** notebook that includes everything you need for emotion detection in one place. No external imports from `src/` needed!

## 🎯 What's Inside

### All Classes Included (No External Dependencies)
- ✅ `Config` - Complete configuration management
- ✅ `TextPreprocessor` - Advanced text preprocessing
- ✅ `EmbeddingHandler` - GloVe and Word2Vec support
- ✅ `ModelBuilder` - LSTM/GRU/Bidirectional architectures
- ✅ `ResultsVisualizer` - Comprehensive visualizations
- ✅ `ExperimentTracker` - Automatic experiment tracking

### Complete Pipeline (24 Sections)
1. **Imports and Setup** - All libraries and configuration
2. **Configuration** - Editable config class
3. **TextPreprocessor Class** - 30+ preprocessing rules
4. **EmbeddingHandler Class** - GloVe/Word2Vec handling
5. **ModelBuilder Class** - Multiple architectures
6. **Visualization Class** - Professional plots
7. **ExperimentTracker** - Auto-tracking callback
8. **Data Loading** - Load train/val datasets
9. **EDA** - Exploratory data analysis with plots
10. **Preprocessing** - Clean and prepare text
11. **Embedding Creation** - Load embeddings and create matrix
12. **Class Weights** - Handle imbalanced data
13. **Model Creation** - Build neural network
14. **Training** - Train with callbacks
15. **Evaluation** - Calculate metrics
16. **Training History** - Plot accuracy/loss curves
17. **Confusion Matrix** - Raw confusion matrix
18. **Normalized Confusion** - Normalized version
19. **Classification Report** - Per-class metrics
20. **Per-Class Accuracy** - Accuracy by emotion
21. **Prediction Examples** - Sample predictions
22. **Interactive Prediction** - Custom text prediction
23. **Test Predictions** - Try example texts
24. **Summary & Guide** - Results and next steps

## 🚀 Quick Start

### 1. Open the Notebook
```bash
jupyter notebook notebooks/unified_complete_pipeline.ipynb
```

### 2. Run All Cells
- Click: `Cell → Run All`
- Or: Press `Shift + Enter` through each cell

### 3. Watch the Complete Pipeline Execute!
The notebook will:
- Load and preprocess data ✓
- Create embeddings ✓
- Build model ✓
- Train with callbacks ✓
- Evaluate and visualize ✓
- Save results ✓

## ⚙️ Configuration

### Easy Customization (Section 2)

Change these settings in the `Config` class:

```python
config = Config()

# Quick changes:
config.model_type = 'gru'        # 'lstm', 'gru', or 'bilstm'
config.rnn_units = 256           # 64, 128, 256
config.num_layers = 2            # 1, 2, 3
config.embedding_type = 'word2vec'  # 'glove' or 'word2vec'
config.batch_size = 64           # 16, 32, 64
config.epochs = 100              # 10, 50, 100
config.learning_rate = 0.0005    # 0.0001, 0.001, 0.01
```

### Available Model Types

| Model Type | Description | Speed | Accuracy |
|-----------|-------------|-------|----------|
| `lstm` | Standard LSTM | Medium | Good |
| `gru` | GRU (faster) | Fast | Good |
| `bilstm` | Bidirectional LSTM | Slow | Best |

## 🎯 Common Workflows

### Workflow 1: Quick Baseline
```python
# Use defaults (Section 2)
# Just run all cells!
# Expected: 78-82% accuracy in 5-10 minutes
```

### Workflow 2: Try Different Models
```python
# Section 2 - Change config
config.experiment_name = "gru_experiment"
config.model_type = 'gru'

# Then run from Section 13 (Model Creation) onwards
```

### Workflow 3: Larger Model
```python
# Section 2
config.experiment_name = "large_bilstm"
config.model_type = 'bilstm'
config.rnn_units = 256
config.num_layers = 2

# Run from Section 13 onwards
```

### Workflow 4: Word2Vec Instead of GloVe
```python
# Section 2
config.experiment_name = "lstm_word2vec"
config.embedding_type = 'word2vec'

# Run from Section 11 (Embedding Creation) onwards
```

### Workflow 5: Fine-tune Embeddings
```python
# Section 2
config.experiment_name = "lstm_finetuned"
config.trainable_embeddings = True

# Run from Section 13 onwards
```

## 📊 Understanding the Output

### During Training (Section 14)
```
Epoch 1/50
500/500 [==============================] - 45s - loss: 1.2345 - accuracy: 0.5123 - val_loss: 1.1234 - val_accuracy: 0.5456
```
- Watch `val_accuracy` increase
- Watch `val_loss` decrease
- Training stops early if no improvement (patience=5)

### After Training
You'll see:
- ✅ Training history plots (accuracy & loss)
- ✅ Confusion matrices (2 versions)
- ✅ Classification report (precision, recall, F1)
- ✅ Per-class accuracy bars
- ✅ Sample predictions
- ✅ Final summary with metrics

## 🧪 Testing Custom Text

### In Section 22 - Interactive Prediction
```python
# Try your own text!
predict_emotion("I'm so excited about this project!")
predict_emotion("This makes me really angry")
predict_emotion("I'm scared of failing")
```

Output example:
```
Input: I'm so excited about this project!
Cleaned: i am so excited about this project

🎯 Predicted Emotion: Joy (confidence: 87.34%)

📊 All probabilities:
  Sadness     :                                          2.34%
  Joy         : ███████████████████████████████████████ 87.34%
  Love        : ██                                       4.56%
  Anger       :                                          1.23%
  Fear        :                                          3.12%
  Surprise    : █                                        1.41%
```

## 📁 Files Created

After running the notebook, you'll have:

```
saved_models/
  └── emotion_detection_unified_best.keras    # Best model

logs/
  └── emotion_detection_unified_training.csv  # Training log

results/
  ├── emotion_detection_unified_results.json  # Full results
  └── emotion_detection_unified_summary.json  # Quick summary
```

## 🎓 Learning Path

### Beginner
1. Run all cells without changes
2. Understand the output
3. Try different texts in Section 22
4. Read the code comments

### Intermediate
1. Modify config (Section 2)
2. Try different model types
3. Experiment with hyperparameters
4. Compare results

### Advanced
1. Modify the classes (Sections 3-7)
2. Add new preprocessing rules
3. Implement new architectures
4. Create custom visualizations

## 🔧 Troubleshooting

### Problem: Out of Memory
**Solution**: In Section 2:
```python
config.batch_size = 16  # Reduce from 32
config.rnn_units = 64   # Reduce from 128
```

### Problem: GloVe File Not Found
**Solution**: Update path in Section 2:
```python
config.glove_path = "/path/to/your/glove.6B.100d.txt"
```

### Problem: Slow Training
**Solution**: In Section 2:
```python
config.model_type = 'gru'  # Faster than LSTM
config.batch_size = 64     # Larger batches
```

### Problem: Poor Accuracy (<70%)
**Solution**: In Section 2:
```python
config.model_type = 'bilstm'  # Better performance
config.rnn_units = 256         # More capacity
config.epochs = 100            # More training
```

## 💡 Pro Tips

### Tip 1: Compare Multiple Experiments
Run the notebook multiple times with different configs:
```python
# Run 1
config.experiment_name = "exp_lstm_128"
config.model_type = 'lstm'
config.rnn_units = 128

# Run 2 (change name and config)
config.experiment_name = "exp_gru_256"
config.model_type = 'gru'
config.rnn_units = 256

# Compare results in results/ directory
```

### Tip 2: Save Your Best Config
When you find a good configuration:
```python
# Copy the Config class values
# Paste into a text file for reference
```

### Tip 3: Quick Iteration
Don't always run from the beginning:
- **Changed model only?** → Run from Section 13
- **Changed embeddings?** → Run from Section 11
- **Just want new predictions?** → Run Section 22

### Tip 4: Monitor Training
Watch for:
- ✅ `val_accuracy` increasing steadily
- ✅ `val_loss` decreasing
- ❌ `val_loss` increasing while `train_loss` decreases = overfitting

### Tip 5: Use the Summary
Check Section 23 output for quick overview:
```json
{
  "final_val_accuracy": 0.8234,
  "best_val_accuracy": 0.8456,
  "embedding_coverage": 96.78,
  "oov_rate_val": 2.34
}
```

## 📊 Expected Results

| Configuration | Accuracy | Time | Notes |
|--------------|----------|------|-------|
| LSTM 128 (default) | 78-82% | 5-10 min | Good baseline |
| GRU 128 | 76-80% | 4-8 min | Faster |
| Bi-LSTM 128 | 80-84% | 8-15 min | Best performance |
| LSTM 256 | 79-83% | 8-12 min | More capacity |
| Deep LSTM (2 layers) | 79-83% | 10-20 min | Complex |

*Times for GPU training with 16k samples*

## 🎯 Success Checklist

After running, verify:
- [ ] Training completed without errors
- [ ] Validation accuracy ≥ 75%
- [ ] Training and validation curves look good (no severe overfitting)
- [ ] Confusion matrix shows reasonable performance
- [ ] All 6 emotions have F1-score > 0.5
- [ ] Predictions on custom text make sense
- [ ] Files saved in saved_models/, logs/, results/

## 🆚 Comparison: This vs Other Notebooks

| Feature | Unified Notebook | improved_pipeline.ipynb | Original full_pipeline.ipynb |
|---------|------------------|-------------------------|------------------------------|
| **Self-contained** | ✅ All classes included | ❌ Imports from src/ | ✅ All in notebook |
| **Professional Classes** | ✅ All 6 classes | ✅ All 6 classes | ❌ Basic code |
| **Easy Config** | ✅ Config class at top | ✅ Config class | ❌ Hardcoded |
| **Multiple Models** | ✅ LSTM/GRU/BiLSTM | ✅ LSTM/GRU/BiLSTM | ❌ LSTM only |
| **Visualizations** | ✅ 10+ plots | ✅ 10+ plots | ✅ 3 plots |
| **Experiment Tracking** | ✅ Auto-save results | ✅ Auto-save results | ❌ Manual |
| **Best For** | Quick experiments | Learning modular code | Understanding basics |

## 🚀 Next Steps

1. **Run the baseline** (default config)
2. **Try 3-5 experiments** with different settings
3. **Compare results** in results/ directory
4. **Pick the best model** (highest val_accuracy)
5. **Test with your own texts** (Section 22)
6. **Deploy or share** your best model

## 📞 Need Help?

- **Config issues?** → Check Section 2
- **Training slow?** → Reduce batch_size or use GRU
- **Poor accuracy?** → Try bidirectional or larger model
- **Want to modify?** → All classes are in Sections 3-7
- **General issues?** → Check [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)

---

## 🎉 Summary

This notebook is **perfect for**:
- ✅ Quick experimentation
- ✅ Learning the complete pipeline
- ✅ No external dependencies
- ✅ Easy configuration changes
- ✅ Professional results

**Just open and run!** Everything you need is in one place. 🚀

---

**Created with all your original work + professional enhancements!**
