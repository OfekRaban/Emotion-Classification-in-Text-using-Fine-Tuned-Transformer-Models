# Troubleshooting Guide

Common issues and their solutions for the Emotion Detection project.

---

## Installation Issues

### Problem: `pip install` fails

**Solution 1** - Update pip:
```bash
pip install --upgrade pip
```

**Solution 2** - Install with verbose:
```bash
pip install -r requirements.txt -v
```

**Solution 3** - Install individually:
```bash
pip install tensorflow
pip install gensim
# etc.
```

### Problem: TensorFlow GPU not working

**Check GPU availability**:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

**If empty**:
1. Check CUDA/cuDNN installation
2. Reinstall TensorFlow GPU:
   ```bash
   pip uninstall tensorflow
   pip install tensorflow-gpu
   ```
3. Or use CPU (slower but works):
   ```bash
   pip install tensorflow
   ```

---

## Data Loading Issues

### Problem: FileNotFoundError for CSV files

**Check paths**:
```python
import os
print(os.path.exists('data/raw/train.csv'))
print(os.path.exists('data/raw/validation.csv'))
```

**Fix**:
- Update paths in config file
- Or move CSV files to correct location:
  ```bash
  mkdir -p data/raw
  mv train.csv data/raw/
  mv validation.csv data/raw/
  ```

### Problem: GloVe file not found

**Download GloVe**:
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
mkdir -p embeddings
mv glove.6B.*.txt embeddings/
```

**Update config**:
```yaml
embedding:
  embedding_path: embeddings/glove.6B.100d.txt
```

---

## Memory Issues

### Problem: OOM (Out of Memory) during training

**Solution 1** - Reduce batch size:
```bash
python run_experiment.py --batch-size 16
```

**Solution 2** - Reduce vocabulary:
Edit config:
```yaml
data:
  max_words: 10000  # Instead of 20000
```

**Solution 3** - Reduce sequence length:
```yaml
data:
  max_len: 40  # Instead of 60
```

**Solution 4** - Smaller model:
```bash
python run_experiment.py --units 64
```

**Solution 5** - Use smaller embeddings:
```yaml
embedding:
  embedding_dim: 50  # Instead of 100
  embedding_path: embeddings/glove.6B.50d.txt
```

### Problem: OOM during data loading

**Solution** - Load data in chunks:
```python
# Modify preprocessor to use chunks
df = pd.read_csv(path, chunksize=1000)
```

---

## Training Issues

### Problem: Model not converging (accuracy stuck)

**Check 1** - Verify data loaded correctly:
```python
print(train_df.head())
print(train_df['label'].value_counts())
```

**Check 2** - Verify embeddings loaded:
```python
print(f"Coverage: {stats['coverage_percent']:.2f}%")
print(f"OOV rate: {embedding_handler.get_oov_rate(X_train):.2f}%")
```

**Solution 1** - Increase learning rate:
```bash
python run_experiment.py --learning-rate 0.01
```

**Solution 2** - Increase model capacity:
```bash
python run_experiment.py --units 256
```

**Solution 3** - Add more layers:
```bash
python run_experiment.py --model lstm --units 128 # Edit config for num_layers: 2
```

### Problem: Training too slow

**Solution 1** - Use GRU instead of LSTM:
```bash
python run_experiment.py --model gru
```

**Solution 2** - Increase batch size:
```bash
python run_experiment.py --batch-size 64
```

**Solution 3** - Reduce vocabulary:
```yaml
data:
  max_words: 10000
```

**Solution 4** - Disable TensorBoard:
Edit config:
```yaml
training:
  tensorboard: false
```

**Solution 5** - Use GPU:
- Check GPU availability
- Install tensorflow-gpu

### Problem: Overfitting (val_loss >> train_loss)

**Solution 1** - Increase dropout:
Edit config:
```yaml
model:
  dropout: 0.5
  spatial_dropout: 0.3
```

**Solution 2** - Reduce model size:
```bash
python run_experiment.py --units 64
```

**Solution 3** - Early stopping (already enabled):
Verify in config:
```yaml
training:
  early_stopping: true
  patience: 5
```

**Solution 4** - Add regularization:
```python
# In model architecture (custom modification)
model.add(layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
```

### Problem: Underfitting (low train accuracy)

**Solution 1** - Increase model capacity:
```bash
python run_experiment.py --units 256
```

**Solution 2** - Add layers:
Edit config:
```yaml
model:
  num_layers: 2
```

**Solution 3** - Use bidirectional:
```bash
python run_experiment.py --bidirectional
```

**Solution 4** - Increase epochs:
```bash
python run_experiment.py --epochs 100
```

**Solution 5** - Fine-tune embeddings:
Edit config:
```yaml
embedding:
  trainable: true
```

---

## Performance Issues

### Problem: Poor performance on minority classes

**Check class distribution**:
```python
print(train_df['label'].value_counts())
```

**Solution 1** - Verify class weights enabled:
```yaml
training:
  use_class_weights: true
```

**Solution 2** - Oversample minority classes:
```python
from sklearn.utils import resample

# Oversample minority class
minority = train_df[train_df['label'] == 5]  # Surprise
oversampled = resample(minority, n_samples=1000, random_state=42)
train_df = pd.concat([train_df, oversampled])
```

**Solution 3** - Adjust prediction threshold:
```python
# Instead of argmax, use threshold
predictions = model.predict(X_val)
# Custom threshold for minority classes
```

### Problem: Low overall accuracy (<70%)

**Checklist**:
- [ ] Check data preprocessing
- [ ] Verify embedding coverage (>90%)
- [ ] Try larger model (units: 256)
- [ ] Use bidirectional layers
- [ ] Increase epochs
- [ ] Check for data leakage

**Debug Steps**:
```python
# 1. Check data quality
print(train_df.head(20))
print(train_df['text'].apply(len).describe())

# 2. Check embedding coverage
print(f"Coverage: {stats['coverage_percent']:.2f}%")

# 3. Check model predictions
y_pred = model.predict(X_val[:10])
print(y_pred)

# 4. Check confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val.argmax(1), y_pred.argmax(1))
print(cm)
```

---

## Jupyter Notebook Issues

### Problem: Kernel dies during training

**Solution 1** - Reduce batch size:
```python
config.training.batch_size = 16
```

**Solution 2** - Restart kernel and clear outputs:
- Kernel → Restart & Clear Output

**Solution 3** - Run in separate process:
Instead of notebook, use:
```bash
python run_experiment.py
```

### Problem: Import errors in notebook

**Solution 1** - Add path:
```python
import sys
sys.path.append('../')
```

**Solution 2** - Install in editable mode:
```bash
pip install -e .
```

**Solution 3** - Restart kernel:
- Kernel → Restart

### Problem: Plots not showing

**Solution**:
```python
import matplotlib.pyplot as plt
%matplotlib inline

# Or
%matplotlib widget  # For interactive plots
```

---

## Configuration Issues

### Problem: Config file not loading

**Check syntax**:
```bash
# Validate YAML
python -c "import yaml; yaml.safe_load(open('configs/sample_config.yaml'))"
```

**Check indentation**:
- YAML is sensitive to indentation
- Use 2 spaces (not tabs)

**Example fix**:
```yaml
# Wrong
model:
model_type: lstm  # Missing indentation

# Correct
model:
  model_type: lstm
```

### Problem: Config parameters not applied

**Verify config loaded**:
```python
config = ExperimentConfig.load('configs/my_config.yaml')
print(config.model.model_type)
print(config.training.batch_size)
```

**Check override order**:
CLI arguments override config file:
```bash
# This will use batch_size=64, not config value
python run_experiment.py --config my_config.yaml --batch-size 64
```

---

## Results and Logging Issues

### Problem: Results not saving

**Check directories exist**:
```bash
mkdir -p results logs saved_models configs
```

**Check permissions**:
```bash
chmod -R 755 results logs saved_models configs
```

**Check disk space**:
```bash
df -h
```

### Problem: TensorBoard not working

**Launch TensorBoard**:
```bash
tensorboard --logdir logs/ --port 6006
```

**Check logs exist**:
```bash
ls -la logs/
```

**Access TensorBoard**:
- Local: http://localhost:6006
- Remote: Use SSH tunneling:
  ```bash
  ssh -L 6006:localhost:6006 user@server
  ```

### Problem: Plots not generating

**Check matplotlib backend**:
```python
import matplotlib
print(matplotlib.get_backend())

# If 'agg', change to:
matplotlib.use('TkAgg')  # Or 'Qt5Agg'
```

**Install missing dependencies**:
```bash
pip install matplotlib seaborn
```

---

## Prediction Issues

### Problem: Poor predictions on new data

**Check preprocessing**:
```python
# Same preprocessing as training
text = "I love this!"
cleaned = preprocessor.clean_text(text)
print(f"Original: {text}")
print(f"Cleaned: {cleaned}")
```

**Check sequence generation**:
```python
sequence = embedding_handler.texts_to_sequences([cleaned])
print(f"Sequence shape: {sequence.shape}")
print(f"Sequence: {sequence[0][:20]}")  # First 20 tokens
```

**Check OOV**:
```python
oov_rate = embedding_handler.get_oov_rate(sequence)
print(f"OOV rate: {oov_rate:.2f}%")
# If high, text has many unknown words
```

### Problem: Model predictions don't make sense

**Verify model loaded correctly**:
```python
from tensorflow import keras
model = keras.models.load_model('saved_models/experiment_best.keras')
model.summary()
```

**Check prediction probabilities**:
```python
pred = model.predict(sequence)
print(f"Probabilities: {pred[0]}")
print(f"Sum: {pred[0].sum()}")  # Should be ~1.0
```

**Check emotion mapping**:
```python
emotion_map = {0: 'Sadness', 1: 'Joy', 2: 'Love',
               3: 'Anger', 4: 'Fear', 5: 'Surprise'}
pred_idx = pred[0].argmax()
print(f"Predicted: {emotion_map[pred_idx]}")
```

---

## Common Errors and Fixes

### Error: "ValueError: could not convert string to float"

**Cause**: Data not properly preprocessed

**Fix**:
```python
# Check for non-numeric labels
print(train_df['label'].dtype)
train_df['label'] = train_df['label'].astype(int)
```

### Error: "ResourceExhaustedError: OOM when allocating tensor"

**Cause**: Out of GPU memory

**Fix**: Reduce batch size or model size (see Memory Issues above)

### Error: "InvalidArgumentError: indices are not valid"

**Cause**: Tokenizer mismatch between training and inference

**Fix**: Use same tokenizer:
```python
# Save tokenizer
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(embedding_handler.tokenizer, f)

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
```

### Error: "AttributeError: 'NoneType' object has no attribute"

**Cause**: Component not initialized

**Fix**: Follow correct order:
```python
# 1. Create preprocessor
preprocessor = TextPreprocessor()

# 2. Load data
train_df, val_df, _ = load_and_preprocess_data(...)

# 3. Create embedding handler
handler = EmbeddingHandler()

# 4. Prepare sequences
handler.create_tokenizer(train_df['text'])
handler.load_glove_embeddings(glove_path)
```

---

## Performance Optimization Tips

### Speed up training

1. **Use GRU** (30% faster than LSTM)
2. **Increase batch size** (better GPU utilization)
3. **Reduce vocabulary** (faster embedding lookup)
4. **Disable TensorBoard** (small overhead)
5. **Use mixed precision** (for newer GPUs):
   ```python
   from tensorflow.keras import mixed_precision
   policy = mixed_precision.Policy('mixed_float16')
   mixed_precision.set_global_policy(policy)
   ```

### Improve accuracy

1. **Use larger embeddings** (100d → 300d)
2. **Bidirectional layers** (+2-3% accuracy)
3. **Add depth** (2 layers)
4. **Fine-tune embeddings** (trainable=True)
5. **Ensemble** multiple models

### Reduce overfitting

1. **Increase dropout** (0.2 → 0.5)
2. **Add spatial dropout** (0.2-0.3)
3. **Reduce model size**
4. **Early stopping** (already enabled)
5. **Data augmentation**

---

## Getting Help

If you're still stuck:

1. **Check logs**:
   ```bash
   cat experiment.log
   cat logs/experiment_name_training.csv
   ```

2. **Review TensorBoard**:
   ```bash
   tensorboard --logdir logs/
   ```

3. **Verify configuration**:
   ```python
   config = ExperimentConfig.load('configs/experiment_config.yaml')
   print(config.to_dict())
   ```

4. **Test minimal example**:
   Run baseline preset:
   ```bash
   python run_experiment.py --preset lstm_glove
   ```

5. **Check dependencies**:
   ```bash
   pip list | grep tensorflow
   pip list | grep gensim
   ```

---

## Quick Diagnostic Script

Run this to check your setup:

```python
import sys
import os

print("=== Environment Check ===")

# Python version
print(f"Python: {sys.version}")

# Packages
try:
    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
except ImportError as e:
    print(f"TensorFlow: NOT INSTALLED - {e}")

try:
    import gensim
    print(f"Gensim: {gensim.__version__}")
except ImportError as e:
    print(f"Gensim: NOT INSTALLED - {e}")

try:
    import pandas as pd
    print(f"Pandas: {pd.__version__}")
except ImportError as e:
    print(f"Pandas: NOT INSTALLED - {e}")

# Data files
print(f"\nTrain CSV exists: {os.path.exists('data/raw/train.csv')}")
print(f"Val CSV exists: {os.path.exists('data/raw/validation.csv')}")

# GloVe
glove_path = 'embeddings/glove.6B.100d.txt'
print(f"GloVe exists: {os.path.exists(glove_path)}")

# Directories
for dir_name in ['results', 'logs', 'saved_models', 'configs']:
    exists = os.path.exists(dir_name)
    print(f"{dir_name}/: {'EXISTS' if exists else 'MISSING'}")

print("\n=== Setup Complete ===")
```

Save as `check_setup.py` and run:
```bash
python check_setup.py
```

---

**Still have issues?** Check the full documentation in `README.md` or review the example notebook in `notebooks/improved_pipeline.ipynb`.
