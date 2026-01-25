# Quick Start Guide

## Option 1: Using Jupyter Notebook (Recommended for Beginners)

1. **Activate environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```

3. **Open**: `notebooks/improved_pipeline.ipynb`

4. **Run**: Execute all cells (Cell → Run All)

The notebook will:
- Load and preprocess data
- Create GloVe embeddings
- Build and train LSTM model
- Generate visualizations
- Save results

## Option 2: Using Python Script (Quick Experiments)

### Basic Usage

```bash
# Activate environment
source venv/bin/activate

# Run with default settings (LSTM + GloVe)
python run_experiment.py --experiment-name my_first_exp

# Use a preset configuration
python run_experiment.py --preset lstm_glove

# Use a config file
python run_experiment.py --config configs/sample_config.yaml
```

### Custom Experiments

```bash
# LSTM with 256 units, 50 epochs
python run_experiment.py --model lstm --units 256 --epochs 50 --experiment-name lstm_256

# Bidirectional GRU with Word2Vec
python run_experiment.py --model gru --bidirectional --embedding word2vec --experiment-name bigru_w2v

# Custom learning rate and batch size
python run_experiment.py --learning-rate 0.0005 --batch-size 64 --epochs 100 --experiment-name custom_lr
```

## Option 3: Interactive Python Session

```python
# Start Python
python

# Import and run
from src.utils.config import get_lstm_glove_config
from run_experiment import main
import argparse

# Create arguments
args = argparse.Namespace(
    preset='lstm_glove',
    config=None,
    experiment_name='interactive_test',
    model=None, units=None, bidirectional=False,
    embedding=None, epochs=None, batch_size=None, learning_rate=None
)

# Run experiment
main(args)
```

## Viewing Results

### TensorBoard (Real-time Training Monitoring)

```bash
tensorboard --logdir logs/
# Open: http://localhost:6006
```

### Result Files

Check these directories:
- `results/`: Plots and visualizations
- `saved_models/`: Trained model checkpoints
- `logs/`: Training logs and CSV files
- `configs/`: Saved configurations

### View Specific Results

```python
import json

# Load experiment results
with open('results/my_experiment_results.json', 'r') as f:
    results = json.load(f)

print(f"Best Validation Accuracy: {results['best_metrics']['val_accuracy']:.4f}")
print(f"Training Time: {results['training_time_seconds']:.2f}s")
```

## Common Workflows

### 1. Baseline Experiment

```bash
# Quick baseline with LSTM + GloVe
python run_experiment.py --preset lstm_glove --experiment-name baseline
```

### 2. Compare Multiple Models

```bash
# LSTM
python run_experiment.py --model lstm --experiment-name exp_lstm

# GRU
python run_experiment.py --model gru --experiment-name exp_gru

# Bidirectional LSTM
python run_experiment.py --model lstm --bidirectional --experiment-name exp_bilstm
```

### 3. Hyperparameter Search

```bash
# Different units
for units in 64 128 256; do
    python run_experiment.py --units $units --experiment-name lstm_units_${units}
done

# Different batch sizes
for bs in 16 32 64; do
    python run_experiment.py --batch-size $bs --experiment-name lstm_bs_${bs}
done
```

### 4. Compare Embeddings

```bash
# GloVe 100d
python run_experiment.py --embedding glove --experiment-name glove_100d

# Word2Vec (trained on corpus)
python run_experiment.py --embedding word2vec --experiment-name word2vec_custom
```

## Testing Your Model

```python
from tensorflow import keras
from src.data.preprocessor import TextPreprocessor
from src.data.embeddings import EmbeddingHandler

# Load model
model = keras.models.load_model('saved_models/my_experiment_best.keras')

# Load preprocessor and embedding handler (must match training)
preprocessor = TextPreprocessor()
embedding_handler = EmbeddingHandler()
# ... (setup as in notebook)

# Test new text
test_text = "I am so happy and excited!"
cleaned = preprocessor.clean_text(test_text)
sequence = embedding_handler.texts_to_sequences([cleaned])
prediction = model.predict(sequence)

emotion_map = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'}
predicted_emotion = emotion_map[prediction.argmax()]
confidence = prediction.max() * 100

print(f"Emotion: {predicted_emotion} ({confidence:.1f}%)")
```

## Troubleshooting

### GPU Not Detected

```bash
# Check TensorFlow GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If empty, install tensorflow-gpu or use CPU (slower but works)
```

### Out of Memory

Edit config or use command line:
```bash
python run_experiment.py --batch-size 16  # Reduce batch size
```

### GloVe File Not Found

Download GloVe embeddings:
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d embeddings/
```

Update path in config or use:
```bash
python run_experiment.py --config configs/sample_config.yaml
# (Make sure embedding_path is correct in config)
```

## Next Steps

1. **Review Results**: Check `results/` for visualizations
2. **Tune Hyperparameters**: Experiment with different settings
3. **Try Different Architectures**: LSTM vs GRU vs Bidirectional
4. **Compare Embeddings**: GloVe vs Word2Vec
5. **Fine-tune**: Set `trainable_embeddings=True` in config
6. **Deploy**: Use saved model for inference

## Getting Help

- Read [README.md](README.md) for detailed documentation
- Check experiment logs in `logs/`
- Review TensorBoard for training curves
- Open an issue if you find bugs

## Performance Benchmarks

Expected results (may vary):

| Model | Embedding | Accuracy | Training Time |
|-------|-----------|----------|---------------|
| LSTM (128) | GloVe 100d | 78-82% | 5-10 min |
| GRU (128) | GloVe 100d | 76-80% | 4-8 min |
| Bi-LSTM (128) | GloVe 100d | 80-84% | 8-15 min |
| Deep LSTM (2x128) | GloVe 100d | 79-83% | 10-20 min |

*Times are approximate for GPU training with 16k samples*

## Best Practices

1. **Always start with baseline**: Run `lstm_glove` preset first
2. **One change at a time**: Modify single hyperparameter per experiment
3. **Monitor overfitting**: Watch validation loss diverging from training
4. **Use early stopping**: Already enabled by default
5. **Save configs**: Each experiment saves its config automatically
6. **Version results**: Use meaningful experiment names

Happy experimenting! 🚀
