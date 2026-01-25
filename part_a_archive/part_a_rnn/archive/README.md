# Emotion Detection in Text using Deep Learning

A professional, production-ready system for detecting emotions in Twitter text using LSTM and GRU neural networks with pre-trained word embeddings.

## Project Overview

This project implements a comprehensive NLP pipeline for classifying short text snippets (Twitter responses) into one of six emotion categories:

- **0: Sadness** 😢
- **1: Joy** 😊
- **2: Love** ❤️
- **3: Anger** 😠
- **4: Fear** 😨
- **5: Surprise** 😲

### Key Features

- **Advanced Text Preprocessing**: Contraction expansion, typo correction, elongation normalization
- **Multiple Embedding Support**: GloVe and Word2Vec pre-trained embeddings
- **Flexible Model Architectures**: LSTM, GRU, Bidirectional variants with configurable layers
- **Professional Training Pipeline**: Callbacks, checkpointing, early stopping, learning rate scheduling
- **Comprehensive Evaluation**: Confusion matrices, per-class metrics, visualizations
- **Experiment Tracking**: Automatic logging and result saving
- **Configuration Management**: YAML/JSON-based configuration system
- **Class Imbalance Handling**: Computed class weights for balanced training

## Project Structure

```
Emotion_Detection_DL/
├── src/
│   ├── data/
│   │   ├── preprocessor.py      # Text preprocessing and data loading
│   │   └── embeddings.py        # Embedding handling (GloVe/Word2Vec)
│   ├── models/
│   │   └── architectures.py     # Model architectures (LSTM/GRU)
│   ├── training/
│   │   └── trainer.py           # Training pipeline and callbacks
│   └── utils/
│       ├── config.py            # Configuration management
│       └── visualization.py     # Visualization utilities
├── notebooks/
│   ├── full_pipeline.ipynb      # Original pipeline
│   └── improved_pipeline.ipynb  # Professional pipeline ⭐
├── data/
│   └── raw/
│       ├── train.csv            # Training data
│       └── validation.csv       # Validation data
├── configs/                     # Saved configurations
├── logs/                        # Training logs and TensorBoard
├── saved_models/                # Model checkpoints
├── results/                     # Evaluation results and plots
├── requirements.txt             # Project dependencies
├── setup.py                     # Installation script
└── README.md                    # This file
```

## Installation

### 1. Clone the Repository

```bash
cd /path/to/Emotion_Detection_DL
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download GloVe Embeddings

Download GloVe embeddings from [Stanford NLP](https://nlp.stanford.edu/projects/glove/):

```bash
# Download glove.6B.zip (contains 50d, 100d, 200d, 300d)
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d embeddings/
```

Recommended: `glove.6B.100d.txt` for this project.

## Quick Start

### Using Jupyter Notebook (Recommended)

1. Launch Jupyter:
```bash
jupyter notebook
```

2. Open `notebooks/improved_pipeline.ipynb`

3. Run all cells to:
   - Load and preprocess data
   - Create embeddings
   - Train LSTM model
   - Evaluate and visualize results

### Using Python Scripts

```python
from src.data.preprocessor import load_and_preprocess_data
from src.data.embeddings import create_embeddings
from src.models.architectures import create_model
from src.training.trainer import train_model
from src.utils.config import get_lstm_glove_config

# Load configuration
config = get_lstm_glove_config()

# Load and preprocess data
train_df, val_df, preprocessor = load_and_preprocess_data(
    "data/raw/train.csv",
    "data/raw/validation.csv"
)

# Create embeddings
X_train, X_val, embedding_matrix, handler, stats = create_embeddings(
    train_df['text'].tolist(),
    val_df['text'].tolist(),
    embedding_type='glove',
    embedding_path='embeddings/glove.6B.100d.txt'
)

# Prepare labels
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(train_df['label'], num_classes=6)
y_val = to_categorical(val_df['label'], num_classes=6)

# Create model
model = create_model(
    model_type='lstm',
    vocab_size=handler.vocab_size,
    embedding_dim=100,
    embedding_matrix=embedding_matrix,
    config={'lstm_units': 128, 'dropout': 0.2}
)

# Train
trainer, history = train_model(
    model, X_train, y_train, X_val, y_val,
    experiment_name='my_experiment'
)
```

## Model Architectures

### LSTM Model
```python
# Simple LSTM
config = {
    'lstm_units': 128,
    'num_layers': 1,
    'dropout': 0.2,
    'spatial_dropout': 0.2
}

# Bidirectional LSTM
config = {
    'lstm_units': 128,
    'bidirectional': True,
    'dropout': 0.2
}

# Deep LSTM (2 layers)
config = {
    'lstm_units': 128,
    'num_layers': 2,
    'dropout': 0.2
}
```

### GRU Model
```python
config = {
    'gru_units': 128,
    'num_layers': 1,
    'dropout': 0.2,
    'bidirectional': False
}
```

## Configuration System

### Using Predefined Configs

```python
from src.utils.config import get_all_configs

configs = get_all_configs()
# Available: lstm_glove, lstm_word2vec, gru_glove, bilstm_glove, deep_lstm_glove
```

### Creating Custom Config

```python
from src.utils.config import ExperimentConfig

config = ExperimentConfig(experiment_name="custom_exp")
config.model.model_type = "lstm"
config.model.units = 256
config.model.bidirectional = True
config.training.batch_size = 64
config.training.epochs = 100

# Save configuration
config.save("configs/custom_exp.yaml")

# Load configuration
config = ExperimentConfig.load("configs/custom_exp.yaml")
```

## Hyperparameter Tuning

Key hyperparameters to experiment with:

### Model Architecture
- `units`: Number of LSTM/GRU units (64, 128, 256)
- `num_layers`: Number of recurrent layers (1, 2)
- `bidirectional`: Use bidirectional layers (True/False)
- `dropout`: Dropout rate (0.1, 0.2, 0.3, 0.5)

### Training
- `batch_size`: Batch size (16, 32, 64)
- `learning_rate`: Learning rate (0.0001, 0.001, 0.01)
- `epochs`: Maximum epochs (50, 100)

### Embeddings
- `embedding_type`: 'glove' or 'word2vec'
- `embedding_dim`: Dimension (50, 100, 200, 300 for GloVe)
- `trainable`: Fine-tune embeddings (False recommended initially)

## Results and Visualization

The pipeline automatically generates:

1. **Training History Plots**: Accuracy and loss curves
2. **Confusion Matrix**: Both raw counts and normalized
3. **Classification Report**: Precision, recall, F1-score per class
4. **Per-Class Accuracy**: Visual comparison
5. **Prediction Distribution**: True vs predicted label distribution

All visualizations are saved in `results/` directory.

## Performance Expectations

Target performance metrics:

- **Overall Accuracy**: ≥ 75%
- **Per-Class Metrics**:
  - Joy/Sadness: 80-85% (well-represented classes)
  - Anger/Fear: 70-75%
  - Love/Surprise: 60-70% (minority classes)

### Tips for Improving Performance

1. **Increase model capacity**: Use more units (256) or add layers
2. **Use bidirectional layers**: Captures context from both directions
3. **Fine-tune embeddings**: Set `trainable_embeddings=True`
4. **Adjust class weights**: Already implemented for imbalanced data
5. **Data augmentation**: Implement synonym replacement or back-translation
6. **Ensemble methods**: Combine multiple models

## Class Imbalance Handling

The dataset is imbalanced:
- Joy: ~33%
- Sadness: ~29%
- Anger: ~13%
- Fear: ~12%
- Love: ~8%
- Surprise: ~4%

**Solution**: Class weights are automatically computed and applied during training.

## Experiment Tracking

Each experiment automatically saves:

```
results/
├── {experiment_name}_results.json          # Metrics and hyperparameters
├── {experiment_name}_training_history.png  # Training curves
├── {experiment_name}_confusion_matrix.png  # Confusion matrix
└── ...

saved_models/
└── {experiment_name}_best.keras            # Best model checkpoint

logs/
├── {experiment_name}_training.csv          # Training log
└── {experiment_name}/                      # TensorBoard logs
```

## TensorBoard

Monitor training in real-time:

```bash
tensorboard --logdir logs/
```

Visit: http://localhost:6006

## Testing on New Data

```python
# Load trained model
from tensorflow import keras
model = keras.models.load_model('saved_models/my_experiment_best.keras')

# Preprocess new text
new_text = "I am so excited about this!"
cleaned_text = preprocessor.clean_text(new_text)
sequence = embedding_handler.texts_to_sequences([cleaned_text])

# Predict
prediction = model.predict(sequence)
emotion_idx = np.argmax(prediction)
emotion = EMOTION_MAP[emotion_idx]
confidence = prediction[0][emotion_idx] * 100

print(f"Emotion: {emotion} (confidence: {confidence:.2f}%)")
```

## Model Comparison

Run multiple experiments and compare:

```python
from src.utils.visualization import ResultsVisualizer

results = {
    'LSTM': {'accuracy': 0.82, 'loss': 0.45},
    'GRU': {'accuracy': 0.80, 'loss': 0.48},
    'Bi-LSTM': {'accuracy': 0.85, 'loss': 0.42}
}

visualizer = ResultsVisualizer()
visualizer.compare_models(results)
```

## API Deployment (Future Work)

For production deployment, consider:

1. **Flask/FastAPI**: Create REST API
2. **Docker**: Containerize the application
3. **Model serving**: TensorFlow Serving or ONNX
4. **Monitoring**: Track inference latency and accuracy

## Contributing

To extend this project:

1. Add new model architectures in `src/models/architectures.py`
2. Implement custom preprocessing in `src/data/preprocessor.py`
3. Add new embeddings in `src/data/embeddings.py`
4. Create visualization utilities in `src/utils/visualization.py`

## Common Issues and Solutions

### 1. CUDA/GPU Issues
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 2. Memory Errors
- Reduce `batch_size`
- Reduce `max_words` vocabulary size
- Use smaller embedding dimensions

### 3. Poor Performance
- Check class weights are applied
- Verify embeddings loaded correctly (check coverage)
- Ensure no data leakage
- Try different architectures

### 4. Overfitting
- Increase dropout rates
- Add more regularization
- Use early stopping (already implemented)
- Reduce model capacity

## Citation

If you use this project, please cite:

```bibtex
@misc{emotion_detection_dl,
  title={Emotion Detection in Text using Deep Learning},
  author={Emotion Detection Team},
  year={2025},
  howpublished={\url{https://github.com/yourusername/emotion-detection-dl}}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- GloVe embeddings: [Stanford NLP Group](https://nlp.stanford.edu/projects/glove/)
- Dataset: Twitter emotion classification dataset
- Inspired by modern NLP best practices and research

## Contact

For questions or issues, please open an issue on GitHub or contact the project maintainers.

---

**Happy Emotion Detection! 😊**
