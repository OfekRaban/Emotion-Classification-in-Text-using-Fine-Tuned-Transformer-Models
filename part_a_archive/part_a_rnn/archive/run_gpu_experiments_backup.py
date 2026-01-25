#!/usr/bin/env python3
"""
GPU Training Script for Emotion Detection on H100
Runs multiple experiments comparing LSTM vs GRU with different hyperparameters
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List

# Set environment for GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from gensim.models import Word2Vec
import re
from collections import Counter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for each experiment"""
    experiment_name: str
    random_seed: int = 42

    # Data paths
    train_path: str = "/home/lab/rabanof/projects/Emotion_Detection_DL/data/raw/train.csv"
    val_path: str = "/home/lab/rabanof/projects/Emotion_Detection_DL/data/raw/validation.csv"
    glove_path: str = "/home/lab/rabanof/Emotion_Detection_DL/glove/glove.6B.100d.txt"

    # Data parameters
    max_len: int = 60
    max_words: int = 20000
    text_column: str = "text"
    label_column: str = "label"
    num_classes: int = 6

    # Model architecture
    model_type: str = "lstm"  # 'lstm', 'gru', 'bilstm', 'bigru'
    rnn_units: int = 128
    num_rnn_layers: int = 1

    # Embeddings
    embedding_type: str = "glove"  # 'glove' or 'word2vec'
    embedding_dim: int = 100
    trainable_embeddings: bool = False

    # Regularization
    dropout: float = 0.2
    spatial_dropout: float = 0.2
    use_class_weights: bool = True

    # Training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    patience: int = 5

    # Directories
    save_dir: str = "saved_models"
    log_dir: str = "logs"
    result_dir: str = "results"


class TextPreprocessor:
    """Text preprocessing with statistics"""

    def __init__(self):
        self.specific_contractions = {
            "didnt": "did not", "dont": "do not", "cant": "cannot",
            "wont": "will not", "wouldnt": "would not", "shouldnt": "should not",
            "couldnt": "could not", "im": "i am", "ive": "i have",
            "id": "i would", "ill": "i will", "hadnt": "had not",
            "youve": "you have", "werent": "were not", "theyve": "they have",
            "theyll": "they will", "itll": "it will", "couldve": "could have",
            "shouldve": "should have", "wouldve": "would have", "hadn": "had not"
        }

        self.general_contractions = {
            "n't": " not", "'re": " are", "'s": " is",
            "'d": " would", "'ll": " will", "'t": " not",
            "'ve": " have", "'m": " am"
        }

        self.slang_corrections = {
            "idk": "i do not know", "yknow": "you know",
            "becuz": "because", "alittle": "a little", "incase": "in case"
        }

        self.typo_corrections = {
            "vunerable": "vulnerable", "percieve": "perceive",
            "definetly": "definitely", "writting": "writing"
        }

    def clean_text(self, text: str) -> str:
        """Clean text with all preprocessing"""
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        for key, value in self.specific_contractions.items():
            text = re.sub(rf'\b{re.escape(key)}\b', value, text)

        for key, value in self.general_contractions.items():
            text = text.replace(key, value)

        for corrections in [self.slang_corrections, self.typo_corrections]:
            for key, value in corrections.items():
                text = re.sub(rf'\b{re.escape(key)}\b', value, text)

        text = re.sub(r"([!?.,])\1+", r"\1", text)
        text = re.sub(r"\.{2,}", ".", text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text


class EmbeddingHandler:
    """Handle embeddings with GloVe or Word2Vec"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tokenizer = None
        self.embedding_matrix = None
        self.embeddings_index = {}
        self.word2vec_model = None

    def create_tokenizer(self, texts: List[str]):
        """Create tokenizer"""
        self.tokenizer = Tokenizer(num_words=self.config.max_words, oov_token="<UNK>", lower=True)
        self.tokenizer.fit_on_texts(texts)
        logger.info(f"Tokenizer created. Vocabulary size: {min(len(self.tokenizer.word_index), self.config.max_words)}")
        return self.tokenizer

    def texts_to_sequences(self, texts: List[str]) -> np.ndarray:
        """Convert texts to padded sequences"""
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.config.max_len, padding='post', truncating='post')
        return padded

    def load_glove_embeddings(self):
        """Load GloVe embeddings"""
        logger.info(f"Loading GloVe embeddings from {self.config.glove_path}...")
        with open(self.config.glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = vector
        logger.info(f"Loaded {len(self.embeddings_index)} word vectors")

    def train_word2vec(self, texts: List[str]):
        """Train Word2Vec"""
        logger.info("Training Word2Vec embeddings...")
        tokenized_texts = [text.split() for text in texts]
        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.config.embedding_dim,
            window=5,
            min_count=1,
            workers=4,
            seed=self.config.random_seed
        )
        logger.info(f"Word2Vec trained. Vocabulary size: {len(self.word2vec_model.wv)}")

    def create_embedding_matrix(self) -> np.ndarray:
        """Create embedding matrix"""
        word_index = self.tokenizer.word_index
        vocab_size = min(len(word_index) + 1, self.config.max_words + 1)

        self.embedding_matrix = np.random.normal(0, 0.1, (vocab_size, self.config.embedding_dim))
        self.embedding_matrix[0] = np.zeros(self.config.embedding_dim)

        found_count = 0
        for word, i in word_index.items():
            if i >= self.config.max_words:
                continue

            embedding_vector = None
            if self.config.embedding_type == 'glove':
                embedding_vector = self.embeddings_index.get(word)
            elif self.config.embedding_type == 'word2vec' and self.word2vec_model:
                try:
                    embedding_vector = self.word2vec_model.wv[word]
                except KeyError:
                    pass

            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector
                found_count += 1

        coverage = (found_count / (vocab_size - 1)) * 100
        logger.info(f"Embedding matrix created. Coverage: {coverage:.2f}%")

        return self.embedding_matrix


class ModelBuilder:
    """Build RNN models"""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def build_model(self, embedding_matrix: np.ndarray) -> keras.Model:
        """Build model based on configuration"""
        vocab_size, embedding_dim = embedding_matrix.shape

        inputs = layers.Input(shape=(self.config.max_len,), name='input')

        x = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=self.config.max_len,
            trainable=self.config.trainable_embeddings,
            name='embedding'
        )(inputs)

        if self.config.spatial_dropout > 0:
            x = layers.SpatialDropout1D(self.config.spatial_dropout)(x)

        for layer_idx in range(self.config.num_rnn_layers):
            return_sequences = (layer_idx < self.config.num_rnn_layers - 1)

            if self.config.model_type in ['lstm', 'bilstm']:
                rnn_layer = layers.LSTM(
                    units=self.config.rnn_units,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout if layer_idx == 0 else 0
                )
            elif self.config.model_type in ['gru', 'bigru']:
                rnn_layer = layers.GRU(
                    units=self.config.rnn_units,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout if layer_idx == 0 else 0
                )

            if self.config.model_type in ['bilstm', 'bigru']:
                rnn_layer = layers.Bidirectional(rnn_layer)

            x = rnn_layer(x)

            if self.config.dropout > 0 and layer_idx < self.config.num_rnn_layers - 1:
                x = layers.Dropout(self.config.dropout)(x)

        outputs = layers.Dense(self.config.num_classes, activation='softmax', name='output')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info(f"Model built: {self.config.model_type.upper()}, Parameters: {model.count_params():,}")
        return model


def run_experiment(config: ExperimentConfig) -> Dict:
    """Run a complete experiment"""
    logger.info("="*80)
    logger.info(f"STARTING EXPERIMENT: {config.experiment_name}")
    logger.info("="*80)

    # Set random seeds
    np.random.seed(config.random_seed)
    tf.random.set_seed(config.random_seed)

    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"GPUs Available: {len(gpus)}")
    if gpus:
        for gpu in gpus:
            logger.info(f"  GPU: {gpu}")
            tf.config.experimental.set_memory_growth(gpu, True)

    # Create directories
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    train_df = pd.read_csv(config.train_path)
    val_df = pd.read_csv(config.val_path)
    logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Preprocess
    logger.info("Preprocessing...")
    preprocessor = TextPreprocessor()
    train_df['text'] = train_df['text'].apply(preprocessor.clean_text)
    val_df['text'] = val_df['text'].apply(preprocessor.clean_text)

    train_texts = train_df['text'].values.tolist()
    val_texts = val_df['text'].values.tolist()
    y_train = train_df['label'].values
    y_val = val_df['label'].values

    # Create embeddings
    logger.info(f"Creating {config.embedding_type.upper()} embeddings...")
    embedding_handler = EmbeddingHandler(config)
    embedding_handler.create_tokenizer(train_texts)

    if config.embedding_type == 'glove':
        embedding_handler.load_glove_embeddings()
    elif config.embedding_type == 'word2vec':
        embedding_handler.train_word2vec(train_texts)

    embedding_matrix = embedding_handler.create_embedding_matrix()

    # Convert to sequences
    logger.info("Converting to sequences...")
    X_train = embedding_handler.texts_to_sequences(train_texts)
    X_val = embedding_handler.texts_to_sequences(val_texts)

    y_train_cat = to_categorical(y_train, num_classes=config.num_classes)
    y_val_cat = to_categorical(y_val, num_classes=config.num_classes)

    # Compute class weights
    class_weights = None
    if config.use_class_weights:
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        logger.info(f"Class weights: {class_weights}")

    # Build model
    logger.info("Building model...")
    model_builder = ModelBuilder(config)
    model = model_builder.build_model(embedding_matrix)

    # Setup callbacks
    checkpoint_path = os.path.join(config.save_dir, f'{config.experiment_name}_best.h5')
    csv_path = os.path.join(config.log_dir, f'{config.experiment_name}_training.csv')

    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=config.patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path)
    ]

    # Train
    logger.info(f"Training {config.model_type.upper()} model...")
    start_time = time.time()

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=config.epochs,
        batch_size=config.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # Evaluate
    logger.info("Evaluating...")
    y_val_pred = model.predict(X_val, verbose=0)
    y_val_pred_labels = y_val_pred.argmax(axis=1)

    val_accuracy = accuracy_score(y_val, y_val_pred_labels)
    precision, recall, f1, support = precision_recall_fscore_support(y_val, y_val_pred_labels, labels=list(range(config.num_classes)))

    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    # Results
    results = {
        'experiment_name': config.experiment_name,
        'model_type': config.model_type,
        'embedding_type': config.embedding_type,
        'rnn_units': config.rnn_units,
        'num_layers': config.num_rnn_layers,
        'dropout': config.dropout,
        'learning_rate': config.learning_rate,
        'trainable_embeddings': config.trainable_embeddings,
        'val_accuracy': float(val_accuracy),
        'val_loss': float(history.history['val_loss'][-1]),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'training_time': float(training_time),
        'epochs_trained': len(history.history['loss']),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'per_class_f1': [float(f) for f in f1]
    }

    # Save results
    results_path = os.path.join(config.result_dir, f'{config.experiment_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("="*80)
    logger.info(f"EXPERIMENT COMPLETED: {config.experiment_name}")
    logger.info(f"  Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    logger.info(f"  Macro F1: {macro_f1:.4f}")
    logger.info(f"  Training Time: {training_time:.2f}s")
    logger.info("="*80)

    return results


def main():
    """Run all experiments"""
    logger.info("="*80)
    logger.info("GPU EMOTION DETECTION EXPERIMENTS")
    logger.info("="*80)

    # Check TensorFlow GPU
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"GPU Available: {tf.test.is_gpu_available()}")
    logger.info(f"GPU Devices: {tf.config.list_physical_devices('GPU')}")

    # Define experiments
    experiments = [
        # Baseline: LSTM with GloVe
        ExperimentConfig(
            experiment_name="lstm_glove_baseline",
            model_type="lstm",
            embedding_type="glove",
            rnn_units=128,
            num_rnn_layers=1,
            dropout=0.2
        ),

        # GRU with GloVe
        ExperimentConfig(
            experiment_name="gru_glove_baseline",
            model_type="gru",
            embedding_type="glove",
            rnn_units=128,
            num_rnn_layers=1,
            dropout=0.2
        ),

        # LSTM with Word2Vec
        ExperimentConfig(
            experiment_name="lstm_word2vec_baseline",
            model_type="lstm",
            embedding_type="word2vec",
            rnn_units=128,
            num_rnn_layers=1,
            dropout=0.2
        ),

        # GRU with Word2Vec
        ExperimentConfig(
            experiment_name="gru_word2vec_baseline",
            model_type="gru",
            embedding_type="word2vec",
            rnn_units=128,
            num_rnn_layers=1,
            dropout=0.2
        ),

        # BiLSTM with GloVe
        ExperimentConfig(
            experiment_name="bilstm_glove",
            model_type="bilstm",
            embedding_type="glove",
            rnn_units=128,
            num_rnn_layers=1,
            dropout=0.2
        ),

        # Hyperparameter variation: More units
        ExperimentConfig(
            experiment_name="lstm_glove_256units",
            model_type="lstm",
            embedding_type="glove",
            rnn_units=256,
            num_rnn_layers=1,
            dropout=0.2
        ),

        # Hyperparameter variation: 2 layers
        ExperimentConfig(
            experiment_name="lstm_glove_2layers",
            model_type="lstm",
            embedding_type="glove",
            rnn_units=128,
            num_rnn_layers=2,
            dropout=0.2
        ),

        # Hyperparameter variation: Higher dropout
        ExperimentConfig(
            experiment_name="lstm_glove_dropout05",
            model_type="lstm",
            embedding_type="glove",
            rnn_units=128,
            num_rnn_layers=1,
            dropout=0.5
        ),
    ]

    # Run all experiments
    all_results = []
    for i, config in enumerate(experiments, 1):
        logger.info(f"\n\nRunning experiment {i}/{len(experiments)}: {config.experiment_name}")
        try:
            results = run_experiment(config)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Experiment {config.experiment_name} failed: {e}")
            continue

    # Create comparison table
    logger.info("\n\n" + "="*80)
    logger.info("ALL EXPERIMENTS COMPLETED - COMPARISON TABLE")
    logger.info("="*80)

    comparison_df = pd.DataFrame(all_results)
    comparison_df = comparison_df.sort_values('val_accuracy', ascending=False)

    print("\n" + comparison_df.to_string(index=False))

    # Save comparison
    comparison_path = os.path.join('results', 'all_experiments_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"\nComparison table saved to: {comparison_path}")

    # Summary
    best_exp = comparison_df.iloc[0]
    logger.info("\n" + "="*80)
    logger.info("BEST MODEL")
    logger.info("="*80)
    logger.info(f"Experiment: {best_exp['experiment_name']}")
    logger.info(f"Model: {best_exp['model_type'].upper()}")
    logger.info(f"Embedding: {best_exp['embedding_type'].upper()}")
    logger.info(f"Accuracy: {best_exp['val_accuracy']:.4f} ({best_exp['val_accuracy']*100:.2f}%)")
    logger.info(f"Macro F1: {best_exp['macro_f1']:.4f}")
    logger.info("="*80)

    logger.info("\nAll experiments completed successfully!")


if __name__ == "__main__":
    main()
