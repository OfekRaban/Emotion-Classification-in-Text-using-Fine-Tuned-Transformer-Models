#!/usr/bin/env python3
"""
GPU Training Script for Emotion Detection on H100
Uses EXACT preprocessing from full_pipeline.ipynb to ensure high accuracy
"""

import os
import sys
import json
import time
import logging
import re
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List
from collections import Counter

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
    glove_path: str = "/home/lab/rabanof/Emotion_Detection_DL/glove/glove.6B.50d.txt"

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
    embedding_dim: int = 50  # Using 50d like full_pipeline
    trainable_embeddings: bool = False

    # Regularization
    dropout: float = 0.2
    spatial_dropout: float = 0.2
    recurrent_dropout: float = 0.2
    use_class_weights: bool = True

    # Training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    patience: int = 15

    # Directories
    save_dir: str = "saved_models"
    log_dir: str = "logs"
    result_dir: str = "results"


def aggressive_text_normalization(text):
    """
    EXACT preprocessing from your full_pipeline.ipynb
    This is crucial for achieving 85-90% accuracy!
    """
    if not isinstance(text, str):
        return ""

    # 1. Elongation Normalization (sooo -> soo)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # 2. Comprehensive contraction expansion (from your full_pipeline)
    contractions_and_slang = {
        # Original contractions
        "won't": "will not", "can't": "cannot", "n't": " not", "'re": " are", "'s": " is",
        "'d": " would", "'ll": " will", "'ve": " have", "'m": " am",

        # Specific contractions (from your pipeline)
        "hadnt": "had not", "youve": "you have", "hadn": "had not", "werent": "were not",
        "theyve": "they have", "theyll": "they will", "itll": "it will", "couldve": "could have",
        "shouldve": "should have", "wouldve": "would have", "didnt": "did not", "dont": "do not",
        "wont": "will not", "wouldnt": "would not", "shouldnt": "should not",
        "couldnt": "could not", "im": "i am", "ive": "i have",
        "id": "i would", "ill": "i will",

        # Slang/Shortcuts
        "incase": "in case", "alittle": "a little", "becuz": "because",
        "idk": "i do not know", "yknow": "you know",

        # Typos
        "vunerable": "vulnerable", "percieve": "perceive",
        "definetly": "definitely", "writting": "writing"
    }

    # Apply replacements with word boundaries
    for key, value in contractions_and_slang.items():
        text = re.sub(r'\b' + re.escape(key) + r'\b', value, text)

    # 3. Reduce repeated punctuation
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    text = re.sub(r"\.{2,}", ".", text)

    # 4. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess dataframe with your exact pipeline"""
    df = df.copy()

    # Lowercase
    df['text'] = df['text'].str.lower()

    # Apply aggressive normalization
    df['text'] = df['text'].apply(aggressive_text_normalization)

    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=['text'], keep='first')
    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"  Removed {removed} duplicates")

    return df.reset_index(drop=True)


def check_data_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    """Remove overlapping texts (from your pipeline)"""
    train_texts = set(train_df['text'])
    val_texts = set(val_df['text'])
    overlaps = val_texts.intersection(train_texts)

    if len(overlaps) > 0:
        logger.warning(f"  Data leakage: {len(overlaps)} overlapping texts found, removing from train")
        train_df_clean = train_df[~train_df['text'].isin(overlaps)].copy()
        return train_df_clean.reset_index(drop=True)

    logger.info("  No data leakage detected")
    return train_df


class EmbeddingHandler:
    """Handle embeddings with GloVe or Word2Vec"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tokenizer = None
        self.embedding_matrix = None
        self.embeddings_index = {}
        self.word2vec_model = None

    def create_tokenizer(self, texts: List[str]):
        """Create tokenizer - using exact settings from your pipeline"""
        self.tokenizer = Tokenizer(
            num_words=self.config.max_words,
            oov_token="<UNK>",
            lower=True
        )
        self.tokenizer.fit_on_texts(texts)

        vocab_size = min(len(self.tokenizer.word_index), self.config.max_words)
        logger.info(f"  Tokenizer created. Vocabulary size: {vocab_size}")
        return self.tokenizer

    def texts_to_sequences(self, texts: List[str]) -> np.ndarray:
        """Convert texts to padded sequences"""
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.config.max_len, padding='post', truncating='post')
        return padded

    def load_glove_embeddings(self):
        """Load GloVe embeddings"""
        logger.info(f"  Loading GloVe from {self.config.glove_path}...")
        with open(self.config.glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = vector
        logger.info(f"  Loaded {len(self.embeddings_index)} word vectors")

    def train_word2vec(self, texts: List[str]):
        """Train Word2Vec"""
        logger.info("  Training Word2Vec embeddings...")
        tokenized_texts = [text.split() for text in texts]
        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.config.embedding_dim,
            window=5,
            min_count=1,
            workers=4,
            seed=self.config.random_seed
        )
        logger.info(f"  Word2Vec trained. Vocabulary: {len(self.word2vec_model.wv)}")

    def create_embedding_matrix(self) -> np.ndarray:
        """Create embedding matrix - matching your pipeline exactly"""
        word_index = self.tokenizer.word_index
        vocab_size = min(len(word_index) + 1, self.config.max_words + 1)

        # Initialize with zeros (matching your pipeline)
        self.embedding_matrix = np.zeros((vocab_size, self.config.embedding_dim))

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
        logger.info(f"  Embedding coverage: {coverage:.2f}% ({found_count}/{vocab_size-1})")

        return self.embedding_matrix


class ModelBuilder:
    """Build RNN models - matching your architecture"""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def build_model(self, embedding_matrix: np.ndarray) -> keras.Model:
        """Build model - EXACTLY matching your full_pipeline architecture"""
        vocab_size, embedding_dim = embedding_matrix.shape

        if self.config.model_type in ['lstm', 'gru']:
            # Sequential API (like your pipeline)
            from tensorflow.keras.models import Sequential
            model = Sequential()

            # Embedding layer (frozen)
            model.add(layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                weights=[embedding_matrix],
                input_length=self.config.max_len,
                trainable=self.config.trainable_embeddings
            ))

            # Spatial dropout
            model.add(layers.SpatialDropout1D(self.config.spatial_dropout))

            # RNN layer
            if self.config.model_type == 'lstm':
                model.add(layers.LSTM(
                    units=self.config.rnn_units,
                    dropout=self.config.dropout,
                    recurrent_dropout=self.config.recurrent_dropout
                ))
            else:  # GRU
                model.add(layers.GRU(
                    units=self.config.rnn_units,
                    dropout=self.config.dropout,
                    recurrent_dropout=self.config.recurrent_dropout
                ))

            # Output layer
            model.add(layers.Dense(self.config.num_classes, activation='softmax'))

        else:  # Bidirectional models
            from tensorflow.keras.models import Sequential
            model = Sequential()

            model.add(layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                weights=[embedding_matrix],
                input_length=self.config.max_len,
                trainable=self.config.trainable_embeddings
            ))

            model.add(layers.SpatialDropout1D(self.config.spatial_dropout))

            if 'lstm' in self.config.model_type:
                rnn = layers.LSTM(self.config.rnn_units, dropout=self.config.dropout)
            else:
                rnn = layers.GRU(self.config.rnn_units, dropout=self.config.dropout)

            model.add(layers.Bidirectional(rnn))
            model.add(layers.Dense(self.config.num_classes, activation='softmax'))

        # Compile
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info(f"  Model built: {self.config.model_type.upper()}, Parameters: {model.count_params():,}")
        return model


def run_experiment(config: ExperimentConfig) -> Dict:
    """Run a complete experiment"""
    logger.info("="*80)
    logger.info(f"EXPERIMENT: {config.experiment_name}")
    logger.info("="*80)

    # Set random seeds
    np.random.seed(config.random_seed)
    tf.random.set_seed(config.random_seed)

    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"GPUs Available: {len(gpus)}")
    if gpus:
        for gpu in gpus:
            logger.info(f"  {gpu}")
            tf.config.experimental.set_memory_growth(gpu, True)

    # Create directories
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)

    # Load data
    logger.info("Step 1: Loading data...")
    train_df = pd.read_csv(config.train_path)
    val_df = pd.read_csv(config.val_path)
    logger.info(f"  Train: {len(train_df)}, Val: {len(val_df)}")

    # Preprocess (EXACT pipeline)
    logger.info("Step 2: Preprocessing (full_pipeline exact method)...")
    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)

    # Check data leakage
    train_df = check_data_leakage(train_df, val_df)

    train_texts = train_df['text'].values.tolist()
    val_texts = val_df['text'].values.tolist()
    y_train = train_df['label'].values
    y_val = val_df['label'].values

    # Create embeddings
    logger.info(f"Step 3: Creating {config.embedding_type.upper()} embeddings...")
    embedding_handler = EmbeddingHandler(config)
    embedding_handler.create_tokenizer(train_texts)

    if config.embedding_type == 'glove':
        embedding_handler.load_glove_embeddings()
    elif config.embedding_type == 'word2vec':
        embedding_handler.train_word2vec(train_texts)

    embedding_matrix = embedding_handler.create_embedding_matrix()

    # Convert to sequences
    logger.info("Step 4: Converting to sequences...")
    X_train = embedding_handler.texts_to_sequences(train_texts)
    X_val = embedding_handler.texts_to_sequences(val_texts)

    y_train_cat = to_categorical(y_train, num_classes=config.num_classes)
    y_val_cat = to_categorical(y_val, num_classes=config.num_classes)

    logger.info(f"  X_train shape: {X_train.shape}, y_train shape: {y_train_cat.shape}")

    # Compute class weights
    class_weights = None
    if config.use_class_weights:
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        logger.info(f"  Class weights: {class_weights}")

    # Build model
    logger.info("Step 5: Building model...")
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
    logger.info(f"Step 6: Training {config.model_type.upper()} model...")
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
    logger.info(f"  Training completed in {training_time:.2f} seconds")

    # Evaluate
    logger.info("Step 7: Evaluating...")
    y_val_pred = model.predict(X_val, verbose=0)
    y_val_pred_labels = y_val_pred.argmax(axis=1)

    val_accuracy = accuracy_score(y_val, y_val_pred_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_val, y_val_pred_labels, labels=list(range(config.num_classes))
    )

    macro_f1 = np.mean(f1)

    # Results
    results = {
        'experiment_name': config.experiment_name,
        'model_type': config.model_type,
        'embedding_type': config.embedding_type,
        'embedding_dim': config.embedding_dim,
        'rnn_units': config.rnn_units,
        'dropout': config.dropout,
        'val_accuracy': float(val_accuracy),
        'val_loss': float(history.history['val_loss'][-1]),
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
    logger.info(f"COMPLETED: {config.experiment_name}")
    logger.info(f"  Val Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    logger.info(f"  Macro F1: {macro_f1:.4f}")
    logger.info(f"  Training Time: {training_time:.2f}s")
    logger.info("="*80)

    return results


def main():
    """Run all experiments"""
    logger.info("="*80)
    logger.info("GPU EMOTION DETECTION - EXACT PIPELINE")
    logger.info("="*80)

    # Check TensorFlow GPU
    logger.info(f"TensorFlow: {tf.__version__}")
    logger.info(f"GPU Devices: {tf.config.list_physical_devices('GPU')}")

    # Define experiments (using 50d embeddings like your pipeline)
    experiments = [
        # Core comparisons
        ExperimentConfig(
            experiment_name="lstm_glove50_baseline",
            model_type="lstm",
            embedding_type="glove",
            embedding_dim=50,
            rnn_units=128
        ),

        ExperimentConfig(
            experiment_name="gru_glove50_baseline",
            model_type="gru",
            embedding_type="glove",
            embedding_dim=50,
            rnn_units=128
        ),

        ExperimentConfig(
            experiment_name="lstm_word2vec_baseline",
            model_type="lstm",
            embedding_type="word2vec",
            embedding_dim=50,
            rnn_units=128
        ),

        ExperimentConfig(
            experiment_name="gru_word2vec_baseline",
            model_type="gru",
            embedding_type="word2vec",
            embedding_dim=50,
            rnn_units=128
        ),

        ExperimentConfig(
            experiment_name="bilstm_glove50",
            model_type="bilstm",
            embedding_type="glove",
            embedding_dim=50,
            rnn_units=128
        ),

        # Hyperparameter variations
        ExperimentConfig(
            experiment_name="lstm_glove50_256units",
            model_type="lstm",
            embedding_type="glove",
            embedding_dim=50,
            rnn_units=256
        ),

        ExperimentConfig(
            experiment_name="lstm_glove50_dropout05",
            model_type="lstm",
            embedding_type="glove",
            embedding_dim=50,
            rnn_units=128,
            dropout=0.5
        ),
    ]

    # Run all experiments
    all_results = []
    for i, config in enumerate(experiments, 1):
        logger.info(f"\n\nEXPERIMENT {i}/{len(experiments)}: {config.experiment_name}")
        try:
            results = run_experiment(config)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Experiment {config.experiment_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create comparison table
    logger.info("\n\n" + "="*80)
    logger.info("ALL EXPERIMENTS COMPLETED - COMPARISON TABLE")
    logger.info("="*80)

    comparison_df = pd.DataFrame(all_results)
    comparison_df = comparison_df.sort_values('val_accuracy', ascending=False)

    print("\n" + comparison_df[['experiment_name', 'model_type', 'embedding_type',
                                  'val_accuracy', 'macro_f1', 'training_time']].to_string(index=False))

    # Save comparison
    comparison_path = os.path.join('results', 'all_experiments_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"\nComparison saved to: {comparison_path}")

    # Best model
    best_exp = comparison_df.iloc[0]
    logger.info("\n" + "="*80)
    logger.info("BEST MODEL")
    logger.info("="*80)
    logger.info(f"Experiment: {best_exp['experiment_name']}")
    logger.info(f"Model: {best_exp['model_type'].upper()}")
    logger.info(f"Embedding: {best_exp['embedding_type'].upper()} {best_exp['embedding_dim']}d")
    logger.info(f"Accuracy: {best_exp['val_accuracy']:.4f} ({best_exp['val_accuracy']*100:.2f}%)")
    logger.info(f"Macro F1: {best_exp['macro_f1']:.4f}")
    logger.info("="*80)

    logger.info("\nAll experiments completed successfully!")


if __name__ == "__main__":
    main()
