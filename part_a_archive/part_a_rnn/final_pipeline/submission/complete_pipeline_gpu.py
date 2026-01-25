#!/usr/bin/env python
# coding: utf-8

# # Emotion Detection Pipeline
# ## Professional Deep Learning System with Full EDA and Features
# ## Ofek Raban Ron Gabay
# 
# **This notebook includes:**
# - EDA and preprocessing 
# -  All  classes and functions (self-contained)
# -  Advanced features: ablation studies, detailed logging, model comparison
# -  Complete experiment tracking and reproducibility
# -  Comprehensive visualizations and metrics
# 
# ### Emotion Classes:
# 0. Sadness  | 1. Joy  | 2. Love  | 3. Anger  | 4. Fear  | 5. Surprise 

# ##  Section 1: Imports and Setup

# In[2]:


# Standard library
import os
import re
import json
import time
import pickle
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from collections import Counter

# Data processing
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Machine Learning
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_fscore_support, accuracy_score
)

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, CSVLogger, Callback
)

# Embeddings
from gensim.models import Word2Vec
import emoji

# Settings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Random seed set to: {RANDOM_SEED}")
print(" All imports successful!")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"Random Seed: {RANDOM_SEED}")


# ##  Section 2: Advanced Configuration

# In[3]:


@dataclass
class ExperimentConfig:
    """Complete configuration with ablation flags and advanced options."""
    
    # ========== Experiment Info ==========
    experiment_name: str = "ultimate_emotion_detection"
    random_seed: int = RANDOM_SEED
    
    # ========== Data Paths ==========
    train_path: str = "/home/lab/rabanof/projects/Emotion_Detection_DL/data/raw/train.csv"
    val_path: str = "/home/lab/rabanof/projects/Emotion_Detection_DL/data/raw/validation.csv"
    glove_path: str = "/home/lab/rabanof/Emotion_Detection_DL/glove/glove.6B.100d.txt"
    
    # ========== Data Parameters ==========
    max_len: int = 60
    max_words: int = 20000
    text_column: str = "text"
    label_column: str = "label"
    num_classes: int = 6
    
    # ========== Preprocessing Ablation Flags ==========
    enable_aggressive_normalization: bool = True  # Slang, typos, etc.
    enable_elongation_normalization: bool = True  # sooo -> soo
    enable_contraction_expansion: bool = True     # don't -> do not
    
    # ========== Embedding Configuration ==========
    embedding_type: str = "glove"  # 'glove' or 'word2vec'
    embedding_dim: int = 100
    trainable_embeddings: bool = False  # Ablation: True vs False
    oov_token: str = "<UNK>"
    oov_init_std: float = 0.1  # Std dev for OOV random initialization
    
    # ========== Model Architecture ==========
    model_type: str = "lstm"  # 'lstm', 'gru', 'bilstm', 'bigru'
    rnn_units: int = 128
    num_rnn_layers: int = 1  # Number of recurrent layers
    
    # ========== Regularization ==========
    spatial_dropout: float = 0.2  # After embedding
    dropout: float = 0.2          # After RNN
    recurrent_dropout: float = 0.0  # Within RNN (set to 0 for GPU efficiency)
    use_layer_norm: bool = False   # Layer normalization after RNN
    
    # ========== Training ==========
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    use_class_weights: bool = True
    
    # ========== Callbacks ==========
    early_stopping: bool = True
    patience: int = 5
    reduce_lr: bool = True
    lr_factor: float = 0.5
    lr_patience: int = 3
    min_lr: float = 1e-7
    
    # ========== Logging & Saving ==========
    verbose: int = 1
    save_tokenizer: bool = True
    save_config: bool = True
    save_embedding_matrix: bool = True
    
    # ========== Directories ==========
    save_dir: str = "saved_models"
    log_dir: str = "logs"
    result_dir: str = "results"
    config_dir: str = "configs"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def save_to_json(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {filepath}")

# Create default configuration
config = ExperimentConfig()

# Log configuration
logger.info("="*80)
logger.info("EXPERIMENT CONFIGURATION")
logger.info("="*80)
logger.info(f"Experiment: {config.experiment_name}")
logger.info(f"Model: {config.model_type.upper()}")
logger.info(f"RNN Units: {config.rnn_units}, Layers: {config.num_rnn_layers}")
logger.info(f"Embedding: {config.embedding_type.upper()}, Dim: {config.embedding_dim}, Trainable: {config.trainable_embeddings}")
logger.info(f"Ablation - Aggressive Norm: {config.enable_aggressive_normalization}")
logger.info(f"Ablation - Elongation Norm: {config.enable_elongation_normalization}")
logger.info(f"Random Seed: {config.random_seed}")
logger.info("="*80)

print("\n Configuration created!")
print(f" Experiment: {config.experiment_name}")
print(f" Model: {config.model_type.upper()} ({config.rnn_units} units, {config.num_rnn_layers} layers)")
print(f" Embedding: {config.embedding_type.upper()} {config.embedding_dim}d (Trainable: {config.trainable_embeddings})")


# ---
# 
# ##  How to Use This Notebook
# 
# ### Quick Start:
# 1. **Run All Cells**: Execute cells in order from top to bottom
# 2. **Modify Config**: Change parameters in Section 2 to experiment
# 3. **Re-run Sections**: After changing config, re-run relevant sections
# 
# ### Sections Overview:
# - **Sections 1-2**: Setup and configuration (always run first)
# - **Sections 3-7**: Professional classes (run once)
# - **Sections 8-13**: Data loading and EDA (run once)
# - **Sections 14**: Preprocessing (run once per config change)
# - **Sections 15-16**: Tokenization and embeddings (re-run if embedding changes)
# - **Sections 17-19**: Model building and training (re-run for each experiment)
# - **Sections 20-23**: Evaluation (re-run after each training)
# - **Sections 24-25**: Model comparison (run to compare experiments)
# - **Sections 26-27**: Predictions and testing (run anytime after training)
# - **Section 28**: Final summary
# 
# ### Hyperparameter Experiments:
# To compare different configurations:
# ```python
# # Experiment 1: LSTM with GloVe
# config.model_type = 'lstm'
# config.embedding_type = 'glove'
# config.rnn_units = 128
# # Run sections 15-23
# 
# # Experiment 2: GRU with Word2Vec  
# config.model_type = 'gru'
# config.embedding_type = 'word2vec'
# config.rnn_units = 128
# config.experiment_name = 'gru_word2vec'
# # Run sections 15-23
# # Add to comparer: comparer.add_experiment(...)
# 
# # View comparison
# comparer.create_comparison_table()
# ```
# 
# ### Ablation Studies:
# Test impact of preprocessing:
# ```python
# config.enable_elongation_normalization = False
# config.enable_contraction_expansion = False
# # Re-run from Section 14
# ```
# 
# ### Key Configuration Parameters:
# 
# **Model Architecture:**
# - `model_type`: 'lstm', 'gru', 'bilstm', 'bigru'
# - `rnn_units`: 64, 128, 256
# - `num_rnn_layers`: 1, 2, 3
# - `use_layer_norm`: True/False
# 
# **Embeddings:**
# - `embedding_type`: 'glove', 'word2vec'
# - `embedding_dim`: 50, 100, 200, 300
# - `trainable_embeddings`: True/False
# 
# **Regularization:**
# - `dropout`: 0.0 to 0.5
# - `spatial_dropout`: 0.0 to 0.5
# - `use_class_weights`: True/False
# 
# **Training:**
# - `epochs`: 20-100
# - `batch_size`: 16, 32, 64
# - `learning_rate`: 0.0001 to 0.01
# - `patience`: 3-10 (early stopping)
# 
# ---

# ---
# #  HYPERPARAMETER EXPERIMENTATION
# 
# **To compare different models:**
# 1. Modify the configuration in Section 2 (change model_type, rnn_units, embedding_type, etc.)
# 2. Re-run all cells from Section 15 onwards
# 3. Use the ModelComparer below to compare results

# ## Section 3: Advanced Text Preprocessor with Statistics

# In[5]:


class AdvancedTextPreprocessor:
    
    # Advanced text preprocessing with ablation flags and statistics logging.
    
    
    def __init__(self, config: ExperimentConfig):
        self.config = config

        # specific contractions by examaples we saw in the data
        self.specific_contractions = {
            "didnt": "did not", "dont": "do not", "cant": "cannot",
            "wont": "will not", "wouldnt": "would not", "shouldnt": "should not",
            "couldnt": "could not", "im": "i am", "ive": "i have",
            "id": "i would", "ill": "i will", "hadnt": "had not",
            "youve": "you have", "werent": "were not", "theyve": "they have",
            "theyll": "they will", "itll": "it will", "couldve": "could have",
            "shouldve": "should have", "wouldve": "would have", "hadn": "had not"
        }
        # General contraction patterns
        self.general_contractions = {
            "n't": " not", "'re": " are", "'s": " is",
            "'d": " would", "'ll": " will", "'t": " not",
            "'ve": " have", "'m": " am"
        }
        
        # slang and typo corrections
        self.slang_corrections = {
            "idk": "i do not know", "yknow": "you know",
            "becuz": "because", "alittle": "a little", "incase": "in case"
        }
        # corrections for common typos
        self.typo_corrections = {
            "vunerable": "vulnerable", "percieve": "perceive",
            "definetly": "definitely", "writting": "writing"
        }
        
        # Statistics
        self.stats = {
            'tokens_before': 0,
            'tokens_after': 0,
            'texts_modified': 0,
            'total_texts': 0
        }
    
    def clean_text(self, text: str) -> str:

        # Apply comprehensive text cleaning with ablation flags.
        
        #check if text is empty
        if not isinstance(text, str):
            return ""
        
        original_text = text
        
        # Lowercase
        text = text.lower()
        
        # Elongation normalization (e.g., sooo -> soo) , save the intense feeling, for 2 chars.
        if self.config.enable_elongation_normalization:
            text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Contraction expansion (e.g., don't -> do not)
        if self.config.enable_contraction_expansion:
            # Specific contractions first (word boundaries)
            for key, value in self.specific_contractions.items():
                text = re.sub(rf'\b{re.escape(key)}\b', value, text)   #re.sub(pattern, replacement, text)
            
            # General patterns
            for key, value in self.general_contractions.items():
                text = text.replace(key, value)
        
        # Aggressive normalization (slang + typos)
        if self.config.enable_aggressive_normalization:
            for corrections in [self.slang_corrections, self.typo_corrections]:
                for key, value in corrections.items():
                    text = re.sub(rf'\b{re.escape(key)}\b', value, text)
        
        # Reduce repeated punctuation (e.g., !!! -> !)
        text = re.sub(r"([!?.,])\1+", r"\1", text)
        text = re.sub(r"\.{2,}", ".", text)
        
        # Normalize whitespace (remove extra spaces)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Track if modified
        if text != original_text.lower().strip():
            self.stats['texts_modified'] += 1
        
        return text
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # Preprocess dataframe and collect statistics.
        
        # save a copy to avoid modifying original
        df = df.copy()
        self.stats['total_texts'] = len(df)
        
        # log start
        logger.info(f"Preprocessing {len(df)} samples...")
        
        # Count tokens before
        tokens_before = df[self.config.text_column].str.split().str.len().sum()
        self.stats['tokens_before'] = tokens_before
        
        # Apply cleaning
        df[self.config.text_column] = df[self.config.text_column].apply(self.clean_text)
        
        # Count tokens after
        tokens_after = df[self.config.text_column].str.split().str.len().sum()
        self.stats['tokens_after'] = tokens_after
        
        # Add text length column
        df['text_len'] = df[self.config.text_column].str.split().str.len()
        
        # Log statistics
        avg_before = tokens_before / len(df)
        avg_after = tokens_after / len(df)
        pct_modified = (self.stats['texts_modified'] / len(df)) * 100
        
        logger.info(f"Preprocessing Statistics:")
        logger.info(f"  Avg tokens before: {avg_before:.2f}")
        logger.info(f"  Avg tokens after: {avg_after:.2f}")
        logger.info(f"  Texts modified: {self.stats['texts_modified']} ({pct_modified:.1f}%)")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate texts."""

        initial_len = len(df)
        df = df.drop_duplicates(subset=[self.config.text_column], keep='first')
        removed = initial_len - len(df)
        if removed > 0:
            logger.warning(f"Removed {removed} duplicates")
        return df.reset_index(drop=True)
    
    def check_data_leakage(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Check and remove overlapping texts."""
        train_texts = set(train_df[self.config.text_column])
        val_texts = set(val_df[self.config.text_column])
        overlaps = val_texts.intersection(train_texts)
        
        if len(overlaps) > 0:
            logger.warning(f"Data leakage: {len(overlaps)} overlapping texts found")
            val_df_clean = val_df[~val_df[self.config.text_column].isin(overlaps)].copy()
            return val_df_clean.reset_index(drop=True), len(overlaps)
        """"avoid data leakage, overlapping samples are removed from the validation set rather than the training set, 
        in order to preserve the integrity and size of the training data while ensuring a fair and unbiased evaluation on the validation set."""
        
        logger.info("No data leakage detected")
        return val_df, 0
    
    def compute_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """Compute balanced class weights."""
        # Calculate class weights to handle class imbalance, using sklearn utility for weighted loss.
        classes = np.unique(labels)
        weights = compute_class_weight('balanced', classes=classes, y=labels)
        class_weights = dict(zip(classes, weights))
        
        logger.info(f"Class weights computed: {class_weights}")
        return class_weights
    
    def log_class_distribution(self, labels: np.ndarray, emotion_map: Dict):
        """
        Log class only for distribution with counts, percentages, and imbalance ratio.
        """
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        logger.info("="*80)
        logger.info("CLASS DISTRIBUTION")
        logger.info("="*80)
        
        distribution_data = []
        for label, count in zip(unique, counts):
            pct = (count / total) * 100
            distribution_data.append({
                'label': label,
                'emotion': emotion_map[label],
                'count': count,
                'percentage': pct
            })
            logger.info(f"  {emotion_map[label]:12s}: {count:5d} ({pct:5.2f}%)")
        
        # Calculate imbalance ratio
        max_count = counts.max()
        min_count = counts.min()
        imbalance_ratio = max_count / min_count
        
        logger.info(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
        logger.info(f"(Max: {max_count}, Min: {min_count})")
        logger.info("="*80)
        
        return pd.DataFrame(distribution_data)

print(" Advanced Text Preprocessor class created!")


# ##  Section 4: Advanced Embedding Handler

# ##  Section 5: Advanced Model Builder

# In[ ]:


class AdvancedEmbeddingHandler:
    """
    Advanced embedding handler with GloVe/Word2Vec support and detailed analytics.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tokenizer = None
        self.embedding_matrix = None
        self.embeddings_index = {}
        self.word2vec_model = None
        
        # Statistics
        self.stats = {
            'vocab_size': 0,
            'coverage_count': 0,
            'coverage_percent': 0.0,
            'oov_count': 0,
            'oov_percent': 0.0,
            'oov_words': []
        }
    
    def create_tokenizer(self, texts: List[str]) -> Tokenizer:
        """
        Create and fit tokenizer on texts.
        """
        logger.info(f"Creating tokenizer with max_words={self.config.max_words}...")
        
        self.tokenizer = Tokenizer(
            num_words=self.config.max_words,
            oov_token=self.config.oov_token,
            lower=True
        )
        self.tokenizer.fit_on_texts(texts)
        
        word_index = self.tokenizer.word_index
        self.stats['vocab_size'] = min(len(word_index), self.config.max_words)
        
        logger.info(f"Tokenizer created. Vocabulary size: {self.stats['vocab_size']}")
        logger.info(f"Total unique words: {len(word_index)}")
        
        return self.tokenizer
    
    def texts_to_sequences(self, texts: List[str]) -> np.ndarray:
        """
        Convert texts to padded sequences.
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.config.max_len, padding='post', truncating='post')
        
        logger.info(f"Converted {len(texts)} texts to sequences of length {self.config.max_len}")
        
        return padded
    
    def analyze_sequence_lengths(self, texts: List[str]) -> Dict:
        """
        Analyze sequence lengths and truncation.
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        lengths = [len(seq) for seq in sequences]
        
        truncated = sum(1 for l in lengths if l > self.config.max_len)
        truncation_pct = (truncated / len(lengths)) * 100
        
        stats = {
            'mean_length': np.mean(lengths),
            'median_length': np.median(lengths),
            'max_length': max(lengths),
            'min_length': min(lengths),
            'truncated_count': truncated,
            'truncation_percent': truncation_pct
        }
        
        logger.info(f"Sequence Length Analysis:")
        logger.info(f"  Mean: {stats['mean_length']:.2f}, Median: {stats['median_length']:.0f}")
        logger.info(f"  Max: {stats['max_length']}, Min: {stats['min_length']}")
        logger.info(f"  Truncated: {truncated} ({truncation_pct:.1f}%)")
        
        return stats, lengths
    
    def load_glove_embeddings(self, glove_path: str):
        """
        Load pre-trained GloVe embeddings.
        """
        logger.info(f"Loading GloVe embeddings from {glove_path}...")
        
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = vector
        
        logger.info(f"Loaded {len(self.embeddings_index)} word vectors")
    
    def train_word2vec(self, texts: List[str], min_count: int = 1):
        """
        Train Word2Vec embeddings on the corpus.
        """
        logger.info("Training Word2Vec embeddings...")
        
        # Tokenize texts into word lists
        tokenized_texts = [text.split() for text in texts]
        
        # Train Word2Vec
        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.config.embedding_dim,
            window=5,
            min_count=min_count,
            workers=4,
            seed=self.config.random_seed
        )
        
        logger.info(f"Word2Vec trained on {len(texts)} texts")
        logger.info(f"Vocabulary size: {len(self.word2vec_model.wv)}")
    
    def create_embedding_matrix(self) -> np.ndarray:
        """
        Create embedding matrix from GloVe or Word2Vec.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be created first")
        
        word_index = self.tokenizer.word_index
        vocab_size = min(len(word_index) + 1, self.config.max_words + 1)
        
        # Initialize with small random values
        self.embedding_matrix = np.random.normal(
            0, self.config.oov_init_std, 
            (vocab_size, self.config.embedding_dim)
        )
        
        # Set padding vector to zeros
        self.embedding_matrix[0] = np.zeros(self.config.embedding_dim)
        
        # Fill with pre-trained vectors
        found_count = 0
        oov_words = []
        
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
            else:
                oov_words.append(word)
        
        # Update statistics
        self.stats['coverage_count'] = found_count
        self.stats['coverage_percent'] = (found_count / (vocab_size - 1)) * 100
        self.stats['oov_count'] = len(oov_words)
        self.stats['oov_percent'] = (len(oov_words) / (vocab_size - 1)) * 100
        self.stats['oov_words'] = oov_words[:100]  # Store first 100
        
        logger.info(f"Embedding Matrix Created:")
        logger.info(f"  Shape: {self.embedding_matrix.shape}")
        logger.info(f"  Coverage: {found_count}/{vocab_size-1} ({self.stats['coverage_percent']:.2f}%)")
        logger.info(f"  OOV: {len(oov_words)} ({self.stats['oov_percent']:.2f}%)")
        
        return self.embedding_matrix
    
    def get_oov_rate(self, sequences: np.ndarray) -> float:
        """
        Calculate OOV rate in sequences.
        """
        if self.tokenizer is None:
            return 0.0
        
        oov_index = self.tokenizer.word_index.get(self.config.oov_token, 1)
        total_tokens = sequences.size
        oov_tokens = np.sum(sequences == oov_index)
        
        oov_rate = (oov_tokens / total_tokens) * 100
        return oov_rate
    
    def save_tokenizer(self, filepath: str):
        """Save tokenizer to JSON file."""
        if self.tokenizer is None:
            logger.warning("No tokenizer to save")
            return
        
        tokenizer_json = self.tokenizer.to_json()
        with open(filepath, 'w') as f:
            f.write(tokenizer_json)
        logger.info(f"Tokenizer saved to {filepath}")
    
    def save_embedding_matrix(self, filepath: str):
        """Save embedding matrix to numpy file."""
        if self.embedding_matrix is None:
            logger.warning("No embedding matrix to save")
            return
        
        np.save(filepath, self.embedding_matrix)
        logger.info(f"Embedding matrix saved to {filepath}")

print(" AdvancedEmbeddingHandler class created!")


# ##  Section 6: Results Visualizer

# In[ ]:


class AdvancedModelBuilder:
    """
    Advanced model builder supporting LSTM, GRU, and Bidirectional variants.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
    
    def build_model(self, embedding_matrix: np.ndarray) -> keras.Model:
        """
        Build model based on configuration.
        """
        logger.info(f"Building {self.config.model_type.upper()} model...")
        
        vocab_size, embedding_dim = embedding_matrix.shape
        
        # Input layer
        inputs = layers.Input(shape=(self.config.max_len,), name='input')
        
        # Embedding layer
        x = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=self.config.max_len,
            trainable=self.config.trainable_embeddings,
            name='embedding'
        )(inputs)
        
        # Spatial dropout after embedding
        if self.config.spatial_dropout > 0:
            x = layers.SpatialDropout1D(self.config.spatial_dropout, name='spatial_dropout')(x)
        
        # Recurrent layers
        for layer_idx in range(self.config.num_rnn_layers):
            return_sequences = (layer_idx < self.config.num_rnn_layers - 1)
            
            # Choose RNN type
            if self.config.model_type in ['lstm', 'bilstm']:
                rnn_layer = layers.LSTM(
                    units=self.config.rnn_units,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout if layer_idx == 0 else 0,
                    recurrent_dropout=self.config.recurrent_dropout,
                    name=f'lstm_{layer_idx+1}'
                )
            elif self.config.model_type in ['gru', 'bigru']:
                rnn_layer = layers.GRU(
                    units=self.config.rnn_units,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout if layer_idx == 0 else 0,
                    recurrent_dropout=self.config.recurrent_dropout,
                    name=f'gru_{layer_idx+1}'
                )
            else:
                raise ValueError(f"Unknown model type: {self.config.model_type}")
            
            # Apply bidirectional wrapper if needed
            if self.config.model_type in ['bilstm', 'bigru']:
                rnn_layer = layers.Bidirectional(rnn_layer, name=f'bidirectional_{layer_idx+1}')
            
            x = rnn_layer(x)
            
            # Layer normalization if enabled
            if self.config.use_layer_norm:
                x = layers.LayerNormalization(name=f'layer_norm_{layer_idx+1}')(x)
            
            # Dropout after RNN (except for last layer)
            if self.config.dropout > 0 and layer_idx < self.config.num_rnn_layers - 1:
                x = layers.Dropout(self.config.dropout, name=f'dropout_{layer_idx+1}')(x)
        
        # Output layer
        outputs = layers.Dense(
            self.config.num_classes,
            activation='softmax',
            name='output'
        )(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name=f'{self.config.model_type}_model')
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model built successfully!")
        logger.info(f"  Type: {self.config.model_type.upper()}")
        logger.info(f"  Units: {self.config.rnn_units}, Layers: {self.config.num_rnn_layers}")
        logger.info(f"  Trainable embeddings: {self.config.trainable_embeddings}")
        logger.info(f"  Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def get_model_summary(self) -> str:
        """Get model summary as string."""
        if self.model is None:
            return "No model built yet"
        
        # Capture summary
        from io import StringIO
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()

print(" AdvancedModelBuilder class created!")


# ##  Section 7: Experiment Tracker and Callback

# In[ ]:


class ResultsVisualizer:
    """
    Comprehensive visualization class for model results and analysis.
    """
    
    def __init__(self, emotion_map: Dict[int, str]):
        self.emotion_map = emotion_map
        self.emotion_names = [emotion_map[i] for i in sorted(emotion_map.keys())]
    
    def plot_training_history(self, history, save_path: Optional[str] = None):
        """Plot training and validation accuracy/loss curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=False, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        # Convert one-hot to labels if needed
        if len(y_true.shape) > 1:
            y_true = y_true.argmax(axis=1)
        if len(y_pred.shape) > 1:
            y_pred = y_pred.argmax(axis=1)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                    xticklabels=self.emotion_names, yticklabels=self.emotion_names,
                    cbar_kws={'label': 'Percentage' if normalize else 'Count'})
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_classification_report(self, y_true, y_pred, save_path: Optional[str] = None):
        """Plot classification report as heatmap."""
        # Convert one-hot to labels if needed
        if len(y_true.shape) > 1:
            y_true = y_true.argmax(axis=1)
        if len(y_pred.shape) > 1:
            y_pred = y_pred.argmax(axis=1)
        
        # Get metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(self.emotion_names)))
        )
        
        # Create dataframe
        metrics_df = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }, index=self.emotion_names)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.heatmap(metrics_df.T, annot=True, fmt='.3f', cmap='RdYlGn', 
                    vmin=0, vmax=1, cbar_kws={'label': 'Score'})
        plt.title('Classification Report by Emotion', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Metric', fontsize=12)
        plt.xlabel('Emotion', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return metrics_df
    
    def plot_per_class_metrics(self, y_true, y_pred, metric='f1', save_path: Optional[str] = None):
        """Plot per-class metrics as bar chart."""
        # Convert one-hot to labels if needed
        if len(y_true.shape) > 1:
            y_true = y_true.argmax(axis=1)
        if len(y_pred.shape) > 1:
            y_pred = y_pred.argmax(axis=1)
        
        # Get metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(self.emotion_names)))
        )
        
        # Select metric
        metric_map = {
            'precision': ('Precision', precision),
            'recall': ('Recall', recall),
            'f1': ('F1-Score', f1)
        }
        
        metric_name, values = metric_map.get(metric.lower(), ('F1-Score', f1))
        
        # Plot
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.emotion_names)))
        bars = plt.bar(self.emotion_names, values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title(f'Per-Class {metric_name}', fontsize=14, fontweight='bold')
        plt.ylabel(metric_name, fontsize=12)
        plt.xlabel('Emotion', fontsize=12)
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_label_distribution(self, labels, title='Label Distribution', save_path: Optional[str] = None):
        """Plot label distribution."""
        unique, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
        bars = plt.bar([self.emotion_map[i] for i in unique], counts, color=colors, 
                      edgecolor='black', linewidth=1.5)
        
        # Add percentages
        total = sum(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            pct = (count / total) * 100
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('Count', fontsize=12)
        plt.xlabel('Emotion', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_text_length_distribution(self, text_lengths, max_len=None, save_path: Optional[str] = None):
        """Plot text length distribution."""
        plt.figure(figsize=(12, 5))
        
        plt.hist(text_lengths, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        plt.axvline(np.mean(text_lengths), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(text_lengths):.1f}')
        plt.axvline(np.median(text_lengths), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(text_lengths):.0f}')
        
        if max_len:
            plt.axvline(max_len, color='orange', linestyle='--', 
                       linewidth=2, label=f'MAX_LEN: {max_len}')
        
        plt.title('Text Length Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Text Length (words)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_word_cloud(self, texts, title='Word Cloud', save_path: Optional[str] = None):
        """Generate and plot word cloud."""
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(combined_text)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

print(" ResultsVisualizer class created!")


# ##  Section 8: Load Data

# ---
# #  DATA LOADING AND EXPLORATION
# 
# This section includes all original EDA from your full_pipeline.ipynb

# In[ ]:


class ExperimentTracker(Callback):
    """
    Custom Keras callback to track experiment metrics and training time.
    """
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.experiment_results = {
            'config': config.to_dict(),
            'training_time_per_epoch': [],
            'total_training_time': 0,
            'best_val_accuracy': 0,
            'best_epoch': 0
        }
        self.epoch_start_time = None
    
    def on_epoch_begin(self, epoch, logs=None):
        """Record epoch start time."""
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        """Record epoch metrics and time."""
        epoch_time = time.time() - self.epoch_start_time
        self.experiment_results['training_time_per_epoch'].append(epoch_time)
        
        # Track best validation accuracy
        val_acc = logs.get('val_accuracy', 0)
        if val_acc > self.experiment_results['best_val_accuracy']:
            self.experiment_results['best_val_accuracy'] = val_acc
            self.experiment_results['best_epoch'] = epoch + 1
        
        if self.config.verbose:
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s - "
                       f"val_acc: {val_acc:.4f}")
    
    def on_train_end(self, logs=None):
        """Record total training time."""
        self.experiment_results['total_training_time'] = sum(
            self.experiment_results['training_time_per_epoch']
        )
        
        logger.info(f"Training completed!")
        logger.info(f"  Total time: {self.experiment_results['total_training_time']:.2f}s")
        logger.info(f"  Best val accuracy: {self.experiment_results['best_val_accuracy']:.4f} "
                   f"at epoch {self.experiment_results['best_epoch']}")
    
    def get_results(self) -> Dict:
        """Get experiment results."""
        return self.experiment_results
    
    def save_results(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.experiment_results, f, indent=2)
        logger.info(f"Experiment results saved to {filepath}")


class ModelComparer:
    """
    Compare multiple model configurations and results.
    """
    
    def __init__(self, emotion_map: Dict[int, str]):
        self.emotion_map = emotion_map
        self.experiments = []
    
    def add_experiment(self, name: str, config: ExperimentConfig, history, 
                      metrics: Dict, predictions=None, y_true=None):
        """Add experiment results for comparison."""
        experiment = {
            'name': name,
            'config': config.to_dict(),
            'history': history.history if hasattr(history, 'history') else history,
            'metrics': metrics,
            'predictions': predictions,
            'y_true': y_true
        }
        self.experiments.append(experiment)
        logger.info(f"Added experiment: {name}")
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create comparison table of all experiments."""
        comparison_data = []
        
        for exp in self.experiments:
            config = exp['config']
            metrics = exp['metrics']
            
            comparison_data.append({
                'Experiment': exp['name'],
                'Model': config['model_type'].upper(),
                'RNN Units': config['rnn_units'],
                'Layers': config['num_rnn_layers'],
                'Embedding': config['embedding_type'].upper(),
                'Embed Dim': config['embedding_dim'],
                'Trainable Emb': config['trainable_embeddings'],
                'Dropout': config['dropout'],
                'Val Accuracy': metrics.get('val_accuracy', 0),
                'Val Loss': metrics.get('val_loss', 0),
                'Macro F1': metrics.get('macro_f1', 0),
                'Training Time': metrics.get('training_time', 0)
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('Val Accuracy', ascending=False)
    
    def plot_comparison(self, metric='val_accuracy', save_path: Optional[str] = None):
        """Plot comparison of experiments."""
        if not self.experiments:
            print("No experiments to compare")
            return
        
        names = [exp['name'] for exp in self.experiments]
        values = [exp['metrics'].get(metric, 0) for exp in self.experiments]
        
        plt.figure(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        bars = plt.bar(names, values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title(f'Model Comparison: {metric}', fontsize=14, fontweight='bold')
        plt.ylabel(metric, fontsize=12)
        plt.xlabel('Experiment', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_comparison(self, filepath: str):
        """Save comparison table to CSV."""
        df = self.create_comparison_table()
        df.to_csv(filepath, index=False)
        logger.info(f"Comparison table saved to {filepath}")

print(" ExperimentTracker and ModelComparer classes created!")


# ##  Section 9: Class Distribution Analysis

# In[ ]:


# Emotion mapping (your original mapping)
emotion_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

# Load datasets
logger.info("Loading datasets...")
train_df = pd.read_csv(config.train_path)
val_df = pd.read_csv(config.val_path)

print("="*80)
print("TRAINING DATA")
print("="*80)
print(f"Shape: {train_df.shape}")
print(f"\nFirst 5 rows:")
print(train_df.head())
print(f"\nData Info:")
print(train_df.info())
print(f"\nMissing values:")
print(train_df.isnull().sum())
print(f"\nLabel distribution:")
print(train_df['label'].value_counts().sort_index())

print("\n" + "="*80)
print("VALIDATION DATA")
print("="*80)
print(f"Shape: {val_df.shape}")
print(f"\nFirst 5 rows:")
print(val_df.head())
print(f"\nData Info:")
print(val_df.info())
print(f"\nMissing values:")
print(val_df.isnull().sum())
print(f"\nLabel distribution:")
print(val_df['label'].value_counts().sort_index())

# Store original data size
original_train_size = len(train_df)
original_val_size = len(val_df)

logger.info(f"Loaded {original_train_size} training and {original_val_size} validation samples")


# ##  Section 10: Text Length Analysis

# In[ ]:


# Initialize visualizer
visualizer = ResultsVisualizer(emotion_map)

# Create detailed class distribution table
print("="*80)
print("CLASS DISTRIBUTION TABLE")
print("="*80)

train_labels = train_df['label'].values
val_labels = val_df['label'].values

# Training set distribution
train_unique, train_counts = np.unique(train_labels, return_counts=True)
train_total = len(train_labels)

train_dist = []
for label, count in zip(train_unique, train_counts):
    pct = (count / train_total) * 100
    train_dist.append({
        'Label': label,
        'Emotion': emotion_map[label],
        'Train Count': count,
        'Train %': f"{pct:.2f}%"
    })

# Validation set distribution
val_unique, val_counts = np.unique(val_labels, return_counts=True)
val_total = len(val_labels)

for i, (label, count) in enumerate(zip(val_unique, val_counts)):
    pct = (count / val_total) * 100
    train_dist[i]['Val Count'] = count
    train_dist[i]['Val %'] = f"{pct:.2f}%"

dist_df = pd.DataFrame(train_dist)
print(dist_df.to_string(index=False))

# Calculate imbalance ratio
max_count = train_counts.max()
min_count = train_counts.min()
imbalance_ratio = max_count / min_count

print(f"\n{'='*80}")
print(f"CLASS IMBALANCE RATIO: {imbalance_ratio:.2f}:1")
print(f"Most common: {emotion_map[train_unique[train_counts.argmax()]].upper()} ({max_count} samples)")
print(f"Least common: {emotion_map[train_unique[train_counts.argmin()]].upper()} ({min_count} samples)")
print(f"{'='*80}\n")

# Visualize distributions
visualizer.plot_label_distribution(train_labels, title='Training Set - Label Distribution')
visualizer.plot_label_distribution(val_labels, title='Validation Set - Label Distribution')


# ##  Section 11: Word Clouds by Emotion

# In[ ]:


# Calculate text lengths (in words)
train_df['text_len'] = train_df['text'].str.split().str.len()
val_df['text_len'] = val_df['text'].str.split().str.len()

print("="*80)
print("TEXT LENGTH STATISTICS (Before Preprocessing)")
print("="*80)
print("\nTraining Set:")
print(train_df['text_len'].describe())
print("\nValidation Set:")
print(val_df['text_len'].describe())
print("="*80)

# Visualize text length distribution
visualizer.plot_text_length_distribution(
    train_df['text_len'].values, 
    max_len=config.max_len,
    save_path=None
)

# Text length by emotion
print("\n" + "="*80)
print("AVERAGE TEXT LENGTH BY EMOTION")
print("="*80)
for label in sorted(emotion_map.keys()):
    emotion = emotion_map[label]
    avg_len = train_df[train_df['label'] == label]['text_len'].mean()
    print(f"{emotion.capitalize():12s}: {avg_len:.2f} words")
print("="*80)


# ##  Section 12: Most Common Words Analysis

# In[ ]:


# Generate word clouds for each emotion
print("Generating word clouds for each emotion...")

for label in sorted(emotion_map.keys()):
    emotion = emotion_map[label]
    texts = train_df[train_df['label'] == label]['text'].values
    visualizer.plot_word_cloud(texts, title=f'Word Cloud - {emotion.capitalize()}')


# ##  Section 13: Check for Twitter Noise (Emojis, Hashtags, Mentions)

# In[ ]:


# Most common words per emotion
print("="*80)
print("TOP 10 MOST COMMON WORDS BY EMOTION")
print("="*80)

for label in sorted(emotion_map.keys()):
    emotion = emotion_map[label]
    texts = train_df[train_df['label'] == label]['text'].values
    
    # Combine all texts and split into words
    all_words = ' '.join(texts).split()
    word_counts = Counter(all_words)
    
    print(f"\n{emotion.upper()} (Label {label}):")
    for word, count in word_counts.most_common(10):
        print(f"  {word:15s}: {count:5d}")

print("\n" + "="*80)


# ##  Section 14: Text Preprocessing with Statistics

# ---
# #  PREPROCESSING AND DATA PREPARATION

# In[ ]:


# Check for Twitter-specific elements
print("="*80)
print("CHECKING FOR TWITTER NOISE")
print("="*80)

# Check for emojis
def contains_emoji(text):
    return any(char in emoji.EMOJI_DATA for char in text)

# Check for hashtags
def contains_hashtag(text):
    return '#' in text

# Check for mentions
def contains_mention(text):
    return '@' in text

# Analyze training set
all_texts = train_df['text'].values
emoji_count = sum(contains_emoji(text) for text in all_texts)
hashtag_count = sum(contains_hashtag(text) for text in all_texts)
mention_count = sum(contains_mention(text) for text in all_texts)

print(f"\nTexts with emojis: {emoji_count} ({emoji_count/len(all_texts)*100:.2f}%)")
print(f"Texts with hashtags: {hashtag_count} ({hashtag_count/len(all_texts)*100:.2f}%)")
print(f"Texts with mentions: {mention_count} ({mention_count/len(all_texts)*100:.2f}%)")

# Check for URLs
url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
url_count = sum(1 for text in all_texts if url_pattern.search(text))
print(f"Texts with URLs: {url_count} ({url_count/len(all_texts)*100:.2f}%)")

print("\n" + "="*80)
print("CONCLUSION: Dataset is clean - minimal Twitter noise detected")
print("="*80)


# ##  Section 15: Tokenization and Sequence Analysis

# ---
# #  TOKENIZATION AND EMBEDDING

# In[ ]:


# Initialize preprocessor
preprocessor = AdvancedTextPreprocessor(config)

print("="*80)
print("PREPROCESSING CONFIGURATION")
print("="*80)
print(f"Aggressive Normalization: {config.enable_aggressive_normalization}")
print(f"Elongation Normalization: {config.enable_elongation_normalization}")
print(f"Contraction Expansion: {config.enable_contraction_expansion}")
print("="*80)

# Show before/after examples
print("\n" + "="*80)
print("PREPROCESSING EXAMPLES (Before → After)")
print("="*80)

sample_texts = [
    "i didnt feel good about this sooo bad situation!!!",
    "im feeling amazing today cant wait",
    "i love you sooooo much!!!",
    "dont be angry with me pleeease",
    "im scared idk what to do",
    "omg this is soooo surprising!!"
]

for i, text in enumerate(sample_texts, 1):
    cleaned = preprocessor.clean_text(text)
    print(f"\n{i}. Original: {text}")
    print(f"   Cleaned:  {cleaned}")

print("\n" + "="*80)

# Preprocess datasets
print("\nPreprocessing training data...")
train_df_clean = preprocessor.preprocess_dataframe(train_df.copy())

print("\nPreprocessing validation data...")
val_df_clean = preprocessor.preprocess_dataframe(val_df.copy())

# Check for duplicates
print("\nChecking for duplicates...")
train_df_clean = preprocessor.remove_duplicates(train_df_clean)

# Check for data leakage
print("\nChecking for data leakage between train and validation...")
val_df_clean, leakage_count = preprocessor.check_data_leakage(train_df_clean, val_df_clean)

if leakage_count > 0:
    print(f"  Removed {leakage_count} overlapping texts from validation set")
else:
    print(" No data leakage detected")

# Display preprocessing statistics
print("\n" + "="*80)
print("PREPROCESSING STATISTICS SUMMARY")
print("="*80)
print(f"Training set:")
print(f"  Original size: {original_train_size}")
print(f"  Final size: {len(train_df_clean)}")
print(f"  Texts modified: {preprocessor.stats['texts_modified']}")
print(f"\nValidation set:")
print(f"  Original size: {original_val_size}")
print(f"  Final size: {len(val_df_clean)}")
print("="*80)


# ##  Section 16: Load/Train Embeddings and Create Embedding Matrix

# In[ ]:


# Initialize embedding handler
embedding_handler = AdvancedEmbeddingHandler(config)

# Extract texts
train_texts = train_df_clean['text'].values.tolist()
val_texts = val_df_clean['text'].values.tolist()

# Create tokenizer
print("="*80)
print("CREATING TOKENIZER")
print("="*80)
tokenizer = embedding_handler.create_tokenizer(train_texts)

# Analyze sequence lengths
print("\n" + "="*80)
print("SEQUENCE LENGTH ANALYSIS")
print("="*80)

seq_stats, seq_lengths = embedding_handler.analyze_sequence_lengths(train_texts)

# Plot sequence length distribution
plt.figure(figsize=(14, 5))

# Histogram
plt.subplot(1, 2, 1)
plt.hist(seq_lengths, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
plt.axvline(seq_stats['mean_length'], color='red', linestyle='--', 
           linewidth=2, label=f"Mean: {seq_stats['mean_length']:.1f}")
plt.axvline(seq_stats['median_length'], color='green', linestyle='--', 
           linewidth=2, label=f"Median: {seq_stats['median_length']:.0f}")
plt.axvline(config.max_len, color='orange', linestyle='--', 
           linewidth=2, label=f"MAX_LEN: {config.max_len}")
plt.title('Sequence Length Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Sequence Length (tokens)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Box plot
plt.subplot(1, 2, 2)
plt.boxplot(seq_lengths, vert=True)
plt.axhline(config.max_len, color='orange', linestyle='--', 
           linewidth=2, label=f"MAX_LEN: {config.max_len}")
plt.title('Sequence Length Box Plot', fontsize=14, fontweight='bold')
plt.ylabel('Sequence Length (tokens)', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Justify MAX_LEN choice
print("\n" + "="*80)
print("MAX_LEN JUSTIFICATION")
print("="*80)
percentiles = [50, 75, 90, 95, 99]
for p in percentiles:
    val = np.percentile(seq_lengths, p)
    print(f"{p}th percentile: {val:.0f} tokens")

print(f"\nChosen MAX_LEN: {config.max_len}")
print(f"This covers {(sum(1 for l in seq_lengths if l <= config.max_len) / len(seq_lengths) * 100):.1f}% of sequences")
print(f"Sequences truncated: {seq_stats['truncated_count']} ({seq_stats['truncation_percent']:.1f}%)")
print("="*80)

# Convert to sequences
print("\n" + "="*80)
print("CONVERTING TEXTS TO SEQUENCES")
print("="*80)

X_train = embedding_handler.texts_to_sequences(train_texts)
X_val = embedding_handler.texts_to_sequences(val_texts)

print(f"\nTraining sequences shape: {X_train.shape}")
print(f"Validation sequences shape: {X_val.shape}")

# Prepare labels
y_train = train_df_clean['label'].values
y_val = val_df_clean['label'].values

# Convert to one-hot encoding
y_train_cat = to_categorical(y_train, num_classes=config.num_classes)
y_val_cat = to_categorical(y_val, num_classes=config.num_classes)

print(f"\nTraining labels shape: {y_train_cat.shape}")
print(f"Validation labels shape: {y_val_cat.shape}")

# Save tokenizer if configured
if config.save_tokenizer:
    os.makedirs(config.config_dir, exist_ok=True)
    tokenizer_path = os.path.join(config.config_dir, 'tokenizer.json')
    embedding_handler.save_tokenizer(tokenizer_path)


# ##  Section 17: Build Model

# ---
# #  MODEL BUILDING AND TRAINING

# In[ ]:


# Load or train embeddings based on configuration
print("="*80)
print(f"LOADING {config.embedding_type.upper()} EMBEDDINGS")
print("="*80)

if config.embedding_type == 'glove':
    # Load GloVe embeddings
    embedding_handler.load_glove_embeddings(config.glove_path)
elif config.embedding_type == 'word2vec':
    # Train Word2Vec on our corpus
    embedding_handler.train_word2vec(train_texts, min_count=1)

# Create embedding matrix
print("\n" + "="*80)
print("CREATING EMBEDDING MATRIX")
print("="*80)

embedding_matrix = embedding_handler.create_embedding_matrix()

# Display embedding coverage statistics
print("\n" + "="*80)
print("EMBEDDING COVERAGE ANALYSIS")
print("="*80)
print(f"Vocabulary size: {embedding_handler.stats['vocab_size']}")
print(f"Words with embeddings: {embedding_handler.stats['coverage_count']} ({embedding_handler.stats['coverage_percent']:.2f}%)")
print(f"OOV words: {embedding_handler.stats['oov_count']} ({embedding_handler.stats['oov_percent']:.2f}%)")
print(f"\nFirst 20 OOV words:")
for i, word in enumerate(embedding_handler.stats['oov_words'][:20], 1):
    print(f"  {i:2d}. {word}")

# Calculate OOV rate in actual sequences
oov_rate = embedding_handler.get_oov_rate(X_train)
print(f"\nOOV tokens in training sequences: {oov_rate:.2f}%")
print("="*80)

# Visualize embedding coverage
vocab_size = embedding_handler.stats['vocab_size']
coverage_count = embedding_handler.stats['coverage_count']
oov_count = embedding_handler.stats['oov_count']

plt.figure(figsize=(10, 6))
colors = ['#2ecc71', '#e74c3c']
sizes = [coverage_count, oov_count]
labels = [f'Covered\n{coverage_count} words\n({embedding_handler.stats["coverage_percent"]:.1f}%)',
          f'OOV\n{oov_count} words\n({embedding_handler.stats["oov_percent"]:.1f}%)']
explode = (0.05, 0.05)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
plt.title(f'{config.embedding_type.upper()} Embedding Coverage', fontsize=14, fontweight='bold', pad=20)
plt.axis('equal')
plt.show()

# Save embedding matrix if configured
if config.save_embedding_matrix:
    os.makedirs(config.save_dir, exist_ok=True)
    embedding_path = os.path.join(config.save_dir, 'embedding_matrix.npy')
    embedding_handler.save_embedding_matrix(embedding_path)

print(f"\n Embedding matrix ready: {embedding_matrix.shape}")


# ##  Section 18: Train Model with Callbacks

# In[ ]:


# Initialize model builder
model_builder = AdvancedModelBuilder(config)

# Build model
print("="*80)
print("BUILDING MODEL")
print("="*80)

model = model_builder.build_model(embedding_matrix)

# Display model architecture
print("\n" + "="*80)
print("MODEL ARCHITECTURE")
print("="*80)
model.summary()

# Visualize model architecture (if possible)
try:
    from tensorflow.keras.utils import plot_model
    plot_path = os.path.join(config.save_dir, 'model_architecture.png')
    os.makedirs(config.save_dir, exist_ok=True)
    plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
    print(f"\n Model architecture diagram saved to {plot_path}")
except:
    print("\n  Could not generate model diagram (requires graphviz)")

# Compute class weights
if config.use_class_weights:
    class_weights = preprocessor.compute_class_weights(y_train)
else:
    class_weights = None
    logger.info("Class weights disabled")

print("\n Model ready for training!")


# ##  Section 19: Visualize Training History

# In[ ]:


# Prepare directories
os.makedirs(config.save_dir, exist_ok=True)
os.makedirs(config.log_dir, exist_ok=True)
os.makedirs(config.result_dir, exist_ok=True)

# Setup callbacks
callbacks = []

# Experiment tracker
experiment_tracker = ExperimentTracker(config)
callbacks.append(experiment_tracker)

# Model checkpoint
checkpoint_path = os.path.join(config.save_dir, f'{config.experiment_name}_best_model.h5')
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
callbacks.append(checkpoint)

# Early stopping
if config.early_stopping:
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=config.patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stop)

# Reduce LR on plateau
if config.reduce_lr:
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=config.lr_factor,
        patience=config.lr_patience,
        min_lr=config.min_lr,
        verbose=1
    )
    callbacks.append(reduce_lr)

# TensorBoard
tensorboard_dir = os.path.join(config.log_dir, config.experiment_name)
tensorboard = TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)
callbacks.append(tensorboard)

# CSV Logger
csv_path = os.path.join(config.log_dir, f'{config.experiment_name}_training.csv')
csv_logger = CSVLogger(csv_path)
callbacks.append(csv_logger)

print("="*80)
print("TRAINING MODEL")
print("="*80)
print(f"Experiment: {config.experiment_name}")
print(f"Model: {config.model_type.upper()}")
print(f"Epochs: {config.epochs}")
print(f"Batch Size: {config.batch_size}")
print(f"Learning Rate: {config.learning_rate}")
print(f"Class Weights: {'Enabled' if config.use_class_weights else 'Disabled'}")
print(f"Early Stopping: {'Enabled (patience={})'.format(config.patience) if config.early_stopping else 'Disabled'}")
print("="*80)

# Train model
start_time = time.time()

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=config.epochs,
    batch_size=config.batch_size,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=config.verbose
)

training_time = time.time() - start_time

print("\n" + "="*80)
print("TRAINING COMPLETED!")
print("="*80)
print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
print(f"Best val accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"Best val loss: {min(history.history['val_loss']):.4f}")
print(f"Model saved to: {checkpoint_path}")
print("="*80)


# ##  Section 20: Make Predictions and Calculate Metrics

# ---
# #  MODEL EVALUATION

# In[ ]:


# Plot training history
visualizer.plot_training_history(
    history,
    save_path=os.path.join(config.result_dir, f'{config.experiment_name}_training_history.png')
)

# Display training statistics
print("\n" + "="*80)
print("TRAINING STATISTICS")
print("="*80)
print(f"Epochs trained: {len(history.history['loss'])}")
print(f"\nFinal metrics:")
print(f"  Train accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  Train loss: {history.history['loss'][-1]:.4f}")
print(f"  Val accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"  Val loss: {history.history['val_loss'][-1]:.4f}")
print(f"\nBest metrics:")
print(f"  Best val accuracy: {max(history.history['val_accuracy']):.4f} (epoch {np.argmax(history.history['val_accuracy'])+1})")
print(f"  Best val loss: {min(history.history['val_loss']):.4f} (epoch {np.argmin(history.history['val_loss'])+1})")
print("="*80)


# ##  Section 21: Confusion Matrix

# In[ ]:


# Make predictions
print("="*80)
print("MAKING PREDICTIONS")
print("="*80)

y_train_pred = model.predict(X_train, verbose=0)
y_val_pred = model.predict(X_val, verbose=0)

# Convert to class labels
y_train_pred_labels = y_train_pred.argmax(axis=1)
y_val_pred_labels = y_val_pred.argmax(axis=1)

# Calculate comprehensive metrics
print("\n" + "="*80)
print("VALIDATION SET METRICS")
print("="*80)

# Overall accuracy
val_accuracy = accuracy_score(y_val, y_val_pred_labels)
print(f"\nOverall Accuracy: {val_accuracy:.4f}")

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_val, y_val_pred_labels, labels=list(range(config.num_classes))
)

# Macro averages
macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1 = np.mean(f1)

print(f"\nMacro-averaged metrics:")
print(f"  Precision: {macro_precision:.4f}")
print(f"  Recall: {macro_recall:.4f}")
print(f"  F1-Score: {macro_f1:.4f}")

# Per-class breakdown
print(f"\nPer-class metrics:")
print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
print("-" * 60)
for i, emotion in emotion_map.items():
    print(f"{emotion.capitalize():<12} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<12}")

print("="*80)

# Store metrics for later use
metrics = {
    'val_accuracy': val_accuracy,
    'val_loss': history.history['val_loss'][-1],
    'macro_precision': macro_precision,
    'macro_recall': macro_recall,
    'macro_f1': macro_f1,
    'per_class_precision': precision.tolist(),
    'per_class_recall': recall.tolist(),
    'per_class_f1': f1.tolist(),
    'training_time': training_time
}

# Save metrics to JSON
metrics_path = os.path.join(config.result_dir, f'{config.experiment_name}_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\n Metrics saved to {metrics_path}")


# ##  Section 22: Classification Report Visualization

# In[ ]:


# Plot raw confusion matrix
visualizer.plot_confusion_matrix(
    y_val, y_val_pred_labels, 
    normalize=False,
    save_path=os.path.join(config.result_dir, f'{config.experiment_name}_confusion_matrix.png')
)

# Plot normalized confusion matrix
visualizer.plot_confusion_matrix(
    y_val, y_val_pred_labels, 
    normalize=True,
    save_path=os.path.join(config.result_dir, f'{config.experiment_name}_confusion_matrix_normalized.png')
)


# ##  Section 23: Per-Class F1 Scores

# In[ ]:


# Plot classification report heatmap
metrics_df = visualizer.plot_classification_report(
    y_val, y_val_pred_labels,
    save_path=os.path.join(config.result_dir, f'{config.experiment_name}_classification_report.png')
)

print("\nClassification Report DataFrame:")
print(metrics_df)

# Save to CSV
metrics_df.to_csv(os.path.join(config.result_dir, f'{config.experiment_name}_classification_report.csv'))


# ##  Section 24: Model Comparison (Run multiple experiments)

# ##  Section 25: View Comparison Table

# In[ ]:


# Initialize model comparer
comparer = ModelComparer(emotion_map)

# Add current experiment
comparer.add_experiment(
    name=config.experiment_name,
    config=config,
    history=history,
    metrics=metrics,
    predictions=y_val_pred_labels,
    y_true=y_val
)

print(" Current experiment added to comparer")
print("\nTo add more experiments:")
print("1. Modify config in Section 2 (e.g., config.model_type = 'gru')")
print("2. Re-run sections 15-23")
print("3. Run: comparer.add_experiment(name='experiment_2', config=config, history=history, metrics=metrics)")
print("4. View comparison: comparer.create_comparison_table()")

# Example: Uncomment and modify to add more experiments
# config.model_type = 'gru'
# config.experiment_name = 'gru_128'
# # Re-run training...
# comparer.add_experiment(name='gru_128', config=config, history=history, metrics=metrics)


# ##  Section 26: Prediction Function

# ---
# #  PREDICTIONS AND INTERACTIVE TESTING

# In[ ]:


# Display comparison table
comparison_df = comparer.create_comparison_table()
print("="*80)
print("MODEL COMPARISON TABLE")
print("="*80)
print(comparison_df.to_string(index=False))
print("="*80)

# Plot comparison
if len(comparer.experiments) > 1:
    comparer.plot_comparison(metric='val_accuracy')
    comparer.plot_comparison(metric='macro_f1')
    
    # Save comparison
    comparison_path = os.path.join(config.result_dir, 'model_comparison.csv')
    comparer.save_comparison(comparison_path)
else:
    print("\n Add more experiments to enable comparison visualizations")


# ##  Section 27: Test Predictions with Examples

# In[ ]:


def predict_emotion(text: str, show_probabilities: bool = True) -> Dict:
    """
    Predict emotion for a given text.
    
    Args:
        text: Input text
        show_probabilities: Whether to show probabilities for all classes
        
    Returns:
        Dictionary with prediction results
    """
    # Preprocess text
    cleaned_text = preprocessor.clean_text(text)
    
    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=config.max_len, padding='post', truncating='post')
    
    # Predict
    predictions = model.predict(padded, verbose=0)[0]
    predicted_label = predictions.argmax()
    predicted_emotion = emotion_map[predicted_label]
    confidence = predictions[predicted_label]
    
    result = {
        'original_text': text,
        'cleaned_text': cleaned_text,
        'predicted_emotion': predicted_emotion,
        'predicted_label': int(predicted_label),
        'confidence': float(confidence)
    }
    
    if show_probabilities:
        result['all_probabilities'] = {
            emotion_map[i]: float(predictions[i]) 
            for i in range(len(predictions))
        }
    
    return result


def display_prediction(text: str):
    """Display prediction with formatting."""
    result = predict_emotion(text, show_probabilities=True)
    
    print("="*80)
    print("EMOTION PREDICTION")
    print("="*80)
    print(f"Original Text: {result['original_text']}")
    print(f"Cleaned Text:  {result['cleaned_text']}")
    print(f"\n Predicted Emotion: {result['predicted_emotion'].upper()}")
    print(f"   Confidence: {result['confidence']*100:.2f}%")
    print(f"\nAll Probabilities:")
    for emotion, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
        bar = '' * int(prob * 50)
        print(f"  {emotion.capitalize():<12} {prob*100:5.2f}% {bar}")
    print("="*80)


print(" Prediction functions created!")
print("\nUsage:")
print("  display_prediction('I am so happy today!')")
print("  result = predict_emotion('I feel terrible', show_probabilities=True)")


