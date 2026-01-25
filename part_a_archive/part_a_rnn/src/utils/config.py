"""
Configuration management for experiments and hyperparameters.
"""

import yaml
import json
from typing import Dict, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data processing configuration."""
    train_path: str = "data/raw/train.csv"
    val_path: str = "data/raw/validation.csv"
    test_path: str = "data/raw/test.csv"
    max_len: int = 60
    max_words: int = 20000
    text_column: str = "text"
    label_column: str = "label"


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    embedding_type: str = "glove"  # 'glove' or 'word2vec'
    embedding_dim: int = 100
    embedding_path: str = "embeddings/glove.6B.100d.txt"
    trainable: bool = False
    oov_token: str = "<UNK>"

    # Word2Vec specific
    w2v_window: int = 5
    w2v_min_count: int = 2
    w2v_epochs: int = 10


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_type: str = "lstm"  # 'lstm', 'gru', or 'hybrid'
    units: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    recurrent_dropout: float = 0.0
    spatial_dropout: float = 0.2
    bidirectional: bool = False
    dense_units: int = 0
    num_classes: int = 6


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss: str = "categorical_crossentropy"
    use_class_weights: bool = True

    # Callbacks
    early_stopping: bool = True
    patience: int = 5
    reduce_lr: bool = True
    lr_factor: float = 0.5
    lr_patience: int = 3
    min_lr: float = 1e-7
    tensorboard: bool = True
    save_best_only: bool = True
    monitor: str = "val_accuracy"
    mode: str = "max"

    verbose: int = 1


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_name: str = "emotion_detection_exp"
    data: DataConfig = None
    embedding: EmbeddingConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'experiment_name': self.experiment_name,
            'data': asdict(self.data),
            'embedding': asdict(self.embedding),
            'model': asdict(self.model),
            'training': asdict(self.training)
        }

    def save(self, filepath: str):
        """Save configuration to file."""
        config_dict = self.to_dict()

        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError("File must be .yaml, .yml, or .json")

        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from file."""
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError("File must be .yaml, .yml, or .json")

        # Create config object
        config = cls(
            experiment_name=config_dict.get('experiment_name', 'experiment'),
            data=DataConfig(**config_dict.get('data', {})),
            embedding=EmbeddingConfig(**config_dict.get('embedding', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {}))
        )

        logger.info(f"Configuration loaded from {filepath}")
        return config


# Predefined configurations for different experiments

def get_lstm_glove_config() -> ExperimentConfig:
    """Get LSTM with GloVe configuration."""
    return ExperimentConfig(
        experiment_name="lstm_glove_baseline",
        embedding=EmbeddingConfig(embedding_type="glove", embedding_dim=100),
        model=ModelConfig(model_type="lstm", units=128, num_layers=1),
        training=TrainingConfig(epochs=50, batch_size=32)
    )


def get_lstm_word2vec_config() -> ExperimentConfig:
    """Get LSTM with Word2Vec configuration."""
    return ExperimentConfig(
        experiment_name="lstm_word2vec",
        embedding=EmbeddingConfig(embedding_type="word2vec", embedding_dim=100),
        model=ModelConfig(model_type="lstm", units=128, num_layers=1),
        training=TrainingConfig(epochs=50, batch_size=32)
    )


def get_gru_glove_config() -> ExperimentConfig:
    """Get GRU with GloVe configuration."""
    return ExperimentConfig(
        experiment_name="gru_glove_baseline",
        embedding=EmbeddingConfig(embedding_type="glove", embedding_dim=100),
        model=ModelConfig(model_type="gru", units=128, num_layers=1),
        training=TrainingConfig(epochs=50, batch_size=32)
    )


def get_bidirectional_lstm_config() -> ExperimentConfig:
    """Get Bidirectional LSTM configuration."""
    return ExperimentConfig(
        experiment_name="bilstm_glove",
        embedding=EmbeddingConfig(embedding_type="glove", embedding_dim=100),
        model=ModelConfig(model_type="lstm", units=128, num_layers=1, bidirectional=True),
        training=TrainingConfig(epochs=50, batch_size=32)
    )


def get_deep_lstm_config() -> ExperimentConfig:
    """Get Deep (2-layer) LSTM configuration."""
    return ExperimentConfig(
        experiment_name="deep_lstm_glove",
        embedding=EmbeddingConfig(embedding_type="glove", embedding_dim=100),
        model=ModelConfig(model_type="lstm", units=128, num_layers=2),
        training=TrainingConfig(epochs=50, batch_size=32)
    )


def get_all_configs() -> Dict[str, ExperimentConfig]:
    """Get all predefined configurations."""
    return {
        'lstm_glove': get_lstm_glove_config(),
        'lstm_word2vec': get_lstm_word2vec_config(),
        'gru_glove': get_gru_glove_config(),
        'bilstm_glove': get_bidirectional_lstm_config(),
        'deep_lstm_glove': get_deep_lstm_config()
    }
