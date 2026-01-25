#!/usr/bin/env python3
"""
Quick start script to run an emotion detection experiment.

Usage:
    python run_experiment.py --config configs/sample_config.yaml
    python run_experiment.py --model lstm --embedding glove --epochs 30
"""

import argparse
import sys
import logging
import numpy as np
from tensorflow.keras.utils import to_categorical

from src.data.preprocessor import load_and_preprocess_data
from src.data.embeddings import create_embeddings
from src.models.architectures import create_model
from src.training.trainer import train_model
from src.utils.config import ExperimentConfig, get_all_configs
from src.utils.visualization import create_comprehensive_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main(args):
    """Run emotion detection experiment."""

    logger.info("="*80)
    logger.info("Starting Emotion Detection Experiment")
    logger.info("="*80)

    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = ExperimentConfig.load(args.config)
    elif args.preset:
        logger.info(f"Using preset configuration: {args.preset}")
        all_configs = get_all_configs()
        if args.preset not in all_configs:
            logger.error(f"Unknown preset: {args.preset}")
            logger.info(f"Available presets: {list(all_configs.keys())}")
            return
        config = all_configs[args.preset]
    else:
        logger.info("Creating default configuration")
        config = ExperimentConfig()

    # Override with command line arguments
    if args.model:
        config.model.model_type = args.model
    if args.embedding:
        config.embedding.embedding_type = args.embedding
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.units:
        config.model.units = args.units
    if args.bidirectional:
        config.model.bidirectional = True
    if args.experiment_name:
        config.experiment_name = args.experiment_name

    logger.info(f"\nExperiment: {config.experiment_name}")
    logger.info(f"Model: {config.model.model_type}")
    logger.info(f"Embedding: {config.embedding.embedding_type}")
    logger.info(f"Epochs: {config.training.epochs}")
    logger.info(f"Batch Size: {config.training.batch_size}")

    # Step 1: Load and preprocess data
    logger.info("\n" + "="*80)
    logger.info("Step 1: Loading and preprocessing data")
    logger.info("="*80)

    train_df, val_df, preprocessor = load_and_preprocess_data(
        config.data.train_path,
        config.data.val_path
    )

    # Step 2: Create embeddings
    logger.info("\n" + "="*80)
    logger.info("Step 2: Creating embeddings")
    logger.info("="*80)

    X_train, X_val, embedding_matrix, handler, stats = create_embeddings(
        train_df['text'].tolist(),
        val_df['text'].tolist(),
        embedding_type=config.embedding.embedding_type,
        embedding_path=config.embedding.embedding_path,
        embedding_dim=config.embedding.embedding_dim,
        max_words=config.data.max_words,
        max_len=config.data.max_len
    )

    # Prepare labels
    y_train = to_categorical(train_df['label'].values, num_classes=6)
    y_val = to_categorical(val_df['label'].values, num_classes=6)

    logger.info(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"Embedding coverage: {stats['coverage_percent']:.2f}%")

    # Step 3: Create model
    logger.info("\n" + "="*80)
    logger.info("Step 3: Creating model")
    logger.info("="*80)

    model_config = {
        'lstm_units' if config.model.model_type == 'lstm' else 'gru_units': config.model.units,
        'num_layers': config.model.num_layers,
        'dropout': config.model.dropout,
        'recurrent_dropout': config.model.recurrent_dropout,
        'spatial_dropout': config.model.spatial_dropout,
        'bidirectional': config.model.bidirectional,
        'dense_units': config.model.dense_units,
        'trainable_embeddings': config.embedding.trainable,
        'learning_rate': config.training.learning_rate,
        'optimizer': config.training.optimizer,
        'loss': config.training.loss,
        'metrics': ['accuracy']
    }

    model = create_model(
        model_type=config.model.model_type,
        vocab_size=handler.vocab_size,
        embedding_dim=config.embedding.embedding_dim,
        embedding_matrix=embedding_matrix,
        config=model_config
    )

    model.summary()

    # Step 4: Compute class weights
    if config.training.use_class_weights:
        class_weights = preprocessor.compute_class_weights(train_df['label'].values)
    else:
        class_weights = None

    # Step 5: Train model
    logger.info("\n" + "="*80)
    logger.info("Step 4: Training model")
    logger.info("="*80)

    training_config = {
        'epochs': config.training.epochs,
        'batch_size': config.training.batch_size,
        'class_weight': class_weights,
        'early_stopping': config.training.early_stopping,
        'patience': config.training.patience,
        'reduce_lr': config.training.reduce_lr,
        'lr_factor': config.training.lr_factor,
        'lr_patience': config.training.lr_patience,
        'min_lr': config.training.min_lr,
        'tensorboard': config.training.tensorboard,
        'save_best_only': config.training.save_best_only,
        'monitor': config.training.monitor,
        'mode': config.training.mode,
        'verbose': config.training.verbose
    }

    trainer, history = train_model(
        model, X_train, y_train, X_val, y_val,
        config.experiment_name,
        training_config
    )

    # Step 6: Evaluate and visualize
    logger.info("\n" + "="*80)
    logger.info("Step 5: Evaluating and creating visualizations")
    logger.info("="*80)

    val_results = trainer.evaluate(X_val, y_val)
    logger.info(f"\nFinal Validation Results:")
    for metric, value in val_results.items():
        logger.info(f"  {metric}: {value:.4f}")

    # Get predictions
    y_pred = trainer.predict(X_val)

    # Create comprehensive report
    create_comprehensive_report(
        y_val, y_pred, history,
        config.experiment_name,
        save_dir='results'
    )

    # Save configuration
    config.save(f'configs/{config.experiment_name}_config.yaml')

    logger.info("\n" + "="*80)
    logger.info("Experiment Complete!")
    logger.info("="*80)
    logger.info(f"Results saved in: results/")
    logger.info(f"Model saved in: saved_models/{config.experiment_name}_best.keras")
    logger.info(f"Config saved in: configs/{config.experiment_name}_config.yaml")
    logger.info(f"Logs saved in: logs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run emotion detection experiment")

    # Configuration
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--preset', type=str, choices=['lstm_glove', 'lstm_word2vec', 'gru_glove', 'bilstm_glove', 'deep_lstm_glove'],
                       help='Use a preset configuration')

    # Experiment settings
    parser.add_argument('--experiment-name', type=str, help='Experiment name')

    # Model settings
    parser.add_argument('--model', type=str, choices=['lstm', 'gru', 'hybrid'], help='Model type')
    parser.add_argument('--units', type=int, help='Number of RNN units')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional layers')

    # Embedding settings
    parser.add_argument('--embedding', type=str, choices=['glove', 'word2vec'], help='Embedding type')

    # Training settings
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        sys.exit(1)
