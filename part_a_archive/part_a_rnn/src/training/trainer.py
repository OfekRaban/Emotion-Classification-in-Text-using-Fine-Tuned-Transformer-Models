"""
Training pipeline with callbacks, checkpointing, and experiment tracking.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, CSVLogger, Callback
)
import logging

logger = logging.getLogger(__name__)


class ExperimentTracker(Callback):
    """Custom callback to track experiment metrics and save results."""

    def __init__(self, experiment_name: str, results_dir: str = 'results'):
        super().__init__()
        self.experiment_name = experiment_name
        self.results_dir = results_dir
        self.start_time = None
        self.metrics_history = {'train': {}, 'val': {}}

        os.makedirs(results_dir, exist_ok=True)

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        logger.info(f"Training started for experiment: {self.experiment_name}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Store metrics
        for key, value in logs.items():
            if key.startswith('val_'):
                metric_name = key[4:]
                if metric_name not in self.metrics_history['val']:
                    self.metrics_history['val'][metric_name] = []
                self.metrics_history['val'][metric_name].append(float(value))
            else:
                if key not in self.metrics_history['train']:
                    self.metrics_history['train'][key] = []
                self.metrics_history['train'][key].append(float(value))

    def on_train_end(self, logs=None):
        training_time = time.time() - self.start_time

        # Prepare results summary
        results = {
            'experiment_name': self.experiment_name,
            'training_time_seconds': training_time,
            'total_epochs': len(self.metrics_history['train'].get('loss', [])),
            'metrics_history': self.metrics_history,
            'final_metrics': {
                'train': {k: v[-1] for k, v in self.metrics_history['train'].items() if v},
                'val': {k: v[-1] for k, v in self.metrics_history['val'].items() if v}
            },
            'best_metrics': {
                'val_accuracy': max(self.metrics_history['val'].get('accuracy', [0])),
                'val_loss': min(self.metrics_history['val'].get('loss', [float('inf')]))
            }
        }

        # Save results
        results_file = os.path.join(self.results_dir, f'{self.experiment_name}_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Training completed in {training_time:.2f}s")
        logger.info(f"Best val_accuracy: {results['best_metrics']['val_accuracy']:.4f}")
        logger.info(f"Results saved to {results_file}")


class ModelTrainer:
    """Comprehensive model training pipeline."""

    def __init__(self, model: keras.Model, experiment_name: str,
                 checkpoint_dir: str = 'saved_models',
                 logs_dir: str = 'logs',
                 results_dir: str = 'results'):
        """
        Initialize trainer.

        Args:
            model: Keras model to train
            experiment_name: Name for this experiment
            checkpoint_dir: Directory to save model checkpoints
            logs_dir: Directory for TensorBoard logs
            results_dir: Directory to save results
        """
        self.model = model
        self.experiment_name = experiment_name
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir
        self.results_dir = results_dir

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        self.history = None

    def create_callbacks(self, config: Dict) -> List[Callback]:
        """
        Create training callbacks based on configuration.

        Args:
            config: Configuration dictionary containing callback settings

        Returns:
            List of Keras callbacks
        """
        callbacks = []

        # Model checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'{self.experiment_name}_best.keras'
        )
        callbacks.append(ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=config.get('monitor', 'val_accuracy'),
            mode=config.get('mode', 'max'),
            save_best_only=config.get('save_best_only', True),
            verbose=1
        ))

        # Early stopping
        if config.get('early_stopping', True):
            callbacks.append(EarlyStopping(
                monitor=config.get('monitor', 'val_loss'),
                patience=config.get('patience', 5),
                restore_best_weights=True,
                verbose=1
            ))

        # Reduce learning rate on plateau
        if config.get('reduce_lr', True):
            callbacks.append(ReduceLROnPlateau(
                monitor=config.get('monitor', 'val_loss'),
                factor=config.get('lr_factor', 0.5),
                patience=config.get('lr_patience', 3),
                min_lr=config.get('min_lr', 1e-7),
                verbose=1
            ))

        # TensorBoard
        if config.get('tensorboard', True):
            tensorboard_dir = os.path.join(self.logs_dir, self.experiment_name)
            callbacks.append(TensorBoard(
                log_dir=tensorboard_dir,
                histogram_freq=1,
                write_graph=True
            ))

        # CSV Logger
        csv_path = os.path.join(self.logs_dir, f'{self.experiment_name}_training.csv')
        callbacks.append(CSVLogger(csv_path, append=False))

        # Experiment tracker
        callbacks.append(ExperimentTracker(self.experiment_name, self.results_dir))

        logger.info(f"Created {len(callbacks)} callbacks")

        return callbacks

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              config: Dict) -> keras.callbacks.History:
        """
        Train the model.

        Args:
            X_train: Training sequences
            y_train: Training labels (one-hot encoded)
            X_val: Validation sequences
            y_val: Validation labels (one-hot encoded)
            config: Training configuration containing:
                - epochs: Number of epochs
                - batch_size: Batch size
                - class_weight: Class weights (optional)
                - validation_split: Validation split (if not using separate val set)
                - callbacks config

        Returns:
            Training history object
        """
        logger.info(f"Starting training for {self.experiment_name}")
        logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

        callbacks = self.create_callbacks(config)

        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.get('epochs', 50),
            batch_size=config.get('batch_size', 32),
            class_weight=config.get('class_weight'),
            callbacks=callbacks,
            verbose=config.get('verbose', 1)
        )

        self.history = history

        return history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set.

        Args:
            X_test: Test sequences
            y_test: Test labels (one-hot encoded)

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model...")

        results = self.model.evaluate(X_test, y_test, verbose=0, return_dict=True)

        logger.info(f"Test results: {results}")

        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Input sequences

        Returns:
            Predicted class probabilities
        """
        return self.model.predict(X, verbose=0)

    def save_model(self, filepath: Optional[str] = None):
        """Save the model."""
        if filepath is None:
            filepath = os.path.join(self.checkpoint_dir, f'{self.experiment_name}_final.keras')

        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a saved model."""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self.model


def train_model(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                experiment_name: str, config: Optional[Dict] = None) -> Tuple:
    """
    Convenience function to train a model.

    Args:
        model: Keras model to train
        X_train: Training sequences
        y_train: Training labels
        X_val: Validation sequences
        y_val: Validation labels
        experiment_name: Name for this experiment
        config: Training configuration

    Returns:
        Tuple of (trainer, history)
    """
    if config is None:
        config = {
            'epochs': 50,
            'batch_size': 32,
            'early_stopping': True,
            'patience': 5
        }

    trainer = ModelTrainer(model, experiment_name)
    history = trainer.train(X_train, y_train, X_val, y_val, config)

    return trainer, history
