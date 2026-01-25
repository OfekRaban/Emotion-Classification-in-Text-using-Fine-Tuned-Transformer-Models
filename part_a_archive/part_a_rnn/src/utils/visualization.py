"""
Visualization utilities for model evaluation and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class ResultsVisualizer:
    """Comprehensive visualization for model results."""

    def __init__(self, emotion_labels: Optional[List[str]] = None):
        """
        Initialize visualizer.

        Args:
            emotion_labels: List of emotion label names
        """
        if emotion_labels is None:
            self.emotion_labels = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
        else:
            self.emotion_labels = emotion_labels

    def plot_training_history(self, history, save_path: Optional[str] = None):
        """
        Plot training and validation metrics over epochs.

        Args:
            history: Keras History object
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
        axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss
        axes[1].plot(history.history['loss'], label='Train Loss', marker='o')
        axes[1].plot(history.history['val_loss'], label='Val Loss', marker='s')
        axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")

        plt.show()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             normalize: bool = False, save_path: Optional[str] = None):
        """
        Plot confusion matrix.

        Args:
            y_true: True labels (one-hot or class indices)
            y_pred: Predicted labels (one-hot or class indices)
            normalize: Whether to normalize the matrix
            save_path: Path to save figure (optional)
        """
        # Convert one-hot to class indices if needed
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels,
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'})

        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.show()

    def plot_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   save_path: Optional[str] = None):
        """
        Generate and display classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save report (optional)
        """
        # Convert one-hot to class indices if needed
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)

        report = classification_report(y_true, y_pred, target_names=self.emotion_labels,
                                      output_dict=True)

        # Convert to DataFrame for visualization
        df_report = pd.DataFrame(report).transpose()

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Select metrics to plot
        metrics_to_plot = ['precision', 'recall', 'f1-score']
        df_plot = df_report.loc[self.emotion_labels, metrics_to_plot]

        df_plot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Classification Metrics by Emotion', fontsize=14, fontweight='bold')
        ax.set_xlabel('Emotion', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim([0, 1.0])
        ax.legend(title='Metric', loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Classification report plot saved to {save_path}")

        plt.show()

        # Print text report
        print("\nDetailed Classification Report:")
        print("=" * 70)
        print(classification_report(y_true, y_pred, target_names=self.emotion_labels))

        return report

    def plot_prediction_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     save_path: Optional[str] = None):
        """
        Plot distribution of predictions vs true labels.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save figure (optional)
        """
        # Convert one-hot to class indices if needed
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # True distribution
        true_counts = pd.Series(y_true).value_counts().sort_index()
        axes[0].bar(range(len(self.emotion_labels)),
                   [true_counts.get(i, 0) for i in range(len(self.emotion_labels))],
                   color='skyblue', edgecolor='black')
        axes[0].set_title('True Label Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Emotion')
        axes[0].set_ylabel('Count')
        axes[0].set_xticks(range(len(self.emotion_labels)))
        axes[0].set_xticklabels(self.emotion_labels, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Predicted distribution
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        axes[1].bar(range(len(self.emotion_labels)),
                   [pred_counts.get(i, 0) for i in range(len(self.emotion_labels))],
                   color='lightcoral', edgecolor='black')
        axes[1].set_title('Predicted Label Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Emotion')
        axes[1].set_ylabel('Count')
        axes[1].set_xticks(range(len(self.emotion_labels)))
        axes[1].set_xticklabels(self.emotion_labels, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction distribution plot saved to {save_path}")

        plt.show()

    def plot_per_class_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray,
                                save_path: Optional[str] = None):
        """
        Plot per-class accuracy.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save figure (optional)
        """
        # Convert one-hot to class indices if needed
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)

        accuracies = []
        for i in range(len(self.emotion_labels)):
            mask = y_true == i
            if mask.sum() > 0:
                acc = (y_pred[mask] == i).sum() / mask.sum()
                accuracies.append(acc)
            else:
                accuracies.append(0)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.emotion_labels, accuracies, color='steelblue', edgecolor='black')

        # Color bars by performance
        for i, bar in enumerate(bars):
            if accuracies[i] >= 0.8:
                bar.set_color('green')
            elif accuracies[i] >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Emotion', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim([0, 1.0])
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (label, acc) in enumerate(zip(self.emotion_labels, accuracies)):
            plt.text(i, acc + 0.02, f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Per-class accuracy plot saved to {save_path}")

        plt.show()

    def compare_models(self, results_dict: Dict[str, Dict], save_path: Optional[str] = None):
        """
        Compare multiple models' performance.

        Args:
            results_dict: Dictionary mapping model names to their metrics
            save_path: Path to save figure (optional)
        """
        metrics = ['accuracy', 'loss']
        model_names = list(results_dict.keys())

        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))

        for idx, metric in enumerate(metrics):
            values = [results_dict[name].get(metric, 0) for name in model_names]
            axes[idx].bar(model_names, values, color='skyblue', edgecolor='black')
            axes[idx].set_title(f'Model Comparison: {metric.capitalize()}',
                              fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Model')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')

            # Add value labels
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")

        plt.show()


def create_comprehensive_report(y_true: np.ndarray, y_pred: np.ndarray,
                                history, experiment_name: str,
                                save_dir: str = 'results'):
    """
    Create comprehensive visualization report.

    Args:
        y_true: True labels
        y_pred: Predicted labels or probabilities
        history: Training history object
        experiment_name: Name of the experiment
        save_dir: Directory to save visualizations
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    visualizer = ResultsVisualizer()

    # Convert predictions if needed
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = y_pred

    # Training history
    visualizer.plot_training_history(
        history,
        save_path=os.path.join(save_dir, f'{experiment_name}_training_history.png')
    )

    # Confusion matrix
    visualizer.plot_confusion_matrix(
        y_true, y_pred_classes,
        save_path=os.path.join(save_dir, f'{experiment_name}_confusion_matrix.png')
    )

    # Normalized confusion matrix
    visualizer.plot_confusion_matrix(
        y_true, y_pred_classes, normalize=True,
        save_path=os.path.join(save_dir, f'{experiment_name}_confusion_matrix_normalized.png')
    )

    # Classification report
    visualizer.plot_classification_report(
        y_true, y_pred_classes,
        save_path=os.path.join(save_dir, f'{experiment_name}_classification_report.png')
    )

    # Distribution
    visualizer.plot_prediction_distribution(
        y_true, y_pred_classes,
        save_path=os.path.join(save_dir, f'{experiment_name}_distribution.png')
    )

    # Per-class accuracy
    visualizer.plot_per_class_accuracy(
        y_true, y_pred_classes,
        save_path=os.path.join(save_dir, f'{experiment_name}_per_class_accuracy.png')
    )

    logger.info(f"Comprehensive report created in {save_dir}")
