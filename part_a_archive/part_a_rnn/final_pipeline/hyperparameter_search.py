#!/usr/bin/env python3
"""
Comprehensive Hyperparameter Search for Emotion Detection
Tests all combinations of hyperparameters to find the best configuration
"""

import os
import sys
import json
import logging
from datetime import datetime
from itertools import product

# Add parent directory to path to import the pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_gpu_experiments import ExperimentConfig, run_experiment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hyperparameter_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Run comprehensive hyperparameter search"""

    logger.info("="*80)
    logger.info("HYPERPARAMETER SEARCH - EMOTION DETECTION")
    logger.info("="*80)
    logger.info("Testing all combinations of hyperparameters")
    logger.info("")

    # Define hyperparameter grid
    hyperparameters = {
        'model_type': ['lstm', 'gru', 'bilstm'],
        'rnn_units': [64, 128, 256],
        'dropout': [0.0, 0.2, 0.4, 0.8],
        'batch_size': [16, 32, 64, 128],
        'learning_rate': [0.01, 0.001, 0.0001],
        'epochs': [25, 50]
    }

    # Calculate total number of experiments
    total_experiments = 1
    for key, values in hyperparameters.items():
        total_experiments *= len(values)

    logger.info("Hyperparameter Grid:")
    for key, values in hyperparameters.items():
        logger.info(f"  {key}: {values}")
    logger.info(f"\nTotal experiments to run: {total_experiments}")
    logger.info("="*80)
    logger.info("")

    # Generate all combinations
    param_names = list(hyperparameters.keys())
    param_values = [hyperparameters[name] for name in param_names]

    all_results = []
    successful_experiments = 0
    failed_experiments = 0

    # Run all experiments
    for i, combination in enumerate(product(*param_values), 1):
        # Create parameter dict
        params = dict(zip(param_names, combination))

        # Create experiment name
        experiment_name = (
            f"{params['model_type']}_"
            f"units{params['rnn_units']}_"
            f"drop{params['dropout']}_"
            f"bs{params['batch_size']}_"
            f"lr{params['learning_rate']}_"
            f"ep{params['epochs']}"
        )

        logger.info("")
        logger.info("="*80)
        logger.info(f"EXPERIMENT {i}/{total_experiments}: {experiment_name}")
        logger.info("="*80)
        logger.info(f"Parameters:")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")
        logger.info("")

        try:
            # Create config with current hyperparameters
            config = ExperimentConfig(
                experiment_name=experiment_name,
                model_type=params['model_type'],
                rnn_units=params['rnn_units'],
                dropout=params['dropout'],
                spatial_dropout=params['dropout'],  # Use same dropout value
                recurrent_dropout=params['dropout'],  # Use same dropout value
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                epochs=params['epochs'],

                # Fixed parameters (GloVe 100d only)
                embedding_type='glove',
                embedding_dim=100,
                glove_path='/home/lab/rabanof/Emotion_Detection_DL/glove/glove.6B.100d.txt',

                # Other fixed parameters
                patience=15,
                max_len=60,
                max_words=None,
                num_classes=6,
                use_class_weights=True,
                trainable_embeddings=False,

                # Directories
                save_dir='saved_models',
                log_dir='logs',
                result_dir='results'
            )

            # Run experiment
            results = run_experiment(config)

            # Add hyperparameters to results
            results['hyperparameters'] = params
            all_results.append(results)
            successful_experiments += 1

            logger.info(f"✓ Experiment completed successfully")
            logger.info(f"  Validation Accuracy: {results['val_accuracy']:.4f} ({results['val_accuracy']*100:.2f}%)")
            logger.info(f"  Macro F1: {results['macro_f1']:.4f}")

        except Exception as e:
            logger.error(f"✗ Experiment {experiment_name} failed: {e}")
            failed_experiments += 1
            import traceback
            traceback.print_exc()
            continue

        # Save intermediate results after each experiment
        save_results(all_results, successful_experiments, failed_experiments, total_experiments)

    # Final summary
    logger.info("")
    logger.info("="*80)
    logger.info("HYPERPARAMETER SEARCH COMPLETED")
    logger.info("="*80)
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Successful: {successful_experiments}")
    logger.info(f"Failed: {failed_experiments}")
    logger.info("")

    if all_results:
        # Find best model
        best_result = max(all_results, key=lambda x: x['val_accuracy'])

        logger.info("="*80)
        logger.info("BEST MODEL FOUND")
        logger.info("="*80)
        logger.info(f"Experiment: {best_result['experiment_name']}")
        logger.info(f"Validation Accuracy: {best_result['val_accuracy']:.4f} ({best_result['val_accuracy']*100:.2f}%)")
        logger.info(f"Macro F1: {best_result['macro_f1']:.4f}")
        logger.info("")
        logger.info("Best Hyperparameters:")
        for key, value in best_result['hyperparameters'].items():
            logger.info(f"  {key}: {value}")
        logger.info("="*80)

        # Save final summary
        save_final_summary(all_results, best_result)

    logger.info("")
    logger.info("All results saved to:")
    logger.info("  - results/hyperparameter_search_all_results.json")
    logger.info("  - results/hyperparameter_search_summary.csv")
    logger.info("  - results/hyperparameter_search_best_model.json")


def save_results(all_results, successful, failed, total):
    """Save intermediate results"""
    os.makedirs('results', exist_ok=True)

    # Save all results as JSON
    with open('results/hyperparameter_search_all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Save as CSV for easy viewing
    import pandas as pd

    if all_results:
        # Flatten results for CSV
        csv_data = []
        for result in all_results:
            row = {
                'experiment_name': result['experiment_name'],
                'val_accuracy': result['val_accuracy'],
                'macro_f1': result['macro_f1'],
                'training_time': result['training_time'],
                'epochs_trained': result['epochs_trained'],
            }
            # Add hyperparameters
            if 'hyperparameters' in result:
                row.update(result['hyperparameters'])
            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        # Sort by validation accuracy
        df = df.sort_values('val_accuracy', ascending=False)
        df.to_csv('results/hyperparameter_search_summary.csv', index=False)

        logger.info(f"\nProgress: {successful}/{total} experiments completed ({failed} failed)")


def save_final_summary(all_results, best_result):
    """Save final summary and best model info"""

    # Save best model
    with open('results/hyperparameter_search_best_model.json', 'w') as f:
        json.dump(best_result, f, indent=2)

    # Create top 10 summary
    sorted_results = sorted(all_results, key=lambda x: x['val_accuracy'], reverse=True)
    top_10 = sorted_results[:10]

    logger.info("")
    logger.info("="*80)
    logger.info("TOP 10 MODELS")
    logger.info("="*80)
    for i, result in enumerate(top_10, 1):
        logger.info(f"\n{i}. {result['experiment_name']}")
        logger.info(f"   Accuracy: {result['val_accuracy']:.4f} ({result['val_accuracy']*100:.2f}%)")
        logger.info(f"   F1: {result['macro_f1']:.4f}")
        if 'hyperparameters' in result:
            logger.info(f"   Params: {result['hyperparameters']}")


if __name__ == "__main__":
    main()
