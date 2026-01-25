#!/usr/bin/env python3
"""
Ablation Study for Emotion Detection - LSTM Baseline
Tests each hyperparameter independently using LSTM as baseline
(Skips model architecture comparison)
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add parent directory to path to import the pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_gpu_experiments import ExperimentConfig, run_experiment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ablation_study_lstm.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Run ablation study - test each hyperparameter independently with LSTM baseline"""

    logger.info("="*80)
    logger.info("ABLATION STUDY - EMOTION DETECTION (LSTM BASELINE)")
    logger.info("="*80)
    logger.info("Testing each hyperparameter independently with LSTM")
    logger.info("Then testing optimal combination of best parameters")
    logger.info("")

    # ===== BASELINE CONFIGURATION (LSTM) =====
    baseline = {
        'model_type': 'lstm',  # LSTM baseline
        'rnn_units': 128,
        'dropout': 0.2,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 50,

        # Fixed for all experiments
        'embedding_type': 'glove',
        'embedding_dim': 100,
        'glove_path': '/home/lab/rabanof/Emotion_Detection_DL/glove/glove.6B.100d.txt',
        'patience': 15,
        'max_len': 60,
        'max_words': None,
        'num_classes': 6,
        'use_class_weights': True,
        'trainable_embeddings': False,
    }

    logger.info("BASELINE CONFIGURATION (LSTM):")
    for key, value in baseline.items():
        if key not in ['glove_path', 'max_words', 'num_classes', 'use_class_weights', 'trainable_embeddings', 'patience', 'max_len']:
            logger.info(f"  {key}: {value}")
    logger.info("")

    # ===== DEFINE ABLATION EXPERIMENTS =====
    ablation_experiments = []

    # 1. Test different RNN UNITS (keeping other params at baseline)
    logger.info("Study 1: RNN Units")
    for units in [64, 128, 256]:
        ablation_experiments.append({
            'study': 'rnn_units',
            'param_name': 'rnn_units',
            'param_value': units,
            'config': {**baseline, 'rnn_units': units}
        })

    # 2. Test different DROPOUT values (keeping other params at baseline)
    logger.info("Study 2: Dropout Rate")
    for dropout in [0.0, 0.2, 0.4]:
        ablation_experiments.append({
            'study': 'dropout',
            'param_name': 'dropout',
            'param_value': dropout,
            'config': {**baseline, 'dropout': dropout}
        })

    # 3. Test different BATCH SIZES (keeping other params at baseline)
    logger.info("Study 3: Batch Size")
    for bs in [16, 32, 64, 128]:
        ablation_experiments.append({
            'study': 'batch_size',
            'param_name': 'batch_size',
            'param_value': bs,
            'config': {**baseline, 'batch_size': bs}
        })

    # 4. Test different LEARNING RATES (keeping other params at baseline)
    logger.info("Study 4: Learning Rate")
    for lr in [0.01, 0.001, 0.0001]:
        ablation_experiments.append({
            'study': 'learning_rate',
            'param_name': 'learning_rate',
            'param_value': lr,
            'config': {**baseline, 'learning_rate': lr}
        })

    # 5. Test different EPOCHS (keeping other params at baseline)
    logger.info("Study 5: Training Epochs")
    for ep in [25, 50]:
        ablation_experiments.append({
            'study': 'epochs',
            'param_name': 'epochs',
            'param_value': ep,
            'config': {**baseline, 'epochs': ep}
        })

    total_experiments = len(ablation_experiments)
    logger.info(f"\nTotal experiments: {total_experiments}")
    logger.info("="*80)
    logger.info("")

    # ===== RUN ALL EXPERIMENTS =====
    all_results = []
    successful_experiments = 0
    failed_experiments = 0

    for i, exp in enumerate(ablation_experiments, 1):
        study = exp['study']
        param_name = exp['param_name']
        param_value = exp['param_value']
        config_dict = exp['config']

        # Create experiment name
        experiment_name = f"lstm_{study}_{param_name}_{param_value}"

        logger.info("")
        logger.info("="*80)
        logger.info(f"EXPERIMENT {i}/{total_experiments}: {experiment_name}")
        logger.info("="*80)
        logger.info(f"Study: {study}")
        logger.info(f"Testing: {param_name} = {param_value}")
        logger.info("")

        try:
            # Create config
            config = ExperimentConfig(
                experiment_name=experiment_name,
                model_type=config_dict['model_type'],
                rnn_units=config_dict['rnn_units'],
                dropout=config_dict['dropout'],
                spatial_dropout=config_dict['dropout'],
                recurrent_dropout=config_dict['dropout'],
                batch_size=config_dict['batch_size'],
                learning_rate=config_dict['learning_rate'],
                epochs=config_dict['epochs'],
                embedding_type=config_dict['embedding_type'],
                embedding_dim=config_dict['embedding_dim'],
                glove_path=config_dict['glove_path'],
                patience=config_dict['patience'],
                max_len=config_dict['max_len'],
                max_words=config_dict['max_words'],
                num_classes=config_dict['num_classes'],
                use_class_weights=config_dict['use_class_weights'],
                trainable_embeddings=config_dict['trainable_embeddings'],
                save_dir='saved_models',
                log_dir='logs',
                result_dir='results'
            )

            # Run experiment
            results = run_experiment(config)

            # Add study information
            results['study'] = study
            results['param_name'] = param_name
            results['param_value'] = param_value
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

        # Save intermediate results
        save_results(all_results, successful_experiments, failed_experiments, total_experiments)

    # ===== ANALYZE INDIVIDUAL STUDIES =====
    logger.info("")
    logger.info("="*80)
    logger.info("INDIVIDUAL STUDIES COMPLETED")
    logger.info("="*80)
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Successful: {successful_experiments}")
    logger.info(f"Failed: {failed_experiments}")
    logger.info("")

    if all_results:
        best_params = analyze_results(all_results)
        save_results(all_results, successful_experiments, failed_experiments, total_experiments)

        # ===== OPTIMAL COMBINATION =====
        logger.info("")
        logger.info("="*80)
        logger.info("OPTIMAL COMBINATION (LSTM)")
        logger.info("="*80)
        logger.info("Testing the best values from each individual study combined together")
        logger.info("")

        if best_params:
            logger.info("Optimal parameters identified:")
            for key, value in best_params.items():
                logger.info(f"  {key}: {value}")
            logger.info("")

            # Run optimal combination experiment
            try:
                optimal_config = {**baseline}
                optimal_config.update(best_params)

                experiment_name = "lstm_optimal_combination_" + "_".join([f"{k}{v}" for k, v in best_params.items()])

                logger.info(f"Running optimal combination experiment: {experiment_name}")
                logger.info("")

                config = ExperimentConfig(
                    experiment_name=experiment_name,
                    model_type='lstm',  # Force LSTM
                    rnn_units=optimal_config['rnn_units'],
                    dropout=optimal_config['dropout'],
                    spatial_dropout=optimal_config['dropout'],
                    recurrent_dropout=optimal_config['dropout'],
                    batch_size=optimal_config['batch_size'],
                    learning_rate=optimal_config['learning_rate'],
                    epochs=optimal_config['epochs'],
                    embedding_type=optimal_config['embedding_type'],
                    embedding_dim=optimal_config['embedding_dim'],
                    glove_path=optimal_config['glove_path'],
                    patience=optimal_config['patience'],
                    max_len=optimal_config['max_len'],
                    max_words=optimal_config['max_words'],
                    num_classes=optimal_config['num_classes'],
                    use_class_weights=optimal_config['use_class_weights'],
                    trainable_embeddings=optimal_config['trainable_embeddings'],
                    save_dir='saved_models',
                    log_dir='logs',
                    result_dir='results'
                )

                results = run_experiment(config)
                results['study'] = 'optimal_combination'
                results['param_name'] = 'combined'
                results['param_value'] = 'optimal'
                results['optimal_params'] = best_params
                all_results.append(results)

                logger.info("")
                logger.info("="*80)
                logger.info("OPTIMAL COMBINATION RESULTS (LSTM)")
                logger.info("="*80)
                logger.info(f"Validation Accuracy: {results['val_accuracy']:.4f} ({results['val_accuracy']*100:.2f}%)")
                logger.info(f"Macro F1: {results['macro_f1']:.4f}")
                logger.info("")

                # Compare with baseline
                baseline_results = [r for r in all_results if r['study'] == 'rnn_units' and r['param_value'] == 128]
                if baseline_results:
                    baseline_acc = baseline_results[0]['val_accuracy']
                    improvement = (results['val_accuracy'] - baseline_acc) * 100
                    logger.info(f"Improvement over baseline: {improvement:+.2f}%")
                    logger.info("")

            except Exception as e:
                logger.error(f"Optimal combination experiment failed: {e}")
                import traceback
                traceback.print_exc()

        # ===== FINAL SUMMARY =====
        logger.info("")
        logger.info("="*80)
        logger.info("ABLATION STUDY COMPLETED (LSTM)")
        logger.info("="*80)

        save_final_summary(all_results)

    logger.info("")
    logger.info("All results saved to:")
    logger.info("  - results/ablation_study_lstm_all_results.json")
    logger.info("  - results/ablation_study_lstm_summary.csv")
    logger.info("  - results/ablation_study_lstm_by_parameter.json")


def analyze_results(all_results):
    """Analyze results for each parameter study and return best parameters"""

    # Group results by study
    studies = {}
    for result in all_results:
        study = result['study']
        if study not in studies:
            studies[study] = []
        studies[study].append(result)

    logger.info("="*80)
    logger.info("ANALYSIS BY HYPERPARAMETER (LSTM)")
    logger.info("="*80)

    study_summaries = {}
    best_params = {}

    # Mapping from study name to config parameter name
    study_to_param = {
        'rnn_units': 'rnn_units',
        'dropout': 'dropout',
        'batch_size': 'batch_size',
        'learning_rate': 'learning_rate',
        'epochs': 'epochs'
    }

    for study_name, results in studies.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"STUDY: {study_name.upper().replace('_', ' ')}")
        logger.info(f"{'='*80}")

        # Sort by accuracy
        results_sorted = sorted(results, key=lambda x: x['val_accuracy'], reverse=True)

        study_summaries[study_name] = []

        for result in results_sorted:
            param_value = result['param_value']
            acc = result['val_accuracy']
            f1 = result['macro_f1']

            logger.info(f"  {result['param_name']} = {param_value:>6}  →  Accuracy: {acc:.4f} ({acc*100:>6.2f}%)  F1: {f1:.4f}")

            study_summaries[study_name].append({
                'param_value': param_value,
                'accuracy': acc,
                'f1': f1,
                'experiment': result['experiment_name']
            })

        # Find best and worst
        best = results_sorted[0]
        worst = results_sorted[-1]
        improvement = (best['val_accuracy'] - worst['val_accuracy']) * 100

        logger.info(f"\n  BEST:  {best['param_name']} = {best['param_value']} ({best['val_accuracy']*100:.2f}%)")
        logger.info(f"  WORST: {worst['param_name']} = {worst['param_value']} ({worst['val_accuracy']*100:.2f}%)")
        logger.info(f"  IMPACT: {improvement:.2f}% difference")

        # Store best parameter value for optimal combination
        if study_name in study_to_param:
            param_key = study_to_param[study_name]
            best_params[param_key] = best['param_value']

    # Save study summaries
    with open('results/ablation_study_lstm_by_parameter.json', 'w') as f:
        json.dump(study_summaries, f, indent=2)

    # Overall best model from individual studies
    logger.info(f"\n{'='*80}")
    logger.info("OVERALL BEST FROM INDIVIDUAL STUDIES (LSTM)")
    logger.info(f"{'='*80}")
    best_overall = max(all_results, key=lambda x: x['val_accuracy'])
    logger.info(f"Experiment: {best_overall['experiment_name']}")
    logger.info(f"Study: {best_overall['study']}")
    logger.info(f"Parameter: {best_overall['param_name']} = {best_overall['param_value']}")
    logger.info(f"Accuracy: {best_overall['val_accuracy']:.4f} ({best_overall['val_accuracy']*100:.2f}%)")
    logger.info(f"Macro F1: {best_overall['macro_f1']:.4f}")

    return best_params


def save_results(all_results, successful, failed, total):
    """Save intermediate results"""
    os.makedirs('results', exist_ok=True)

    # Save all results as JSON
    with open('results/ablation_study_lstm_all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Save as CSV
    import pandas as pd

    if all_results:
        csv_data = []
        for result in all_results:
            row = {
                'experiment_name': result['experiment_name'],
                'study': result['study'],
                'param_name': result['param_name'],
                'param_value': result['param_value'],
                'val_accuracy': result['val_accuracy'],
                'macro_f1': result['macro_f1'],
                'training_time': result['training_time'],
                'epochs_trained': result['epochs_trained'],
            }
            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        df.to_csv('results/ablation_study_lstm_summary.csv', index=False)

        logger.info(f"\nProgress: {successful}/{total} experiments completed ({failed} failed)")


def save_final_summary(all_results):
    """Save final summary"""

    summary = {
        'total_experiments': len(all_results),
        'best_model': max(all_results, key=lambda x: x['val_accuracy']),
        'timestamp': datetime.now().isoformat()
    }

    with open('results/ablation_study_lstm_final_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
