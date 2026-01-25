#!/usr/bin/env python3
"""
Generate LSTM vs GRU Comparison Graphs for Each Hyperparameter
Creates side-by-side comparison plots showing how each hyperparameter affects LSTM vs GRU
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['font.family'] = 'sans-serif'

def load_results():
    """Load both LSTM and GRU ablation study results"""
    # Load GRU results
    with open('results/ablation_study_all_results.json', 'r') as f:
        gru_results = json.load(f)
    df_gru = pd.read_csv('results/ablation_study_summary.csv')

    # Load LSTM results
    with open('results/ablation_study_lstm_all_results.json', 'r') as f:
        lstm_results = json.load(f)
    df_lstm = pd.read_csv('results/ablation_study_lstm_summary.csv')

    return gru_results, df_gru, lstm_results, df_lstm


def plot_hyperparameter_comparison(df_gru, df_lstm, param_study, param_name, title, filename):
    """
    Create side-by-side comparison: LSTM (left) vs GRU (right) for a specific hyperparameter
    Line plots with markers and grid
    """
    # Filter data for this study
    gru_data = df_gru[df_gru['study'] == param_study].copy()
    lstm_data = df_lstm[df_lstm['study'] == param_study].copy()

    # Sort by parameter value for proper line connection
    gru_data = gru_data.sort_values('param_value')
    lstm_data = lstm_data.sort_values('param_value')

    # Create figure with 2 subplots (LSTM left, GRU right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ========== LEFT PLOT: LSTM ==========
    x_values_lstm = range(len(lstm_data))
    y_acc_lstm = lstm_data['val_accuracy'] * 100
    y_f1_lstm = lstm_data['macro_f1']

    # Accuracy line (blue)
    ax1.plot(x_values_lstm, y_acc_lstm,
             color='#3498db', linewidth=3,
             marker='o', markersize=12, markeredgecolor='black',
             markeredgewidth=2, label='Accuracy', zorder=3)

    # F1 line (red) - secondary y-axis
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x_values_lstm, y_f1_lstm,
                  color='#e74c3c', linewidth=3,
                  marker='s', markersize=12, markeredgecolor='black',
                  markeredgewidth=2, label='Macro F1', zorder=3)

    # Highlight best and worst for accuracy
    best_idx = y_acc_lstm.idxmax()
    worst_idx = y_acc_lstm.idxmin()
    best_pos = lstm_data.index.get_loc(best_idx)
    worst_pos = lstm_data.index.get_loc(worst_idx)

    ax1.plot(best_pos, y_acc_lstm.iloc[best_pos],
            marker='*', markersize=20, color='#2ecc71',
            markeredgecolor='black', markeredgewidth=2, zorder=5)
    ax1.plot(worst_pos, y_acc_lstm.iloc[worst_pos],
            marker='X', markersize=15, color='#e74c3c',
            markeredgecolor='black', markeredgewidth=2, zorder=5)

    # Labels and formatting
    ax1.set_xlabel(param_name.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=13, fontweight='bold', color='#3498db')
    ax1_twin.set_ylabel('Macro F1 Score', fontsize=13, fontweight='bold', color='#e74c3c')
    ax1.set_title(f'LSTM - {title}', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x_values_lstm)
    ax1.set_xticklabels(lstm_data['param_value'], rotation=45 if len(lstm_data) > 3 else 0)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1.5, zorder=0)
    ax1.set_ylim(0, 100)
    ax1_twin.set_ylim(0, 1)

    # Tick colors
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1_twin.tick_params(axis='y', labelcolor='#e74c3c')

    # Add value labels for accuracy
    for i, (x, y) in enumerate(zip(x_values_lstm, y_acc_lstm)):
        ax1.text(x, y + 2, f'{y:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=9, color='#3498db')

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=10)

    # ========== RIGHT PLOT: GRU ==========
    x_values_gru = range(len(gru_data))
    y_acc_gru = gru_data['val_accuracy'] * 100
    y_f1_gru = gru_data['macro_f1']

    # Accuracy line (blue)
    ax2.plot(x_values_gru, y_acc_gru,
             color='#3498db', linewidth=3,
             marker='o', markersize=12, markeredgecolor='black',
             markeredgewidth=2, label='Accuracy', zorder=3)

    # F1 line (red) - secondary y-axis
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_values_gru, y_f1_gru,
                  color='#e74c3c', linewidth=3,
                  marker='s', markersize=12, markeredgecolor='black',
                  markeredgewidth=2, label='Macro F1', zorder=3)

    # Highlight best and worst for accuracy
    best_idx = y_acc_gru.idxmax()
    worst_idx = y_acc_gru.idxmin()
    best_pos = gru_data.index.get_loc(best_idx)
    worst_pos = gru_data.index.get_loc(worst_idx)

    ax2.plot(best_pos, y_acc_gru.iloc[best_pos],
            marker='*', markersize=20, color='#2ecc71',
            markeredgecolor='black', markeredgewidth=2, zorder=5)
    ax2.plot(worst_pos, y_acc_gru.iloc[worst_pos],
            marker='X', markersize=15, color='#e74c3c',
            markeredgecolor='black', markeredgewidth=2, zorder=5)

    # Labels and formatting
    ax2.set_xlabel(param_name.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=13, fontweight='bold', color='#3498db')
    ax2_twin.set_ylabel('Macro F1 Score', fontsize=13, fontweight='bold', color='#e74c3c')
    ax2.set_title(f'GRU - {title}', fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(x_values_gru)
    ax2.set_xticklabels(gru_data['param_value'], rotation=45 if len(gru_data) > 3 else 0)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1.5, zorder=0)
    ax2.set_ylim(0, 100)
    ax2_twin.set_ylim(0, 1)

    # Tick colors
    ax2.tick_params(axis='y', labelcolor='#3498db')
    ax2_twin.tick_params(axis='y', labelcolor='#e74c3c')

    # Add value labels for accuracy
    for i, (x, y) in enumerate(zip(x_values_gru, y_acc_gru)):
        ax2.text(x, y + 2, f'{y:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=9, color='#3498db')

    # Legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=10)

    # Overall title
    fig.suptitle(f'{title} - LSTM vs GRU Comparison',
                 fontsize=16, fontweight='bold', y=1.00)

    plt.tight_layout()
    plt.savefig(f'visualizations/comparison_{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created comparison_{filename}.png")


def main():
    """Generate all LSTM vs GRU comparison plots"""

    print("="*80)
    print("GENERATING LSTM vs GRU COMPARISON VISUALIZATIONS")
    print("="*80)

    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)

    # Load results
    print("\nLoading results...")
    gru_results, df_gru, lstm_results, df_lstm = load_results()
    print(f"✓ Loaded GRU results: {len(df_gru)} experiments")
    print(f"✓ Loaded LSTM results: {len(df_lstm)} experiments")

    # Generate comparison plots for each hyperparameter
    print("\nGenerating comparison plots...")
    print("-"*80)

    # Study 1: RNN Units
    plot_hyperparameter_comparison(
        df_gru, df_lstm,
        param_study='rnn_units',
        param_name='rnn_units',
        title='RNN Units',
        filename='rnn_units'
    )

    # Study 2: Dropout Rate
    plot_hyperparameter_comparison(
        df_gru, df_lstm,
        param_study='dropout',
        param_name='dropout',
        title='Dropout Rate',
        filename='dropout'
    )

    # Study 3: Batch Size
    plot_hyperparameter_comparison(
        df_gru, df_lstm,
        param_study='batch_size',
        param_name='batch_size',
        title='Batch Size',
        filename='batch_size'
    )

    # Study 4: Learning Rate
    plot_hyperparameter_comparison(
        df_gru, df_lstm,
        param_study='learning_rate',
        param_name='learning_rate',
        title='Learning Rate',
        filename='learning_rate'
    )

    # Study 5: Epochs
    plot_hyperparameter_comparison(
        df_gru, df_lstm,
        param_study='epochs',
        param_name='epochs',
        title='Training Epochs',
        filename='epochs'
    )

    print("-"*80)
    print("\n" + "="*80)
    print("ALL COMPARISON VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files in visualizations/:")
    print("  - comparison_rnn_units.png")
    print("  - comparison_dropout.png")
    print("  - comparison_batch_size.png")
    print("  - comparison_learning_rate.png")
    print("  - comparison_epochs.png")
    print("\nEach file shows LSTM (left) vs GRU (right) side-by-side comparison")
    print("="*80)


if __name__ == "__main__":
    main()
