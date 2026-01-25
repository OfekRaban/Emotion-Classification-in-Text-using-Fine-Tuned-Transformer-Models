#!/bin/bash
#
# GPU Training Script for H100 Cluster
# Runs all emotion detection experiments on GPU
#

echo "========================================================================"
echo "GPU EMOTION DETECTION - TRAINING SCRIPT"
echo "========================================================================"
echo ""

# Check CUDA availability
echo "Checking CUDA/GPU setup..."
nvidia-smi
echo ""

# Activate environment if needed
# source /path/to/your/venv/bin/activate

# Set environment variables for GPU
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "Starting GPU training experiments..."
echo "Date: $(date)"
echo ""

# Run the experiments
python3 run_gpu_experiments.py 2>&1 | tee gpu_training_output.log

echo ""
echo "========================================================================"
echo "TRAINING COMPLETED"
echo "========================================================================"
echo ""
echo "Results saved in:"
echo "  - results/all_experiments_comparison.csv"
echo "  - results/*_results.json"
echo "  - logs/*_training.csv"
echo "  - saved_models/*_best.h5"
echo "  - gpu_training.log"
echo ""
echo "To view comparison:"
echo "  cat results/all_experiments_comparison.csv"
echo ""
