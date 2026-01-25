#!/bin/bash
#
# Execute complete_pipeline.ipynb on GPU 3
# This script runs the entire notebook and saves results
#

set -e  # Exit on error

echo "================================================================"
echo "  COMPLETE PIPELINE - GPU 3 EXECUTION"
echo "================================================================"
echo ""
echo "Starting notebook execution on GPU 3..."
echo "This will take approximately 20-40 minutes."
echo ""
echo "The notebook will:"
echo "  1. Load and preprocess data"
echo "  2. Train the model on GPU 3"
echo "  3. Evaluate and generate visualizations"
echo "  4. Save all results"
echo ""
echo "Output will be saved to:"
echo "  - notebooks/complete_pipeline_executed.ipynb"
echo ""
echo "================================================================"
echo ""

# Set GPU (use environment variable if already set, otherwise default to 3)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=3
fi
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Check GPU
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi 2>/dev/null && echo "" || echo "nvidia-smi found but couldn't query GPUs"
else
    echo "NOTE: nvidia-smi not available (you may be on login node)"
    echo "The script will work when run on a GPU node."
fi
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run notebook
echo "Executing notebook..."
echo ""
venv/bin/python3 run_notebook_on_gpu.py 2>&1 | tee notebook_execution.log

echo ""
echo "================================================================"
echo "  EXECUTION COMPLETE"
echo "================================================================"
echo ""
echo "Check results in:"
echo "  - notebooks/complete_pipeline_executed.ipynb (executed notebook)"
echo "  - notebook_execution.log (execution log)"
echo "  - results/ (metrics and plots)"
echo "  - saved_models/ (trained models)"
echo "  - logs/ (training logs)"
echo ""
