#!/bin/bash
#SBATCH --job-name=emotion_notebook
#SBATCH --output=notebook_job_%j.log
#SBATCH --error=notebook_job_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

echo "================================================================"
echo "  EMOTION DETECTION NOTEBOOK - BATCH JOB"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo "================================================================"
echo ""

echo "GPU Information:"
nvidia-smi
echo ""

echo "================================================================"
echo "Starting notebook execution..."
echo "================================================================"
echo ""

cd /home/lab/rabanof/projects/Emotion_Detection_DL

# Run notebook
./run_notebook_gpu3.sh

echo ""
echo "================================================================"
echo "Job completed: $(date)"
echo "================================================================"
echo ""
echo "Results saved to:"
echo "  - notebooks/complete_pipeline_executed.ipynb"
echo "  - results/ultimate_emotion_detection_metrics.json"
echo "  - saved_models/ultimate_emotion_detection_best_model.h5"
echo ""
