#!/bin/bash
#SBATCH --job-name=gemma_train
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=fit
#SBATCH --qos=fitq
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=ml23

module load miniforge3

# Activate conda environment from scratch area
conda activate /scratch/ml23/bnge/conda/envs/fyp

# export HF_TOKEN="ENTER_YOUR_HUGGINGFACE_TOKEN_HERE"

# Set ALL cache directories to scratch area under gemma/ subdirectory
export HF_HOME=/scratch/ml23/bnge/gemma/hf_cache
export HF_HUB_CACHE=/scratch/ml23/bnge/gemma/hf_cache
export HUGGINGFACE_HUB_CACHE=/scratch/ml23/bnge/gemma/hf_cache
export XDG_CACHE_HOME=/scratch/ml23/bnge/gemma/xdg_cache

# Fix matplotlib cache issue
export MPLCONFIGDIR=/scratch/ml23/bnge/gemma/matplotlib_cache

# Additional cache directories
export TORCH_HOME=/scratch/ml23/bnge/gemma/torch_cache
export TMPDIR=/scratch/ml23/bnge/gemma/tmp

# CREATE ALL CACHE DIRECTORIES
echo "Creating Gemma cache directories under gemma/..."
mkdir -p $HF_HOME
mkdir -p $HF_HUB_CACHE
mkdir -p $HUGGINGFACE_HUB_CACHE
mkdir -p $XDG_CACHE_HOME
mkdir -p $MPLCONFIGDIR
mkdir -p $TORCH_HOME
mkdir -p $TMPDIR

# Verify directories were created
echo "Cache directories created:"
echo "HF_HOME: $HF_HOME"
echo "MPLCONFIGDIR: $MPLCONFIGDIR"
echo "TMPDIR: $TMPDIR"

# Check available space on scratch
echo "Available space on scratch:"
df -h /scratch/ml23/bnge

# GPU and performance settings
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Absolute path to your working directory on scratch
cd /projects/ml23/bnge0001/gemma

DATASETS="merged_corpus.csv merged_reddit_gab.csv merged_stormfront.csv unified_hate_speech_dataset.csv"

for DATASET in $DATASETS; do
    BASENAME=$(basename "$DATASET" .csv)
    OUTPUT_DIR="gemma_baseline_${BASENAME}_model"
    echo "======================================"
    echo "Running Gemma training for $DATASET"
    echo "======================================"
    python train_gemma.py "$DATASET" "$OUTPUT_DIR" > "train_gemma_${BASENAME}.log" 2>&1
done

conda deactivate
