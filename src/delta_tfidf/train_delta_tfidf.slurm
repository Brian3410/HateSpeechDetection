#!/bin/bash
#SBATCH --job-name=delta_tfidf_svm
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=ml23

module load miniforge3

conda activate /scratch/ml23/bnge/conda/envs/fyp

mkdir -p /scratch/ml23/bnge/delta_tfidf/models

cd /projects/ml23/bnge0001/delta_tfidf

DATASETS="merged_corpus.csv merged_reddit_gab.csv merged_stormfront.csv unified_hate_speech_dataset.csv"

for DATASET in $DATASETS; do
    BASENAME=$(basename "$DATASET" .csv)
    echo "======================================"
    echo "Running Delta TFIDF SVM for $DATASET"
    echo "======================================"
    python delta_tfidf.py "$DATASET" > "delta_tfidf_${BASENAME}.log" 2>&1
done

conda deactivate