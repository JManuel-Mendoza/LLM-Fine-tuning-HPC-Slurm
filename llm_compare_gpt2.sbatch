#!/bin/bash
#SBATCH -J llm-compare
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH -t 00:15:00
#SBATCH -o /data/nfs-hpc/llm-express/logs/%x-%j.out
#SBATCH -e /data/nfs-hpc/llm-express/logs/%x-%j.err

set -euo pipefail

BASE_DIR="/data/nfs-hpc/llm-express"
CKPT="$BASE_DIR/checkpoints/gpt2-tinystories-express/checkpoint-2000"
BASE_MODEL="openai-community/gpt2"

echo "JobID: ${SLURM_JOB_ID}"
echo "Node:  ${SLURM_NODELIST}"
nvidia-smi

export HF_HOME="$BASE_DIR/hf_cache"
export TRANSFORMERS_CACHE="$BASE_DIR/hf_cache/transformers"
export HF_DATASETS_CACHE="$BASE_DIR/hf_cache/datasets"

CKPT="$CKPT" BASE_MODEL="$BASE_MODEL" micromamba run -n llm \
  python "$BASE_DIR/code/compare_base_vs_ckpt.py"
