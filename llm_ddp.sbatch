#!/bin/bash
#SBATCH -J llm-ddp
#SBATCH -p gpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -o /data/nfs-hpc/llm-express/logs/%x-%j.out
#SBATCH -e /data/nfs-hpc/llm-express/logs/%x-%j.err

set -euo pipefail

BASE="/data/nfs-hpc/llm-express"
mkdir -p "$BASE"/{logs,checkpoints,hf_cache,code}

echo "=== SLURM ==="
echo "JobID:   ${SLURM_JOB_ID}"
echo "Nodos:   ${SLURM_NODELIST}"
echo "N nodos: ${SLURM_NNODES}"
echo "============="

export HF_HOME="$BASE/hf_cache"
export TRANSFORMERS_CACHE="$BASE/hf_cache/transformers"
export HF_DATASETS_CACHE="$BASE/hf_cache/datasets"
export TOKENIZERS_PARALLELISM=false

# NCCL: excluir loopback y redes virtuales; usar la interfaz interna
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="^lo,docker0,virbr0"
export NCCL_DEBUG=WARN

# El primer nodo de la lista actua como coordinador del rendez-vous
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# Log periodico de GPU por nodo (cada 30 segundos)
GPU_LOG="$BASE/logs/gpu-${SLURM_JOB_ID}-${SLURM_NODEID}.log"
(while true; do
    date
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
               --format=csv,noheader,nounits
    echo "----"
    sleep 30
done) > "$GPU_LOG" 2>&1 &
GPU_WATCH_PID=$!
trap 'kill $GPU_WATCH_PID 2>/dev/null || true' EXIT

OUTDIR="$BASE/checkpoints/gpt2-tinystories-ddp"

timeout 170m srun --kill-on-bad-exit=1 \
  micromamba run -n llm \
    torchrun \
      --nnodes="${SLURM_NNODES}" \
      --nproc_per_node=1 \
      --rdzv_backend=c10d \
      --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
      "$BASE/code/train_distributed.py" \
        --model_name_or_path openai-community/gpt2 \
        --output_dir "$OUTDIR" \
        --cache_dir "$BASE/hf_cache" \
        --block_size 256 \
        --max_steps 2000 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-5 \
        --warmup_steps 100 \
        --logging_steps 25 \
        --save_steps 500 \
        --save_total_limit 2

echo "=== DONE ==="
