#!/bin/bash
#SBATCH -J llm-express
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -o /data/nfs-hpc/llm-express/logs/%x-%j.out
#SBATCH -e /data/nfs-hpc/llm-express/logs/%x-%j.err

set -euo pipefail

BASE="/data/nfs-hpc/llm-express"
mkdir -p "$BASE"/{hf_cache,checkpoints,logs,code}

echo "=== SLURM ==="
echo "JobID: ${SLURM_JOB_ID}"
echo "Node:  ${SLURM_NODELIST}"

nvidia-smi

export HF_HOME="$BASE/hf_cache"
export TRANSFORMERS_CACHE="$BASE/hf_cache/transformers"
export HF_DATASETS_CACHE="$BASE/hf_cache/datasets"

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=enp3s0f0
export NCCL_DEBUG=WARN

# Verificar dependencias antes de consumir tiempo de GPU
micromamba run -n llm python - <<'PY'
mods = ["torch", "transformers", "datasets", "accelerate"]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        missing.append((m, str(e)))
if missing:
    print("Missing/import errors:", missing)
    raise SystemExit(2)
print("Deps OK")
PY

OUTDIR="$BASE/checkpoints/gpt2-tinystories-express"

timeout --signal=TERM --kill-after=60s 170m \
  srun --kill-on-bad-exit=1 micromamba run -n llm \
    python "$BASE/code/run_clm.py" \
      --model_name_or_path openai-community/gpt2 \
      --dataset_name roneneldan/TinyStories \
      --do_train \
      --block_size 256 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 16 \
      --learning_rate 5e-5 \
      --warmup_steps 200 \
      --fp16 \
      --max_steps 3500 \
      --save_steps 500 \
      --save_total_limit 3 \
      --logging_steps 20 \
      --output_dir "$OUTDIR" \
      --report_to none

echo "DONE"
