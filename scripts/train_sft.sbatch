#!/bin/bash
#SBATCH --job-name=beyond-sft
#SBATCH --account=cpsc4880
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --constraint="gpu:a40|gpu:l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/sft-%j.log

# Usage (from the repository root):
#   sbatch scripts/train_sft.sbatch --data outputs/sft-data/<variant> --output outputs/sft-runs/<run> [options]
# Options are the sft.train settings, e.g. --epochs 2 --learning-rate 1e-4 --lora-r 16.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
module load Python/3.12.3-GCCcore-13.3.0
if [[ ! -x .venv/bin/python ]]; then
  echo "Run bash scripts/prepare_model.sh from the repository first." >&2
  exit 1
fi
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .venv/bin/python -u -m sft.train "$@"
