#!/bin/bash
#SBATCH --job-name=beyond-pilot
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/pilot-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
module load Python/3.12.3-GCCcore-13.3.0 Java/21 PostgreSQL/16.4-GCCcore-13.3.0
if [[ ! -x .venv/bin/python ]]; then
  echo "Run bash scripts/prepare_pilot.sh from the repository first." >&2
  exit 1
fi
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
exec .venv/bin/python -u -m pilot.run run "$@"
