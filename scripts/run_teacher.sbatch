#!/bin/bash
#SBATCH --job-name=beyond-teacher
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/teacher-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load Python/3.12.3-GCCcore-13.3.0 Java/21 PostgreSQL/16.4-GCCcore-13.3.0
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export USE_TORCH=0
export USE_TF=0
exec .venv/bin/python -u -m teacher.serve "$@"
