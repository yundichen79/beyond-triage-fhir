#!/bin/bash
#SBATCH --job-name=beyond-synthea
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/synthea-%j.log

set -euo pipefail

# Submit from the repository root; add site account/partition flags if required.
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
module load Java/17
python3 -m dataset.cohort generate --batch "${1:-1}"
