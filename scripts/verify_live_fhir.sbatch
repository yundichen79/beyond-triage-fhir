#!/bin/bash
#SBATCH --job-name=beyond-fhir
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/fhir-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
module load Java/21 PostgreSQL/16.4-GCCcore-13.3.0
# exec allows the Python runner to receive termination and stop both servers.
exec python3 -u -m dataset.live run "$@"
