#!/bin/bash
#SBATCH --job-name=generic_gpu_job
#SBATCH --output=slurm_logs/generic_gpu_%j.out
#SBATCH --error=slurm_logs/generic_gpu_%j.err
#SBATCH --partition=L4-4h        # Partition name
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --cpus-per-task=4        # Request 4 CPU cores
#SBATCH --mem=16G                # Request 16GB of RAM
#SBATCH --time=04:00:00          # Time limit (hh:mm:ss)
#SBATCH --mail-type=BEGIN,END,FAIL

# ======================================================================
# GENERIC GPU SLURM SCRIPT
# Usage: sbatch slarm_code/run_generic_gpu.slurm <script_path> [args...]
# ======================================================================

set -e

if [ $# -eq 0 ]; then
    echo "❌ Error: No python script provided."
    echo "Usage: sbatch run_generic_gpu.slurm <script_path> [arguments...]"
    exit 1
fi

SCRIPT_PATH=$1
shift # Remove the script path from arguments list, leaving the rest

echo "🚀 Starting GPU job: $SCRIPT_PATH"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Arguments: $@"

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs
source .venv/bin/activate

python "$SCRIPT_PATH" "$@"

echo "✅ GPU Job completed at $(date)"
