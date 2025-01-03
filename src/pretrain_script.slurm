#!/bin/bash
#SBATCH --job-name=DNABART_Pretrain             # Job name
#SBATCH --mail-type=ALL                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mpg8nm@umsystem.edu         # Where to send mail
#SBATCH --output=output_%j.log                  # Standard output and error log (%j = Job ID)
#SBATCH --error=error_%j.log                    # Error log (%j = Job ID)
#SBATCH --time=72:00:00                         # Time limit (HH:MM:SS)
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks=1                              # Number of tasks (processes)
#SBATCH --cpus-per-task=4                       # Number of CPU cores per task
#SBATCH --mem=64G                               # Memory per node
#SBATCH --gres=gpu:H100:1                       # Request one GPU (adjust as needed)
#SBATCH --partition=gpu                         # Partition to use (change to appropriate GPU partition name)

pwd; hostname;
echo "Started at $(date)"


# Load necessary modules
source activate dnabart

# Change to the project directory
cd /share/vast/spxzb-lab/mattStuff/DNABART/src

# Run your training script
python train.py

# Optionally, add commands for cleanup or logging here
