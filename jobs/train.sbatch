#!/bin/bash
#
#SBATCH --job-name=e_ma
#SBATCH --cpus-per-task=2
#SBATCH --partition=week
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/job.%J.out
#SBATCH --error=jobs/job.%J.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=johannes.zenn@student.uni-tuebingen.de

MAIN="$HOME/MomentAlignment/main.py"
CONFIGURATION="$1"

singularity exec --nv /common/singularityImages/TCML-Cuda10_0Tensorflow2_0_0.simg python -u "$MAIN" "$CONFIGURATION"
