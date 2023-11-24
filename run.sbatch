#!/bin/bash
#SBATCH --job-name=toxic_comment_classification
#SBATCH --partition=gpu-5h
#SBATCH --gpus-per-node=40gb:2
#SBATCH --ntasks-per-node=4
#SBATCH --output=./output_folder/output.txt
#SBATCH --error=./output_folder/error.txt

module load apptainer

./run.sh
