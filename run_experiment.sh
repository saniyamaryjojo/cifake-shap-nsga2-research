#!/bin/bash
#SBATCH --job-name=cifake_nsga2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=nsga2_%j.out

module load StdEnv/2023 python/3.11 arrow/19.0.1
source ~/ai-image-env/bin/activate

cd ~/AI_Image_Detection_Project

export OMP_NUM_THREADS=4

python -u run_nsga2_acenet.py \
    --dataset-path cifake_dataset \
    --pop-size 6 \
    --generations 3 \
    --output nsga2_results.csv
