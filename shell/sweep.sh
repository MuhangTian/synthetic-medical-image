#!/bin/bash
#SBATCH --job-name=S-DDPM
#SBATCH --time=10-00:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:A5000:1
#SBATCH --mem=50G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=logs/%j.out
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

srun python \
    train.py \
    --sweep_id $sweep_id \
    --load_path $load_path \