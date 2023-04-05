#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --partition=devel
#SBATCH --time=0-00:10:00
#SBATCH --mem=128GB
#SBATCH --job-name=mvitv2_b_single_apes_k400_32x48
#SBATCH --output=out/%x-%j.out


module load cuda/11.2

source ~/.bashrc
conda activate slowfast_torch113_cu116 

cd ~/SlowFast

export PYTHONWARNINGS="ignore"

python tools/run_net.py --cfg configs/Kinetics/single_ape/MViTv2_B_32x48_k400_SINGLE_APES.yaml