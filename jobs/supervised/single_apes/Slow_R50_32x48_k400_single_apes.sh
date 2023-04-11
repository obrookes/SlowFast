#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --partition=devel
#SBATCH --time=0-00:10:00
#SBATCH --mem=128GB
#SBATCH --job-name=slow_r50_single-apes_32x48_k400
#SBATCH --output=%x-%j.out


module load cuda/11.2

source ~/.bashrc
conda activate slowfast_torch113_cu116 

cd ~/SlowFast

export PYTHONWARNINGS="ignore"

python tools/run_net.py --cfg configs/Kinetics/single_ape/Slow_R50_32x48_k400_SINGLE_APES.yaml