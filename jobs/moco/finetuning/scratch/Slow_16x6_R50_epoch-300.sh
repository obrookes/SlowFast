#!/bin/bash    

#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --partition=big
#SBATCH --time=0-08:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=moco_ft_epoch-300
#SBATCH --output=%x-%j.out

module load cuda/11.2

source ~/.bashrc
conda activate slowfast_torch113_cu116 

cd ~/SlowFast

export PYTHONWARNINGS="ignore"

python tools/run_net.py --cfg configs/contrastive_ssl/MoCo/finetuning/scratch/Slow_16x6_R50_epoch-300.yaml

