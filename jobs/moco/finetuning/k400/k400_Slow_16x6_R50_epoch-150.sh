#!/bin/bash    

#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --partition=big
#SBATCH --time=1-00:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=moco_ft_k400_epoch-150
#SBATCH --output=%x-%j.out

module load cuda/11.2

source ~/.bashrc
conda activate slowfast_torch113_cu116 

cd ~/SlowFast

export PYTHONWARNINGS="ignore"

python tools/run_net.py --cfg configs/contrastive_ssl/MoCo/finetuning/k400/k400_Slow_16x6_R50_epoch-150.yaml