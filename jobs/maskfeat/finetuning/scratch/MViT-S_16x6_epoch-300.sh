#!/bin/bash    

#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --partition=big
#SBATCH --time=0-04:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=maskfeat_ft_epoch-300
#SBATCH --output=%x-%j.out

module load cuda/11.2

source ~/.bashrc
conda activate slowfast_torch113_cu116 

cd ~/SlowFast

export PYTHONWARNINGS="ignore"

python tools/run_net.py --cfg configs/masked_ssl/MaskFeat/scratch/MVITv2_S_16x4_FT_epoch-300.yaml
