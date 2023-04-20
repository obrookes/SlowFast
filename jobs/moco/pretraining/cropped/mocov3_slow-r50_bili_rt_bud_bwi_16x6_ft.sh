#!/bin/bash    

#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --partition=devel
#SBATCH --time=0-00:10:00
#SBATCH --mem=256GB
#SBATCH --job-name=mocov3_slow-r50_bili_rt_bud_bwi_16x6_epoch-300_ft
#SBATCH --output=%x-%j.out

module load cuda/11.2

source ~/.bashrc
conda activate slowfast_torch113_cu116 

cd ~/SlowFast

export PYTHONWARNINGS="ignore"

python tools/run_net.py --cfg configs/contrastive_ssl/MoCo/pretraining/MoCo_K400_SlowR50_Cropped_Bili_RT_Bud_Bwi_16x6_FT.yaml