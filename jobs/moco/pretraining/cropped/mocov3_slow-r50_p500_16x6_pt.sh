#!/bin/bash    

#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --partition=big
#SBATCH --time=1-00:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=mocov3_slow-r50_p500_16x24_pt
#SBATCH --output=%x-%j.out

module load cuda/11.2

source ~/.bashrc
conda activate slowfast_torch113_cu116 

cd ~/SlowFast

export PYTHONWARNINGS="ignore"

python tools/run_net.py --cfg configs/contrastive_ssl/MoCo/pretraining/MoCo_K400_SlowR50_Cropped_PanAf500_16x6.yaml