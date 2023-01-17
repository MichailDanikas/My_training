#!/usr/local_rwth/bin/zsh
 
#SBATCH -J gpu_s
#SBATCH -o gpu_.%J.log
#SBATCH -t 1-0:00:00
#SBATCH --mem=96GB
#SBATCH --gres=gpu:volta:2
 
module load cuda/11.0
module load python
 
#print some debug informations...
nvidia-smi;

echo "Job started at $(date +%Y-%m-%d_%H:%M:%S)"
python3.9 save_model.py
echo "Job ended at $(date +%Y-%m-%d_%H:%M:%S)"

