#!/bin/bash
# ----------------SLURM Parameters----------------
#SBATCH -p gpu
#SBATCH -n 8
#SBATCH --mem=18g
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -J deepzip
#SBATCH -o deepzip
# ----------------Load Modules--------------------
# module load CUDA/9.0.176-IGB-gcc-4.9.4
# module load Keras/2.2.4-IGB-gcc-4.9.4-Python-3.6.1
# module load Tensorflow-GPU/1.14.0-IGB-gcc-4.9.4-Python-3.6.1
source activate p2
# ----------------Commands------------------------
bash compress.sh
