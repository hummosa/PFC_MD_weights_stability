#!/bin/bash
#SBATCH -t 05:00:00
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
module load openmind/anaconda/3-2019.10
python reservoir_PFCMD_gridsearch.py $1 $2
