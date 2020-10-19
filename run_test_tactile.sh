#!/bin/bash
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH -n 1 
#SBATCH --mem=14G
##S;sdflfjeisla    TCH --gres=gpu:1
#module load openmind/anaconda/3-2019.10
python test_reservoir_PFCMD.py $1 $2 $3
