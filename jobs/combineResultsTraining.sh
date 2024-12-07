#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=512g
#SBATCH --cpus-per-task=16
#SBATCH --job-name=combineResultsTraining

module load python
python /home/SECONDGRAM/evaluation/combineResultsTraining.py