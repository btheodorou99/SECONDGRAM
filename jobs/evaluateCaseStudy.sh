#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=512g
#SBATCH --cpus-per-task=16
#SBATCH --job-name=combineResults

module load python
python /home/ImageGen/evaluation/evaluate_case_study.py
python /home/ImageGen/evaluation/plotCaseStudy.py