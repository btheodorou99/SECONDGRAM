#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=evaluate_modeling_pretrained
#SBATCH --array=1-100
#SBATCH --error=job_errors_%A_%a.err
#SBATCH --output=job_outputs_%A_%a.out

module load python
python /home/ImageGen/evaluation/evaluate_modeling_pretrained_seed.py $SLURM_ARRAY_TASK_ID