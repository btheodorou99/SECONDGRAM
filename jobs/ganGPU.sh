#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16g
#SBATCH --job-name=gan
#SBATCH --array=1-100
#SBATCH --error=job_errors_%A_%a.err
#SBATCH --output=job_outputs_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1

module load python
python /home/ImageGen/trainGAN_seed.py $SLURM_ARRAY_TASK_ID
python /home/ImageGen/generateGAN_seed.py $SLURM_ARRAY_TASK_ID
python /home/ImageGen/evaluation/evaluate_training_gan_seed.py $SLURM_ARRAY_TASK_ID
python /home/ImageGen/evaluation/evaluate_training_unscaled_gan_seed.py $SLURM_ARRAY_TASK_ID