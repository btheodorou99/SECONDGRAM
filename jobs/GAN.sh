#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32g
#SBATCH --job-name=gan
#SBATCH --array=1-100
#SBATCH --error=job_errors_%A_%a.err
#SBATCH --output=job_outputs_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1

module load python
python /home/SECONDGRAM/trainGAN_seed.py $SLURM_ARRAY_TASK_ID
python /home/SECONDGRAM/generateGAN_seed.py $SLURM_ARRAY_TASK_ID
python /home/SECONDGRAM/evaluation/evaluate_training_gan_seed.py $SLURM_ARRAY_TASK_ID
python /home/SECONDGRAM/evaluation/evaluate_training_unscaled_gan_seed.py $SLURM_ARRAY_TASK_ID
python /home/SECONDGRAM/evaluation/evaluate_modeling_gan_seed.py $SLURM_ARRAY_TASK_ID