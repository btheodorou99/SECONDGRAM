# SECONDGRAM: Self-Conditioned Diffusion with Gradient Manipulation
## For Longitudinal MRI Imputation

### Overview

This repository contains the implementation and experimental pipelines for evaluating SECONDGRAM, our proposed method for longitudinal MRI imputation, alongside three baseline methods: Vanilla Diffusion, Pretrained Diffusion, and GAN along with two ablations NoGrad and NoCombined. Each method has its own training, generation, and evaluation pipeline, culminating in a comprehensive combination of results for downstream analysis.

The repository is structured to support both horizontal scaling using SLURM for multiple experimental runs with different random seeds and manual execution for individual runs. It also includes utilities for working with the UK Biobank dataset, including support for generating dummy data to test the pipelines.

### Repository Structure
* `jobs/`: Contains SLURM job scripts for running the experimental pipelines for each method ({key}.sh) and scripts for combining and plotting results.
* `utils/`: Includes scripts for generating datasets, preprocessing data, and other auxiliary utilities.
* `evaluation/`: Contains scripts for evaluating model evaluation and result plotting.
* `models/`: Implementation of different model architectures
* `./`: Training and generation scripts for each method.

### Methods

The repository implements five experimental methods:
1.	SECONDGRAM (our proposed method)
2.	Vanilla
3.  GAN
4.	Pretrained
5.	NoGrad
6.	NoCombined

Each method has an independent pipeline but follows the same general steps:
1.	Training
2.	Generation
3.	Evaluation (Training, Modeling, and Case Study)

### Running Experiments

#### Using SLURM for Batch Execution

Each method’s end-to-end process (training, generation, and evaluation) can be executed using SLURM job scripts located in the jobs/ directory:

`sbatch jobs/{key}.sh`

Adjust the placeholder home in each job script to the base path pointing to the folder containing this SECONDGRAM directory.

Key scripts:
* Training, generation, and evaluation for each method: `jobs/{key}.sh`
* Case study evaluation: `jobs/evaluateCaseStudy.sh`

Combining and plotting results:
* `jobs/combineResults.sh`
* `jobs/combineResultsModeling.sh`
* `jobs/combineResultsTraining.sh`
* `jobs/covariateResults.sh`

To adjust the number of experimental runs:
1. Modify the line `#SBATCH --array=1-100` in each job script.
2. Change `num_runs = 100` in the corresponding configuration files.

#### Manual Execution for Individual Runs

You can execute individual runs without SLURM by running the following commands, replacing {key} with the method and {seed} with the random seed:

`python /home/SECONDGRAM/train{key}_seed.py {seed}`
`python /home/SECONDGRAM/generate{key}_seed.py {seed}`
`python /home/SECONDGRAM/evaluation/evaluate_training_{key}_seed.py {seed}`
`python /home/SECONDGRAM/evaluation/evaluate_training_unscaled_{key}_seed.py {seed}`
`python /home/SECONDGRAM/evaluation/evaluate_modeling_{key}_seed.py {seed}`

Use the printed results directly or adjust the combination scripts (covariateResults and evaluateCaseStudy) to generate plots.

### Dataset Setup

#### Using UK Biobank Dataset

Update the dataset path in `utils/genDataset.py` by changing the path `"/data/UKB_Data/ukb672504_imaging.csv"`

UK Biobank data is available for researchers. Refer to the UK Biobank website for access instructions.

#### Using Dummy Data

For testing, you can generate dummy data by running:

`python utils/genDummyDataset.py`

All subsequent scripts will work with the dummy data output.
