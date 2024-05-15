#!/usr/bin/env bash

#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --job-name=SampleSize
#SBATCH --output=logs/python-%x.%j.out
#SBATCH --error=logs/python-%x.%j.err

# Move to directory from which files were submitted
cd $SLURM_SUBMIT_DIR

# Load the miniconda module
module load miniconda3/4.11.0-py38 

# If dpt-sim environment is not created:
conda create -n dpt-sim python=3.9 -y
source activate dpt-sim
pip install -r requirements.txt

# If dpt-sim environment has been previously created:
# source activate dpt-sim


base_seed=50924000
experiment_count=$4

for target_power in 0 20 40 60 80 95
do
    experiment_count=$(($experiment_count + 1))
    python sample_size_experiment.py \
        --target-power $target_power \
        -n $1 \
        -p 6 \
        -B 999 \
        --trials 1000 \
        --covariate-distribution $2 \
        --epsilon-distribution $3 \
        --seed $(($base_seed + $experiment_count)) \
        --monte-carlo-trials 40000 \
        --f-test-tolerance 0.00001
done
