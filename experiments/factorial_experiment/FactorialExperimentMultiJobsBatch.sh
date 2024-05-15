#!/usr/bin/env bash

#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --job-name=FactorialExperiment
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


base_seed=3324000
experiment_count=$3

for p in 2 6 16
do
    for target_power in 0 20 40 60 80 95
    do
        experiment_count=$(($experiment_count + 1))
        python factorial_experiment.py \
            --target-power $target_power \
            -n 100 \
            -p $p \
            -B 999 \
            --trials 1000 \
            --covariate-distribution $1 \
            --epsilon-distribution $2 \
            --seed $(($base_seed + $experiment_count)) \
            --monte-carlo-trials 40000 \
            --f-test-tolerance 0.00001
    done
done
