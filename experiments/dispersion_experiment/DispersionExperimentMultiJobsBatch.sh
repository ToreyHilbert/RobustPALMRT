#!/usr/bin/env bash

#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --job-name=DispersionExperiment
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


base_seed=112924000
experiment_count=$3

for beta in 0 0.5 1.0 1.5 2.0
do
    experiment_count=$(($experiment_count + 1))
    python dispersion_experiment.py \
        -n $1 \
        -p 6 \
        -B 999 \
        --trials 1000 \
        --beta $beta \
        --epsilon-distribution $2 \
        --seed $(($base_seed + $experiment_count))
done
