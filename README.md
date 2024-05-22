# Robust conformal tests for the linear model

This repository is the official implementation of "Robust conformal tests for the linear model".

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

To run the MY-LC analysis, you will need to download the [MY-LC dataset from Klein et al.](https://doi.org/10.1038/s41586-023-06651-y), and place the file `41586_2023_6651_MOESM4_ESM.csv` in the folder `./MY-LC/MY-LC-Data/`.


## Running new simulations

To run simulations similar to ours with new parameters, see `./experiments/simple_experiment/` folder. The file `./experiments/simple_experiment/simple_experiment.py` is an easily modifiable version of our simulations, where one can replace a single function (`gen_data` on line 32) to test different settings for data generation. 

Here is an example usage:
```sh
python simple_experiment.py --target-power 20 -B 999 --trials 1000 --seed 52224000 --monte-carlo-trials 10000 --f-test-tolerance 0.00001 --name abs-t3 --cpus 5
```

## Reproducing our exact simulations

We ran the simulations on a cluster with 40 CPUs @ 2.60GHz, using a Slurm system using the `sbatch` command. The total computation time was approximately 109 hours. For reproducibility, we additionally provide commands to exactly rerun our simulations on such a system.

To exactly run the factorial experiment, run the following commands:
```sh
cd ./experiments/factorial_experiment
mkdir logs
. FactorialExperimentMultiJobs.sh
```

To exactly run the sample size experiment, run the following commands:
```sh
cd ./experiments/sample_size_experiment
mkdir logs
. SampleSizeMultiJobs.sh
```

To reproduce the figures from the simulations, we provide two Jupyter Notebooks: create_main_paper_simulation_figure.ipynb and create_appendix_simulation_figures.ipynb

## Running the MY-LC analysis

We ran the MY-LC real data analysis on a personal computer 5 CPUs @ 2.60GHz (using Windows). The total computation time was approximately 1 hour.

To reproduce the computations and figures for the MY-LC analysis, use the following commands:
```sh
cd ./MY-LC/
python MY-LC-Computations.py
python MY-LC-Figures.py
```
