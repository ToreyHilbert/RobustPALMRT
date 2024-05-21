# Robust conformal tests for the linear model

This repository is the official implementation of "Robust conformal tests for the linear model".

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

To run the MY-LC analysis, you will need to download the [MY-LC dataset from Klein et al.](https://doi.org/10.1038/s41586-023-06651-y), and place the file `41586_2023_6651_MOESM4_ESM.csv` in the folder `./MY-LC/MY-LC-Data/`.


## Running new simulations

To run simulations similar to ours with new parameters, ...........

## Reproducing our exact simulations

We ran the simulations on a cluster with 40 CPUs @ 2.60GHz, using a Slurm system using the `sbatch` command. For reproducibility, we additionally provide commands to exactly rerun our simulations on such a system.

To exactly run the sample size experiment, run the following commands:
```sample_size_sims
cd ./experiments/sample_size_experiment
mkdir logs
. SampleSizeMultiJobs.sh
. SampleSizeMultiJobs-FollowUp.sh
```

To exactly run the factorial experiment, run the following commands:
```sample_size_sims
cd ./experiments/factorial_experiment
mkdir logs
. FactorialExperimentMultiJobs.sh
```

## Running the MY-LC analysis




## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 