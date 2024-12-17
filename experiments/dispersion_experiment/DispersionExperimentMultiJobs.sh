#!/usr/bin/env bash

experiment_count=0
for n in 100 200 400
do
    for eps in Normal Cauchy LogNormal
    do
        sbatch -p stat DispersionExperimentMultiJobsBatch.sh $n $eps $experiment_count
        experiment_count=$(($experiment_count + 5))
    done
done
