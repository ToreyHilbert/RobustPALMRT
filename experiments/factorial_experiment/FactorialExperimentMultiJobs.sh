#!/usr/bin/env bash

experiment_count=0
for cov in Normal t3 Cauchy BalancedAnova
do
    for eps in Normal t3 Cauchy Multinomial LogNormal
    do
        sbatch -p stat FactorialExperimentMultiJobsBatch.sh $cov $eps $experiment_count
        experiment_count=$(($experiment_count + 20))
    done
done
