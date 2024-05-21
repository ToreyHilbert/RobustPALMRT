#!/usr/bin/env bash

experiment_count=0
for n in 25 50 200 400
    for cov in Normal t3 Cauchy BalancedAnova
    do
        for eps in Normal t3 Cauchy Multinomial LogNormal
        do
            sbatch -p stat SampleSizeMultiJobsBatch.sh $n $cov $eps $experiment_count
            experiment_count=$(($experiment_count + 6))
        done
    done
done