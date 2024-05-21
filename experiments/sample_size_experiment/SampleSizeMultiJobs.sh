#!/usr/bin/env bash

experiment_count=0
for n in 25 50 200 400
do
    for i in \
        "Normal Normal" \
        "Normal t3" \
        "Normal Cauchy" \
        "Normal LogNormal" \
        "Normal Multinomial" \
        "Cauchy Normal" \
        "Cauchy t3" \
        "Cauchy Cauchy" \
        "Cauchy LogNormal" \
        "Cauchy Multinomial" \
        "t3 LogNormal" \
        "BalancedAnova LogNormal"
    do
        cov_eps=( $i )
        sbatch -p stat SampleSizeMultiJobsBatch.sh $n ${cov_eps[0]} ${cov_eps[1]} $experiment_count
        experiment_count=$(($experiment_count + 12))
    done
done