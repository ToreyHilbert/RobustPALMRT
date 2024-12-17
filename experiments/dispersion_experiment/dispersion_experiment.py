import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from itertools import repeat

import time
import argparse
from pathlib import Path

from multiprocess import Pool, freeze_support


# Floating point safe vectorized comparison function
W = lambda M_1, M_2: (~np.isclose(M_1, M_2) & (M_1 > M_2)) + 0.5 * np.isclose(M_1, M_2)

############################################
# Put code to generate the data here!
############################################

def standardize_columns(mat):
    mat = mat - np.mean(mat, axis = 0)
    mat = np.sqrt(mat.shape[0]) * mat / np.linalg.norm(mat, ord = 2, axis = 0)
    return mat

log_normal_sigma = 1.
log_normal_sd = np.sqrt((np.exp(log_normal_sigma**2) - 1) * np.exp(log_normal_sigma**2))

# Note: The first column of Z *MUST* be an all ones column (due to M calculations)
def gen_data(beta, n, p, epsilon_distribution, thread_rng):
    X = np.concatenate([
        np.ones((n // 2)),
        np.zeros((n - n // 2))
    ], axis = 0)
    Z = standardize_columns(thread_rng.standard_normal(size = (n, p - 1)))
    Z = np.concatenate([np.ones((n, 1)), Z], axis = 1)

    # Generate epsilon
    if epsilon_distribution == "Normal":
        e = thread_rng.standard_normal(size = (n,))
    elif epsilon_distribution == "Cauchy":
        e = thread_rng.standard_cauchy(size = (n,))
    elif epsilon_distribution == "LogNormal":
        e = thread_rng.lognormal(mean = 0., sigma = log_normal_sigma, size = (n,))/log_normal_sd

    Y = (1 + beta * X) * e
    
    return X.reshape(-1, 1), Z, Y

############################################
# Run the experiment
############################################

# This is the function that will be parallelized across TRIALS many threads
def compute_pvals_for_trial(B, n, p, epsilon_distribution, beta, gen_data, thread_rng):
    X, Z, Y = gen_data(beta, n, p, epsilon_distribution, thread_rng)
    n = Y.shape[0]

    qlow = 0.10
    qhigh = 0.90
    fit_quantile_regression = lambda X, Y, quantile: QuantileRegressor(
        quantile = quantile, alpha = 0, solver = "highs").fit(X, Y).predict(X)

    dispersion_accept_count = 1
    for _ in range(B):
        pi = thread_rng.permutation(Y.shape[0])
        
        # X Original Calculations
        X_full_pi1 = np.concatenate([X, Z[pi], Z[:,1:]], axis = 1)
        Y_pred_low_pi1 = fit_quantile_regression(X_full_pi1, Y, qlow)
        Y_pred_high_pi1 = fit_quantile_regression(X_full_pi1, Y, qhigh)
        pi1_mask = X.reshape(-1) != 0

        group_0_scale_mean_pi1 = np.mean(np.abs(Y_pred_high_pi1[pi1_mask] - Y_pred_low_pi1[pi1_mask]))
        group_1_scale_mean_pi1 = np.mean(np.abs(Y_pred_high_pi1[~pi1_mask] - Y_pred_low_pi1[~pi1_mask]))
        scale_ratio_pi1 = -np.abs(np.log(group_1_scale_mean_pi1/group_0_scale_mean_pi1))

        # X Permuted Calculations
        X_full_pi2 = np.concatenate([X[pi], Z[pi], Z[:,1:]], axis = 1)
        Y_pred_low_pi2 = fit_quantile_regression(X_full_pi2, Y, qlow)
        Y_pred_high_pi2 = fit_quantile_regression(X_full_pi2, Y, qhigh)

        pi2_mask = X.reshape(-1)[pi] != 0
        group_0_scale_mean_pi2 = np.mean(np.abs(Y_pred_high_pi2[pi2_mask] - Y_pred_low_pi2[pi2_mask]))
        group_1_scale_mean_pi2 = np.mean(np.abs(Y_pred_high_pi2[~pi2_mask] - Y_pred_low_pi2[~pi2_mask]))
        scale_ratio_pi2 = -np.abs(np.log(group_1_scale_mean_pi2/group_0_scale_mean_pi2))

        dispersion_accept_count += W(scale_ratio_pi1, scale_ratio_pi2)

    dispersion_pval = dispersion_accept_count / (B + 1)

    return {
        ""
        "dispersion-pval" : dispersion_pval
    }

if __name__ == "__main__":
    freeze_support()

    # For recording the time that the simulation was run
    start_time = time.time()
    stamp = int(start_time)

    # First we read in the command line information
    parser = argparse.ArgumentParser(
        description = "Runs a simple experiment using the gen_data function as the data generating mechanism"
    )
    parser.add_argument("-B", type = int, required = True, help = "Number of permutations to use")
    parser.add_argument("-t", "--trials", type = int, required = True, help = "Number of design matrices to draw")

    parser.add_argument("-n", type = int, required = True, help = "Sample size to use")
    parser.add_argument("-p", type = int, required = True, help = "Number of nuisance covariates, including intercept")
    parser.add_argument("--epsilon-distribution", choices = [
        "Normal", "Cauchy", "LogNormal"
    ], required = True, help = "Distribution for epsilons")
    parser.add_argument("--beta", type = float, help = "Beta to use in dispersion")

    parser.add_argument("--seed", type = int, help = "Seed for random numbers")
    parser.add_argument("--name", help = "Give a name of the experiment")

    args = parser.parse_args()

    trials, seed = args.trials, args.seed
    B, n, p, beta  = args.B, args.n, args.p, args.beta
    epsilon_distribution = args.epsilon_distribution

    sim_id = f"{epsilon_distribution}-{n}-{beta}"

    rng = np.random.default_rng(seed)

    # Create location for results
    folder_path = "./results"
    folder_dir = Path(folder_path)
    folder_dir.mkdir(parents = True, exist_ok = True)
    file_name = f"{folder_path}/Power-{sim_id}_stamp{stamp}_t{trials}_B{B}_n{n}_p{p}_beta{beta}_edist{epsilon_distribution}_seed{seed}.csv"

    print(f"\nRunning simulation ID-{sim_id}, beta-{beta} at {time.ctime()}")
    print(f"Results will be outputted to file", file_name)

    with Pool() as pool:
        all_pvals = pool.starmap(
            compute_pvals_for_trial, zip(
                repeat(B),
                repeat(n),
                repeat(p),
                repeat(epsilon_distribution),
                repeat(beta),
                repeat(gen_data),
                rng.spawn(trials)
            )
        )
    all_pvals_df = pd.DataFrame(all_pvals)
    all_pvals_df.to_csv(file_name)

    end_time = time.time()
    print(f"Simulations done! Total ellapsed time: {end_time - start_time:.2f}")
