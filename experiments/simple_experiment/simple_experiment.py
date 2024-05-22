import numpy as np
import pandas as pd

from scipy.special import huber
from scipy.optimize import brentq
import scipy.stats.distributions as distributions

from itertools import repeat

import time
import argparse
from pathlib import Path

from multiprocess import Pool, freeze_support


huber_delta = 1.345
mad_scale_factor = 0.7649
alpha = 0.05

# Floating point safe vectorized comparison function
W = lambda M_1, M_2: (~np.isclose(M_1, M_2) & (M_1 > M_2)) + 0.5 * np.isclose(M_1, M_2)

############################################
# Put code to generate the data here!
############################################

n = 100
p = 6

# Note: The first column of Z *MUST* be an all ones column (due to M calculations)
def gen_data(beta, thread_rng):
    X = standardize_columns(thread_rng.standard_normal(size = (n,)))
    Z = standardize_columns(thread_rng.standard_normal(size = (n, p - 1)))
    Z = np.concatenate([np.ones((n, 1)), Z], axis = 1)

    e = np.abs(thread_rng.standard_t(3, size = (n,)))
    Y = X * beta + e
    
    return X.reshape(-1, 1), Z, Y

def standardize_columns(mat):
    mat = mat - np.mean(mat, axis = 0)
    mat = np.sqrt(mat.shape[0]) * mat / np.linalg.norm(mat, ord = 2, axis = 0)
    return mat

############################################
# Methods
############################################

def OLS_L2_PALMRT_omega(pi, X, Z, Y):
    X_full_Orig = np.concatenate([X,      Z[pi], Z[:,1:]], axis = 1)
    X_full_Perm = np.concatenate([X[pi],  Z[pi], Z[:,1:]], axis = 1)

    M_Orig = Y - X_full_Orig @ np.linalg.lstsq(X_full_Orig, Y, rcond = None)[0]
    M_Perm = Y - X_full_Perm @ np.linalg.lstsq(X_full_Perm, Y, rcond = None)[0]

    return W(
        np.linalg.norm(M_Orig, ord = 2),
        np.linalg.norm(M_Perm, ord = 2)
    )

def Huber_Huber_RobustPALMRT_omega(pi, X, Z, Y):
    Z_full = np.concatenate([Z[pi], Z[:,1:]], axis = 1)
    _, scale, _ = fit_huber_regression(Y, Z_full)

    X_full_Orig = np.concatenate([X,      Z[pi], Z[:,1:]], axis = 1)
    X_full_Perm = np.concatenate([X[pi],  Z[pi], Z[:,1:]], axis = 1)

    M_Orig, _, _ = fit_huber_regression(Y, X_full_Orig, scale = scale)
    M_Perm, _, _ = fit_huber_regression(Y, X_full_Perm, scale = scale)

    return W(
        np.sum(huber(huber_delta * scale, M_Orig)),
        np.sum(huber(huber_delta * scale, M_Perm))
    )

def F_test_pval(X, Z, Y):
    X_full = np.concatenate([X, Z], axis = 1)
    degrees_of_freedom = X_full.shape[0] - X_full.shape[1]

    e1 = np.zeros((X_full.shape[1],))
    e1[0] = 1.

    pinv = np.linalg.pinv(X_full.T @ X_full) 
    A = e1.T @ pinv @ X_full.T / np.sqrt(e1.T @ pinv @ e1)
    sigma_hat = np.sqrt((Y.T @ Y - Y.T @ X_full @ pinv @ X_full.T @ Y) / degrees_of_freedom)
    
    return 2 * distributions.t(df = degrees_of_freedom).cdf(-np.abs((A @ Y) / sigma_hat))


# Fits a Huber regression of Y on X_full via IWLS, possibly with scale estimation.
# - If scale = None, then perform scale estimation via MAD.
# - If scale is a number, then perform Huber regression with fixed scale.
def fit_huber_regression(Y, X_full, scale = None,
                   mad_scale_factor = mad_scale_factor, huber_delta = huber_delta,
                   acc = 1e-8, maxiters = 200):
    Yhat = X_full @ ((np.linalg.pinv(X_full.T @ X_full) @ X_full.T) @ Y)
    R_old = Y - Yhat

    converged = False
    for _ in range(maxiters):
        if scale:
            s = scale
        else:
            # Median absolute deviation (MAD) estimator
            s = np.median(np.abs(R_old))/mad_scale_factor

        weights = np.minimum(1, huber_delta * s / np.abs(R_old))
        params = ((np.linalg.pinv((X_full.T * weights) @ X_full) @ X_full.T) @ (weights * Y))
        R = Y - X_full @ params
        if np.sum(np.square(R - R_old)) / np.sum(np.square(R_old)) < acc:
            converged = True
            break
        R_old = R
    if not converged:
        print(f"Huber regression failed to converge after {maxiters} iterations!")
    return R, s, params


############################################
# Root finding to find the value of beta
############################################

def compute_power(trials, beta, gen_data, thread_rng):
    e1 = np.zeros(p + 1)
    e1[0] = 1.

    l = []
    for _ in range(trials):
        X, Z, Y = gen_data(beta, thread_rng)
        X_full = np.concatenate([X, Z], axis = 1)

        degrees_of_freedom = X_full.shape[0] - X_full.shape[1]

        pinv = np.linalg.pinv(X_full.T @ X_full) 
        A = e1.T @ pinv @ X_full.T / np.sqrt(e1.T @ pinv @ e1)

        sigma_hat = np.sqrt((Y.T @ Y - Y.T @ X_full @ pinv @ X_full.T @ Y)  / degrees_of_freedom)
        l.append(np.abs((A @ Y) / sigma_hat))
    l = np.array(l)

    t_critical = np.abs(distributions.t(df = degrees_of_freedom).ppf(alpha/2))
    return np.mean(l > t_critical)

def find_beta_for_params(target_power, gen_data, thread_rng,
                         monte_carlo_trials, tolerance):
    beta_max = 1000.

    f = lambda beta: compute_power(
        monte_carlo_trials, beta, gen_data, thread_rng
    ) - target_power/100.
    res = brentq(f, 0., beta_max, rtol = tolerance)
    return res


############################################
# Run the experiment
############################################

# This is the function that will be parallelized across TRIALS many threads
def compute_pvals_for_trial(B, beta, gen_data, thread_rng):
    X, Z, Y = gen_data(beta, thread_rng)

    huber_huber_pval = 1
    ols_l2_pval = 1
    for _ in range(B):
        pi = thread_rng.permutation(Y.shape[0])
        huber_huber_pval += Huber_Huber_RobustPALMRT_omega(pi, X, Z, Y)
        ols_l2_pval += OLS_L2_PALMRT_omega(pi, X, Z, Y)

    huber_huber_pval = huber_huber_pval / (B + 1)
    ols_l2_pval = ols_l2_pval / (B + 1)

    return {
        "Huber-Huber-RobustPALMRT-pval" : huber_huber_pval,
        "OLS-L2-PALMRT-pval" : ols_l2_pval,
        "F-test-pval" : F_test_pval(X, Z, Y)
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
    parser.add_argument("--seed", type = int, help = "Seed for random numbers")
    parser.add_argument("--target-power", type = float, help = "Set beta by target F-test power; 0 means effect size is zero")
    parser.add_argument("--monte-carlo-trials", type = int, help = "The number of Monte Carlo trials to use for setting the correct F-test power")
    parser.add_argument("--f-test-tolerance", type = float, help = "Tolerance to use in search for correct F-test power")
    parser.add_argument("--cpus", type = int, help = "Number of CPUs to use for parallelization")
    parser.add_argument("--name", help = "Give a name of the experiment")

    args = parser.parse_args()

    sim_id, cpus = args.name, args.cpus
    trials, B, seed = args.trials, args.B, args.seed
    target_power = args.target_power
    monte_carlo_trials, tolerance = args.monte_carlo_trials, args.f_test_tolerance

    rng = np.random.default_rng(seed)

    # After reading in the command line information,
    #  we set the value of beta so that the F-test will have the correct target power.
    if target_power == 0:
        beta = np.array([0.])
    else:
        print(f"\nComputing beta for simulation ID-{sim_id}, power-{target_power:.2f}, p-{p}, n-{n} at {time.ctime()}")
        beta = find_beta_for_params(target_power, gen_data, rng, monte_carlo_trials, tolerance)
        verify_power = compute_power(4 * monte_carlo_trials, beta, gen_data, rng)
        print(f"Using beta = {beta}, with power {verify_power} (target power {target_power}), elapsed time {time.time() - start_time:.2f}")

        parameter_files_path = "computed_betas.csv"
        with open(parameter_files_path, "a+") as f:
            f.write(f"{n},{p},{sim_id},{target_power:.1f},{beta},{verify_power},{stamp}\n")

    # Create location for results
    folder_path = "./results"
    folder_dir = Path(folder_path)
    folder_dir.mkdir(parents = True, exist_ok = True)
    file_name = f"{folder_path}/Power-{sim_id}_stamp{stamp}_t{trials}_B{B}_n{n}_p{p}_target{target_power}_seed{seed}.csv"

    print(f"\nRunning simulation ID-{sim_id}, target-{target_power} at {time.ctime()}")
    print(f"Results will be outputted to file", file_name)

    with Pool(cpus) as pool:
        all_pvals = pool.starmap(
            compute_pvals_for_trial, zip(
                repeat(B),
                repeat(beta),
                repeat(gen_data),
                rng.spawn(trials)
            )
        )
    all_pvals_df = pd.DataFrame(all_pvals)
    all_pvals_df.to_csv(file_name)

    end_time = time.time()
    print(f"Simulations done! Total ellapsed time: {end_time - start_time:.2f}")
