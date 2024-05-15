import numpy as np
import pandas as pd

from scipy.special import huber
from scipy.optimize import brentq
import scipy.stats.distributions as distributions

from collections import defaultdict
import itertools

from multiprocess import Pool

import argparse
import time
from pathlib import Path


NUM_CPUS = 40
huber_delta = 1.345
mad_scale_factor = 0.7649
alpha = 0.05

############################################
############################################

# This is the main runner function.
# Accepts a list of T-methods to use, and a list of omega-methods to use.
#   Tries every pair of combinations.
# Also tries any direct methods.
# Returns a dataframe with each column the collection of p-values for that method.
def run_experiment(trials, B, gen_data, T_methods, omega_methods, direct_methods, rng):
    p_vals = defaultdict(lambda: [])
    prev_time = time.time()
    for trial_num in range(trials):
        if trial_num % 10 == 9:
            new_time = time.time()
            print(f"Starting trial {trial_num + 1}, elapsed {new_time - prev_time:.2f}")
            prev_time = new_time

        _, X, Z, Y = gen_data(trial_num, rng)

        with Pool(NUM_CPUS) as pool:
            omega_vals_array = pool.starmap(
                run_comps_for_perm, itertools.product(
                    [rng.permutation(Y.shape[0]) for _ in range(B)],
                    [X],
                    [Z],
                    [Y],
                    [T_methods],
                    [omega_methods]
                )
            )

        for key in omega_vals_array[0].keys():
            p_vals[key].append((1 + sum([
                omega_vals[key] for omega_vals in omega_vals_array
            ]))/(B + 1))

        for method, name in direct_methods:
            p_vals[name].append(method(X, Z, Y))
        
    return pd.DataFrame(p_vals)

def run_comps_for_perm(pi, X, Z, Y, T_methods, omega_methods):
    omega_vals = {}
    for T, T_name in T_methods:
        T_pi1, T_pi2, scale = T(pi, X, Z, Y)
        for omega, omega_name in omega_methods:
            omega_vals[T_name + "__" + omega_name] = omega(pi, X, Z, T_pi1, T_pi2, scale)
    
    return omega_vals

############################################
############################################
# T Methods
############################################
############################################

def T_OLS(pi, X, Z, Y, *args):
    Z_full_pi = np.concatenate([Z[pi], Z[:,1:]], axis = 1)
    T_pi_Z = Y - Z_full_pi @ np.linalg.lstsq(Z_full_pi, Y, rcond = None)[0]
    scale = np.linalg.norm(T_pi_Z, ord = 1) / (X.shape[0] * mad_scale_factor)

    X_full_pi1 = np.concatenate([X,      Z[pi], Z[:,1:]], axis = 1)
    X_full_pi2 = np.concatenate([X[pi],  Z[pi], Z[:,1:]], axis = 1)

    T_pi1 = Y - X_full_pi1 @ np.linalg.lstsq(X_full_pi1, Y, rcond = None)[0]
    T_pi2 = Y - X_full_pi2 @ np.linalg.lstsq(X_full_pi2, Y, rcond = None)[0]

    return T_pi1, T_pi2, scale


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

def T_Huber(pi, X, Z, Y, *args):
    Z_full_pi = np.concatenate([Z[pi], Z[:,1:]], axis = 1)
    _, scale, _ = fit_huber_regression(Y, Z_full_pi)

    X_full_pi1 = np.concatenate([X,      Z[pi], Z[:,1:]], axis = 1)
    X_full_pi2 = np.concatenate([X[pi],  Z[pi], Z[:,1:]], axis = 1)

    T_pi1, _, _ = fit_huber_regression(Y, X_full_pi1, scale = scale)
    T_pi2, _, _ = fit_huber_regression(Y, X_full_pi2, scale = scale)

    return T_pi1, T_pi2, scale

def t_test(X, Z, Y):
    X_full = np.concatenate([X, Z], axis = 1)
    degrees_of_freedom = X_full.shape[0] - X_full.shape[1]

    e1 = np.zeros((X_full.shape[1],))
    e1[0] = 1.

    pinv = np.linalg.pinv(X_full.T @ X_full) 
    A = e1.T @ pinv @ X_full.T / np.sqrt(e1.T @ pinv @ e1)
    sigma_hat = np.sqrt((Y.T @ Y - Y.T @ X_full @ pinv @ X_full.T @ Y) / degrees_of_freedom)
    
    return 2 * distributions.t(df = degrees_of_freedom).cdf(-np.abs((A @ Y) / sigma_hat))


T_methods = [
    (
        T_OLS, "OLS"
    ),
    (
        T_Huber, "Huber"
    )
]

direct_methods = [
    (
        t_test, "t-test"
    )
]


############################################
############################################
# Omega Methods
############################################
############################################

# Floating point safe vectorized comparison function
W = lambda T_pi1, T_pi2: (~np.isclose(T_pi1, T_pi2) & (T_pi1 > T_pi2)) + 0.5 * np.isclose(T_pi1, T_pi2)

def omega_L1(pi, X, Z, T_pi1, T_pi2, scale):
    gT_pi1 = np.linalg.norm(T_pi1, ord = 1)
    gT_pi2 = np.linalg.norm(T_pi2, ord = 1)
    return W(gT_pi1, gT_pi2)

def omega_L2(pi, X, Z, T_pi1, T_pi2, scale):
    gT_pi1 = np.linalg.norm(T_pi1, ord = 2)
    gT_pi2 = np.linalg.norm(T_pi2, ord = 2)
    return W(gT_pi1, gT_pi2)

def omega_Huber(pi, X, Z, T_pi1, T_pi2, scale):
    gT_pi1 = np.sum(huber(huber_delta * scale, T_pi1))
    gT_pi2 = np.sum(huber(huber_delta * scale, T_pi2))
    return W(gT_pi1, gT_pi2)


omega_methods = [
    (
        omega_L1, "L1"
    ),
    (
        omega_L2, "L2"
    ),
    (
        omega_Huber, "Huber"
    )
]


############################################
############################################
# Generate the data
############################################
############################################

def standardize_columns(mat):
    mat = mat - np.mean(mat, axis = 0)
    mat = np.sqrt(mat.shape[0]) * mat / np.linalg.norm(mat, ord = 2, axis = 0)
    return mat

log_normal_sigma = 1.
log_normal_sd = np.sqrt((np.exp(log_normal_sigma**2) - 1) * np.exp(log_normal_sigma**2))

# Note: The first column of Z *MUST* be an all ones column (due to T calculations)
def gen_data(beta, n, p, covariate_distribution, epsilon_distribution, rng):
    # Generate covariates
    if covariate_distribution == "Normal":
        X = standardize_columns(rng.standard_normal(size = (n,)))
        Z = standardize_columns(rng.standard_normal(size = (n, p - 1)))
        Z = np.concatenate([np.ones((n, 1)), Z], axis = 1)
    elif covariate_distribution == "t3":
        X = standardize_columns(rng.standard_t(df = 3, size = (n,)))
        Z = standardize_columns(rng.standard_t(df = 3, size = (n, p - 1)))
        Z = np.concatenate([np.ones((n, 1)), Z], axis = 1)
    elif covariate_distribution == "Cauchy":
        X = standardize_columns(rng.standard_cauchy(size = (n,)))
        Z = standardize_columns(rng.standard_cauchy(size = (n, p - 1)))
        Z = np.concatenate([np.ones((n, 1)), Z], axis = 1)
    elif covariate_distribution == "BalancedAnova":
        m = int(n/(p + 1))
        X_full = np.concatenate([np.eye(p + 1) for _ in range(m)])
        X = standardize_columns(X_full[:,-1])
        Z = standardize_columns(X_full[:,:-1])
        Z[:,0] = 1.
    n = X.shape[0]

    # Generate epsilon
    if epsilon_distribution == "Normal":
        e = rng.standard_normal(size = (n,))
    elif epsilon_distribution == "t3":
        e = rng.standard_t(3, size = (n,))
    elif epsilon_distribution == "Cauchy":
        e = rng.standard_cauchy(size = (n,))
    elif epsilon_distribution == "Multinomial":
        e = rng.normal(size = (n,)) + \
            1e4 * rng.multinomial(1, [1/n] * n) * (-1)**(rng.uniform() < 0.5)
    elif epsilon_distribution == "LogNormal":
        e = rng.lognormal(mean = 0., sigma = log_normal_sigma, size = (n,))/log_normal_sd

    X_full = np.concatenate([X.reshape(-1, 1), Z], axis = 1)
    Y = X * beta + e
    
    return X_full, X.reshape(-1, 1), Z, Y

def compute_power(trials, beta, n, p, covariate_distribution, epsilon_distribution, rng):
    e1 = np.zeros(p + 1)
    e1[0] = 1.

    l = []
    for _ in range(trials):
        X_full, _, _, Y = gen_data(beta, n, p, covariate_distribution, epsilon_distribution, rng)

        degrees_of_freedom = X_full.shape[0] - X_full.shape[1]

        pinv = np.linalg.pinv(X_full.T @ X_full) 
        A = e1.T @ pinv @ X_full.T / np.sqrt(e1.T @ pinv @ e1)

        sigma_hat = np.sqrt((Y.T @ Y - Y.T @ X_full @ pinv @ X_full.T @ Y)  / degrees_of_freedom)
        l.append(np.abs((A @ Y) / sigma_hat))
    l = np.array(l)

    t_critical = np.abs(distributions.t(df = degrees_of_freedom).ppf(alpha/2))
    return np.mean(l > t_critical)

def find_beta_for_params(p, target_power, covariate_distribution, epsilon_distribution, rng,
                         monte_carlo_trials, tolerance):
    beta_max = 100. if epsilon_distribution != "Multinomial" else 10000.

    f = lambda beta: compute_power(
        monte_carlo_trials, beta, n, p, covariate_distribution, epsilon_distribution, rng
    ) - target_power/100.
    res = brentq(f, 0., beta_max, rtol = tolerance)
    return res


############################################
############################################
# Run the experiment
############################################
############################################


if __name__ == "__main__":
    # For recording the time that the simulation was run
    start_time = time.time()
    stamp = int(start_time)

    # First we read in the command line information
    parser = argparse.ArgumentParser(
        description = "Runs the core factorial experiment simuations."
    )
    parser.add_argument("-n", type = int, required = True, help = "Sample size to use")
    parser.add_argument("-p", type = int, required = True, help = "Number of nuisance covariates, including intercept")
    parser.add_argument("-B", type = int, required = True, help = "Number of permutations to use")
    parser.add_argument("-t", "--trials", type = int, required = True, help = "Number of design matrices to draw")
    parser.add_argument("--covariate-distribution", choices = [
        "Normal", "t3", "Cauchy", "BalancedAnova"
    ], required = True, help = "Distribution for covariates")
    parser.add_argument("--epsilon-distribution", choices = [
        "Normal", "t3", "Cauchy", "Multinomial", "LogNormal"
    ], required = True, help = "Distribution for epsilons")
    parser.add_argument("--seed", type = int, help = "Seed for random numbers")
    parser.add_argument("--target-power", type = float, help = "Set beta by target F-test power; 0 means effect size is zero")
    parser.add_argument("--monte-carlo-trials", type = int, help = "The number of Monte Carlo trials to use for setting the correct F-test power")
    parser.add_argument("--f-test-tolerance", type = float, help = "Tolerance to use in search for correct F-test power")

    args = parser.parse_args()

    sim_id = f"{args.covariate_distribution}-{args.epsilon_distribution}"
    trials, n, p, B, seed = args.trials, args.n, args.p, args.B, args.seed
    covariate_distribution, epsilon_distribution, target_power = args.covariate_distribution, args.epsilon_distribution, args.target_power
    monte_carlo_trials, tolerance = args.monte_carlo_trials, args.f_test_tolerance

    rng = np.random.default_rng(seed)

    # After reading in the command line information,
    #  we set the value of beta so that the F-test will have the correct target power.
    if target_power == 0:
        beta = np.array([0.])
    else:
        print(f"\nComputing beta for simulation ID-{sim_id}, power-{target_power:.2f}, p-{p}, n-{n} at {time.ctime()}")
        beta = find_beta_for_params(p, target_power, covariate_distribution, epsilon_distribution, rng, monte_carlo_trials, tolerance)
        verify_power = compute_power(4 * monte_carlo_trials, beta, n, p, covariate_distribution, epsilon_distribution, rng)
        print(f"Using beta = {beta}, with power {verify_power} (target power {target_power}), elapsed time {time.time() - start_time:.2f}")

        parameter_files_path = "computed_betas.csv"
        with open(parameter_files_path, "a+") as f:
            f.write(f"{n},{p},{covariate_distribution},{epsilon_distribution},{target_power:.1f},{beta},{verify_power},{stamp}\n")
    
    # Create location for results
    folder_path = "./results"
    folder_dir = Path(folder_path)
    folder_dir.mkdir(parents = True, exist_ok = True)
    file_name = f"{folder_path}/Power-{sim_id}_stamp{stamp}_t{trials}_B{B}_n{n}_p{p}_target{target_power}_seed{seed}.csv"

    print(f"\nRunning simulation ID-{sim_id}, target-{target_power}, p-{p} at {time.ctime()}")
    print(f"Results will be outputted to file", file_name)

    gen_data_local = lambda trial, rng: gen_data(
        beta, n, p, args.covariate_distribution, args.epsilon_distribution, rng
    )

    p_vals = run_experiment(
        trials = trials,
        B = B,
        gen_data = gen_data_local,
        T_methods = T_methods,
        omega_methods = omega_methods,
        direct_methods = direct_methods,
        rng = rng
    )

    p_vals.to_csv(file_name, index = False)

    end_time = time.time()
    print(f"Simulations done! Total ellapsed time: {end_time - start_time:.2f}")
