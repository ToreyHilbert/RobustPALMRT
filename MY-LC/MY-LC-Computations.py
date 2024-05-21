import numpy as np
import pandas as pd

from scipy.special import huber
import scipy.stats
import scipy.stats.distributions as distributions
import scipy.optimize

from multiprocess import Pool, freeze_support

from pathlib import Path

from sklearn.linear_model import QuantileRegressor

import time
import re
from itertools import repeat

NUM_CPUS = 5
BASE_SEED = 5142024
rng = np.random.default_rng(BASE_SEED)

# Floating point safe vectorized comparison function
W = lambda T_pi1, T_pi2: (~np.isclose(T_pi1, T_pi2) & (T_pi1 > T_pi2)) + 0.5 * np.isclose(T_pi1, T_pi2)

#######################################################
# F-test p-values for comparison
#######################################################

def FtestPval(X_unfiltered, Z_unfiltered, Y):
    na_filter_idx = ~np.isnan(Y)
    Y = Y[na_filter_idx]
    X = X_unfiltered[na_filter_idx]
    Z = Z_unfiltered[na_filter_idx]
    n = Y.shape[0]

    X_full = np.concatenate([X, Z], axis = 1)

    e1 = np.zeros(X_full.shape[1])
    e1[0] = 1.

    degrees_of_freedom = X_full.shape[0] - X_full.shape[1]

    pinv = np.linalg.pinv(X_full.T @ X_full) 
    A = e1.T @ pinv @ X_full.T / np.sqrt(e1.T @ pinv @ e1)

    sigma_hat = np.sqrt((Y.T @ Y - Y.T @ X_full @ pinv @ X_full.T @ Y)  / degrees_of_freedom)
    tval = (A @ Y) / sigma_hat
    return 2 * distributions.t(df = degrees_of_freedom).cdf(-np.abs(tval))

#######################################################
# Huber-Huber RobustPALMRT intervals
#######################################################

# Fits a Huber regression of Y on X_full via IRLS, possibly with scale estimation.
# - If scale = None, then perform scale estimation via MAD.
# - If scale is a number, then perform Huber regression with fixed scale.
def fit_huber_regression(Y, X_full, scale = None,
                   scale_factor = 0.7649, huber_delta = 1.345,
                   acc = 1e-8, maxiters = 50):
    Yhat = X_full @ ((np.linalg.pinv(X_full.T @ X_full) @ X_full.T) @ Y)
    R_old = Y - Yhat

    converged = False
    for _ in range(maxiters):
        if scale:
            s = scale
        else:
            # Median absolute deviation (MAD) estimator
            s = np.median(np.abs(R_old))/scale_factor

        weights = np.minimum(1, huber_delta * s / np.abs(R_old))
        params = ((np.linalg.pinv((X_full.T * weights) @ X_full) @ X_full.T) @ (weights * Y))
        R = Y - X_full @ params
        if np.sum(np.square(R - R_old)) / np.sum(np.square(R_old)) < acc:
            converged = True
            break
        R_old = R
    if not converged:
        print("Failed to converge!")
    return R, s, params

# Utility function for computing Huber norms
def huber_norm(x, delta = 1.345):
    return np.sum(
        huber(delta, x)
    )

# Computes the Robust PALMRT p-value for a given set of data
def compute_pval_HuberHuberRobustPALMRT(X, Z, Y, B, thread_rng):
    n = Y.shape[0]

    pval = 1
    for _ in range(B):
        pi = thread_rng.permutation(n)

        Z_full_pi = np.concatenate([Z[pi], Z[:,1:]], axis = 1)

        X_full_pi1 = np.concatenate([X,      Z[pi], Z[:,1:]], axis = 1)
        X_full_pi2 = np.concatenate([X[pi],  Z[pi], Z[:,1:]], axis = 1)

        _, scale, _ = fit_huber_regression(Y, Z_full_pi)
        T_pi1, _, _ = fit_huber_regression(Y, X_full_pi1, scale = scale)
        T_pi2, _, _ = fit_huber_regression(Y, X_full_pi2, scale = scale)
        pval += W(huber_norm(T_pi1/scale), huber_norm(T_pi2/scale))
    return pval / (B + 1)


def HuberHuberRobustPALMRTInterval(X_unfiltered, Z_unfiltered, Y, thread_rng):
    na_filter_idx = ~np.isnan(Y)
    Y = Y[na_filter_idx]
    X = X_unfiltered[na_filter_idx]
    Z = Z_unfiltered[na_filter_idx]
    n = Y.shape[0]

    B = 999
    tolerance = 1e-3
    seed_trials = 499

    scales = []
    betas = []
    for _ in range(seed_trials):
        pi = thread_rng.permutation(n)
        X_full_pi = np.concatenate([X, Z[pi], Z[:,1:]], axis = 1)
        _, scale, params = fit_huber_regression(Y, X_full_pi)
        scales.append(scale)
        betas.append(params[0])
    estimated_beta = np.mean(betas)
    estimated_scale = np.mean(scales)

    target = 0.05
    f = lambda beta: compute_pval_HuberHuberRobustPALMRT(
        X, Z, Y - X @ np.array([beta]), B, thread_rng
    ) - target

    upper_point = scipy.optimize.brentq(
        f, estimated_beta, estimated_beta + 2 * estimated_scale, rtol = tolerance)
    lower_point = scipy.optimize.brentq(
        f, estimated_beta - 2 * estimated_scale, estimated_beta, rtol = tolerance)

    return (lower_point, upper_point, estimated_beta, estimated_scale)


#######################################################
# OLS-L2 PALMRT intervals
#######################################################

# Computes the Robust PALMRT p-value for a given set of data
def compute_pval_OLSPALMRT(X, Z, Y, B, thread_rng):
    n = Y.shape[0]

    pval = 1
    for _ in range(B):
        pi = thread_rng.permutation(n)

        X_full_pi1 = np.concatenate([X,      Z[pi], Z[:,1:]], axis = 1)
        X_full_pi2 = np.concatenate([X[pi],  Z[pi], Z[:,1:]], axis = 1)

        T_pi1 = Y - X_full_pi1 @ np.linalg.pinv(X_full_pi1.T @ X_full_pi1) @ X_full_pi1.T @ Y
        T_pi2 = Y - X_full_pi2 @ np.linalg.pinv(X_full_pi2.T @ X_full_pi2) @ X_full_pi2.T @ Y
        pval += W(np.linalg.norm(T_pi1, ord = 2), np.linalg.norm(T_pi2, ord = 2))
    return pval / (B + 1)

def OLSL2PALMRTInterval(X_unfiltered, Z_unfiltered, Y, thread_rng):
    na_filter_idx = ~np.isnan(Y)
    Y = Y[na_filter_idx]
    X = X_unfiltered[na_filter_idx]
    Z = Z_unfiltered[na_filter_idx]
    n = Y.shape[0]

    B = 999
    tolerance = 1e-3
    seed_trials = 499

    scales = []
    betas = []
    for _ in range(seed_trials):
        pi = thread_rng.permutation(n)
        X_full_pi = np.concatenate([X,      Z[pi], Z[:,1:]], axis = 1)
        beta = np.linalg.pinv(X_full_pi.T @ X_full_pi) @ X_full_pi.T @ Y
        sigma = np.linalg.norm(Y - X_full_pi @ beta, ord = 2) / np.sqrt(n)
        scales.append(sigma)
        betas.append(beta[0])
    estimated_beta = np.mean(betas)
    estimated_scale = np.mean(scales)

    target = 0.05
    f = lambda beta: compute_pval_OLSPALMRT(
        X, Z, Y - X @ np.array([beta]), B, thread_rng
    ) - target

    upper_point = scipy.optimize.brentq(
        f, estimated_beta, estimated_beta + 5 * estimated_scale, rtol = tolerance)
    lower_point = scipy.optimize.brentq(
        f, estimated_beta - 5 * estimated_scale, estimated_beta, rtol = tolerance)

    return (lower_point, upper_point, estimated_beta, estimated_scale)

#######################################################
# Scale based RobustPALMRT p-values
#######################################################

def ScaleRobustPALMRTPval(X_unfiltered, Z_unfiltered, Y, thread_rng):
    na_filter_idx = ~np.isnan(Y)
    Y = Y[na_filter_idx]
    X = X_unfiltered[na_filter_idx]
    Z = Z_unfiltered[na_filter_idx]
    n = Y.shape[0]

    qlow = 0.10
    qhigh = 0.90
    fit_quantile_regression = lambda X, Y, quantile: QuantileRegressor(
        quantile = quantile, alpha = 0, solver = "highs").fit(X, Y).predict(X)

    B = 3999
    pval = 1
    for _ in range(B):
        pi = thread_rng.permutation(n)

        X_full_pi1 = np.concatenate([X,      Z[pi], Z[:,1:]], axis = 1)
        Y_pred_low_pi1 = fit_quantile_regression(X_full_pi1, Y, qlow)
        Y_pred_high_pi1 = fit_quantile_regression(X_full_pi1, Y, qhigh)
        pi1_mask = X.reshape(-1) != 0

        group_0_scale_mean_pi1 = np.mean(np.abs(Y_pred_high_pi1[pi1_mask] - Y_pred_low_pi1[pi1_mask]))
        group_1_scale_mean_pi1 = np.mean(np.abs(Y_pred_high_pi1[~pi1_mask] - Y_pred_low_pi1[~pi1_mask]))
        scale_ratio_pi1 = -np.abs(np.log(group_1_scale_mean_pi1/group_0_scale_mean_pi1))

        X_full_pi2 = np.concatenate([X[pi],  Z[pi], Z[:,1:]], axis = 1)
        Y_pred_low_pi2 = fit_quantile_regression(X_full_pi2, Y, qlow)
        Y_pred_high_pi2 = fit_quantile_regression(X_full_pi2, Y, qhigh)
        pi2_mask = X.reshape(-1)[pi] != 0
        group_0_scale_mean_pi2 = np.mean(np.abs(Y_pred_high_pi2[pi2_mask] - Y_pred_low_pi2[pi2_mask]))
        group_1_scale_mean_pi2 = np.mean(np.abs(Y_pred_high_pi2[~pi2_mask] - Y_pred_low_pi2[~pi2_mask]))
        scale_ratio_pi2 = -np.abs(np.log(group_1_scale_mean_pi2/group_0_scale_mean_pi2))

        pval += W(scale_ratio_pi1, scale_ratio_pi2)

    return pval / (B + 1)

#######################################################
# Run the code
#######################################################


if __name__ == '__main__':
    freeze_support()

    start_time = time.time()

    #######################################################
    # Read in the data and process it into a usable form
    #######################################################

    print(f"Beginning experiment at {time.ctime()}")

    raw_df = pd.read_csv("./MY-LC-Data/41586_2023_6651_MOESM4_ESM.csv")

    df = raw_df.copy()
    df = df[df["x0_Censor_Complete"] == 0] # Remove censored patients
    df["LC"] = (df["x0_Censor_Cohort_ID"] == 3).astype(np.float32) # Retrieve Long Covid Status

    # Select and rename the features of interest
    cell_features_names = list(filter(lambda x: re.search(r"Flow_Cyt_.*ML$", x), df.columns))
    other_features = [
        "LC",
        "x0_Demographics_Age",
        "x0_Demographics_Sex",
        "x0_Demographics_BMI",
    ]
    all_features = other_features + cell_features_names

    def feature_renaming(name):
        parts = name.split("_")
        if (len(parts) == 1):
            return(parts[0])
        elif (len(parts) == 3):
            return(parts[2])
        else:
            return(parts[3])
    df = df[all_features].rename(feature_renaming, axis = "columns")

    cell_features_names = list(map(feature_renaming, cell_features_names))

    # Create the design matrix
    df = df.dropna(subset = ["Age", "Sex", "BMI"]) # First drop any rows that have missing demographics

    df["Sex"] = df["Sex"] - 1 # Convert Sex to {0, 1} rather than {1, 2}
    df["Age*BMI"] = df["Age"] * df["BMI"]
    df["Sex*BMI"] = df["Sex"] * df["BMI"]
    df["Intercept"] = 1.0

    X_unfiltered = df[["LC"]].to_numpy()
    Z_unfiltered = df[["Intercept", "Age", "Sex", "BMI", "Age*BMI", "Sex*BMI"]].to_numpy()

    num_cell_features = len(cell_features_names)

    
    #######################################################
    # Create the folder for the outputs
    #######################################################

    folder_path = "./results"
    folder_dir = Path(folder_path)
    folder_dir.mkdir(parents = True, exist_ok = True)

    #######################################################
    # Run F-test in parallel
    #######################################################

    print(f"Starting F-test at {time.ctime()}")

    with Pool(NUM_CPUS) as pool:
        f_test_pvals = pool.starmap(
            FtestPval, zip(
                repeat(X_unfiltered),
                repeat(Z_unfiltered),
                [df[feature_name].to_numpy() for feature_name in cell_features_names]
            )
        )
    f_test_pvals_series = pd.Series(
        f_test_pvals, index = cell_features_names)
    f_test_pvals_series.to_csv(
        f"{folder_path}/MY-LC-Ftest-pvals.csv",
        columns = ["Ftest-pval"]
    )


    #######################################################
    # Run Huber-Huber RobustPALMRT in parallel
    #######################################################
    
    print(f"Starting Huber-Huber RobustPALMRT at {time.ctime()}")

    with Pool(NUM_CPUS) as pool:
        huber_huber_robust_palmrt_intervals = pool.starmap(
            HuberHuberRobustPALMRTInterval, zip(
                repeat(X_unfiltered),
                repeat(Z_unfiltered),
                [df[feature_name].to_numpy() for feature_name in cell_features_names],
                rng.spawn(num_cell_features)
            )
        )
    huber_huber_robust_palmrt_intervals_df = pd.DataFrame(
        huber_huber_robust_palmrt_intervals,
        index = cell_features_names,
        columns = ["Huber-LowerBound", "Huber-UpperBound", "Huber-EstimatedBeta", "Huber-EstimatedScale"]
    )
    huber_huber_robust_palmrt_intervals_df.to_csv(f"{folder_path}/MY-LC-HuberHuberRobustPALMRT-intervals.csv")

    #######################################################
    # Run OLS-L2 RobustPALMRT in parallel
    #######################################################
    
    print(f"Starting OLS-L2 PALMRT at {time.ctime()}")

    with Pool(NUM_CPUS) as pool:
        ols_l2_palmrt_intervals = pool.starmap(
            OLSL2PALMRTInterval, zip(
                repeat(X_unfiltered),
                repeat(Z_unfiltered),
                [df[feature_name].to_numpy() for feature_name in cell_features_names],
                rng.spawn(num_cell_features)
            )
        )
    ols_l2_palmrt_intervals_df = pd.DataFrame(
        ols_l2_palmrt_intervals,
        index = cell_features_names,
        columns = ["OLS-LowerBound", "OLS-UpperBound", "OLS-EstimatedBeta", "OLS-EstimatedScale"]
    )
    ols_l2_palmrt_intervals_df.to_csv(f"{folder_path}/MY-LC-OLSL2PALMRT-intervals.csv")

    #######################################################
    # Run Scale RobustPALMRT in parallel
    #######################################################

    print(f"Starting Scale RobustPALMRT at {time.ctime()}")

    with Pool(NUM_CPUS) as pool:
        scale_robust_palmrt_pvals = pool.starmap(
            ScaleRobustPALMRTPval, zip(
                repeat(X_unfiltered),
                repeat(Z_unfiltered),
                [df[feature_name].to_numpy() for feature_name in cell_features_names],
                rng.spawn(num_cell_features)
            )
        )
    scale_robust_palmrt_pvals_series = pd.Series(
        scale_robust_palmrt_pvals, index = cell_features_names)
    scale_robust_palmrt_pvals_series.to_csv(
        f"{folder_path}/MY-LC-ScaleRobustPALMRT-pvals.csv",
        columns = ["Scale-pval"]
    )

    print(f"Finished at {time.ctime()}")

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")


