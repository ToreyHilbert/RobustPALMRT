# Import useful libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
import statsmodels.api as sm

import time

############################################
# Generate the data
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
# Run the experiments
############################################

seed = 112920241000
trials = 1000
p = 6
alpha = 0.05
stamp = time.ctime()

rng = np.random.default_rng(seed = seed)

def test_at_setting(beta, n, p, edist, thread_rng):
    X, Z, Y = gen_data(beta, n, p, edist, thread_rng)
    M = np.concatenate([Z, X], axis = 1)

    model = sm.OLS(Y, M)
    res = model.fit()

    aux_reg_variables = np.concatenate([np.ones((X.shape[0], 1)), X], axis = 1)
    _, pval, _, _ = sm.stats.het_breuschpagan(res.resid, aux_reg_variables)
    return pval


all_powers = []
for n in [100, 200, 400]:
    for beta in [0., 0.5, 1.0, 1.5, 2.0]:
        for edist in ["Normal", "Cauchy", "LogNormal"]:
            rejects_at_setting = 0
            for trial in range(trials):
                pval = test_at_setting(beta, n, p, edist, rng)
                rejects_at_setting += pval <= alpha
            all_powers.append({
                "n" : n,
                "p" : p,
                "edist" : edist,
                "alpha" : alpha,
                "beta" : beta,
                "power" : rejects_at_setting / trials
            })
            print(f"Finished {edist} {n} {beta}")

file_path = f"./Breusch-Pagan_t{trials}_stamp{stamp}_seed{seed}.csv"
df = pd.DataFrame(all_powers)
df.to_csv(file_path)
