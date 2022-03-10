# %%
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/so/Documents/Projects/pyStateEstimator')
import sqlite3
import itertools
from functools import partial

# %%
def remove_anomalies(data, tolerance):
    mean = data.mean(axis=0)
    new_data = []
    for i, ithdata in enumerate(data):
        remainder_mean = np.concatenate([data[:i], data[i+1:]]).mean(axis=0)
        distance = np.abs(mean - remainder_mean).max()
        if distance < tolerance:
            new_data.append(ithdata)
        else:
            print(f"Anomaly detected at i = {i}")

    return np.array(new_data)


def get_mean_and_std(data, num_samples, tolerance):
    anomalies_removed = remove_anomalies(data, tolerance)
    mean = anomalies_removed[:num_samples].mean(axis=0)
    anomaly = anomalies_removed[:num_samples] - mean[None]
    variance = (anomaly**2).mean(axis=0)
    std = np.sqrt(variance)
    return mean, std

# %%
con = sqlite3.connect("temp_ungm.db", detect_types=sqlite3.PARSE_DECLTYPES)
cursor = con.cursor()

power_range = [1.0, 1.0, 0.8]
damp_range = [1.0, 0.8, 0.8]
trans_types = ['TT', 'UT', 'MCT']
colors = ['C3', 'C2', 'C0']

query_str = "SELECT {}" \
            " FROM UNGM_EXP" \
            " WHERE Transform='{}' AND Seed={} AND Power={} AND Damping={}"

# %%

num_seeds = 10
fig, axs = plt.subplots(2, 3, figsize=(12,5))
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
for i, params in enumerate(zip(power_range, damp_range)):
    power, damping = params
    print(f"Power: {power}, Damping: {damping}")
    for c, trans_id in zip(colors, trans_types):
        RMSE_data, NLL_data = [], []
        for SEED in range(15):
            row = cursor.execute(query_str.format('RMSE', trans_id, SEED, power, damping)).fetchall()
            RMSE_data.append(np.array(row).squeeze())
            row = cursor.execute(query_str.format('NLL', trans_id, SEED, power, damping)).fetchall()
            NLL_data.append(np.array(row).squeeze())
        RMSE_data = np.array(RMSE_data)
        NLL_data = np.array(NLL_data)

        RMSE_mean, RMSE_std = get_mean_and_std(RMSE_data, num_seeds, 1.0)
        NLL_mean, NLL_std = get_mean_and_std(NLL_data, num_seeds, 30000)

        axs[0, i].plot(np.arange(0, 50), RMSE_mean, c=c, label=trans_id)
        axs[0, i].fill_between(np.arange(0, 50), RMSE_mean-RMSE_std, RMSE_mean+RMSE_std, alpha=0.2, color=c)

        axs[1, i].plot(np.arange(0, 50), NLL_mean, c=c, label=trans_id)
        axs[1, i].fill_between(np.arange(0, 50), NLL_mean-NLL_std, NLL_mean+NLL_std, alpha=0.2, color=c)

    axs[0, i].legend(fontsize=12, loc='upper right')
    if i ==0:
        axs[0, i].set_ylabel("RMSE", fontsize=16)
    xticks = [0, 10, 20, 30, 40, 50]
    axs[0, i].set_xticks(xticks)
    axs[0, i].set_title(f"Power: {power}, Damping: {damping}", fontsize=16)

    axs[1, i].legend(fontsize=12, loc='upper right')
    axs[1, i].set_xlabel("Iterations", fontsize=16)
    if i==0:
        axs[1, i].set_ylabel("NLL", fontsize=16)
    xticks = [0, 10, 20, 30, 40, 50]
    axs[1, i].set_xticks(xticks)
    axs[1, i].ticklabel_format(axis='y', style='sci', scilimits=(4,4))

plt.tight_layout()


# %%
