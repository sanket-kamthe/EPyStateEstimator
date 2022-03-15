# %%
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/so/Documents/Projects/pyStateEstimator')
import sqlite3
import itertools
from functools import partial
import seaborn as sns

# %%
# def remove_anomalies(data, tolerance):
#     mean = data.mean(axis=0)
#     new_data = []
#     for i, ithdata in enumerate(data):
#         remainder_mean = np.concatenate([data[:i], data[i+1:]]).mean(axis=0)
#         distance = np.abs(mean - remainder_mean).max()
#         if distance < tolerance:
#             new_data.append(ithdata)
#         else:
#             print(f"Anomaly detected at i = {i}")

#     return np.array(new_data)

def get_mean_and_std(data):
#def get_mean_and_std(data, num_samples, tolerance=None):
    # if tolerance is None:
    #     data_cleaned = data
    # else:
    #     data_cleaned = remove_anomalies(data, tolerance)
    # mean = data_cleaned[:num_samples].mean(axis=0)
    mean = data.mean(axis=0)
    anomaly = data - mean[None]
    variance = (anomaly**2).mean(axis=0)
    std = np.sqrt(variance)
    return mean, std

# %%
# First plot (figure 4)
con = sqlite3.connect("temp_ungm.db", detect_types=sqlite3.PARSE_DECLTYPES)
cursor = con.cursor()

power_range = [1.0, 1.0, 0.8]
damp_range = [1.0, 0.8, 0.8]
trans_types = ['TT', 'UT', 'MCT']
colors = ['C3', 'C2', 'C0']
Seeds = np.arange(100, 110)

query_str = "SELECT {}" \
            " FROM UNGM_EXP" \
            " WHERE Transform='{}' AND Seed={} AND Power={} AND Damping={}"

# %%
fig, axs = plt.subplots(2, 3, figsize=(12,5))
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
for i, params in enumerate(zip(power_range, damp_range)):
    power, damping = params
    print(f"Power: {power}, Damping: {damping}")
    for c, trans_id in zip(colors, trans_types):
        RMSE_data, NLL_data = [], []
        for SEED in Seeds:
            row = cursor.execute(query_str.format('RMSE', trans_id, int(SEED), power, damping)).fetchall()
            RMSE_data.append(np.array(row).squeeze())
            row = cursor.execute(query_str.format('NLL', trans_id, int(SEED), power, damping)).fetchall()
            NLL_data.append(np.array(row).squeeze())
        RMSE_data = np.array(RMSE_data)
        NLL_data = np.array(NLL_data)

        # RMSE_mean, RMSE_std = get_mean_and_std(RMSE_data, num_seeds, 1.0)
        # NLL_mean, NLL_std = get_mean_and_std(NLL_data, num_seeds, 300_000)
        RMSE_mean, RMSE_std = get_mean_and_std(RMSE_data)
        NLL_mean, NLL_std = get_mean_and_std(NLL_data)

        axs[0, i].plot(np.arange(0, 50), RMSE_mean, c=c, label=trans_id, zorder=1)
        axs[0, i].fill_between(np.arange(0, 50), RMSE_mean-RMSE_std, RMSE_mean+RMSE_std, alpha=0.2, color=c, zorder=2)

        axs[1, i].plot(np.arange(0, 50), NLL_mean, c=c, label=trans_id, zorder=1)
        axs[1, i].fill_between(np.arange(0, 50), NLL_mean-NLL_std, NLL_mean+NLL_std, alpha=0.2, color=c, zorder=2)

    if i ==0:
        axs[0, i].set_ylabel("RMSE", fontsize=16)
    xticks = [0, 10, 20, 30, 40, 50]

    axs[0, i].set_xticks(xticks)
    axs[0, i].set_title(f"Power: {power}, Damping: {damping}", fontsize=16)

    axs[1, i].set_xlabel("Iterations", fontsize=16)
    if i==0:
        axs[1, i].set_ylabel("NLL", fontsize=16)
    xticks = [0, 10, 20, 30, 40, 50]
    axs[1, i].set_xticks(xticks)
    axs[1, i].ticklabel_format(axis='y', style='sci', scilimits=(4,4))

    for n in range(2):
       axs[n, i].set_facecolor('#EBEBEB')
       axs[n, i].grid(True, color='w', linestyle='-', zorder=3, linewidth=1)

axs[0, 0].set_ylim(1, 18)
axs[0, 0].legend(fontsize=12, loc='upper right')
axs[0, 1].set_ylim(1, 25)
axs[0, 1].legend(fontsize=12, loc='upper right')
axs[0, 2].set_ylim(1, 17)
axs[0, 2].legend(fontsize=12, loc='upper right')
axs[1, 0].set_ylim(-100, 2000)
axs[1, 0].legend(fontsize=12, loc='upper right')
axs[1, 1].set_ylim(-500, 2000)
axs[1, 1].legend(fontsize=12, loc='upper right')
axs[1, 2].set_ylim(-500, 3000)
axs[1, 2].legend(fontsize=12, loc='upper right')
plt.tight_layout()
plt.savefig("../figs/ep_comparison.png", dpi=300)


# %%
# Second plot (figure 5)
con = sqlite3.connect("temp_ungm_2.db", detect_types=sqlite3.PARSE_DECLTYPES)
cursor = con.cursor()

damping = 0.4
iter_list = [2, 10, 30]
trans_types = ['TT', 'UT', 'MCT']
colors = ['C3', 'C2', 'C0']

query_str = "SELECT {}" \
            " FROM UNGM_EXP" \
            " WHERE Transform='{}' AND Seed={} AND Damping={} AND Iter={}"

num_seeds = 10
fig, axs = plt.subplots(2, 3, figsize=(12,5))
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
for i, iter in enumerate(iter_list):
    for c, trans_id in zip(colors, trans_types):
        RMSE_data, NLL_data = [], []
        for SEED in range(15):
            row = cursor.execute(query_str.format('RMSE', trans_id, SEED, damping, iter)).fetchall()
            RMSE_data.append(np.array(row).squeeze())
            row = cursor.execute(query_str.format('NLL', trans_id, SEED, damping, iter)).fetchall()
            NLL_data.append(np.array(row).squeeze())
        RMSE_data = np.array(RMSE_data)
        NLL_data = np.array(NLL_data)

        RMSE_mean, RMSE_std = get_mean_and_std(RMSE_data, num_seeds)
        NLL_mean, NLL_std = get_mean_and_std(NLL_data, num_seeds)

        for n in range(2):
            axs[n, i].set_facecolor('#EBEBEB')
            axs[n, i].grid(True, color='w', linestyle='-', zorder=3, linewidth=1)

        axs[0, i].plot(np.arange(0, 20), RMSE_mean, c=c, label=trans_id, zorder=1)
        axs[0, i].fill_between(np.arange(0, 20), RMSE_mean-RMSE_std, RMSE_mean+RMSE_std, alpha=0.2, color=c, zorder=2)

        axs[1, i].plot(np.arange(0, 20), NLL_mean, c=c, label=trans_id, zorder=1)
        axs[1, i].fill_between(np.arange(0, 20), NLL_mean-NLL_std, NLL_mean+NLL_std, alpha=0.2, color=c, zorder=2)

    if i ==0:
        axs[0, i].set_ylabel("RMSE", fontsize=16)
    axs[0, i].set_title(f"Iteration: {iter}, Damping: {damping}", fontsize=16)

    axs[1, i].set_xlabel("Power", fontsize=16)
    if i==0:
        axs[1, i].set_ylabel("NLL", fontsize=16)
    axs[1, i].ticklabel_format(axis='y', style='sci', scilimits=(5,5))

axs[0, 0].set_ylim(1, 25)
axs[0, 0].legend(fontsize=12, loc='upper left')
axs[0, 1].set_ylim(1, 25)
axs[0, 1].legend(fontsize=12, loc='upper left')
axs[0, 2].set_ylim(1, 25)
axs[0, 2].legend(fontsize=12, loc='upper left')
axs[1, 0].set_ylim(-80_000, 500_000)
axs[1, 0].legend(fontsize=12, loc='upper right')
axs[1, 1].set_ylim(-300_000, 1_400_000)
axs[1, 1].legend(fontsize=12, loc='upper right')
axs[1, 2].set_ylim(-300_000, 1_500_000)
axs[1, 2].legend(fontsize=12, loc='upper right')
plt.tight_layout()
plt.savefig("../figs/power_sweep.png", dpi=300)

# %%
# Plot result just for Taylor transform
fig, axs = plt.subplots(2, 3, figsize=(12,5))
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
for i, iter in enumerate(iter_list):
    RMSE_data, NLL_data = [], []
    for SEED in range(15):
        row = cursor.execute(query_str.format('RMSE', 'TT', SEED, damping, iter)).fetchall()
        RMSE_data.append(np.array(row).squeeze())
        row = cursor.execute(query_str.format('NLL', 'TT', SEED, damping, iter)).fetchall()
        NLL_data.append(np.array(row).squeeze())
    RMSE_data = np.array(RMSE_data)
    NLL_data = np.array(NLL_data)

    RMSE_mean, RMSE_std = get_mean_and_std(RMSE_data, num_seeds)
    NLL_mean, NLL_std = get_mean_and_std(NLL_data, num_seeds)

    axs[0, i].plot(np.arange(0, 20), RMSE_mean, c='C3', label='TT')
    axs[0, i].fill_between(np.arange(0, 20), RMSE_mean-RMSE_std, RMSE_mean+RMSE_std, alpha=0.2, color='C3')

    axs[1, i].plot(np.arange(0, 20), NLL_mean, c='C3', label='TT')
    axs[1, i].fill_between(np.arange(0, 20), NLL_mean-NLL_std, NLL_mean+NLL_std, alpha=0.2, color='C3')

    if i ==0:
        axs[0, i].set_ylabel("RMSE", fontsize=16)
    axs[0, i].set_title(f"Iteration: {iter}, Damping: {damping}", fontsize=16)
    axs[0, i].set_ylim(3, 18)

    axs[1, i].legend(fontsize=12, loc='upper right')
    axs[1, i].set_xlabel("Power", fontsize=16)
    if i==0:
        axs[1, i].set_ylabel("NLL", fontsize=16)
    axs[1, i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axs[1, i].set_ylim(-300_000, 1_000_000)

plt.tight_layout()
# %%
