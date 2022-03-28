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
def get_mean_and_std(data):
    mean = data.mean(axis=0)
    anomaly = data - mean[None]
    variance = (anomaly**2).mean(axis=0)
    std = np.sqrt(variance)
    return mean, std

# %%
# First plot (figure 4)
con = sqlite3.connect("ungm_final.db", detect_types=sqlite3.PARSE_DECLTYPES)
cursor = con.cursor()

power_range = [1.0, 1.0, 0.8]
damp_range = [1.0, 0.8, 0.8]
trans_types = ['TT', 'UT', 'MCT']
colors = ['C3', 'C2', 'C0']
Seeds = np.arange(101, 1101, 100)


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
    axs[0, i].legend(fontsize=12, loc='upper right', ncol=3)

    axs[1, i].set_xlabel("Iterations", fontsize=16)
    if i==0:
        axs[1, i].set_ylabel("NLL", fontsize=16)
    xticks = [0, 10, 20, 30, 40, 50]
    axs[1, i].set_xticks(xticks)
    # axs[1, i].ticklabel_format(axis='y', style='sci', scilimits=(3,4))
    axs[1, i].legend(fontsize=12, loc='upper right', ncol=3)

    for n in range(2):
       axs[n, i].set_facecolor('#EBEBEB')
       axs[n, i].grid(True, color='w', linestyle='-', zorder=3, linewidth=1)

axs[0, 0].set_ylim(2, 15)
axs[0, 1].set_ylim(2, 15)
axs[0, 2].set_ylim(2, 15)
axs[1, 0].set_ylim(0, 18)
axs[1, 1].set_ylim(0, 18)
axs[1, 2].set_ylim(0, 18)
plt.tight_layout()
plt.savefig("../figs/ep_comparison_finite_difference_taylor.png", dpi=300)


# %%
# Second plot (figure 5)
con = sqlite3.connect("ungm_final.db", detect_types=sqlite3.PARSE_DECLTYPES)
cursor = con.cursor()

damping = 0.8
iter_list = [2, 10, 50]
trans_types = ['TT', 'UT', 'MCT']
colors = ['C3', 'C2', 'C0']

query_str = "SELECT {}" \
            " FROM UNGM_EXP" \
            " WHERE Transform='{}' AND Seed={} AND Damping={} AND Iter={}"

fig, axs = plt.subplots(2, 3, figsize=(12,5))
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
for i, iter in enumerate(iter_list):
    for c, trans_id in zip(colors, trans_types):
        RMSE_data, NLL_data = [], []
        for SEED in Seeds:
            row = cursor.execute(query_str.format('RMSE', trans_id, SEED, damping, iter)).fetchall()
            RMSE_data.append(np.array(row).squeeze())
            row = cursor.execute(query_str.format('NLL', trans_id, SEED, damping, iter)).fetchall()
            NLL_data.append(np.array(row).squeeze())
        RMSE_data = np.array(RMSE_data)
        NLL_data = np.array(NLL_data)

        RMSE_mean, RMSE_std = get_mean_and_std(RMSE_data)
        NLL_mean, NLL_std = get_mean_and_std(NLL_data)

        for n in range(2):
            axs[n, i].set_facecolor('#EBEBEB')
            axs[n, i].grid(True, color='w', linestyle='-', zorder=3, linewidth=1)

        axs[0, i].plot(np.arange(0, 19), RMSE_mean, c=c, label=trans_id, zorder=1)
        axs[0, i].fill_between(np.arange(0, 19), RMSE_mean-RMSE_std, RMSE_mean+RMSE_std, alpha=0.2, color=c, zorder=2)

        axs[1, i].plot(np.arange(0, 19), NLL_mean, c=c, label=trans_id, zorder=1)
        axs[1, i].fill_between(np.arange(0, 19), NLL_mean-NLL_std, NLL_mean+NLL_std, alpha=0.2, color=c, zorder=2)

    if i ==0:
        axs[0, i].set_ylabel("RMSE", fontsize=16)
    axs[0, i].set_title(f"Iteration: {iter}, Damping: {damping}", fontsize=16)

    axs[1, i].set_xlabel("Power", fontsize=16)
    if i==0:
        axs[1, i].set_ylabel("NLL", fontsize=16)
    
    #axs[1, i].ticklabel_format(axis='y', style='sci', scilimits=(5,5))

    for j in [0, 1]:
        axs[j, i].set_xticks(np.linspace(0, 19, 4))
        axs[j, i].set_xticklabels(np.linspace(0.1, 1.0, 4))

axs[0, 0].set_ylim(1, 15)
axs[0, 0].legend(fontsize=12, loc='upper left', ncol=3)
axs[0, 1].set_ylim(1, 15)
axs[0, 1].legend(fontsize=12, loc='upper left', ncol=3)
axs[0, 2].set_ylim(1, 15)
axs[0, 2].legend(fontsize=12, loc='upper left', ncol=3)
axs[1, 0].set_ylim(0, 30)
axs[1, 0].legend(fontsize=12, loc='upper right', ncol=3)
axs[1, 1].set_ylim(-20, 500)
axs[1, 1].legend(fontsize=12, loc='upper right', ncol=3)
axs[1, 2].set_ylim(-30, 1000)
axs[1, 2].legend(fontsize=12, loc='upper right', ncol=3)
plt.tight_layout()
plt.savefig("../figs/power_sweep.png", dpi=300)

# %%
# Heat map
methods = [None, 'none','nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

def plot_power_sweep(img, methods, title):
    plt.imshow(img, interpolation=methods[6],
               extent=[0.1,1,0.1,1], cmap='jet',
               vmax=min(np.max(img), 10.0), vmin=np.min(img),origin='lower')

    ax = plt.gca()
    ax.set_xlabel('Power', fontweight='bold')
    ax.set_ylabel('Damping', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.grid(False)
    plt.colorbar()


def make_image_data(transform, seed, iters=10):
    all_iters = """
                    SELECT RMSE, NLL, Power, Damping from 
                    UNGM_EXP
                    WHERE Transform = '{}' AND Seed = {} AND Iter = {}
                """
    cursor.execute(all_iters.format(transform, seed, iters))
    data = cursor.fetchall()
    rms, nl, p, d = zip(*data)
    nl = np.array(nl)
    rms = np.array(rms)
    p = np.array(p)
    d = np.array(d)
    p = p.reshape(19, 19)
    d = d.reshape(19, 19)
    p_nll = nl.reshape(19, 19)
    p_rmse = rms.reshape(19, 19)
    
    return p_rmse, p_nll, p, d


def all_seeds_image(transform, iters=50):
  all_prmse = []
  all_pnll = []
  cursor.execute("SELECT DISTINCT Seed from UNGM_EXP")
  seeds = cursor.fetchall()
  for seed in seeds:
    rmse, nll, p, d = make_image_data(transform, *seed, iters=iters)
    all_prmse.append(rmse)
    all_pnll.append(nll)
  prmse = np.array(all_prmse)
  pnll = np.array(all_pnll)

  return np.mean(prmse, axis=0), np.mean(pnll, axis=0), p, d


def plot_sweep(transform, iters=10, kind='rmse'):
    p_rmse, p_nll, p, d = all_seeds_image(transform, iters=iters)
    title = f'Loss = {kind}, Transform = {transform}'
    if kind == 'rmse':
        plot_power_sweep((p_rmse.T), methods, title)
    else:
        plot_power_sweep((p_nll.T), methods, title)


# %%  
tranform_ = 'MCT'
iteration = 50

plt.figure(figsize=(8,5))
plot_sweep(tranform_, iters=iteration, kind='rmse')

# %%
plt.figure(figsize=(8,5))
plot_sweep(tranform_, iters=iteration, kind='nll')

# %%
