# %%
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from functools import partial
import seaborn as sns

# %%
def get_mean_and_std(data):
    mean = data.mean(axis=0)
    anomaly = data - mean[None]
    variance = (anomaly**2).mean(axis=0)
    std = np.sqrt(variance)
    return mean, std


def select_data(experiment):
    if experiment == 'ungm':
        exp_table = 'UNGM_EXP'
        con = sqlite3.connect("ungm_final_2.db", detect_types=sqlite3.PARSE_DECLTYPES)
    elif experiment == 'bot':
        exp_table = 'BOT_EXP'
        con = sqlite3.connect("bot_final.db", detect_types=sqlite3.PARSE_DECLTYPES)

    cursor = con.cursor()
    return exp_table, cursor

# %%
# First plot (figure 4)
experiment = 'ungm'
power_range = [1.0, 1.0, 0.8]
damp_range = [1.0, 0.8, 0.8]
trans_types = ['TT', 'UT', 'MCT']
colors = ['C3', 'C2', 'C0']
Seeds = np.arange(101, 1101, 100)

exp_table, cursor = select_data(experiment)

query_str = "SELECT {}" \
            " FROM {}" \
            " WHERE Transform='{}' AND Seed={} AND Power={} AND Damping={}"

fig, axs = plt.subplots(2, 3, figsize=(12,5))
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
for i, params in enumerate(zip(power_range, damp_range)):
    power, damping = params
    print(f"Power: {power}, Damping: {damping}")
    for c, trans_id in zip(colors, trans_types):
        RMSE_data, NLL_data = [], []
        for SEED in Seeds:
            row = cursor.execute(query_str.format('RMSE', exp_table, trans_id, int(SEED), power, damping)).fetchall()
            RMSE_data.append(np.array(row).squeeze())
            row = cursor.execute(query_str.format('NLL', exp_table, trans_id, int(SEED), power, damping)).fetchall()
            NLL_data.append(np.array(row).squeeze())
        RMSE_data = np.array(RMSE_data)
        NLL_data = np.array(NLL_data)

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
# plt.savefig("../figs/ep_comparison_finite_difference_taylor.png", dpi=300)


# %%
# Second plot (figure 5)
experiment = 'ungm'
exp_table, cursor = select_data(experiment)

damping = 0.4
iter_list = [2, 10, 50]
trans_types = ['TT', 'UT', 'MCT']
colors = ['C3', 'C2', 'C0']
Seeds = np.arange(101, 1101, 100)

query_str = "SELECT {}" \
            " FROM {}" \
            " WHERE Transform='{}' AND Seed={} AND Damping={} AND Iter={}"

fig, axs = plt.subplots(2, 3, figsize=(12,5))
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
for i, iter in enumerate(iter_list):
    for c, trans_id in zip(colors, trans_types):
        RMSE_data, NLL_data = [], []
        for SEED in Seeds:
            row = cursor.execute(query_str.format('RMSE', exp_table, trans_id, SEED, damping, iter)).fetchall()
            RMSE_data.append(np.array(row).squeeze())
            row = cursor.execute(query_str.format('NLL', exp_table, trans_id, SEED, damping, iter)).fetchall()
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
# plt.savefig("../figs/power_sweep.png", dpi=300)

# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import namedtuple

experiment = 'ungm'
exp_table, cursor = select_data(experiment)

# Heat map
methods = [None, 'none','nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

PlotConfig = namedtuple('PlotConfig', ['xlabel_kwargs','ylabel_kwargs',
                                        'title_kwargs', 'vmin', 'vmax'])

def plot_power_sweep(config, img, methods, title, ax=None):
    # plt.imshow(img, interpolation=methods[6],
    #            extent=[0.1,1,0.1,1], cmap='jet',
    #            vmax=1000.0, vmin=np.min(img), origin='lower')

    if ax is None:
        ax = plt.gca()

    im = ax.imshow(img, interpolation=methods[4],
                   extent=[0.1,1,0.1,1], cmap='jet',
                   vmax=config.vmax, vmin=config.vmin, origin='lower')

    ax.set_xlabel('Power', **config.xlabel_kwargs)
    ax.set_ylabel('Damping', **config.ylabel_kwargs)
    ax.set_title(title, **config.title_kwargs)
    ax.grid(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    if config.vmax is not None and config.vmin is not None:
        ticks = np.linspace(0, config.vmax, 6)
        tickslabel = [int(tick) for tick in ticks]
        tickslabel[-1] = f'>{tickslabel[-1]}'
        cbar = plt.colorbar(im, cax=cax, ticks=ticks)
        cbar.ax.set_yticklabels(tickslabel)
    else:
        plt.colorbar(im, cax=cax)
    

def make_image_data(table, transform, seed, iters=10):
    all_iters = """
                    SELECT RMSE, NLL, Power, Damping from 
                    {}
                    WHERE Transform = '{}' AND Seed = {} AND Iter = {}
                """
    cursor.execute(all_iters.format(table, transform, seed, iters))
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


def all_seeds_image(table, transform, iters=50):
  all_prmse = []
  all_pnll = []
  cursor.execute(f"SELECT DISTINCT Seed from {table}")
  seeds = cursor.fetchall()
  for seed in seeds:
    rmse, nll, p, d = make_image_data(table, transform, *seed, iters=iters)
    all_prmse.append(rmse)
    all_pnll.append(nll)
  prmse = np.array(all_prmse)
  pnll = np.array(all_pnll)

  return np.mean(prmse, axis=0), np.mean(pnll, axis=0), p, d


def plot_sweep(config, table, transform, iters=10, kind='rmse', ax=None):
    p_rmse, p_nll, p, d = all_seeds_image(table, transform, iters=iters)
    title = f'Transform = {transform}'
    if kind == 'rmse':
        plot_power_sweep(config, (p_rmse.T), methods, title, ax)
    else:
        plot_power_sweep(config, (p_nll.T), methods, title, ax)


def plot_all_transforms(config, table, iters=10, kind='rmse'):
    transform_list = ['TT', 'UT', 'MCT']
    fig, axs = plt.subplots(1, 3, figsize=(25, 9))
    for i, trans in enumerate(transform_list):
        plot_sweep(config, table, trans, iters, kind, axs[i])
    if kind == 'rmse':
        plt.suptitle('RMSE', fontweight='bold', fontsize=30)
    else:
        plt.suptitle('NLL', fontweight='bold', fontsize=30)
    plt.tight_layout()


# %%  
config = PlotConfig(xlabel_kwargs={'fontsize':30},
                    ylabel_kwargs={'fontsize':30},
                    title_kwargs={'fontsize':30},
                    vmin=2.5, vmax=10.0)

plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plot_all_transforms(config, exp_table, 50, 'rmse')

config=config._replace(vmin=-15.0)
config=config._replace(vmax=500.0)
plot_all_transforms(config, exp_table, 50, 'nll')

# tranform_ = 'MCT'
# iteration = 50

# plt.figure(figsize=(8,5))
# plot_sweep(exp_table, tranform_, iters=iteration, kind='rmse')

# %%
# tranform_ = 'MCT'
# plt.figure(figsize=(8,5))
# plot_sweep(exp_table, tranform_, iters=iteration, kind='nll')

# %%
