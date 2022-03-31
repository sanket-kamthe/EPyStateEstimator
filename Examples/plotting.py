# %%
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterable

# %%
query_str = "SELECT {}" \
            " FROM {}" \
            " WHERE Transform='{}' AND Seed={} AND Power={} AND Damping={} AND Iter={}"

parent_str = "SELECT {} FROM {} WHERE Transform='{}' AND Seed={}"


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
        #con = sqlite3.connect("bot_final.db", detect_types=sqlite3.PARSE_DECLTYPES)
        con = sqlite3.connect("../temp.db", detect_types=sqlite3.PARSE_DECLTYPES)
    elif experiment == 'bott':
        exp_table = 'BOTT_EXP'
        con = sqlite3.connect("bott_final.db", detect_types=sqlite3.PARSE_DECLTYPES)

    cursor = con.cursor()
    return exp_table, cursor


@dataclass
class PlotConfig:
    figsize: Tuple = (12, 5)
    xlabel_fontsize: int = 16
    ylabel_fontsize: int = 16
    title_fontsize: int = 16
    legend_fontsize: int = 12
    xticks: Iterable = None
    xtick_labels: Iterable = None
    xtick_labelsize: int = 14
    ytick_labelsize: int = 14
    vmin_rmse: float = None
    vmax_rmse: float = None
    vmin_nll: float = None
    vmax_nll: float = None


@dataclass
class ExpConfig:
    exp_name: str
    Seeds: Iterable
    exp_tablename : str = field(init=False)
    cursor: sqlite3.Cursor = field(init=False)
    def __post_init__(self):
        assert self.exp_name in ['ungm', 'bot', 'bott'], 'Experiment must be one of (ungm, bot, bott)'
        tablename, cursor = select_data(self.exp_name)
        self.exp_tablename = tablename
        self.cursor = cursor


def plot_multiple_1d(plot_config: PlotConfig, exp_config: ExpConfig, control_vars: Dict,
                     trans_types=['TT', 'UT', 'MCT'], colors=['C3', 'C2', 'C0']):
    # Prepare data
    keys = list(control_vars.keys())
    key1, key2 = keys
    var1, var2 = control_vars[key1], control_vars[key2]
    ncol = len(var1)
    vars = ['Power', 'Damping', 'Iter']
    labels = {'Power': 'Power', 'Damping': 'Damping', 'Iter': 'Iteration'}
    assert len(keys) == 2, 'Must specify two control variables'
    assert len(trans_types) == len(colors), 'Number of transformation types and colors must match'
    for key in keys:
        assert key in vars, 'Variable must be one of (Power, Damping, Iter)'
    query_str = parent_str
    for var in keys:
        query_str = query_str + f" AND {var}" + "={}"
        vars.remove(var)
    key3 = vars[0]
    # Plot results
    Seeds, exp_table, cursor = exp_config.Seeds, exp_config.exp_tablename, exp_config.cursor
    fig, axs = plt.subplots(2, ncol, figsize=plot_config.figsize)
    # Set background grid
    for n in range(2):
        for i in range(ncol):
            axs[n, i].set_facecolor('#EBEBEB')
            axs[n, i].grid(True, color='w', linestyle='-', linewidth=1)
    # Plot
    plt.rcParams['xtick.labelsize'] = plot_config.xtick_labelsize
    plt.rcParams['ytick.labelsize'] = plot_config.ytick_labelsize
    for i, params in enumerate(zip(var1, var2)):
        param1, param2 = params # Control variables. e.g. Power and Damping
        for c, trans_id in zip(colors, trans_types):
            RMSE_data, NLL_data = [], []
            for SEED in Seeds:
                row = cursor.execute(query_str.format('RMSE', exp_table, trans_id, int(SEED), param1, param2)).fetchall()
                RMSE_data.append(np.array(row).squeeze())
                row = cursor.execute(query_str.format('NLL', exp_table, trans_id, int(SEED), param1, param2)).fetchall()
                NLL_data.append(np.array(row).squeeze())
            RMSE_data = np.array(RMSE_data)
            NLL_data = np.array(NLL_data)

            RMSE_mean, RMSE_std = get_mean_and_std(RMSE_data)
            NLL_mean, NLL_std = get_mean_and_std(NLL_data)

            N = RMSE_mean.shape[0]

            axs[0, i].plot(np.arange(0, N), RMSE_mean, c=c, label=trans_id, zorder=1)
            axs[0, i].fill_between(np.arange(0, N), RMSE_mean-RMSE_std, RMSE_mean+RMSE_std, alpha=0.2, color=c, zorder=2)

            axs[1, i].plot(np.arange(0, N), NLL_mean, c=c, label=trans_id, zorder=1)
            axs[1, i].fill_between(np.arange(0, N), NLL_mean-NLL_std, NLL_mean+NLL_std, alpha=0.2, color=c, zorder=2)

        if i ==0:
            axs[0, i].set_ylabel("RMSE", fontsize=plot_config.ylabel_fontsize)
        axs[0, i].set_title(f"{labels[key1]}: {param1}, {labels[key2]}: {param2}", fontsize=plot_config.title_fontsize)
        axs[0, i].legend(fontsize=plot_config.legend_fontsize, loc='upper right', ncol=ncol)
        axs[0, i].set_ylim(plot_config.vmin_rmse, plot_config.vmax_rmse)

        axs[1, i].set_xlabel(f"{labels[key3]}", fontsize=plot_config.xlabel_fontsize)
        if i==0:
            axs[1, i].set_ylabel("NLL", fontsize=plot_config.ylabel_fontsize)
        axs[1, i].legend(fontsize=plot_config.legend_fontsize, loc='upper right', ncol=ncol)
        axs[1, i].set_ylim(plot_config.vmin_nll, plot_config.vmax_nll)

        for j in [0,1]:
            xticks = plot_config.xticks
            xticklabels = plot_config.xtick_labels
            if xticks is not None: axs[j, i].set_xticks(xticks)
            if xticklabels is not None: axs[j, i].set_xticklabels(xticklabels) 

    plt.tight_layout()

    return fig, axs

# %%
# First plot (figure 4)
experiment = 'ungm'
Seeds = np.arange(101, 1101, 100)
plot_config = PlotConfig(xticks=[0, 10, 20, 30, 40, 50],
                         vmin_rmse=2, vmax_rmse=12,
                         vmin_nll=0, vmax_nll=14)
exp_config = ExpConfig(exp_name=experiment, Seeds=Seeds)
control_vars = {'Power': [1.0, 1.0, 0.8], 'Damping': [1.0, 0.8, 0.8]}
fig, axs = plot_multiple_1d(plot_config, exp_config, control_vars)

# %%
# Plot for Taylor transform only
experiment = 'ungm'
Seeds = np.arange(101, 1101, 100)
plot_config = PlotConfig(xticks=[0, 10, 20, 30, 40, 50],
                         vmin_rmse=4, vmax_rmse=12,
                         vmin_nll=-100, vmax_nll=500)
exp_config = ExpConfig(exp_name=experiment, Seeds=Seeds)
control_vars = {'Power': [1.0, 1.0, 0.8], 'Damping': [1.0, 0.8, 0.8]}
_, _ = plot_multiple_1d(plot_config, exp_config, control_vars, trans_types=['TT'], colors=['C3'])

# %%
# Plot for Unscented transform only
experiment = 'bott'
Seeds = [501]
plot_config = PlotConfig(xticks=[0, 10, 20, 30, 40, 50],
                         vmin_rmse=None, vmax_rmse=None,
                         vmin_nll=None, vmax_nll=None)
exp_config = ExpConfig(exp_name=experiment, Seeds=Seeds)
control_vars = {'Power': [1.0, 1.0, 0.8], 'Damping': [1.0, 0.8, 0.8]}
_, _ = plot_multiple_1d(plot_config, exp_config, control_vars, trans_types=['MCT'], colors=['C3'])

# %%
# Second plot (figure 5)
experiment = 'ungm'
plot_config = PlotConfig(xticks=np.linspace(0, 19, 4),
                         xtick_labels=np.linspace(0.1, 1.0, 4),
                         vmin_rmse=2, vmax_rmse=16,
                         vmin_nll=None, vmax_nll=None)
exp_config = ExpConfig(exp_name=experiment, Seeds=Seeds)
control_vars = {'Iter': [2, 10, 50], 'Damping': [0.4, 0.4, 0.4]}
fig, axs = plot_multiple_1d(plot_config, exp_config, control_vars)
axs[1, 0].set_ylim(2, 5)
axs[1, 1].set_ylim(0, 50)
axs[1, 2].set_ylim(-30, 1000)


# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import namedtuple

experiment = 'bott'
exp_table, cursor = select_data(experiment)

# Heat map
methods = [None, 'none','nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

PlotConfig = namedtuple('PlotConfig', ['xlabel_kwargs','ylabel_kwargs',
                                        'title_kwargs', 'vmin', 'vmax'])

def plot_power_sweep(config, img, methods, title, ax=None):

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


def all_seeds_image(table, transform, iters=50, seeds=None):
  all_prmse = []
  all_pnll = []
  cursor.execute(f"SELECT DISTINCT Seed from {table}")
  if seeds is None:
    seeds = cursor.fetchall()
    seeds = [seed[0] for seed in seeds]
  for seed in seeds:
    rmse, nll, p, d = make_image_data(table, transform, seed, iters=iters)
    all_prmse.append(rmse)
    all_pnll.append(nll)
  prmse = np.array(all_prmse)
  pnll = np.array(all_pnll)

  return np.mean(prmse, axis=0), np.mean(pnll, axis=0), p, d


def plot_sweep(config, table, transform, iters=10, kind='rmse', ax=None, seeds=None):
    p_rmse, p_nll, p, d = all_seeds_image(table, transform, iters=iters, seeds=seeds)
    title = f'Transform = {transform}'
    if kind == 'rmse':
        plot_power_sweep(config, (p_rmse.T), methods, title, ax)
    else:
        plot_power_sweep(config, (p_nll.T), methods, title, ax)


def plot_all_transforms(config, table, iters=10, kind='rmse'):
    transform_list = ['TT', 'UT', 'MCT']
    fig, axs = plt.subplots(1, 3, figsize=(26, 9))
    for i, trans in enumerate(transform_list):
        plot_sweep(config, table, trans, iters, kind, axs[i])
    if kind == 'rmse':
        plt.suptitle('RMSE', fontweight='bold', fontsize=30)
    else:
        plt.suptitle('NLL', fontweight='bold', fontsize=30)
    plt.tight_layout()


# %% 
if experiment == 'ungm':
    rmse_vmin = 2.5
    rmse_vmax = 10.0
    nll_vmin = -15.0
    nll_vmax = 500.0
elif experiment == 'bot':
    rmse_vmin = 0.1
    rmse_vmax = 10.0
    nll_vmin = -5.0
    nll_vmax = 1000.0
elif experiment == 'bott':
    rmse_vmin = None
    rmse_vmax = None
    nll_vmin = None
    nll_vmax = None

config = PlotConfig(xlabel_kwargs={'fontsize':30},
                    ylabel_kwargs={'fontsize':30},
                    title_kwargs={'fontsize':30},
                    vmin=rmse_vmin, vmax=rmse_vmax)

plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plot_all_transforms(config, exp_table, 50, 'rmse')

config=config._replace(vmin=nll_vmin)
config=config._replace(vmax=nll_vmax)
plot_all_transforms(config, exp_table, 50, 'nll')

# %%
# Test for bott experiment
tranform_ = 'MCT'
iteration = 50
seeds = [101, 201, 501, 601, 901]
config = PlotConfig(xlabel_kwargs={'fontsize':16},
                    ylabel_kwargs={'fontsize':16},
                    title_kwargs={'fontsize':16},
                    vmin=-2, vmax=1)

plt.figure(figsize=(8,5))
plot_sweep(config, exp_table, tranform_, iters=iteration, kind='nll', seeds=seeds)

# %%
# tranform_ = 'MCT'
# plt.figure(figsize=(8,5))
# plot_sweep(exp_table, tranform_, iters=iteration, kind='nll')

# %%
