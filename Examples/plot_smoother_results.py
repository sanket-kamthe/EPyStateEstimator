# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/so/Documents/Projects/pyStateEstimator')
from Systems import UniformNonlinearGrowthModel, BearingsOnlyTracking
import sqlite3
from MomentMatching.Estimator import Estimator
from MomentMatching import UnscentedTransform, MonteCarloTransform, TaylorTransform
from ExpectationPropagation.Nodes import build_nodes, node_estimator, node_system
from ExpectationPropagation.Iterations import ep_iterations
from Utils.Metrics import node_metrics

# %%
def select_transform(id='UT', dim=1, samples=int(1e4)):

    if id.upper() == 'UT':
        transition_transform = UnscentedTransform(dim=dim, beta=2, alpha=1, kappa=3)
        measurement_transform = UnscentedTransform(dim=dim, beta=2, alpha=1, kappa=2)

    elif id.upper() == 'TT':
        transition_transform = TaylorTransform(dim=dim)
        measurement_transform = TaylorTransform(dim=dim)

    elif id.upper() == 'MCT':
        transition_transform = MonteCarloTransform(dim=dim, number_of_samples=samples)
        measurement_transform = MonteCarloTransform(dim=dim, number_of_samples=samples)

    else:
        transition_transform = UnscentedTransform(dim=dim, beta=2, alpha=1, kappa=3)
        measurement_transform = UnscentedTransform(dim=dim, beta=2, alpha=1, kappa=2)

    return transition_transform, measurement_transform

# %%
# Model configuration
system = UniformNonlinearGrowthModel()
sys_dim = 1
timesteps = 100

# Set parameters
SEED = 201
trans_id = 'MCT'
power = 0.4
damping = 0.8

# Connect to database
con = sqlite3.connect("corrected_ungm.db", detect_types=sqlite3.PARSE_DECLTYPES)
cursor = con.cursor()

query_str = "SELECT {}" \
            " FROM {}" \
            " WHERE Transform='{}' AND Seed={} AND Power={} AND Damping={}"

# Plot RMSE and NLL for corresponding parameters
metrics = ['RMSE', 'NLL']
fig, axs = plt.subplots(1, 2, figsize=(10,4))
for i, metric in enumerate(metrics):
    row = cursor.execute(query_str.format(metric, 'UNGM_EXP', trans_id, SEED, power, damping)).fetchall()
    axs[i].plot(row)
    axs[i].set_ylabel(metric)
    axs[i].set_xlabel('# Iterations')
plt.suptitle(f'Seed: {SEED}')
plt.tight_layout()

# %%
# Run dynamical system
np.random.seed(seed=SEED)
data = system.simulate(timesteps)
x_true, x_noisy, y_true, y_noisy = zip(*data)

# Build EP nodes
num_samples = int(1e4)
transform, meas_transform = select_transform(trans_id, dim=sys_dim, samples=num_samples)
estim = Estimator(trans_map=transform,
                meas_map=meas_transform,
                trans_noise=system.transition_noise.cov,
                meas_noise=system.measurement_noise.cov,
                power=power,
                damping=damping)

nodes = build_nodes(N=timesteps, dim=sys_dim)
nodes = node_estimator(nodes=nodes, estimator=estim)
nodes = node_system(nodes=nodes, system_model=system, measurements=y_noisy)


# Run EP iteration
max_iter = 50
means = np.zeros((max_iter, timesteps))
stds = np.zeros((max_iter, timesteps))
rmse_list, nll_list = [], []
for i in range(max_iter):
    for node in nodes:
        node.fwd_update()
        node.meas_update()
    for node in reversed(nodes):
        node.back_update()
    for j, node in enumerate(nodes):
        means[i, j] = node.marginal.mean
        stds[i, j] = np.sqrt(node.marginal.cov)
    rmse_, nll_ = node_metrics(nodes, x_noisy)
    rmse_list.append(rmse_)
    nll_list.append(nll_)

# Plot EP smoother results
iters = [0, 29, 49]
fig, axs = plt.subplots(3)
for i, iter in enumerate(iters):
    m, s = means[iter], stds[iter]
    axs[i].plot(m, 'C0', label='prediction')
    axs[i].fill_between(np.arange(timesteps), m-1.96*s, m+1.96*s, color='C0', alpha=0.3)
    axs[i].plot(x_true, 'C3', label='truth')
    axs[i].set_title(f'Seed: {SEED}, Iteration: {iter+1}')
plt.tight_layout()


metrics = {"RMSE" : rmse_list, "NLL" : nll_list}

fix, axs = plt.subplots(2)
for i, key in enumerate(metrics.keys()):
    axs[i].plot(metrics[key])
    axs[i].set_ylabel(key)
    axs[i].grid()
    if i == 0:
        axs[i].set_title(f"Power = {power}, Damping = {damping}")
    elif i == 1:
        axs[i].set_xlabel("# Iterations")
    

# %%
