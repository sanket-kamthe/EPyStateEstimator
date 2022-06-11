import numpy as np
from Systems import L96
import sqlite3
from MomentMatching.Estimator import Estimator
from MomentMatching import UnscentedTransform, MonteCarloTransform, TaylorTransform
from ExpectationPropagation.Nodes import build_nodes, node_estimator, node_system
from Utils.Metrics import node_metrics
import itertools

def select_transform(id='UT', dim=1, meas_dim=1, samples=int(1e4)):

    if id.upper() == 'UT':
        transition_transform = UnscentedTransform(dim=dim, beta=2, alpha=1, kappa=3)
        measurement_transform = UnscentedTransform(dim=meas_dim, beta=2, alpha=1, kappa=2)

    elif id.upper() == 'TT':
        transition_transform = TaylorTransform(dim=dim)
        measurement_transform = TaylorTransform(dim=meas_dim)

    elif id.upper() == 'MCT':
        transition_transform = MonteCarloTransform(dim=dim, number_of_samples=samples)
        measurement_transform = MonteCarloTransform(dim=meas_dim, number_of_samples=samples)

    else:
        transition_transform = UnscentedTransform(dim=dim, beta=2, alpha=1, kappa=3)
        measurement_transform = UnscentedTransform(dim=meas_dim, beta=2, alpha=1, kappa=2)

    return transition_transform, measurement_transform


seeds = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001]
dims = [5, 10, 20, 40, 100, 200]

# Model configuration
F = 8
timesteps = 50

# Set parameters
trans_id = 'MCT'
power_list = [1.0, 0.8, 0.1]
damping_list = [1.0, 1.0, 1.0]
iter_list = [10, 10, 10]

total = len(list(itertools.product(dims, seeds, power_list)))

# Create experiment table
con = sqlite3.connect("../log/L96_dim_experiment_3.db", detect_types=sqlite3.PARSE_DECLTYPES)
db = con.cursor()
schema = """ CREATE TABLE IF NOT EXISTS L96_EXP
            (
            Transform TEXT,
            Seed INT,
            dim INT,
            Iter REAL,
            Power REAL,
            Damping REAL,
            RMSE REAL,
            NLL REAL,
            UNIQUE (Transform, Seed, dim, Iter, Power, Damping)
            )"""
db.execute(schema)
con.commit()
query = "INSERT INTO L96_EXP" \
        " (Transform, Seed, dim, Iter, Power, Damping, RMSE, NLL)" \
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"

i = 1
for dim, seed in itertools.product(dims, seeds):
    # Run dynamical system
    fname = f'../log/L96_initial_condition_F_{F}_dim_{dim}.npy'
    system = L96(dim=dim, init_cond_path=fname)
    np.random.seed(seed=seed)
    data = system.simulate(timesteps)
    x, y = zip(*data)
    for power, damping, max_iter in zip(power_list, damping_list, iter_list):
        print(f"{i}/{total}: dim={dim}, seed={seed}, power={power}")

        # Build EP nodes
        transform, meas_transform = select_transform(trans_id, dim=dim, meas_dim=dim)
        estim = Estimator(trans_map=transform,
                        meas_map=meas_transform,
                        trans_noise=system.transition_noise.cov,
                        meas_noise=system.measurement_noise.cov,
                        power=power,
                        damping=damping)

        nodes = build_nodes(N=timesteps, dim=dim)
        nodes = node_estimator(nodes=nodes, estimator=estim)
        nodes = node_system(nodes=nodes, system_model=system, measurements=y)

        # Run EP iteration
        means = np.zeros((max_iter, timesteps, dim))
        stds = np.zeros((max_iter, timesteps, dim))
        rmse_list, nll_list = [], []
        for iter in range(max_iter):
            for node in nodes:
                node.fwd_update()
                node.meas_update()
            for node in reversed(nodes):
                node.back_update()
            for j, node in enumerate(nodes):
                means[iter, j] = node.marginal.mean
                stds[iter, j] = np.sqrt(np.diag(node.marginal.cov))
            rmse_, nll_ = node_metrics(nodes, x)
            rmse_list.append(rmse_)
            nll_list.append(nll_)
            db.execute(query, (trans_id, seed, dim, iter, power, damping, rmse_, nll_))
            con.commit()

        metrics = {"RMSE" : rmse_list, "NLL" : nll_list}

        i += 1