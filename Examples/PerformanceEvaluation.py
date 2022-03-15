# %%
import numpy as np
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/so/Documents/Projects/pyStateEstimator')
from Systems import UniformNonlinearGrowthModel
from MomentMatching import UnscentedTransform, MonteCarloTransform, TaylorTransform
from MomentMatching.Estimator import Estimator
from ExpectationPropagation.Nodes import build_nodes, node_estimator, node_system
from ExpectationPropagation.Iterations import ep_iterations
from Utils.Metrics import node_metrics
from Utils.Database import create_dynamics_table, insert_dynamics_data
import sqlite3
from Utils.Database import create_experiment_table, Exp_Data
import itertools
from functools import partial

# %%
def select_transform(id='UT', dim=1, samples=int(5e4)):

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


def power_sweep(con, x_true, y_meas, trans_id='UT', SEED=0, power=1, damping=1, dim=1, samples=int(1e4)):
    transform, meas_transform = select_transform(id=trans_id, dim=dim, samples=samples)

    exp_data = Exp_Data(Transform=trans_id,
                        Seed=SEED,
                        Iter=0,
                        Power=power,
                        Damping=damping,
                        RMSE=0.0,
                        NLL=0.0,
                        Mean=0.0,
                        Variance=0.0,
                        Nodes=[])

    estim = Estimator(trans_map=transform,
                      meas_map=meas_transform,
                      trans_noise=system.transition_noise.cov,
                      meas_noise=system.measurement_noise.cov,
                      power=power,
                      damping=damping)

    nodes = build_nodes(N=N, dim=dim)
    nodes = node_estimator(nodes=nodes, estimator=estim)
    nodes = node_system(nodes=nodes, system_model=system, measurements=y_meas)

    ep_iterations(nodes,
                  max_iter=50,
                  conn=con,
                  x_true=x_true,
                  exp_data=exp_data,
                  print_result=False) # This does the full EP iteration + log results

# %%
# Set up connection to database
con = sqlite3.connect("temp_ungm.db", detect_types=sqlite3.PARSE_DECLTYPES)
db = con.cursor()
table_name = 'UNGM_SIM'
create_experiment_table(db=con.cursor())
power_range = [1.0, 1.0, 0.8]
damp_range = [1.0, 0.8, 0.8]
trans_types = ['TT', 'UT', 'MCT']
#trans_types = ['MCT']
Seeds = np.arange(100, 110)
total = len(list(itertools.product(Seeds, trans_types, power_range)))

query_str= "SELECT RMSE" \
           " from UNGM_EXP" \
           " WHERE Transform='{}' AND Seed = {} AND Power ={} AND Damping = {} AND Iter = 50"

# %%
system = UniformNonlinearGrowthModel()
N = 100
sys_dim = 1
max_iter = 50
step = 1
for trans_id in trans_types:
    transform, meas_transform = select_transform(id=trans_id)
    for i, SEED in enumerate(Seeds):
        np.random.seed(seed=SEED)
        for power, damping in zip(power_range, damp_range):
            print(f"running {step}/{total}, trans = {trans_id}, SEED = {SEED}, power = {power}, damping = {damping}")
            step += 1
            data = system.simulate(N)
            x_true, x_noisy, y_true, y_noisy = zip(*data)
            query = query_str.format(trans_id, SEED, power, damping)
            db.execute(query)
            exits = db.fetchall()
            try:
                if len(exits) == 0:
                    power_sweep(con, x_noisy, y_noisy, trans_id=trans_id, SEED=int(SEED), power=power, damping=damping, dim=sys_dim)
            except LinAlgError:
                print('failed for seed={}, power={},'
                    ' damping={}, transform={:s}'.format(SEED, power, damping, trans_id))
                continue

con.commit()
con.close()

