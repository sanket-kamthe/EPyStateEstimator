# Copyright 2018 Sanket Kamthe
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
# import matplotlib.pyplot as plt
from numpy.linalg import LinAlgError

from Systems import UniformNonlinearGrowthModel
from MomentMatching import UnscentedTransform, MonteCarloTransform, TaylorTransform
from MomentMatching.Estimator import Estimator
from ExpectationPropagation.Nodes import build_nodes, node_estimator, node_system
from ExpectationPropagation.Iterations import ep_iterations
from Utils.Database import create_dynamics_table, insert_dynamics_data
import sqlite3
from Utils.Database import create_experiment_table, Exp_Data
import itertools


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


def power_sweep(trans_id='UT', power=1, damping=1, dim=1, samples=int(1e4)):
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
    nodes = node_system(nodes=nodes, system_model=system, measurements=y_noisy)

    ep_iterations(nodes, max_iter=50, conn=con, x_true=x_true, exp_data=exp_data)


SEED = 101

np.random.seed(seed=SEED)
sys_dim = 1
N = 100
system = UniformNonlinearGrowthModel()
# system = BearingsOnlyTracking()
# system = BearingsOnlyTrackingTurn()
data = system.simulate(N)
x_true, x_noisy, y_true, y_noisy = zip(*data)


con = sqlite3.connect("temp_ungm.db", detect_types=sqlite3.PARSE_DECLTYPES)
# con = sqlite3.connect("/home/ubuntu/temp_ungm.db", detect_types=sqlite3.PARSE_DECLTYPES)
db = con.cursor()
table_name = 'UNGM_SIM'
create_dynamics_table(db, name=table_name)
insert_dynamics_data(db, table_name=table_name, data=data, seed=SEED)
con.commit()
# con.close()

power = 0.5
damping = 0.5

trans_id = 'UT'
transform, meas_transform = select_transform(id=trans_id)

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

nodes = build_nodes(N=N, dim=sys_dim)
nodes = node_estimator(nodes=nodes, estimator=estim)
nodes = node_system(nodes=nodes, system_model=system, measurements=y_noisy)


# plt.plot(x_true[:, 0], 'r--', label='X_true')
# kalman_filter(nodes)
# filt_mean = [node.marginal.mean for node in nodes]

# plt.plot(filt_mean, 'b-', label='X_ Filtered')




# kalman_smoother(nodes)

# smoothed_mean = [node.marginal.mean for node in nodes]
create_experiment_table(db=con.cursor())
db = con.cursor()
x = 20
y = 20
power_range = np.linspace(0.1, 1.0, num=x)
damp_range = np.linspace(0.1, 1.0, num=y)
trans = ['UT',  'TT']
total = len(list(itertools.product(trans, power_range, damp_range)))
i = 0
query_str= "SELECT RMSE" \
           " from UNGM_EXP" \
           " WHERE Transform='{}' AND Seed = {} AND Power ={} AND Damping = {} AND Iter = 100"
for trans, power, damping in itertools.product(trans, power_range, damp_range):
    # "EXITS(SELECT RMSE from UNGM_EXP WHERE Transform={}, "
    query = query_str.format(trans, SEED, power, damping)
    print('running {}/{} power={}/ Damping={} '.format(i, total, power, damping))

    db.execute(query)
    exits = db.fetchall()
    i += 1
    print('done {}/{} '.format(i, total))
    print('')
    try:
        if len(exits) == 0:
            power_sweep(trans_id=trans, power=power, damping=damping, dim=sys_dim)
    except LinAlgError:
        print('failed for power={},'
              ' damping={}, transform={:s}'.format(power, damping, trans))
        continue

# ep_iterations(nodes, max_iter=100, conn=con, x_true=x_true, exp_data=exp_data)
# EP_mean = [node.marginal.mean for node in nodes]
# plot_gaussian_node(nodes)
# plt.plot(filt_mean, 'b--', label='X_ Filtered')
# plt.plot(smoothed_mean, 'g--', label='X_ Smoothed')
# plt.plot(EP_mean, 'b--', label='X_ S')
# plt.legend(loc='best')
# plt.show()
con.commit()
con.close()
