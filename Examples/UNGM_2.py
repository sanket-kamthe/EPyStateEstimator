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
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

from Systems import UniformNonlinearGrowthModel
from MomentMatching import UnscentedTransform, MonteCarloTransform, TaylorTransform
from MomentMatching.Estimator import Estimator
from MomentMatching.Nodes import build_nodes, node_estimator, node_system
from MomentMatching.Iterations import kalman_filter, kalman_smoother, ep_update, ep_iterations
from Utils.Plot_Helper import plot_gaussian_node
from MomentMatching.Database import create_dynamics_table, insert_dynamics_data
import sqlite3

SEED = 11

np.random.seed(seed=SEED)
sys_dim = 1
N = 50
system = UniformNonlinearGrowthModel()
data = system.simulate(N)
x_true, x_noisy, y_true, y_noisy = zip(*data)


con = sqlite3.connect("temp.db", detect_types=sqlite3.PARSE_DECLTYPES)
db = con.cursor()
create_dynamics_table(db, name='UNGM')
insert_dynamics_data(db, table_name='UNGM', data=data, seed=SEED)
con.commit()
con.close()

power = 1
damping = 1

transform = UnscentedTransform(dim=1,  beta=2,  alpha=1, kappa=3)
meas_transform = UnscentedTransform(dim=1, beta=2,  alpha=1, kappa=2)

# samples = int(1e4)
# transform = MonteCarloTransform(dim=1, number_of_samples=samples)
# meas_transform = MonteCarloTransform(dim=1, number_of_samples=samples)
# #

# transform = TaylorTransform(dim=1)
# meas_transform = TaylorTransform(dim=1)


estim = Estimator(trans_map=transform,
                  meas_map=meas_transform,
                  trans_noise=system.transition_noise.cov,
                  meas_noise=system.measurement_noise.cov,
                  power=power,
                  damping=damping)

nodes = build_nodes(N=N, dim=1)
nodes = node_estimator(nodes=nodes, estimator=estim)
nodes = node_system(nodes=nodes, system_model=system, measurements=y_noisy)


plt.plot(x_true, 'r--', label='X_true')
kalman_filter(nodes)
filt_mean = [node.marginal.mean for node in nodes]

plt.plot(filt_mean, 'b-', label='X_ Filtered')




kalman_smoother(nodes)

smoothed_mean = [node.marginal.mean for node in nodes]

ep_iterations(nodes, max_iter=100, x_true=x_true)
EP_mean = [node.marginal.mean for node in nodes]
plot_gaussian_node(nodes)
# plt.plot(filt_mean, 'b--', label='X_ Filtered')
plt.plot(smoothed_mean, 'g--', label='X_ Smoothed')
plt.plot(EP_mean, 'b--', label='X_ S')
plt.legend(loc='best')
plt.show()