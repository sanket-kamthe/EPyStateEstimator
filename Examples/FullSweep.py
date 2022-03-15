# Copyright 2022 Sanket Kamthe
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
import sys
sys.path.append('/home/so/Documents/Projects/pyStateEstimator')
from Systems import UniformNonlinearGrowthModel, BearingsOnlyTracking
from MomentMatching import UnscentedTransform, MonteCarloTransform, TaylorTransform
from MomentMatching.Estimator import Estimator
from ExpectationPropagation.Nodes import build_nodes, node_estimator, node_system
from ExpectationPropagation.Iterations import ep_iterations
from Utils.Database import create_dynamics_table, insert_dynamics_data
import sqlite3
from Utils.Database import create_experiment_table, Exp_Data
import itertools
import click
from collections import namedtuple


Config = namedtuple('Config', ['con', 'system', 'timesteps', 'sys_dim', 'num_iter', 'dyn_table_name', 'exp_table_name'])


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


def power_sweep(config, x_true, y_meas, trans_id='UT', SEED=0, power=1, damping=1, samples=int(1e4)):
    con, system, timesteps, sys_dim, num_iter = config.con, config.system, config.timesteps, config.sys_dim, config.num_iter
    transform, meas_transform = select_transform(id=trans_id, dim=sys_dim, samples=samples)

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

    nodes = build_nodes(N=timesteps, dim=sys_dim)
    nodes = node_estimator(nodes=nodes, estimator=estim)
    nodes = node_system(nodes=nodes, system_model=system, measurements=y_meas)

    ep_iterations(nodes,
                  max_iter=num_iter,
                  conn=con,
                  x_true=x_true,
                  exp_data=exp_data,
                  table_name=config.exp_table_name,
                  print_result=False) # Full EP sweep + log results


def full_sweep(config, seed_range, trans_types, power_range, damp_range):
    con, system, timesteps = config.con, config.system, config.timesteps
    db = con.cursor()
    total = len(list(itertools.product(seed_range, trans_types, power_range, damp_range)))
    i = 1
    for SEED in seed_range:
        np.random.seed(seed=SEED)
        data = system.simulate(timesteps)
        insert_dynamics_data(db, config.dyn_table_name, data, int(SEED))
        x_true, y_true, y_noisy = zip(*data)
        for trans_id, power, damping in itertools.product(trans_types, power_range, damp_range):
            print(f"running {i}/{total}, SEED = {SEED}, trans = {trans_id}, power = {power}, damping = {damping}")
            power_sweep(config, x_true, y_noisy, trans_id=trans_id, SEED=int(SEED), power=power, damping=damping)
            i += 1
    

@click.command()
@click.option('-l', '--logdir', type=str, default="temp.db", help='Set database directory to log results')
@click.option('-s', '--system', type=click.Choice(['UNGM', 'BOT']), default='UNGM', help='Choose state-space model')
def main(logdir, system):
    con = sqlite3.connect(logdir, detect_types=sqlite3.PARSE_DECLTYPES)
    db = con.cursor()
    if system == 'UNGM':
        system = UniformNonlinearGrowthModel()
        dyn_table_name = 'UNGM_SIM'
        exp_table_name = 'UNGM_EXP'
    elif system == 'BOT':
        system = BearingsOnlyTracking()
        dyn_table_name = 'BOT_SIM'
        exp_table_name = 'BOT_EXP'

    create_dynamics_table(db, name=dyn_table_name)
    create_experiment_table(db, table_name=exp_table_name)

    timesteps = 100
    sys_dim = 1
    num_iter = 50
    config = Config(con=con,
                    system=system,
                    timesteps=timesteps,
                    sys_dim=sys_dim,
                    num_iter=num_iter,
                    dyn_table_name=dyn_table_name,
                    exp_table_name=exp_table_name)

    num_power = 20
    num_damping = 20
    power_range = np.linspace(0.1, 1.0, num=num_power)
    damp_range = np.linspace(0.1, 1.0, num=num_damping)
    trans_types = ['TT', 'UT', 'MCT']
    seed_range = np.arange(15)

    full_sweep(config, seed_range, trans_types, power_range, damp_range)


if __name__ == '__main__':
    main()
