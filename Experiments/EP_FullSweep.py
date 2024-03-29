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
from numpy.linalg import LinAlgError
from Systems import UniformNonlinearGrowthModel, BearingsOnlyTracking, BearingsOnlyTrackingTurn, L96
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


def select_transform(id='UT', dim=1, samples=int(5e4), alpha1=1, alpha2=1, beta1=2, beta2=2, kappa1=3, kappa2=2):

    if id.upper() == 'UT':
        transition_transform = UnscentedTransform(dim=dim, beta=beta1, alpha=alpha1, kappa=kappa1)
        measurement_transform = UnscentedTransform(dim=dim, beta=beta2, alpha=alpha2, kappa=kappa2)

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


query_str= "SELECT RMSE" \
           " from {}" \
           " WHERE Transform='{}' AND Seed = {} AND Power ={} AND Damping = {} AND Iter = 50"


def full_sweep(config, seed_range, trans_types, power_range, damp_range, override=False):
    con, system, timesteps, table = config.con, config.system, config.timesteps, config.exp_table_name
    db = con.cursor()
    total = len(list(itertools.product(seed_range, trans_types, power_range, damp_range)))
    i = 1
    for SEED in seed_range:
        np.random.seed(seed=SEED)
        data = system.simulate(timesteps)
        insert_dynamics_data(db, config.dyn_table_name, data, int(SEED))
        x, y = zip(*data)
        for trans_id, power, damping in itertools.product(trans_types, power_range, damp_range):
            print(f"running {i}/{total}, SEED = {SEED}, trans = {trans_id}, power = {power}, damping = {damping}")
            query = query_str.format(table, trans_id, SEED, power, damping)
            db.execute(query)
            exits = db.fetchall()
            try:
                if override:
                    power_sweep(config, x, y, trans_id=trans_id, SEED=int(SEED), power=power, damping=damping)
                else:
                    if len(exits) == 0: # Skips sweep if result is already computed for the given settings
                        power_sweep(config, x, y, trans_id=trans_id, SEED=int(SEED), power=power, damping=damping)
            except LinAlgError:
                print('failed for seed={}, power={},'
                    ' damping={}, transform={:s}'.format(SEED, power, damping, trans_id))
                continue
            i += 1
    

@click.command()
@click.option('-l', '--logdir', type=str, default="../log/temp.db", help='Set directory to save results')
@click.option('-d', '--dynamic-system', type=click.Choice(['UNGM', 'BOT', 'BOTT', 'L96']), default='UNGM', help='Choose state-space model')
@click.option('-s', '--seeds', type=click.INT, default=[101], multiple=True, help='Random seed for experiment (multiple allowed)')
@click.option('-t', '--trans-types', type=click.Choice(['TT', 'UT', 'MCT']), default=['TT', 'UT', 'MCT'], multiple=True, help='Transformation types (multiple allowed)')
@click.option('-i', '--num-iter', type=int, default=50, help='Number of EP iterations')
@click.option('-o', '--override/--no-override', default=False, help='Override saved results')
def main(logdir, dynamic_system, seeds, trans_types, num_iter, override):
    con = sqlite3.connect(logdir, detect_types=sqlite3.PARSE_DECLTYPES)
    db = con.cursor()
    if dynamic_system == 'UNGM':
        system = UniformNonlinearGrowthModel()
        sys_dim = 1
        timesteps = 100
        dyn_table_name = 'UNGM_SIM'
        exp_table_name = 'UNGM_EXP'
    elif dynamic_system == 'BOT':
        system = BearingsOnlyTracking()
        sys_dim = 4
        timesteps = 50
        dyn_table_name = 'BOT_SIM'
        exp_table_name = 'BOT_EXP'
    elif dynamic_system == 'BOTT':
        system = BearingsOnlyTrackingTurn()
        sys_dim = 5
        timesteps = 50
        dyn_table_name = 'BOTT_SIM'
        exp_table_name = 'BOTT_EXP'
    elif dynamic_system == 'L96':
        system = L96(init_cond_path='../Systems/L96_initial_conditions.npy')
        sys_dim = 40
        timesteps = 50
        dyn_table_name = 'L96_SIM'
        exp_table_name = 'L96_EXP'

    create_dynamics_table(db, name=dyn_table_name)
    create_experiment_table(db, table_name=exp_table_name)

    config = Config(con=con,
                    system=system,
                    timesteps=timesteps,
                    sys_dim=sys_dim,
                    num_iter=num_iter,
                    dyn_table_name=dyn_table_name,
                    exp_table_name=exp_table_name)

    num_power = 19
    num_damping = 19
    # power_range = np.linspace(0.1, 1.0, num=num_power)
    # damp_range = np.linspace(0.1, 1.0, num=num_damping)
    power_range = np.linspace(0.1, 1.0, num=10)
    damp_range = [1.0, 0.8]

    full_sweep(config, seeds, trans_types, power_range, damp_range, override)


if __name__ == '__main__':
    main()
