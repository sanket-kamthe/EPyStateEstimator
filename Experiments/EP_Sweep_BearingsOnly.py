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
from Systems import BearingsOnlyTrackingTurn
from MomentMatching import UnscentedTransform, MonteCarloTransform, TaylorTransform
from MomentMatching.Estimator import Estimator
from ExpectationPropagation.Nodes import build_nodes, node_estimator, node_system
from ExpectationPropagation.Iterations import ep_iterations, ep_fwd_back_updates
from StateModel import Gaussian
from Utils.Metrics import rmse, nll
from Utils.Database import create_dynamics_table, insert_dynamics_data
import sqlite3
import itertools
import click
from collections import namedtuple
from Experiments.FullSweep import Config, select_transform


def create_experiment_table(db, table_name='BOTT_EXP'):
    """
    Custom table for the bearings only tracking turn experiment.
    This has an additional row 'Quantity', which indicates which physical 
    quantity is taken into account. This must be either 'position', 'velocity'
    or 'angle' (more precisely, angular velocity).
    """
    schema = """ CREATE TABLE IF NOT EXISTS {:s} 
                (
                Transform TEXT,
                Quantity TEXT,
                Seed INT,
                Iter REAL,
                Power REAL,
                Damping REAL,
                RMSE REAL, 
                NLL REAL,
                UNIQUE (Transform, Quantity, Seed, Iter, Power, Damping)
                )""".format(table_name)     
    db.execute(schema)


experiment_data_string = "INSERT OR IGNORE INTO {}" \
                         " (Transform, Quantity, Seed, Iter, Power, Damping, RMSE, NLL)" \
                         "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"


def insert_experiment_data(db, table_name, data):

    query = experiment_data_string.format(table_name)

    values = tuple(list(data.values()))
    db.execute(query, values)


def ep_iterations(nodes, max_iter=100, x_true=None, conn=None, exp_data=None, table_name='BOTT_EXP', print_result=True):
    db = conn.cursor()
    exp_data = exp_data._asdict()
    for i in range(max_iter):
        ep_fwd_back_updates(nodes)
        if x_true is not None:
            exp_data['Iter'] += 1
            pos_rmse, pos_nll, vel_rmse, vel_nll, ang_rmse, ang_nll = node_metrics(nodes, x_true=x_true)
            losses = {'position': [pos_rmse, pos_nll], 'velocity': [vel_rmse, vel_nll], 'angle': [ang_rmse, ang_nll]}
            for key, values in losses.items():
                exp_data['RMSE'], exp_data['NLL'] = values
                exp_data['Quantity'] = key
                insert_experiment_data(db=db, table_name=table_name, data=exp_data)
            if print_result:
                print('\n EP Pass {} NLL (pos) = {}, RMSE (pos) = {}'.format(i+1, pos_nll, pos_rmse))
            conn.commit()


def node_metrics(nodes, x_true):
    """
    Custom metric to compute the losses in the position, velocity
    and angular velocity components separately.
    """
    state_list = [nodes.marginal for nodes in nodes]
    mean_list = [state.mean for state in state_list]
    cov_list = [state.cov for state in state_list]

    pos_states = [Gaussian(np.array([m[0], m[2]]), np.array([[c[0,0], c[0,2]], [c[2,0], c[2,2]]])) 
                for m, c in zip(mean_list, cov_list)]
    vel_states = [Gaussian(np.array([m[1], m[3]]), np.array([[c[1,1], c[1,3]], [c[3,1], c[3,3]]])) 
                for m, c in zip(mean_list, cov_list)]
    ang_states = [Gaussian(np.array([m[4]]), np.array([[c[4,4]]])) 
                for m, c in zip(mean_list, cov_list)]

    pos_true = [np.array([x[0,0], x[0,2]]) for x in x_true]
    vel_true = [np.array([x[0,1], x[0,3]]) for x in x_true]
    ang_true = [np.array([x[0,4]]) for x in x_true]

    pos_rmse, pos_nll = rmse(pos_states, pos_true), nll(pos_states, pos_true)
    vel_rmse, vel_nll = rmse(vel_states, vel_true), nll(vel_states, vel_true)
    ang_rmse, ang_nll = rmse(ang_states, ang_true), nll(ang_states, ang_true)

    return pos_rmse, pos_nll, vel_rmse, vel_nll, ang_rmse, ang_nll


Exp_Data = namedtuple('Exp_Data', ['Transform',
                                   'Quantity',
                                   'Seed',
                                   'Iter',
                                   'Power',
                                   'Damping',
                                   'RMSE',
                                   'NLL'])


def power_sweep(config, x_true, y_meas, trans_id='UT', SEED=0, power=1, damping=1, samples=int(1e4)):
    con, system, timesteps, sys_dim, num_iter = config.con, config.system, config.timesteps, config.sys_dim, config.num_iter
    transform, meas_transform = select_transform(id=trans_id, dim=sys_dim, samples=samples)

    exp_data = Exp_Data(Transform=trans_id,
                        Quantity=None,
                        Seed=SEED,
                        Iter=0,
                        Power=power,
                        Damping=damping,
                        RMSE=0.0,
                        NLL=0.0)

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
           " WHERE Transform='{}' AND Seed = {} AND Power = {} AND Damping = {} AND Iter = 50"


def full_sweep(config, seed_range, trans_types, power_range, damp_range, override=False):
    con, system, timesteps, table = config.con, config.system, config.timesteps, config.exp_table_name
    db = con.cursor()
    total = len(list(itertools.product(seed_range, trans_types, power_range, damp_range)))
    i = 1
    for SEED in seed_range:
        np.random.seed(seed=SEED)
        data = system.simulate(timesteps)
        insert_dynamics_data(db, config.dyn_table_name, data, int(SEED))
        X, y = zip(*data)
        for trans_id, power, damping in itertools.product(trans_types, power_range, damp_range):
            print(f"running {i}/{total}, SEED = {SEED}, trans = {trans_id}, power = {power}, damping = {damping}")
            query = query_str.format(table, trans_id, SEED, power, damping)
            db.execute(query)
            exits = db.fetchall()
            try:
                if override:
                    power_sweep(config, X, y, trans_id=trans_id, SEED=int(SEED), power=power, damping=damping)
                else:
                    if len(exits) == 0: # Skips sweep if result is already computed for the given settings
                        power_sweep(config, X, y, trans_id=trans_id, SEED=int(SEED), power=power, damping=damping)
            except LinAlgError:
                print('failed for seed={}, power={},'
                    ' damping={}, transform={:s}'.format(SEED, power, damping, trans_id))
                continue
            i += 1


@click.command()
@click.option('-l', '--logdir', type=str, default="../log/temp.db", help='Directory to save results')
@click.option('-s', '--seeds', type=click.INT, default=[101], multiple=True, help='Random seed for experiment (multiple allowed)')
@click.option('-t', '--trans-types', type=click.Choice(['TT', 'UT', 'MCT']), default=['TT', 'UT', 'MCT'], multiple=True, help='Transformation types (multiple allowed)')
@click.option('-i', '--num-iter', type=int, default=50, help='Number of EP iterations')
@click.option('-o', '--override/--no-override', default=False, help='Override saved results')
def main(logdir, seeds, trans_types, num_iter, override):
    con = sqlite3.connect(logdir, detect_types=sqlite3.PARSE_DECLTYPES)
    db = con.cursor()
    system = BearingsOnlyTrackingTurn()
    sys_dim = 5
    timesteps = 50
    dyn_table_name = 'BOTT_SIM'
    exp_table_name = 'BOTT_EXP'

    create_dynamics_table(db, name=dyn_table_name)
    create_experiment_table(db, table_name=exp_table_name)

    config = Config(con=con,
                    system=system,
                    timesteps=timesteps,
                    sys_dim=sys_dim,
                    num_iter=num_iter,
                    dyn_table_name=dyn_table_name,
                    exp_table_name=exp_table_name)

    power_range = np.linspace(0.1, 1.0, num=10)
    damp_range = [1.0, 0.8]

    full_sweep(config, seeds, trans_types, power_range, damp_range, override)


if __name__ == '__main__':
    main()