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
from Utils.Metrics import rmse, nll
import sqlite3
from StateModel import Gaussian
from Systems import BearingsOnlyTrackingTurn
from Filters.KalmanFilter import IEKS
from Utils.Metrics import rmse, nll


def create_experiment_table(db, table_name='BOTT_EXP'):
    """
    Table to save results of iterated EKS
    """
    schema = """ CREATE TABLE IF NOT EXISTS {:s} 
                (
                Quantity TEXT,
                Seed INT,
                Iter REAL,
                RMSE REAL, 
                NLL REAL,
                UNIQUE (Quantity, Seed, Iter)
                )""".format(table_name) 
    
    experiment_data_string = "INSERT OR IGNORE INTO {}" \
                        " (Quantity, Seed, Iter, RMSE, NLL)" \
                        "VALUES (?, ?, ?, ?, ?)"

    db.execute(schema)


def componentwise_metrics(pred_list, x_true):
    """
    Custom metric to compute the losses in the position, velocity
    and angular velocity components separately.
    """
    mean_list = [state.mean for state in pred_list]
    cov_list = [state.cov for state in pred_list]

    pos_states = [Gaussian(np.array([m[0], m[2]]), np.array([[c[0,0], c[0,2]], [c[2,0], c[2,2]]])) 
                for m, c in zip(mean_list, cov_list)]
    vel_states = [Gaussian(np.array([m[1], m[3]]), np.array([[c[1,1], c[1,3]], [c[3,1], c[3,3]]])) 
                for m, c in zip(mean_list, cov_list)]
    ang_states = [Gaussian(np.array([m[4]]), np.array([[c[4,4]]])) 
                for m, c in zip(mean_list, cov_list)]

    pos_true = [np.array([x[0], x[2]]) for x in x_true]
    vel_true = [np.array([x[1], x[3]]) for x in x_true]
    ang_true = [np.array([x[4]]) for x in x_true]

    pos_rmse, pos_nll = rmse(pos_states, pos_true), nll(pos_states, pos_true)
    vel_rmse, vel_nll = rmse(vel_states, vel_true), nll(vel_states, vel_true)
    ang_rmse, ang_nll = rmse(ang_states, ang_true), nll(ang_states, ang_true)

    return pos_rmse, pos_nll, vel_rmse, vel_nll, ang_rmse, ang_nll


experiment_data_string = "INSERT OR IGNORE INTO {}" \
                        " (Quantity, Seed, Iter, RMSE, NLL)" \
                        "VALUES (?, ?, ?, ?, ?)"


if __name__ == '__main__':
    logdir = '../log/IEKS_bott.db'
    table = 'BOTT_EXP'
    query_str = experiment_data_string.format(table)
    con = sqlite3.connect(logdir, detect_types=sqlite3.PARSE_DECLTYPES)
    db = con.cursor()

    create_experiment_table(db, table_name=table)

    N = 50

    SEEDS = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001]

    num_iter = 50

    for i, seed in enumerate(SEEDS):
        print(f"Experiment {i+1}/{len(SEEDS)}")
        dim = 5
        system = BearingsOnlyTrackingTurn()

        np.random.seed(seed)
        data = system.simulate(N)
        x, y = zip(*data)
        x = np.array(x).squeeze()

        # Apply iterated Kalman smoother
        iterated_smoother = IEKS(system, dim)
        iks_results = iterated_smoother(y, num_iter=num_iter) 

        pos_rmse, pos_nll, vel_rmse, vel_nll, ang_rmse, ang_nll = componentwise_metrics(iks_results, x)

        # record data
        losses = {'position': [pos_rmse, pos_nll], 'velocity': [vel_rmse, vel_nll], 'angle': [ang_rmse, ang_nll]}
        for key, values in losses.items():
            rmse_, nll_ = values
            db.execute(query_str, (key, int(seed), num_iter, rmse_, nll_))
        con.commit()