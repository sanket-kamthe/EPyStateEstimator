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
from Systems import UniformNonlinearGrowthModel
from Filters.KalmanFilter import IEKS
from Utils.Metrics import rmse, nll


def create_experiment_table(db, table_name):
    """
    Table to save results of iterated EKS
    """
    schema = """ CREATE TABLE IF NOT EXISTS {:s} 
                (
                Seed INT,
                Iter REAL,
                RMSE REAL, 
                NLL REAL,
                UNIQUE (Seed, Iter)
                )""".format(table_name)  

    experiment_data_string = "INSERT OR IGNORE INTO {}" \
                        " (Seed, Iter, RMSE, NLL)" \
                        "VALUES (?, ?, ?, ?)"

    db.execute(schema)


experiment_data_string = "INSERT OR IGNORE INTO {}" \
                        " (Seed, Iter, RMSE, NLL)" \
                        "VALUES (?, ?, ?, ?)"


if __name__ == '__main__':
    logdir = '../log/IEKS_ungm.db'
    table = 'UNGM_EXP'
    query_str = experiment_data_string.format(table)
    con = sqlite3.connect(logdir, detect_types=sqlite3.PARSE_DECLTYPES)
    db = con.cursor()

    create_experiment_table(db, table_name=table)

    N = 100

    SEEDS = [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001]

    num_iter = 50

    for i, seed in enumerate(SEEDS):
        print(f"Experiment {i+1}/{len(SEEDS)}")
        dim = 1
        system = UniformNonlinearGrowthModel()

        np.random.seed(seed)
        data = system.simulate(N)
        x, y = zip(*data)
        x = np.array(x).squeeze()

        # Apply iterated Kalman smoother
        iterated_smoother = IEKS(system, dim)
        iks_results = iterated_smoother(y, num_iter=num_iter) 

        rmse_, nll_ = rmse(iks_results, x), nll(iks_results, x)

        # record data
        db.execute(query_str, (int(seed), num_iter, rmse_, nll_))
        con.commit()