# Copyright 2017 Sanket Kamthe
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

from Utils.Metrics import node_metrics
from MomentMatching.Database import insert_experiment_data, Exp_Data



def kalman_filter(nodes):
    for node in nodes:
        node.fwd_update()
        node.meas_update()


def kalman_smoother(nodes):
    for node in reversed(nodes):
        node.back_update()


def ep_fwd_back_updates(nodes):
    for node in nodes:
        node.fwd_update()
        node.meas_update()
    for node in reversed(nodes):
        node.back_update()

def ep_update(nodes):
    for node in nodes:
        node.fwd_update()
        node.meas_update()
        node.back_update()


def ep_iterations(nodes, max_iter=100, x_true=None, conn=None, exp_data=None):
    db = conn.cursor()
    exp_data = exp_data._asdict()
    del exp_data['Nodes']
    for i in range(max_iter):
        # ep_update(nodes)
        ep_fwd_back_updates(nodes)
        if x_true is not None:
            exp_data['Iter'] += 1
            exp_data['RMSE'], exp_data['NLL'] = node_metrics(nodes, x_true=x_true)
            exp_data['Mean'] = [node.marginal.mean for node in nodes]
            exp_data['Variance'] = [node.marginal.cov for node in nodes]
            # exp_data['Nodes'] = nodes
            # data = Exp_Data._make(exp_data.values())
            insert_experiment_data(db=db, table_name='UNGM_EXP', data=exp_data)
            print('\n EP Pass {} NLL = {}, RMSE = {}'.format(i+1,
                                                             exp_data['NLL'],
                                                             exp_data['RMSE']))
            conn.commit()