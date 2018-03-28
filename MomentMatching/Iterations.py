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


def ep_iterations(nodes, max_iter=100, fwd_back=True, x_true=None  ):

    if fwd_back:
        ep_fwd_back_updates(nodes)
        max_iter -= 1
    if x_true is not None:
        rmse, nll = node_metrics(nodes, x_true=x_true)
        print('\n EP Pass {} NLL = {}, RMSE = {}'.format(1, nll, rmse))
        # print()
    for i in range(max_iter):
        # ep_update(nodes)
        ep_fwd_back_updates(nodes)
        if x_true is not None:
            rmse, nll = node_metrics(nodes, x_true=x_true)
            print('\n EP Pass {} NLL = {}, RMSE = {}'.format(i + 2, nll, rmse))