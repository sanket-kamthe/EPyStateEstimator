import numpy as np


def nll(state_list, true_list):
    result = []
    for state, val in zip(state_list, true_list):
        nll_ = state.nll(val)
        result.append(nll_)
    return np.mean(result)


def rmse(state_list, true_list):
    result = []
    for state, val in zip(state_list, true_list):
        se = state.squared_error(val)
        result.append(se)
    mean_squared_err = np.mean(result)
    return np.sqrt(mean_squared_err)


def node_metrics(nodes, x_true):
    state_list = [nodes.marginal for nodes in nodes]
    return rmse(state_list, x_true), nll(state_list, x_true)


