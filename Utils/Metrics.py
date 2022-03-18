import numpy as np


def nll(state_list, true_list):
    result = []
    for state, val in zip(state_list, true_list):
        ll = state.nll(val)
        result.append(ll)
    return sum(result)/len(result)


def rmse(state_list, true_list):
    result = []
    for state, val in zip(state_list, true_list):
        mse = state.rmse(val)
        result.append(mse)

    mean_squared_err = sum(result)/len(result)
    return np.sqrt(mean_squared_err)


def node_metrics(nodes, x_true):

    state_list = [nodes.marginal for nodes in nodes]
    assert len(x_true) == len(nodes), "The length of true values is not same as number of nodes" \
                                      " len(nodes){0:d} != len(x_true){1:d}".format(len(x_true), len(state_list))

    mean_sqrd_err = [state.rmse(val) for state, val in zip(state_list, x_true)]
    neg_log_lik = [state.nll(val) for state, val in zip(state_list, x_true)]

    return np.sqrt(sum(mean_sqrd_err)/len(mean_sqrd_err)),  sum(neg_log_lik)




def _squared_error(x, y):
    return x
