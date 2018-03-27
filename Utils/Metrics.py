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

    mean_sqaured_err = sum(result)/len(result)
    return np.sqrt(mean_sqaured_err)


