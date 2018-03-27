import pytest
import numpy as np
from StateModel import Gaussian
# from MomentMatching.newMomentMatch import UnscentedTransform, MonteCarloTransform, TaylorTransform
from MomentMatching import UnscentedTransform, TaylorTransform, MonteCarloTransform

dim = 3
A = np.random.randn(dim, dim)
A = A @ A.T
B = np.zeros(shape=(dim, ), dtype=float )

def linear(x):
    # dim = x.shape[0]
    # A = np.random.randn(dim, dim)
    # y = np.einsum('ij,kj -> k', A, x)
    y = A @ x
    return y


def sinusoidal(x, t=0, u=0):
    y = np.sin(0.5 * x)
    return 2 * y


def softplus(x):
    y = np.log(1 + np.exp(x)) - np.log(2)
    return 2 * y


@pytest.fixture(scope="module",
                params=[TaylorTransform, MonteCarloTransform, UnscentedTransform])
def transform(request):
    transform = request.param
    return transform


@pytest.fixture(scope="module",
                params=[linear, sinusoidal, softplus], ids=['linear', 'sinusoidal', 'softplus'])
def func(request):
    func = request.param
    return func


def distribution(dim):
    mean = np.random.randn(dim) * 0.0
    cov = np.eye(dim) * 0.1
    return Gaussian(mean, cov)

def test_transforms(transform, func):
    state = distribution(dim)
    transform = transform(state.dim)
    pred_mean, pred_cov, pred_cross_cov = transform(func, state)
    assert pytest.approx(pred_mean, rel=1e-1, abs=1e-1) == state.mean
    assert pytest.approx(pred_cov, rel=1e-1, abs=1e-1) == state.cov

