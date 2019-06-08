import pytest
import numpy as np
from StateModel import Gaussian
# from MomentMatching.newMomentMatch import UnscentedTransform, MonteCarloTransform, TaylorTransform
from MomentMatching import UnscentedTransform, TaylorTransform, MonteCarloTransform

dim = 3
A = np.random.randn(dim, dim)
A = A @ A.T
A = np.eye(dim)
B = np.zeros(shape=(dim, ), dtype=float )


def linear(x, A=A):
    # dim = x.shape[0]
    # A = np.random.randn(dim, dim)
    x = np.atleast_2d(x)
    y = np.einsum('ij,kj -> ki', A, x)
    # y = A @ x
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
    cov = np.eye(dim) * 0.25
    return Gaussian(mean, cov)


def test_transforms(transform, func):
    state = distribution(dim)
    transform = transform(state.dim)
    pred_mean, pred_cov, pred_cross_cov = transform(func, state)
    assert pytest.approx(pred_mean, rel=1e-1, abs=1e-1) == state.mean
    assert pytest.approx(pred_cov, rel=1e-1, abs=1e-1) == state.cov

