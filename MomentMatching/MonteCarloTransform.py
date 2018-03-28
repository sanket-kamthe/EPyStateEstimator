
import numpy as np
from .MomentMatch import MappingTransform


class MonteCarloTransform(MappingTransform):
    def __init__(self, dim=1, number_of_samples=None):

        if number_of_samples is None:
            number_of_samples = int(1e4)
        super().__init__(approximation_method='Monte Carlo Sampling',
                         dimension_of_state=dim,
                         number_of_samples=number_of_samples)

    def _transform(self, func, state):
        # (self, nonlinear_func, distribution, fargs=None):

        # assert isinstance(state, GaussianState)
        # frozen_func = partial(func, t=t, u=u, *args, **kwargs)

        samples = state.sample(self.number_of_samples)

        # Xi = []
        # for x in samples:
        #     x = np.asanyarray(x)
        #     result = np.asanyarray(func(x))
        #     Xi.append(result)

        # Y = np.asarray(Xi)

        # propagated_samples = np.asarray(Xi)
        propagated_samples = func(samples)
        mean = np.mean(propagated_samples, axis=0)
        cov, cross_cov = \
            self.sample_covariance(propagated_samples.T,
                                   samples.T)
        # cov = np.cov(propagated_samples.T, bias=True)
        # cross_cov = np.cov(samples.T, propagated_samples.T)

        return mean, cov, cross_cov

    @staticmethod
    def sample_covariance(y_samples, x_samples):
        x_dim = x_samples.shape[0]
        y_dim = y_samples.shape[0]
        total_cov = np.cov(x_samples, y_samples, bias=True)
        cov = total_cov[x_dim:, x_dim:]
        cross_cov = total_cov[0:x_dim, x_dim:]

        return cov, cross_cov