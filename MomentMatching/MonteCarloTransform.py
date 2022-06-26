
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

        samples = state.sample(self.number_of_samples)
        propagated_samples = func(samples)
        mean = np.mean(propagated_samples, axis=-2)
        cov, cross_cov = \
            self.sample_covariance(propagated_samples.T,
                                   samples.T)

        return mean, cov, cross_cov

    @staticmethod
    def sample_covariance(y_samples, x_samples):
        x_dim = x_samples.shape[0]
        y_dim = y_samples.shape[0]
        total_cov = np.cov(x_samples, y_samples, bias=False)
        cov = total_cov[x_dim:, x_dim:]
        cross_cov = total_cov[0:x_dim, x_dim:]

        return cov, cross_cov