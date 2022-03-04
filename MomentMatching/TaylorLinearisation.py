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

import numpy as np
from MomentMatching.MomentMatch import MappingTransform
from functools import partial
from autograd import jacobian
from StateModel import Gaussian
# from MomentMatching.auto_grad import logpdf
from autograd.scipy.stats import multivariate_normal
# from autograd.scipy.stats import multivariate_normal.logpdf as logpdf

# logpdf = multivariate_normal.logpdf
# from scipy.stats. import log
# from scipy.stats.

EPS = 1e-4


class TaylorTransform(MappingTransform):
    def __init__(self, dim=1, eps=EPS):
        super().__init__(approximation_method='Taylor 1st Order',
                         dimension_of_state=dim,
                         eps=eps)

    @staticmethod
    def numerical_jacobian(f, x, eps=EPS):
        z = f(x)
        #     jacobian  = np.zeros((m,n), dtype=float)
        jacobian = []
        x_list = x.tolist()
        for i, data in enumerate(x_list):
            x1 = x.tolist()
            x1[i] = np.array(data) + eps
            x1 = np.array(x1)
            jacobian.append((f(x1) - z) / eps)

        return np.array(jacobian).T

    def _transform(self, func, state):
        # (self, nonlinear_func, distribution, fargs=None, y_observation=None):
        assert isinstance(state, Gaussian)
        # frozen_func = partial(func, t=t, u=u, *args, **kwargs)
        J_t = self.numerical_jacobian(func, state.mean)
        # J_t = jacobian(func)(state.mean)
        # J_t = np.reshape(J_t, [-1, state.cov.shape[1]])
        mean = func(state.mean)
        mean = np.squeeze(mean)
        cov = J_t @ state.cov @ J_t.T
        cross_cov = state.cov @ J_t.T
        return mean, cov, cross_cov

    def _data_likelihood(self, nonlinear_func, distribution, data=None):
        meanz = nonlinear_func(distribution.mean)  # z = f(x)
        linear_c = jacobian(nonlinear_func)(distribution.mean)  # linear factor A
        covz = linear_c @ distribution.cov @ linear_c.T  # predictive covariance sz = var(f(x)) = A * Sigma * A^T
        logZi = multivariate_normal.logpdf(data, meanz, covz)

        return logZi

    def project(self, nonlinear_func, distribution, data=None):

        logZi = self._data_likelihood(nonlinear_func, distribution, data)
        dlogZidMz = jacobian(self._data_likelihood, argnum=1)(nonlinear_func, distribution, data)
        dlogZidSz = jacobian(self._data_likelihood, argnum=2)(nonlinear_func, distribution, data)

        return logZi, dlogZidMz, dlogZidSz
