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
from .MomentMatch import MappingTransform
from autograd import jacobian
from StateModel import Gaussian
from autograd.scipy.stats import multivariate_normal

EPS = 1e-4


class TaylorTransform(MappingTransform):
    def __init__(self, dim=1, eps=EPS):
        super().__init__(approximation_method='Taylor 1st Order',
                         dimension_of_state=dim,
                         eps=eps)

    @staticmethod
    def numerical_jacobian(f, x, eps=EPS):
        z = f(x)
        jacobian = []
        x_list = x.tolist()
        for i, data in enumerate(x_list):
            x1 = x.tolist()
            x1[i] = np.array(data) + eps
            x1 = np.array(x1)
            
            x2 = x.tolist()
            x2[i] = np.array(data) - eps
            x2 = np.array(x2)

            #jacobian.append((f(x1) - z) / eps) # forward difference
            jacobian.append( (f(x1) - f(x2)) / (2*eps)) # central difference

        return np.array(jacobian).T

    def _transform(self, func, state):
        assert isinstance(state, Gaussian)
        # J_t = self.numerical_jacobian(func, state.mean)
        J_t = jacobian(func)(state.mean)
        if np.ndim(J_t) > 2:
          J_t = np.squeeze(J_t)
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
