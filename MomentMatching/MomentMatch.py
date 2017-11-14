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

from abc import ABCMeta, abstractmethod
import numpy as np
# import functools
from StateModel import GaussianState


class MomentMatching(metaclass=ABCMeta):
    _params = {}

    @abstractmethod
    def project(self, distribution):
        """
        Returns the approximate projection of distribution through a nonlinear function.

        ..math \\int f(x) q (x) dx, where q(x) is the distribution and f(x) is the non-linear function
        The result is exact when f(x) is a linear function.

        Parameters
        -----------------------------------------------------
        :param distribution: distribution of the variable x
        :type distribution: GaussianState object

        Returns
        ---------------------------------------------

        :return: approximate distribution in the same family as input distribution
        :rtype: object of the type 'distribution'
        """
        pass

    @abstractmethod
    def g(self, x, t=None, u=None, *args, **kwargs):
        """
        A curried function for moment matching
        Returns
        -------
        :return : function through which moment is propagated

        """
        pass

    @property
    @abstractmethod
    def noise(self):
        """
        Defines the parameters of noise, e.g., mean & cov for Gaussian

        Returns
        -------
        parameters
        """
        pass


class ProjectTransition(MomentMatching):

    @abstractmethod
    def project(self, distribution, next_state=None):
        pass


class ProjectMeasurement(MomentMatching):

    @abstractmethod
    def project(self, distribution, meas):
        pass


class MappingTransform:

    def __init__(self, approximation_method=None, **kwargs):
        self.params = {}  # Dictionary containing parameters for the moment matching approximation

        self.method = approximation_method
        self.params.update(kwargs)
        # print(self.params)
        for key in self.params:
            self.__setattr__(key, self.params[key])

    def _transform(self, func, state, t=None, u=None, *args, **kwargs):
        """
        Returns the gaussian approximation the integral

        \int f(x) q (x) dx, where q(x) is the distribution and f(x) is the non-linear function
        The result is exact when f(x) is a linear function.
        :param nonlinear_func:
        :param distribution: object of type GaussianState for example
        :return: distribution of y =  \int f(x) q (x) dx in the form of a tuple
        mean (y), variance (yyT), cross_covariance(xyT)
        """
        return NotImplementedError

    def __call__(self, func, state, t=None, u=None, *args, **kwargs):
        return self._transform(func=func, state=state, t=t, u=u, *args, **kwargs)


class KalmanFilterTransitionMapping(ProjectTransition, MappingTransform):
    def __init__(self, f, noise, approximation_method=None, **kwargs):
        self._f = f
        self._noise = noise
        self.transform = super().__init__(approximation_method=approximation_method,
                                          **kwargs)
    
    def project(self, distribution, next_state=None, t=None, u=None, *args, **kwargs):
        if next_state is None:
            return self._predict(distribution, t=t, u=u, *args, **kwargs)
        else:
            return self._smooth(distribution, next_state,  t=None, u=None, *args, **kwargs)
    
    def g(self, x, t=None, u=None, *args, **kwargs):
        return self._f(x, t=t, u=u, *args, **kwargs)
    
    def _predict(self, distribution,  t=None, u=None, *args, **kwargs):
        xx_mean, xx_cov, _ = self.transform(self.g,
                                            distribution,
                                            t=t, u=u,
                                            *args, **kwargs)
        xx_cov += self.noise()
        return GaussianState(xx_mean, xx_cov)
    
    def _smooth(self, state, next_state, t=None, u=None, *args, **kwargs):

        xx_mean, xx_cov, xx_cross_cov = \
            self.transform(self.g,
                           state,
                           t=t, u=u,
                           *args, **kwargs)

        xx_cov += self.noise()

        smoother_gain = np.linalg.solve(xx_cov, xx_cross_cov)
        mean = state.mean + np.dot(smoother_gain, (next_state.mean - xx_mean))
        cov = state.cov + smoother_gain @ (next_state.cov - xx_cov) @ smoother_gain.T

        return GaussianState(mean, cov)
    
    def noise(self):
        return self._noise.cov


class KalmanFilterMeasurementMapping(ProjectMeasurement, MappingTransform):
    def __init__(self, h, noise, approximation_method=None, **kwargs):
        self._f = h
        self._noise = noise
        self.transform = super().__init__(approximation_method=approximation_method,
                                          **kwargs)

    def project(self, distribution, meas, t=None, u=None, *args, **kwargs):
        return self._correct(state=distribution, meas=meas,
                             t=None, u=None, *args, **kwargs)

    def g(self, x, t=None, u=None, *args, **kwargs):
        return self._f(x, t=t, u=u, *args, **kwargs)

    def _correct(self, state, meas, t=None, u=None, *args, **kwargs):

        z_mean, z_cov, xz_cross_cov = \
            self.transform(self.g, state, t=t, u=u, *args, **kwargs)

        z_cov += self.noise()

        kalman_gain = np.linalg.solve(z_cov, xz_cross_cov)
        mean = state.mean + np.dot(kalman_gain, (meas - z_mean))  # equation 15  in Marc's ACC paper
        cov = state.cov - np.dot(kalman_gain, np.transpose(xz_cross_cov))

        return GaussianState(mean, cov)
    
    def noise(self):
        return self._noise.cov