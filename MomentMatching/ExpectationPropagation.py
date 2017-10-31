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


import autograd.numpy as np
from MomentMatching.StateModels import GaussianState
from MomentMatching.baseMomentMatch import MomentMatching
from collections import  namedtuple
from Filters.KalmanFilter import KalmanFilterSmoother
import itertools
from collections.abc import MutableSequence
import logging
from Utils.Metrics import nll, rmse

np.set_printoptions(precision=4)


# FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
FORMAT = "[ %(funcName)10s() ] %(message)s"


logging.basicConfig(filename='Expectation_Propagation.log', level=logging.FATAL, format=FORMAT)
logger = logging.getLogger(__name__)
logger.propagate = False
# class GaussianState(object):
#     def __init__(self, mean, variance):
#         self.mean = mean
#         self.variance = variance
#         self.precision = np.linalg.solve(variance, np.eye(3, dtype=float))
# #         TODO: Add checks for the variance and mean shapes
#WIDTH
#
# def gauss_divide(numerator, denominator, n_alpha=None, damping_factor=None):
#     variance = np.linalg.inv(numerator.precision - denominator.precision)
#     scaled_mean = np.dot(numerator.precision, numerator.mean) - np.dot(denominator.precision, denominator.mean)
#     mean = np.dot(variance, scaled_mean)
#     return GaussianState(mean, variance)


class BaseNode(object):
    """
    Factory to create nodes
    """
    FACTORS = ['Fwd', 'Measurement', 'Back']

    def __init__(self, state, factors=None):
        self.state = state
        if factors is None:
            factors = self.FACTORS
        self._factors = [factor for factor in factors]

    def update(self, factor=None):
        return NotImplementedError

    # def


#
# class TimeSeriesNode:
#     def __init__()
#         self.marginal = None  # This is a Gaussian density for example
#         self.factors =[]
#         self.t = 0
#
#     def forward_update(self):

class BeginNode(BaseNode):
    def __init__(self, state, factors):
        super(BeginNode, self).__init__(state, factors)

    def ep_update(self, factor, project=None, f=None, power=None, damping=None):
        """
        Psuedo code


        :param factor:
        :return:
        """

        #  First calculate the cavity distribution q_back

        # compute projection

        #
    #def
    # def _forward_update(self, mean, variance):
        marginal = self.state
        tilted_marginal = marginal / factor   # PowerEP equation 20
        projected_marginal = project(f, tilted_marginal)  # PowerEP equation 21
        new_factor = factor * ((projected_marginal / marginal) ** damping)  # PowerEP equation 22
        new_marginal = marginal * ((projected_marginal/marginal) ** (power * damping))  # PowerEP equation 23



class TimeSeriesNodeForEP:
    def __init__(self, t, state_dim=1, marginal_init=None, factor_init=None):

        self.t = t
        self.state_dimension=state_dim
        if marginal_init is None:
            self.marginal = self.marginal_init(state_dim)
        else:
            self.marginal = marginal_init

        if factor_init is None:
            self.measurement_factor, self.back_factor, self.forward_factor = (self.initialise_factors(state_dim))
        else:
            self.measurement_factor, self.back_factor, self.forward_factor = factor_init



    # @property
    # def back_factor(self):
    #     return self._back_factor
    #
    # @back_factor.setter
    # def back_factor(self, value):
    #     assert isinstance(value, GaussianState)
    #     self._back_factor = value
    #
    # @property
    # def forward_factor(self):
    #     return self._forward_factor
    #
    # @forward_factor.setter
    # def forward_factor(self, value):
    #     assert isinstance(value, GaussianState)
    #     self._forward_factor = value
    #
    # @property
    # def measurement_factor(self):
    #     return self._measurement_factor
    #
    # @measurement_factor.setter
    # def measurement_factor(self, value):
    #     assert isinstance(value, GaussianState)
    #     self._measurement_factor = value
    #
    #
    @staticmethod
    def marginal_init(state_dim):
        mean = np.zeros((state_dim,), dtype=float)
        cov = np.inf * np.eye(state_dim, dtype=float)
        return GaussianState(mean_vec=mean, cov_matrix=cov)

    # @staticmethod
    # def factor_init(state_dim):
    #
    #     return GaussianState(mean_vec=mean, cov_matrix=cov)

    @staticmethod
    def initialise_factors(state_dim):
        mean = np.zeros((state_dim,), dtype=float)
        cov = np.inf * np.eye(state_dim, dtype=float)
        # self.measurement_factor = self.factor_init(state_dim)
        # self.back_factor = self.factor_init(state_dim)
        # self.forward_factor = self.factor_init(state_dim)
        return (GaussianState(mean_vec=mean, cov_matrix=cov),
                     GaussianState(mean_vec=mean, cov_matrix=cov),
                     GaussianState(mean_vec=mean, cov_matrix=cov))

    def copy(self):
        marginal_init = self.marginal.copy()
        factor_init = self.measurement_factor.copy(), self.back_factor.copy(), self.forward_factor.copy()
        return TimeSeriesNodeForEP(t=self.t,
                                   state_dim=self.state_dimension,
                                   marginal_init=marginal_init,
                                   factor_init=factor_init)

    def __repr__(self):
        str_rep = f'''{self.__class__}.(t={self.t}, state_dim={self.state_dimension},
    marginal_init={self.marginal}, factor_init={(self.measurement_factor, self.back_factor, self.forward_factor)})'''
        return str_rep


class EPbase:

    def __init__(self):
        self.me = 'cool'

    def forward_update(self,  node, previous_node):
        """
        forward_cavity_distribution = node_marginal / forward_factor  # q(x_t) / q_fwd(x_t)
        back_cavity_distribution = previous_node_marginal / previous_back_factor # q(x_t-1) / q_back(x_t-1)

        new_fwd_factor = moment matching with back_cavity_distribution, transition function and noise Q  #proj op

        new_node_marginal = forward_cavity_distribution * new_fwd_factor

        return new_fwd_factor
        """
        assert isinstance(node, TimeSeriesNodeForEP)
        assert isinstance(previous_node, TimeSeriesNodeForEP)

        forward_cavity_distribution = node.marginal / node.forward_factor  # q(x_t) / q_fwd(x_t)
        back_cavity_distribution = previous_node.marginal / previous_node.back_factor  # q(x_t-1) / q_back(x_t-1)

        new_node = node.copy()

        new_node.fwd_factor = self.moment_matching(self.transition, back_cavity_distribution)  # proj op

        new_node.marginal = forward_cavity_distribution * new_node.fwd_factor

        return new_node

    def measurement_update(self, node, measurement):
        """
        measurement cavity distribution = node_marginal / measurement_factor  # q(x_t) / q_up(x_t)

        new marginal = moment matching with measurement cavity distribution

        new_measurment factor = new_marginal / measurement cavity distribution

        return new_measurement_factor
        """
        assert isinstance(node, TimeSeriesNodeForEP)

        measurement_cavity_distribution = node.marginal / node.measurement_factor  # q(x_t) / q_up(x_t)

        new_node = node.copy()

        new_node.marginal = self.moment_matching(self.measurement, measurement_cavity_distribution, self.R, measurement)

        new_node.measurement_factor = new_node.marginal / measurement_cavity_distribution

        return new_node

    def backward_update(self, node, next_node):
        """
        back_cavity_distribution = node_marginal / back_factor # q(x_t) / q_back(x_t)
        forward_cavity_distribution = next_node_marginal / next_forward_factor  # q(x_t) / q_fwd(x_t)

        new_marginal = moment matching with back cavity distribution

        return new_back_factor
        """

        back_cavity_distribution = node.marginal / node.back_factor  # q(x_t) / q_back(x_t)
        forward_cavity_distribution = next_node.marginal / next_node.forward_factor  # q(x_t+1) / q_fwd(x_t+1)

        new_node = node.copy()

        new_node.marginal = self.moment_matching(self.transition, back_cavity_distribution, forward_cavity_distribution)

        new_node.measurement_factor = new_node.marginal / back_cavity_distribution

        return new_node


class EPNodes(MutableSequence):
    def __init__(self, dimension_of_state=1, N=None, marginal_init=None, factors_init=None):
        self._Node = []
        if N is not None:
            for t in range(N):
                self._Node.append(TimeSeriesNodeForEP(t=t,
                                                      state_dim=dimension_of_state,
                                                      marginal_init=marginal_init,
                                                      factor_init=factors_init)
                                  )
        # self._Node = list(itertools.repeat(init_node, N))
        self.mode_select = [1, 1, 1]

    def validate_input(self, value):
        pass

    def __len__(self):
        return len(self._Node)

    def __getitem__(self, key):
        return self._Node[key]

    def __delitem__(self, key):
        del self._Node[key]

    def __setitem__(self, key, value):
        self.validate_input(value)
        self._Node[key] = value

    def insert(self, index, value):

        self.validate_input(value)
        self._Node.insert(index, value)

    def __str__(self):
        return f'EP Nodes with {len(self._Node)} items in list ' + str(self._Node[0])

    def filter_mode(self):
        mode_select = [1, 1, 0]
        for node in self._Node:
            mode = itertools.compress(node.factors, mode_select)
            for factor in mode:
                print(f'In Node{node.t} {factor} factor')

    def smoother_mode(self):
        self.filter_mode()
        mode_select = [0, 0, 1]
        for node in reversed(self._Node):
            mode = itertools.compress(node.factors, mode_select)
            for factor in mode:
                print(f'In Node{node.t} {factor} factor')

    def filter_iter(self):
        previous_node, next_node = itertools.tee(self._Node)
        next(next_node, None)
        return zip(previous_node, next_node)

    def smoother_iter(self):
        node, next_node = itertools.tee(reversed(self._Node))
        next(node, None)
        return zip(node, next_node)


class TopEP:
    def __init__(self, system_model, moment_matching, power=1, damping=1):
        self.system_model = system_model
        self.moment_matching = moment_matching
        self.Q = self.system_model.system_noise.cov
        self.R = self.system_model.measurement_noise.cov
        self.kf = KalmanFilterSmoother(moment_matching=moment_matching,
                                       system_model=system_model)

        self.power = power
        if damping is None:
            self.damping = 1/self.power
        else:
            self.damping = damping
        # self.node = node
    # @profile

    def kalman_filter(self, Nodes, observations, fargs_list):
        prior = Nodes[0].copy()
        if prior.marginal.cov>2000:
            prior.marginal = self.system_model.init_state

            # GaussianState(mean_vec=np.array([0.1]),
            #                            cov_matrix=0.1 * np.eye(1, dtype=float))

        for node, obs, fargs in zip(Nodes, observations, fargs_list):

            # logger.debug('[obs={}][filter: t = {}]'.format(obs, fargs))
            # logger.debug('[prior:: t={} mean={} cov={}]]'.format(node.t, node.marginal.mean, node.marginal.cov))

            pred_state = self.forward_update(node=node, prev_node=prior, fargs=fargs)

            # logger.debug('[prediction::marginal t={} mean={} cov={}]]'.format(node.t,
            #                                                                   pred_state.marginal.mean,
            #                                                                   pred_state.marginal.cov))
            corrected_state = self.measurement_update(pred_state, obs, fargs)

            # logger.debug(f'[correction::marginal t={node.t} \
            #         mean={corrected_state.marginal.mean} cov={corrected_state.marginal.cov}]]')
            yield corrected_state
            prior = corrected_state.copy()


    def power_update(self, projected_marginal, factor, marginal):
        damping = self.damping
        power = self.power
        # projected_marginal = project(f, tilted_marginal)  # PowerEP equation 21
        new_factor = factor * ((projected_marginal / marginal) ** damping)  # PowerEP equation 22
        new_marginal = marginal * ((projected_marginal / marginal) ** (power * damping))  # PowerEP equation 23

        return new_factor, new_marginal

    def kalman_smoother(self, Nodes, t=None):
        reversedNodes = reversed(Nodes)
        N = len(Nodes)
        t = (N - 1) * self.system_model.dt

        result = []
        next_node = next(reversedNodes).copy()
        result.append(next_node)
        # yield next_node

        for node in reversedNodes:
            smoothed_node = self.backward_update(node=node, next_node=next_node, fargs=t)
            next_node = smoothed_node.copy()
            result.append(next_node)
            t -= self.system_model.dt

        return list(reversed(result))


    def forward_backward_iteration(self, iters, Nodes, observations,  fargs_list, x_true):

        for i in range(iters):
            filt = list(self.kalman_filter(Nodes, observations, fargs_list))
            result = self.kalman_smoother(filt)
            Nodes = result
            EP1 = [node.marginal for node in Nodes]
            # assert EP3 == EP2
            print('\n EP Pass {} NLL = {}, RMSE = {}'.format(i+1, nll(EP1, x_true), rmse(EP1, x_true)))

        return result

    # @profile
    def forward_update(self, node, prev_node, fargs):

        forward_cavity = node.marginal / node.forward_factor
        back_cavity = prev_node.marginal / prev_node.back_factor

        # logger.debug('[forward_cavity:: t={} mean={} cov={}]]'.format(node.t,
        #                                                               forward_cavity.mean,
        #                                                               forward_cavity.cov))
        # logger.debug('[back_cavity:: t={} mean={} cov={}]]'.format(node.t,
        #                                                            back_cavity.mean,
        #                                                            back_cavity.cov))

        result_node = node.copy()

        state = self.kf.predict(prior_state=back_cavity, t=fargs)
        # logger.debug('time {} mean={}, cov={}'.format(node.t, node.marginal.mean, node.marginal.cov))
        # print(state.cov)
        if (state.cov > 0) and (state.cov < 1e8):
            result_node.forward_factor = state.copy()
            result_node.marginal = forward_cavity * result_node.forward_factor



        return result_node

    def measurement_update(self, node, obs, fargs):
        measurement_cavity = node.marginal / node.measurement_factor
        # logger.debug('[measurement_cavity:: t={} mean={} cov={}]]'.format(node.t,
        #                                                                   measurement_cavity.mean,
        #                                                                   measurement_cavity.cov))


            # logger.debug('Negative Cavity')
            # logger.debug('[node_marginal:: t={} mean={} cov={}]]'.format(node.t,
            #                                                                   node.marginal.mean,
            #                                                                   node.marginal.cov))
            # logger.debug('[measurement_factor:: t={} mean={} cov={}]]'.format(node.t,
            #                                                                   node.measurement_factor.mean,
            #
        if np.linalg.det(measurement_cavity.cov) < 0:
            return node.copy()
        #                                                              node.measurement_factor.cov))

        result = node.copy()

        # if measurement_cavity.cov < 0:
            # print(node.t)
            # print(node)
            # print(measurement_cavity)

        state = self.kf.correct(state=measurement_cavity, meas=obs)
        # moment_matching(func=self.system_model.measurement,
        #                              state=measurement_cavity,
        #                                        Q=self.R,
        #                                        match_with=obs,
        #                                        fargs=None)



        if (state.cov > 0) and (state.cov < 100):
            # result.marginal = state.copy()
            #
            # result.measurement_factor = result.marginal / measurement_cavity

            result.measurement_factor, result.marginal = \
                self.power_update(projected_marginal=state,
                                  factor=node.measurement_factor,
                                  marginal=node.marginal)
            # logger.debug('[measurement_factor:: t={} mean={} cov={}]]'.format(node.t,
            #                                                            result.measurement_factor.mean,
            #                                                            result.measurement_factor.cov))
        return result

    def backward_update(self, node, next_node, fargs):
        back_cavity = node.marginal / node.back_factor
        forward_cavity = next_node.marginal / next_node.forward_factor

        # logger.debug('[forward_cavity:: t={} mean={} cov={}]]'.format(node.t,
        #                                                               forward_cavity.mean,
        #                                                               forward_cavity.cov))
        # logger.debug('[back_cavity:: t={} mean={} cov={}]]'.format(node.t,
        #                                                            back_cavity.mean,
        #                                                            back_cavity.cov))

        result_node = node.copy()

        state = self.kf.smooth(state=back_cavity, next_state=next_node.marginal, t=fargs)
        # print('in node {} and t is {} '.format(node.t, fargs))
        # moment_matching(nonlinear_func=self.system_model.transition,
        #                                             distribution=back_cavity,
        #                                             Q=self.Q,
        #                                             match_with=forward_cavity,
        #                                             fargs=fargs)

        if (state.cov>0) and (state.cov<100):
            # print(state.cov)

            result_node.marginal = state.copy()
            result_node.back_factor = result_node.marginal / back_cavity
            # result_node.back_factor, result_node.marginal = \
            # self.power_update(projected_marginal=state,
            #                   factor=node.back_factor,
            #                   marginal=node.marginal)
        # result_node.marginal = forward_cavity * result_node.forward_factor

        return result_node

if __name__ == '__main__':

    from MomentMatching.StateModels import GaussianState
    from MomentMatching.newMomentMatch import UnscentedTransform, TaylorTransform, MonteCarloTransform
    from MomentMatching.auto_grad import logpdf
    from MomentMatching.ExpectationPropagation import TimeSeriesNodeForEP, EPbase, EPNodes
    from MomentMatching.TimeSeriesModel import UniformNonlinearGrowthModel, SimpleSinTest

    np.random.seed(seed=101)

    ungm = SimpleSinTest()
    ungm = UniformNonlinearGrowthModel()
    data = ungm.simulate(N=5)

    Q = ungm.system_noise.cov
    print(Q)
    x_true, x_noisy, y_true, y_noisy = list(zip(*data))

    print('#'*10 + '  x_true  '+ '#'*10)
    print(x_true)
    print('#' * 30)

    print('#'*10 + '  x_noisy  '+ '#'*10)
    print(x_noisy)
    print('#' * 30)
    # TT = TaylorTransform(dimension_of_state=1)
    TT = UnscentedTransform(n=1)
    All_nodes = EPNodes(dimension_of_state=1, N=16)

    TestEP = TopEP(system_model=ungm, moment_matching=TT)
    # prior =
    # All_nodes[0] = prior
    prior = All_nodes[0].copy()
    prior.marginal = GaussianState(mean_vec=np.array([0.1]),
                          cov_matrix=1 * np.eye(1, dtype=float))


    fwd = TestEP.forward_update(All_nodes[0], prior, 0.0)
    meas = TestEP.measurement_update(fwd, y_noisy[0], fargs=0.0)

    print(f'RMSE: {meas.marginal.rmse(x_true[0])}')
    print(f'NLL: {meas.marginal.nll(x_true[0])}')
    # print(meas.marginal.mean)
    # print(x_true[0])

    fwd2 = TestEP.forward_update(All_nodes[1], meas, 1.0)
    print('prediction')
    print(f'RMSE: {fwd2.marginal.rmse(x_true[1])}')
    print(f'NLL: {fwd2.marginal.nll(x_true[1])}')

    meas2 = TestEP.measurement_update(fwd2, y_noisy[1], fargs=0.0)
    print('correction')
    print(f'RMSE: {meas2.marginal.rmse(x_true[1])}')
    print(f'NLL: {meas2.marginal.nll(x_true[1])}')
    print(meas2.marginal.rmse(x_true[1]))
    print(x_true[1])

    sms = TestEP.backward_update(node=meas, next_node=meas2, fargs=0.0)
    print('Smoothing')
    print(f'RMSE: {sms.marginal.rmse(x_true[0])}')
    print(f'NLL: {sms.marginal.nll(x_true[0])}')



