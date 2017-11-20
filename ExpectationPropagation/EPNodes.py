from StateModel import GaussianState, GaussianFactor
from collections.abc import MutableSequence
import itertools


class TimeSeriesNodeForEP:
    def __init__(self, t, state_dim=1, marginal_init=None, factor_init=None):

        self.t = t
        self.state_dimension=state_dim
        if marginal_init is None:
            self.marginal = GaussianState.as_marginal(dim=state_dim)
            assert isinstance(self.marginal, GaussianState)
        else:
            self.marginal = marginal_init

        if factor_init is None:
            self.measurement_factor = GaussianState.as_factor(dim=state_dim)
            self.back_factor = GaussianState.as_factor(dim=state_dim)
            self.forward_factor = GaussianState.as_factor(dim=state_dim)
        else:
            self.measurement_factor, self.back_factor, self.forward_factor = factor_init

        self.factors = ['forward_update', 'measurement_update', 'backward_update']

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

    @staticmethod
    def validate_input(value):
        assert isinstance(value, TimeSeriesNodeForEP)

    def __str__(self):
        return f'EP Nodes with {len(self._Node)} items in list ' + str(self._Node[0])

    def filter_mode(self):
        mode_select = [1, 1, 0]
        for node in self._Node:
            yield node
            mode = itertools.compress(node.factors, mode_select)
            for factor in mode:
                print(f'In Node{node.t} {factor}')

    def smoother_mode(self):
        self.filter_mode()
        mode_select = [0, 0, 1]
        for node in reversed(self._Node):
            mode = itertools.compress(node.factors, mode_select)
            for factor in mode:
                print(f'In Node{node.t} {factor}')

    def filter_iter(self):
        previous_node, next_node = itertools.tee(self._Node)
        next(next_node, None)
        return zip(previous_node, next_node)

    def smoother_iter(self):
        node, next_node = itertools.tee(reversed(self._Node))
        next(node, None)
        return zip(node, next_node)