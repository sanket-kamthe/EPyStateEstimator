from StateModel import GaussianState, GaussianFactor



class TimeSeriesNodeForEP:
    def __init__(self, t, state_dim=1, marginal_init=None, factor_init=None):

        self.t = t
        self.state_dimension=state_dim
        if marginal_init is None:
            self.marginal = GaussianFactor(dim=state_dim)
        else:
            self.marginal = marginal_init

        if factor_init is None:
            self.measurement_factor = GaussianFactor(dim=state_dim)
            self.back_factor = GaussianFactor(dim=state_dim)
            self.forward_factor = GaussianFactor(dim=state_dim)
        else:
            self.measurement_factor, self.back_factor, self.forward_factor = factor_init


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