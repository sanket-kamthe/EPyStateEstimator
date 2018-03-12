
import numpy as np
from ExpectationPropagation import DynamicSystemEP, DynamicSystemPowerEP
from Systems import UniformNonlinearGrowthModel, BearingsOnlyTracking
from MomentMatching import KalmanFilterMapping, PowerKalmanFilterMapping
from ExpectationPropagation import EPNodes
from MomentMatching import UnscentedTransform
from Utils.Metrics import nll, rmse

SEED = 100

np.random.seed(seed=SEED)

N = 50
state_dim = 1
system = UniformNonlinearGrowthModel()
# system = BearingsOnlyTracking()
data = system.simulate(N)
x_true, x_noisy, y_true, y_noisy = zip(*data)


power = 0.5
damping = 0.8


Nodes = EPNodes(dimension_of_state=state_dim, N=N)

transition_transform = UnscentedTransform(dim=state_dim,
                                          beta=0,
                                          alpha=1,
                                          kappa=2)

meas_transform = UnscentedTransform(dim=state_dim,
                                    beta=0,
                                    alpha=1,
                                    kappa=2)

ep_projection = KalmanFilterMapping(transition=transition_transform,
                                    measure=meas_transform)

power_ep_projection =\
    PowerKalmanFilterMapping(transition=transition_transform,
                             measure=meas_transform,
                             power=power)

EP_obj = DynamicSystemEP(system=system,
                         ep_project=ep_projection,
                         measurements=y_noisy)

powerEP_obj = DynamicSystemPowerEP(system=system,
                                   ep_project=power_ep_projection,
                                   measurements=y_noisy,
                                   power=power,
                                   damping=damping)


EP_obj.kalman_filter(Nodes)
EP3 = [node.marginal for node in Nodes]
print('\n Filter {} NLL = {}, RMSE = {}'.format(1, nll(EP3, x_true), rmse(EP3, x_true)))

powerEP_obj.kalman_filter(Nodes)
EP3 = [node.marginal for node in Nodes]
print('\n Filter {} NLL = {}, RMSE = {}'.format(1, nll(EP3, x_true), rmse(EP3, x_true)))



EP_obj.kalman_smoother(Nodes)

EP3 = [node.marginal for node in Nodes]
print('\n Filter {} NLL = {}, RMSE = {}'.format(1, nll(EP3, x_true), rmse(EP3, x_true)))


Nodes = EPNodes(dimension_of_state=state_dim, N=N)
powerEP_obj.ep_update(Nodes, max_iter=10)

EP3 = [node.marginal for node in Nodes]
print('\n Filter {} NLL = {}, RMSE = {}'.format(1, nll(EP3, x_true), rmse(EP3, x_true)))
