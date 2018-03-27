
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
# import os
# import sys
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
from MomentMatching.newMomentMatch import MomentMatching, UnscentedTransform, TaylorTransform, MonteCarloTransform
from ExpectationPropagation import EPNodes
from MomentMatching.ExpectationPropagation import TopEP
from Filters.KalmanFilter import KalmanFilterSmoother, PowerKalmanFilterSmoother
from Utils.Metrics import nll, rmse
from Utils.Plot_Helper import plot_gaussian, plot_gaussian_node
from Systems import BearingsOnlyTracking, UniformNonlinearGrowthModel
import logging
# from ExpectationPropagation import EPNodes

logging.basicConfig(level='critical')
plt.ion()

SEED = 11
np.random.seed(seed=SEED)

N = 5
sys_dim = 1
system = UniformNonlinearGrowthModel()
# system = BearingsOnlyTracking()
sys_dim = system.system_dim
meas_dim = system.measurement_dim
data = system.simulate(N)
x_true, x_noisy, y_true, y_noisy = zip(*data)


power = 0.5
damping = 0.5
EP_iters = 50

transform = UnscentedTransform(n=sys_dim,  beta=2,  alpha=1, kappa=3)
meas_transform = UnscentedTransform(n=sys_dim, beta=2,  alpha=1, kappa=2)
# transform = TaylorTransform()
# meas_transform = TaylorTransform()

# transform = MonteCarloTransform(dimension_of_state=sys_dim)
# meas_transform = MonteCarloTransform(dimension_of_state=sys_dim)
def _power_sweep(power, damping):
    transform = UnscentedTransform(n=sys_dim, beta=0, alpha=1, kappa=2)
    meas_transform = UnscentedTransform(n=sys_dim, beta=0, alpha=1, kappa=2)

    Nodes = EPNodes(dimension_of_state=sys_dim, N=N)
    EP = TopEP(system_model=system,
               moment_matching=transform,
               meas_transform=meas_transform,
               power=power,
               damping=damping)

    EPNodesList = EP.forward_backward_iteration(10, Nodes, y_noisy, list(range(0, N)), x_true)

    Node = [node.marginal for node in EPNodesList[-1]]
    return nll(Node, x_true), rmse(Node, x_true)

power_range = np.linspace(0.5, 1.0, num=2)
damp_range = np.linspace(0.5, 1.0, num=2)
results = []
# for power, damping in itertools.product(power_range, damp_range):
#     ans = _power_sweep(power, damping)
#     results.append(ans)

# power_data = _power_sweep(0.59, 0.6)
# NLL, RMSE = power_data
Nodes = EPNodes(dimension_of_state=sys_dim, N=N)
EP = TopEP(system_model=system,
           moment_matching=transform,
           meas_transform=meas_transform,
           power=power,
           damping=damping)

kf = PowerKalmanFilterSmoother(moment_matching=transform,
                               system_model=system,
                               power=power,
                               meas_moment_matching=meas_transform)

EPFilt = EP.kalman_filter(Nodes, y_noisy, list(range(0, N)))

x_filtered = list(EPFilt)
EP3 = [node.marginal for node in x_filtered]
print('\n Filter {} NLL = {}, RMSE = {}'.format(1, nll(EP3, x_true), rmse(EP3, x_true)))

x_filt_mean = [x.marginal.mean for x in x_filtered]
#
result = kf.kalman_filter(y_noisy, prior_state=system.init_state)
#
smoothed = kf.kalman_smoother(result)

EPSmthd = EP.kalman_smoother(x_filtered)

EP3 = [node.marginal for node in EPSmthd]

print('\n Smoother {} NLL = {}, RMSE = {}'.format(1, nll(EP3, x_true), rmse(EP3, x_true)))
# EP2Filt = list(EP.kalman_filter(EPSmthd, y_noisy, list(range(0, N))))
# EP2Smthd = EP.kalman_smoother(EP2Filt)

plt.plot(x_true, 'r--', label='X_true')
# plt.plot([x.mean for x in result],  'b-', label='Kalman Filter')
plt.plot(x_filt_mean, 'b-', label='EP as Kalman Filter')
# plt.plot([x.mean for x in smoothed],  'm-', label='Kalman Smoother')
plt.plot([x.marginal.mean for x in EPSmthd], 'g-', label='EP as Kalman Smoother')

# plt.plot([x.marginal.mean for x in EP2Filt], 'b--', label='EP 2nd pass as Kalman Filter')
# plt.plot([x.marginal.mean for x in EP2Smthd], 'g--', label='EP 2nd pass as Kalman Smoother')
plt.legend()
# plt.show()

EP1 = [node.marginal for node in EPSmthd]
# EP2 = [node.marginal for node in EP2Smthd]

plt.figure()
plt.plot(x_true, 'r--', label='X_true')
plot_gaussian(EP1, label='EP Pass 1')
# plot_gaussian(EP2, label='EP Pass 2')
plt.legend()
plt.show()
EPNodesList = EP.forward_backward_iteration(EP_iters, Nodes, y_noisy, list(range(0, N)), x_true)
for i, Nodes in enumerate(EPNodesList):
    EP3 = [node.marginal for node in Nodes]
    print('\n EP Pass {} NLL = {}, RMSE = {}'.format(i + 1, nll(EP3, x_true), rmse(EP3, x_true)))
# EP1 = [node.marginal for node in Nodes for Nodes in EPNodesList]
            # assert EP3 == EP2
# print('\n EP Pass {} NLL = {}, RMSE = {}'.format(i+1, nll(EP1, x_true), rmse(EP1, x_true)))
EP3 = [node.marginal for node in EPNodesList[-1]]
plot_gaussian(EP3, label='EP Pass {}'.format(EP_iters))
plt.plot(x_true, 'r--', label='X_true')
plt.legend()
plt.show()



# assert EP3 == EP2
# print('\n EP Pass 1 NLL = {}, RMSE = {}'.format(nll(EP1, x_true),
#                                                        rmse(EP1, x_true)))
#
# print('\n EP Pass 2 NLL = {}, RMSE ={}'.format(nll(EP2, x_true),
#                                                       rmse(EP2, x_true)))
#
# print('\n EP Pass 6 NLL = {}, RMSE ={}'.format(nll(EP3, x_true),
#                                                       rmse(EP3, x_true)))