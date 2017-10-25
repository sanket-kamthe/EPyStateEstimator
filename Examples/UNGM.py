
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-poster')

from MomentMatching.newMomentMatch import MomentMatching, UnscentedTransform, TaylorTransform, MonteCarloTransform
from MomentMatching.TimeSeriesModel import TimeSeriesModel, UniformNonlinearGrowthModel
from MomentMatching.StateModels import GaussianState
from MomentMatching.ExpectationPropagation import EPNodes, TopEP
from Filters.KalmanFilter import KalmanFilterSmoother
from Utils.Metrics import nll, rmse

SEED = 130

np.random.seed(seed=SEED)


N = 50
system = UniformNonlinearGrowthModel()
data = system.simulate(N)
x_true, x_noisy, y_true, y_noisy = zip(*data)


transform = UnscentedTransform(n=1)
# transform = TaylorTransform()
Nodes = EPNodes(dimension_of_state=1, N=N)
EP = TopEP(system_model=system, moment_matching=transform)
kf = KalmanFilterSmoother(moment_matching=transform,
                          system_model=system)

EPFilt = EP.kalman_filter(Nodes, y_noisy, list(range(0, N)))

x_filtered = list(EPFilt)
x_filt_mean = [x.marginal.mean for x in x_filtered]

result = kf.kalman_filter(y_noisy, prior_state=system.init_state)

smoothed = kf.kalman_smoother(result)

EPSmthd = EP.kalman_smoother(x_filtered)

EP2Filt = list(EP.kalman_filter(EPSmthd, y_noisy, list(range(0, N))))
EP2Smthd = EP.kalman_smoother(EP2Filt)

plt.plot(x_true, 'r--', label='X_true')
# plt.plot([x.mean for x in result],  'b-', label='Kalman Filter')
plt.plot(x_filt_mean, 'b-', label='EP as Kalman Filter')
# plt.plot([x.mean for x in smoothed],  'm-', label='Kalman Smoother')
plt.plot([x.marginal.mean for x in EPSmthd], 'g-', label='EP as Kalman Smoother')

plt.plot([x.marginal.mean for x in EP2Filt], 'b--', label='EP 2nd pass as Kalman Filter')
plt.plot([x.marginal.mean for x in EP2Smthd], 'g--', label='EP 2nd pass as Kalman Smoother')
plt.legend()
plt.show()

EP1 = [node.marginal for node in EPSmthd]
EP2 = [node.marginal for node in EP2Smthd]

EP3Nodes = EP.forward_backward_iteration(30, Nodes, y_noisy, list(range(0, N)), x_true)
EP3 = [node.marginal for node in EP3Nodes]
# assert EP3 == EP2
# print('\n EP Pass 1 NLL = {}, RMSE = {}'.format(nll(EP1, x_true),
#                                                        rmse(EP1, x_true)))
#
# print('\n EP Pass 2 NLL = {}, RMSE ={}'.format(nll(EP2, x_true),
#                                                       rmse(EP2, x_true)))
#
# print('\n EP Pass 6 NLL = {}, RMSE ={}'.format(nll(EP3, x_true),
#                                                       rmse(EP3, x_true)))