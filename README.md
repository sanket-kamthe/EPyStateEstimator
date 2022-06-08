# Nonlinear State Estimation with Expectation Propagation

State estimation in nonlinear systems is difficult due to the non-Gaussianity of posterior state distributions. For linear systems, an exact solution is attained by running the Kalman filtering/smoother. However for nonlinear systems, one typically relies on either crude Gaussian approximations by linearising the system (e.g. the Extended Kalman filter/smoother) or to use a Monte-Carlo method (particle filter/smoother) that sample the non-Gaussian posterior, but at the cost of more compute.

We propose an intermediate nonlinear state estimation method based on _(approximate) Expectation Propagation (EP)_, which allows for an iterative refinement of the Gaussian approximation based on message passing.
It turns out this method generalises _any_ standard Gaussian smoother such as the Extended Kalman smoother and the Unscented Kalman smoother, in the sense that these well-known smoothers are special cases of (approximate) EP. Moreover, they have the same computational cost up to the number of iterations, making it a practical solution to improving state estimates.

TODO:
- Fix the thing with `x_true`, `x_noisy`, etc (done)
- Fix all files that use these (done)
- Clean up experiments folder and figure out which databases to upload
- Elaborate readme file

## State estimation

The aim of state estimation is to provide an estimate of a time-evolving latent state (given by a probability distribution) based on noisy observations of the dynamical system. This can be formulated mathematically using the state-space model:

$$
x_t = f_{t-1}(x_{t-1}) + w_t, \quad t = 1, \ldots, T, 
$$

$$
y_t = h_t(x_t) + v_t, \quad t = 1, \ldots, T.
$$

Here, $x_t$ is the latent state that we wish to estimate, with initial state distribution $x_0 \sim p(x_0)$ and transition function $f$ (the dynamical system that describe the evolution of the latent state), $y_t$ is the observation of $x_t$, obtained via an observation operator $h$, and $w_t, v_t$ are the model error and measurement error respectively, typically chosen to be Gaussians.

We distinguish between two types of solutions. In _filtering_, a solution to the state estimation problem is given by the probability distribution $p(x_t | y_1, \ldots, y_t)$ of the state $x_t$ conditioned on observations up to the current time $t$. On the other hand, _smoothing_ yields the solution $p(x_t | y_1, \ldots, y_T)$, i.e. the distribution of state $x_t$ conditioned on _all available observations_ up to time $T$.

## Expectation propagation

Expectation propagation [1] provides a way to estimate the marginal distribution of the nodes in a Bayesian network.

## Requirements
Our implementation of approximate EP primarily uses `numpy` and `scipy`. To perform the Taylor linearisation, we also use automatic differentiation with the `autograd` package. We have kept the number of required packages minimal. You can install the necessary packages by running:
```
pip install -r requirements.txt
```

## Basic Usage

State estimation with EP can be done by following the steps below.
1. Set up a state-space model with the `DynamicSystemModel` class in `Systems.DynamicSystem`.
2. Set up the nodes in the Markov chain with `ExpectationPropagation.build_nodes`.
3. Add the state-space model and observations to the nodes using `ExpectationPropagation.node_system`. This completes the information required to form the factor graph for a dynamical system.
4. Equip the nodes with an `Estimator` object using `ExpectationPropagation.node_estimator`. This object contains information about the state estimation procedure, such as linearisation method (e.g. Taylor transformation), and values for $\alpha$ and $\gamma$.
5. Run a single EP sweep with `ExpectationPropagation.Iterations.ep_fwd_back_updates`.
6. Iterate step 5 until a stopping criterion is met.

A jupyter notebook demonstrating this basic procedure is available at `Notebooks/Demo.ipynb`.

## Examples
### Uniform Nonlinear Growth Model
The uniform nonlinear growth model (UNGM) is a well-known 1D nonlinear benchmark for state estimation, given as follows.

$$
f_t(x) = \frac{x}{2} + \frac{25 x}{1 + x^2} + 8 \cos(1.2t), \qquad h_t(x) = \frac{x^2}{20}.
$$

The video below shows the result of EP iterations with the following setup.
- Linearisation method: unscented transform
- Power: $\alpha = 0.9$
- Damping: $\gamma = 0.4$
- Number of iterations: 50

<p align="center">
  <img src="https://github.com/mpd37/pyStateEstimator/blob/finalise_code/Notebooks/figs/ungm_animation.gif" width="80%" height="80%"/>
</p>


### Bearings Only Tracking of a Turning Target
Next, we demonstrate our algorithm on the problem of bearings-only tracking of a turning target. This is a five dimensional nonlinear system in the variables $(x_1, \dot{x}_1, x_2, \dot{x}_2, \omega)$.
We use the following setup for EP iterations:
- Linearisation method: Taylor transform
- Power: $\alpha = 1.0$
- Damping: $\gamma = 0.6$
- Number of iterations: 10

<p align="center">
  <img src="https://github.com/mpd37/pyStateEstimator/blob/finalise_code/Notebooks/figs/bott_animation.gif" width="60%" height="60%"/>
</p>

The video above only displays the spatial components $(x_1, x_2)$. The green dots represent the predictive mean and the ellipses represent the spatial covariance.

### Lorenz 96 Model
The Lorenz 96 model is another well-known benchmark for nonlinear state-estimation. This is governed by the following system of ODEs:

$$
\frac{\mathrm{d} x_i}{\mathrm{d} t} = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F,
$$

for $i = 1, \ldots, d$. We consider a system with $F = 8$ and $d = 200$. The ODE is discretised using the fourth-order Runge-Kutta scheme.
For the observation, we use the quadratic function $h(x) = x^2$ applied to each component.
The following configurations are used for EP iteration:

- Linearisation method: unscented transform
- Power: $\alpha = 1.0$
- Damping: $\gamma = 1.0$
- Number of iterations: 5

The video below displays the Hovmöller representation of a single simulation of the L96 model, the absolute error of the prediction, and componentwise negative log likelihood loss.

<p align="center">
  <img src="https://github.com/mpd37/pyStateEstimator/blob/finalise_code/Notebooks/figs/L96_animation.gif" width="50%" height="50%"/>
</p>

## References
[1] Thomas P Minka. Expectation Propagation for Approximate Bayesian Inference. In Proceedings of the
Conference on Uncertainty in Artificial Intelligence, 2001.
