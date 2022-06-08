# Nonlinear State Estimation with Expectation Propagation

State estimation in nonlinear systems is a difficult problem due to the non-Gaussianity of posterior state distributions. For linear systems, an exact solution is achieved by running the Kalman filtering/smoothing algorithm. However for nonlinear systems, typically one relies on crude Gaussian approximations by linearising the system (e.g. the Extended Kalman filter/smoother) or to use a Monte-Carlo method (particle filter/smoother) that sample the non-Gaussian posterior, which comes at the cost of more compute. Here, we propose an intermediate nonlinear state estimation method based on _(approximate) Expectation Propagation (EP)_, which allows for an iterative refinement of the Gaussian approximation based on message passing.
It turns out that this method generalises _any_ standard Gaussian smoother such as the Extended Kalman smoother and the Unscented Kalman smoother, in the sense that these well-known smoothers are special cases of (approximate) EP. Moreover, they have the same computational cost up to the number of iterations, making it a practical solution to obtaining improved state estimates.

TODO:
- Fix the thing with `x_true`, `x_noisy`, etc (done)
- Fix all files that use these (done)
- Clean up experiments folder and figure out which databases to upload
- Elaborate readme file

## The state estimation problem

The aim of state estimation is to provide an estimate of a time-evolving latent state (given by a probability distribution) based on noisy observations of the dynamical system. This can be formulated mathematically using the state-space model:

$$
x_t = f(x_{t-1}) + w_t, \quad t = 1, \ldots, T, 
$$

$$
y_t = h(x_t) + v_t, \quad t = 1, \ldots, T.
$$

Here, $x_t$ is the latent state that we wish to estimate, with initial state distribution $x_0 \sim p(x_0)$ and transition function $f$ (the dynamical system that describe the evolution of the latent state), $y_t$ is the observation of $x_t$, obtained via an observation operator $h$, and $w_t, v_t$ are the model error and measurement error respectively, typically chosen to be Gaussians.

We distinguish between two types of solutions. In _filtering_, a solution to the state estimation problem is given by the probability distribution $p(x_t | y_1, \ldots, y_t)$ of the state $x_t$ conditioned on observations up to the current time $t$. On the other hand, _smoothing_ yields the solution $p(x_t | y_1, \ldots, y_T)$, i.e. the distribution of state $x_t$ conditioned on _all available observations_ up to time $T$.

## Expectation propagation

Expectation propagation is an approximate inference method suited to estimate the marginal distribution of the nodes in a Bayesian network.

## Requirements
Our implementation of approximate EP primarily uses `numpy` and `scipy`. To perform the Taylor linearisation, we also use automatic differentiation with the `autograd` package. We have kept the number of required packages minimal. You can install the necessary packages by running
```
pip install -r requirements.txt
```

## Usage
- Set up dynamical system + measurement function
- Build EP nodes
- Apply forward-backward sweep
- See `Notebooks/Demo.ipynb`

## Examples
### Uniform Nonlinear Growth Model
<img src="https://github.com/mpd37/pyStateEstimator/blob/finalise_code/Notebooks/figs/ungm_animation.gif" width="70%" height="70%"/>

### Bearings Only Tracking of a Turning Target
<img src="https://github.com/mpd37/pyStateEstimator/blob/finalise_code/Notebooks/figs/bott_animation.gif" width="50%" height="50%"/>

### Lorenz 96 Model
<img src="https://github.com/mpd37/pyStateEstimator/blob/finalise_code/Notebooks/figs/L96_animation.gif" width="50%" height="50%"/>

## References

