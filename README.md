# Nonlinear State Estimation with Expectation Propagation

TODO:
- Fix the thing with `x_true`, `x_noisy`, etc
- Fix all files that use these
- Clean up experiments folder and figure out which databases to upload
- Elaborate readme file

## Requirements
The number of required packages is minimal. Install necessary packages using
```
pip install -r requirements.txt
```

## Examples
### Uniform Nonlinear Growth Model
<img src="https://github.com/mpd37/pyStateEstimator/blob/finalise_code/Notebooks/figs/ungm_animation.gif" width="70%" height="70%"/>

### Bearings Only Tracking of a Turning Target
<img src="https://github.com/mpd37/pyStateEstimator/blob/finalise_code/Notebooks/figs/bott_animation.gif" width="50%" height="50%"/>

### Lorenz 96 Model
<img src="https://github.com/mpd37/pyStateEstimator/blob/finalise_code/Notebooks/figs/L96_animation.gif" width="50%" height="50%"/>

## Usage
- Set up dynamical system + measurement function
- Build EP nodes
- Apply forward-backward sweep
- See `Notebooks/Demo.ipynb`
