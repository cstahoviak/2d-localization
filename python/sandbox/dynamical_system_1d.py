"""
The sandbox file stores the following models for the 1D Dynamical System
example:

1. State transition density, p(x_{k+1} | x_k).
2. Measurement likelihood density, p(y_{k+1} | x_{k+1}).
3. The SIS Importance density, q(x_{k+1} | x_k, y_{k+1}).
4. The prior distribution, p(x_0).

Also included are methods for generating truth and measurement data.
"""
from typing import Callable

import numpy as np
from scipy.stats import uniform

# Shorthand for type hints
nparr = np.ndarray

# Create the random number generator
RNG_STATE = np.random.default_rng(12345)
RNG_MEAS = np.random.default_rng(12345)


def transition_fcn_1d(state: float):
    """
    State transition density for the 1D Dynamical System example.

    p(x_{k+1} | x_k) = U(x_k, x_k + 5)

    Note that Scipy's uniform(loc, scale) defines a uniform distribution on
    [loc, loc + scale].

    Args:
        state: The previous 1D state, x_k.

    Returns:
        The distribution representing the likelihood of transitioning to the
        current state, x_{k+1} from the previous state, x_k. This distribution
        is typically evaluated at x_{k+1}.
    """
    # Should return a value of 1/5 = 0.2
    return uniform(loc=state, scale=5)


def likelihood_fcn_1d(state: float, measurement: float) -> float:
    """
    The data likelihood function for the 1D Dynamical System example.

    p(y_{k+1} | x_{k+1}) = U(x_{k+1}, (x_{k+1} + 1)^(1.2))

    Note that Scipy's uniform(loc, scale) defines a uniform distribution on
    [loc, loc + scale].

    Args:
        state: The current 1D state, x_{k+1}.
        measurement: The current measurement, y_{k+1}.

    Returns:
        likelihood: The likelihood of observing the current measurement given
        the current state.
    """
    if state <= measurement <= (state + 1) ** 1.2:
        likelihood = uniform(loc=state, scale=(state+1)**1.2 - state).pdf(state)
    else:
        likelihood = 0.0

    return likelihood


def importance_fcn_1d(state: float):
    """
    The importance density function. In the case of the 1D Dynamical System,
    the transition density is used as the importance density.

    q(x_{k+1} | x_k, y_{k+1}) = p(x_{k+1} | x_k) = U(x_k, x_k + 5)

    Args:
        state: state: The previous 1D state, x_k.

    Returns:
        The predicted current state, x_{k+1}, sampled from the importance
        density, q(x_{k+1} | x_k, y_{k+1}).
    """
    return transition_fcn_1d(state)


def prior_fcn_1d(dim: int = 1) -> float:
    """
    The prior density for the 1D Dynamical System example, p(x_0).

    Args:
        dim: The dimension of the state vector.

    Returns:
        A sample from the prior density of shape (dim,)
    """
    return RNG_STATE.uniform(low=1, high=4, size=dim)


def truth_fcn_1d(steps: int, prior: Callable = prior_fcn_1d) -> nparr:
    """
    Create truth data for the 1D Dynamical System.

    Args:
        steps: The number of discrete time steps.
        prior:

    Returns:
        truth: The "true" 1D state.
    """
    # Initialize the truth data
    truth = np.zeros(steps+1)

    # The first value will be drawn from the prior, U[1,4]
    truth[0] = prior()

    # Create "noise" added via the state transition model, x_{k+1} = x_k + eta_k
    noise = RNG_MEAS.uniform(low=0, high=5, size=steps)

    for idx in range(steps):
        truth[idx+1] = truth[idx] + noise[idx]

    return truth


def measurements_fcn_1d(truth: nparr, noisy: bool = True) -> nparr:
    """
    Creates a set of measurements for the 1D Dynamical System.

    Args:
        truth: The true 1D state vector.
        noisy: If True, Gaussian noise is added to the measurements.

    Returns:
        measurements: The simulated (noisy) measurements of shape (steps,) There
            will be one less measurement than time steps since the first
            estimated state will be at time step 1, where time step zero is
            represented by the prior. This is represented in the measurement
            vector by the use of a 'NaN' value at index zero.
    """
    n_measurements = len(truth) - 1

    # Initialize the measurements with a NaN value at index zero
    measurements = np.zeros(n_measurements + 1)
    measurements[0] = np.nan

    if noisy:
        # Add Gaussian noise to the truth data
        noise = RNG_MEAS.normal(loc=0, scale=1.0, size=n_measurements)
    else:
        noise = np.zeros(n_measurements)

    # Create the measurement vector
    measurements[1:] = truth[1:] + noise

    return measurements
