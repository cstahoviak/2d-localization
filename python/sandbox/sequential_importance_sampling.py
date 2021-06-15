"""
This sandbox file implements the 1D Dynamical System Sequential Importance
Sampling (SIS) algorithm from Ristic's "Beyond the Kalman Filter: Particle
Filters for Tracking Applications."
"""
from typing import Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt

from dynamical_system_1d import transition_fcn_1d, likelihood_fcn_1d,\
    importance_fcn_1d, prior_fcn_1d, truth_fcn_1d, measurements_fcn_1d
from animate_posterior import AnimatePosterior

# Shorthand for type hints
nparr = np.ndarray


def sis_filter(particles: nparr, weights: nparr, measurement: float,
        importance_fcn: Callable = importance_fcn_1d,
        transition_fcn: Callable = transition_fcn_1d,
        likelihood_fcn: Callable = likelihood_fcn_1d) -> Tuple[nparr, nparr]:
    """
    The Sequential Importance Sampling (SIS) algorithm. Taken from Ristic
    Table 3.1.

    Args:
        particles: Particles containing state estimates at the previous time
            step, k.
        weights: The particle weights at the previous time step, k.
        measurement: The current measurement, y_{k+1}.
        importance_fcn: The importance density function from which to sample
            the current state given the previous state and (optionally a
            measurement).
        transition_fcn: The state transition density to be evaluated in
            computing the updated particle weights.
        likelihood_fcn: The measurement likelihood density to be evaluated in
            computing the updated particle weights.

    Returns:
        new_particles: Particles containing state estimates at the current time
            step, k+1.
        new_weights: The particle weights at the current time step, k+1.
    """
    new_particles = np.zeros_like(particles)
    new_weights = np.zeros_like(weights)

    for idx, (particle, weight) in enumerate(zip(particles, weights)):
        # Sample x_{k+1}^i from the importance density
        new_particles[idx] = importance_fcn(particle).rvs()

        # Evaluate un-normalized IS weights
        new_weights[idx] = weight * likelihood_fcn(particle, measurement) \
            * transition_fcn(particle).pdf(particle) \
            / importance_fcn(particle).pdf(particle)

    # Normalize the weights
    new_weights = new_weights / np.sum(new_weights)
    # try:
    #     weights = weights / np.sum(weights)
    # except RuntimeWarning:
    #     # Count the total number of particles above a threshold weight
    #     thres = 1 / n_particles * 0.001
    #     count = sum(weights[k] >= thres)
    #
    #     raise ZeroDivisionError(f'Division by zero error. Only {count} '
    #                             f'particles still contributing to the'
    #                             f'posterior estimate.')

    return new_particles, new_weights


if __name__ == "__main__":
    """
    Run the Sequential Importance Sampling (SIS) Demo.
    """
    # Define some additional parameters
    n_particles = 300   # Total number of particles
    steps = 10          # Number of time steps to iterate over
    wait_time = 0.5     # Interval between posterior updates

    # Create a set of simulated measurements of shape (steps,)
    measurements = measurements_fcn_1d(truth_fcn_1d(steps))

    # Animate the SIS posterior distribution
    anim = AnimatePosterior(name="SIS", algo=sis_filter, steps=steps,
                            n_particles=n_particles, prior=prior_fcn_1d,
                            measurements=measurements)
    plt.show()

    """
    Visualize a set of pre-defined time steps
    """
    if np.sum(anim.particles[-1]) != 0:
        # Only create this figure below if filter ran to completion
        _particles = anim.particles
        _weights = anim.weights
        Neff = 1 / np.sum(_weights ** 2, axis=1)

        # Create a figure for plotting several time steps together
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(14, 9))

        time_steps = [1, 2, 3, 4, 5, 6]
        indices = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        for step, ax_idx in zip(time_steps, indices):
            # Plot data
            ax[ax_idx].scatter(_particles[step], _weights[step], marker=".")
            ax[ax_idx].vlines(_particles[step], 0, _weights[step], lw=1)

            # Annotate data
            ax[ax_idx].set_title(f"SIS Posterior, time k={step},"
                                 f"Neff={int(Neff[step])}")
            ax[ax_idx].set_xlabel(f"1D State, x")
            ax[ax_idx].set_xlim(left=0.0, right=60)

        plt.show()
