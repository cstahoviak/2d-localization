"""
This sandbox file implements the 1D Dynamical System Sequential Importance
Sampling (SIS) algorithm from Ristic's "Beyond the Kalman Filter: Particle
Filters for Tracking Applications."
"""
import time
from typing import Tuple, Callable

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import style

from dynamical_system_1d import transition_fcn_1d, likelihood_fcn_1d,\
    importance_fcn_1d, prior_fcn_1d, truth_fcn_1d, measurements_fcn_1d

# Shorthand for type hints
nparr = np.ndarray
Generator = np.random._generator.Generator


# style.use('fivethirtyeight')
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})


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
        particles: Particles containing state estimates at the current time
            step, k+1.
        weights: The particle weights at the current time step, k+1.
    """
    # print('SIS - BEGIN')
    for idx, (particle, weight) in enumerate(zip(particles, weights)):
        # Sample x_{k+1}^i from the importance density
        particles[idx] = importance_fcn(particle).rvs()

        # Evaluate un-normalized IS weights
        weights[idx] = weight * likelihood_fcn(particle, measurement) \
            * transition_fcn(particle).pdf(particle) \
            / importance_fcn(particle).pdf(particle)

    # Normalize the weights
    weights = weights / np.sum(weights)
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

    # print('SIS - END')
    return particles, weights


class SISAnimation:
    def __init__(self, ax, steps: int, n_particles: int, prior: nparr,
                 weights: nparr, measurements: nparr):
        """
        Construtor.

        Args:
            ax:
            steps:
            n_particles:
            prior:
            weights:
            measurements:
        """
        # print('CONSTRUCTOR - BEGIN')
        self.line, = ax.plot([], [], 'k-')
        self.ax = ax

        # Store the number of time steps and number of particles
        self.steps = steps
        self.n_particles = n_particles

        # Store particle states and weights for plotting
        self.particles = np.zeros((self.steps, self.n_particles))
        self.particles[0] = prior
        self.weights = np.zeros((self.steps, self.n_particles))
        self.weights[0] = weights

        # Store the measurements
        self.measurements = measurements

        # Store the current index
        self.idx = 0
        # print('CONSTRUCTOR - END')

    def __call__(self, frame):
        """

        Args:
            frame:
            state:
            weights:
        """
        # print('__CALL__ - BEGIN')
        # Update the posterior via the SIS algorithm
        self.update()

        # Count the total number of particles above a threshold weight
        thres = 1 / self.n_particles / 2
        count = sum(self.weights[self.idx] >= thres)

        # Ensure that weights are being normalized
        weight_sum = np.sum(self.weights[self.idx])

        print(f'Iteration: {self.idx}\tSignificant Particles: {count}\tWeight '
              f'Sum: {weight_sum:.2f}')

        # Update the plot
        self.line.set_data(self.particles[self.idx], self.weights[self.idx])
        # print('__CALL__ - END')
        return self.line,

    def update(self):
        """
        Update the particle state estimates and weights via the SIS algorithm.
        """
        # print('UPDATE - BEGIN')
        # Update the particle states and weights
        self.particles[self.idx + 1], self.weights[self.idx + 1] = \
            sis_filter(particles=self.particles[self.idx],
                       weights=self.weights[self.idx],
                       measurement=measurements[self.idx],
                       importance_fcn=importance_fcn_1d,
                       transition_fcn=transition_fcn_1d,
                       likelihood_fcn=likelihood_fcn_1d)

        # Update the internal index
        self.idx += 1
        # print('UPDATE - END')


if __name__ == "__main__":
    """
    Run the Sequential Importance Sampling (SIS) Demo.
    """
    # Define some additional parameters
    n_particles = 300   # Total number of particles
    steps = 10          # Number of time steps to iterate over
    wait_time = 0.5     # Interval between posterior updates

    # Init particles (states) and weights
    _particles = np.zeros((steps, n_particles))
    _weights = np.zeros((steps, n_particles))

    # Initialize the particles based on a uniform prior, U(1,4)
    _particles[0] = prior_fcn_1d(n_particles)
    _weights[0] = (1 / n_particles) * np.ones_like(n_particles)

    # Use these lines if using SIS Animation
    # prior = state_rng.uniform(low=1, high=4, size=n_particles)
    # weights = (1 / n_particles) * np.ones_like(n_particles)

    # Create the "truth" data, of shape (steps+1,)
    truth = truth_fcn_1d(steps)

    # Create a set of simulated measurements of shape (steps,)
    measurements = measurements_fcn_1d(truth)

    # Create a figure for plotting via SISAnimation
    # fig, ax = plt.subplots()
    # plt.title("SIS Posterior at time, k = 0")
    # plt.xlabel("State, x")
    # plt.ylabel("Estimated Posterior PDF")

    # Set the plot animation
    # sis = SISAnimation(ax, steps, n_particles, prior, weights, measurements)
    # anim = FuncAnimation(fig, sis, frames=steps, interval=500, blit=True)
    # plt.show()

    # Store the effective sample size (Neff)
    Neff = np.zeros(n_particles)
    Neff[0] = n_particles

    # Filtering via SIS
    for k in range(steps-1):
        # Update particle states and weights
        _particles[k+1], _weights[k+1] = \
            sis_filter(particles=_particles[k],
                       weights=_weights[k],
                       measurement=measurements[k+1])

        # Compute the effective Sample Size (Neff)
        Neff[k+1] = int(1 / np.sum(_weights[k+1] ** 2))
        print(f'Iteration: {k+1}\tNeff: {Neff[k+1]}')

        # Update the plot...
        # time.sleep(wait_time)

    # Create a figure for plotting several time steps together
    _, ax = plt.subplots(nrows=2, ncols=3, figsize=(14, 9))

    time_steps = [1, 2, 3, 4, 5, 6]
    indices = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for step, ax_idx in zip(time_steps, indices):
        # Plot data
        ax[ax_idx].scatter(_particles[step], _weights[step], marker=".")
        ax[ax_idx].vlines(_particles[step], 0, _weights[step], lw=1)

        # Annotate data
        ax[ax_idx].set_title(f"SIS Posterior, time k={step}, Neff={Neff[step]}")
        ax[ax_idx].set_xlabel(f"1D State, x")
        ax[ax_idx].set_xlim(left=0.0, right=60)

    plt.show()

