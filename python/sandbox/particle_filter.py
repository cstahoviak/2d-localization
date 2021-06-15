"""
This sandbox file explores the implementation of a simple particle filter. The
algorithms implemented are taken from Ristic's "Beyond the Kalman Filter:
Particle Filters for Tracking Applications." and include:

1. Sequential Importance Sampling (SIS)
2. Systematic Resampling
3. "Generic" Particle Filter
4. SIR Particle Filter
5. ...
"""
from dataclasses import dataclass
from typing import Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt

from dynamical_system_1d import transition_fcn_1d, likelihood_fcn_1d,\
    importance_fcn_1d, prior_fcn_1d, truth_fcn_1d, measurements_fcn_1d
from sequential_importance_sampling import sis_filter
from animate_posterior import AnimatePosterior

# Shorthand for type hints
nparr = np.ndarray

RNG_RESAMPLE = np.random.default_rng(117)


def resample(weights, seed: int):
    """
    Resampling algorithm taken from the Scipy Cookbook:
    https://scipy-cookbook.readthedocs.io/items/ParticleFilter.html

    Args:
        weights: Current particle weights, an nparr of size (nParticles,).
        seed: Seed for the random number generator.

    Returns:
        indices: What is this?
    """
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    rng = np.random.default_rng(seed)
    u0, j = rng.uniform(low=0, high=1/n), 0
    for u in [(u0+i)/n for i in range(n)]:
        while u > C[j]:
            j+=1
        indices.append(j-1)
    return indices


def systematic_resample(particles: nparr, weights: nparr) -> \
    Tuple[nparr, nparr]:
    """
    Resamples a degenerate set of particles. Taken from Ristic Table 3.2.
    
    Args:
        particles: The particles (state estimates) at time step k, x_k
        weights: The particle weights at time step k, w_k

    Returns:
        particles: The particles (state estimates) at time step k+1, x_{k+1}
        weights: The particle weights at time step k+1, w_{k+1}
        parents: The parent indices of each resampled particle
    """
    n_particles = len(particles)

    # Initialize the Cumulative Sum of Weights (CSW)
    csw = np.cumsum(weights)

    # Start at the bottom of the CSW
    i = 0

    # Draw a starting point
    u = np.zeros(n_particles)
    u0 = RNG_RESAMPLE.uniform(low=0, high=1/n_particles)

    # Resample particles
    new_particles = np.zeros_like(particles)
    for j in range(n_particles):
        # Move along the CSW
        u[j] = u0 + (j / n_particles)
        while u[j] > csw[i]:
            i += 1

        # Assign sample
        new_particles[j] = particles[i]

        # Assign parent

    # All particles will have the same weight after resampling, 1/n_particles
    new_weights = (1 / n_particles) * np.ones_like(weights)

    return new_particles, new_weights


def generic_particle_filter(particles: nparr, weights: nparr,
                            measurement: float, neff_thres: int = 25) -> \
        Tuple[nparr, nparr]:
    """
    The "Generic" Particle Filter. Taken from Ristic Table 3.3.

    Args:
        particles: Particles containing state estimates at the previous time
            step, k.
        weights: The particle weights at the previous time step, k.
        measurement: The current measurement, y_{k+1}.
        neff_thres: Minimum number of effective particles before resampling
            occurs.

    Returns:
        new_particles: Particles containing state estimates at the current time
            step, k+1.
        new_weights: The particle weights at the current time step, k+!.
    """
    # Filtering vis SIS
    new_particles, new_weights = sis_filter(particles, weights, measurement)

    # Note that any expected value calculations (MMSE, MAP, etc.) should occur
    # before resampling.

    # Calculate N_eff using Arulampalam Eqn. (51)
    Neff = 1 / np.sum(new_weights ** 2)

    if Neff < neff_thres:
        # Resample particles via "systematic resampling"
        print('RESAMPLING PARTICLES')
        new_particles, new_weights = systematic_resample(new_particles,
                                                         new_weights)

    return new_particles, new_weights


def sir_filter(particles: nparr, measurement: float,
               importance_fcn: Callable = importance_fcn_1d,
               likelihood_fcn: Callable = likelihood_fcn_1d) -> nparr:
    """
    The Sequential Importance Resampling (SIR) particle filter, also known as
    the "Bootstrap" particle filter. Taken from Ristic Table 3.4. The SIR filter
    makes 2 important choices:

    1. The importance density, q(x_{k+1} | x_k, y_{k+1}) is chosen to be the
        state transition density, p(x_{k+1} | x_k). This choice of importance
        density is considered "suboptimal" as it does not exploit knowledge of
        the current measurement, y_{k+1}.
    2. Resampling occurs at every time step, and therefore particle weights
        need not be carried forward from one time step to the next (weights
        after resampling as the same, w_{k+1}^i = 1/Nparticles.

    It can be seen that this particular choice of importance density simplifies
    the calculation of the importance weight such that the updated importance
    weight, w_{k+1}^i, is now simply proportional to the likelihood.

    Args:
        particles: Particles containing state estimates at the previous time
            step, k.
        measurement: The current measurement, y_{k+1}.

    Returns:
        new_particles: Particles containing state estimates at the current time
            step, k+1.
    """
    # Initialize the weights
    n_particles = len(particles)
    weights = np.zeros(n_particles)

    # Filter via the SIR-PF
    new_particles = np.zeros_like(particles)
    for idx, particle in enumerate(particles):
        # Sample x_{k+1}^i from the transition density
        new_particles[idx] = importance_fcn(particle).rvs()

        # Calculate the importance weight
        weights[idx] = likelihood_fcn(particle, measurement)

    # Normalize the weights
    weights = weights / np.sum(weights)

    # Resample particles (note that particle weights are not stored or carried
    # from one time step to the next).
    new_particles, _ = systematic_resample(new_particles, weights)

    return new_particles

@dataclass
class Particle:
    """
    Not currently used.
    """
    _id: int = 0
    dim: int = 2
    state: nparr = np.zeros((1, dim))
    weight: float = 0.0
    parent: int = None

    @property
    def ID(self):
        """
        Returns the particle ID.
        """
        return self._id


if __name__ == '__main__':
    """
    Run the Generic Particle Filter Demo.
    """
    # Define some additional parameters
    n_particles = 300   # Total number of particles
    steps = 20          # Number of time steps to iterate over
    wait_time = 0.5     # Interval between posterior updates

    # Create a set of simulated measurements of shape (steps,)
    measurements = measurements_fcn_1d(truth_fcn_1d(steps))

    # Animate the Generic PF posterior distribution
    anim = AnimatePosterior(name="PF", algo=generic_particle_filter,
                            steps=steps, n_particles=n_particles,
                            prior=prior_fcn_1d, measurements=measurements)
    plt.show()

    """
    Visualize a set of pre-defined time steps
    """
    pf_particles = anim.particles
    pf_weights = anim.weights

    # Init particles states and weights (SIS Filter, Generic PF and SIR-PF)
    sis_particles = np.zeros((steps, n_particles))
    sis_weights = np.zeros((steps, n_particles))
    sir_particles = np.zeros((steps, n_particles))

    # Use the same prior and weights for the SIS filter and the SIR-PF
    sis_particles[0] = pf_particles[0]
    sis_weights[0] = pf_weights[0]
    sir_particles[0] = pf_particles[0]

    # Store the effective sample size (Neff) for the SIS filter
    Neff = np.zeros(n_particles)
    Neff[0] = n_particles

    # Filtering via SIS, the Generic Particle Filter and the SIR-PF
    for k in range(steps-1):
        # Filtering via SIS
        sis_particles[k+1], sis_weights[k+1] = \
            sis_filter(particles=sis_particles[k],
                       weights=sis_weights[k],
                       measurement=measurements[k+1])

        # # Filtering via the "generic" particle filter
        # pf_particles[k+1], pf_weights[k+1] = \
        #     generic_particle_filter(particles=pf_particles[k],
        #                             weights=pf_weights[k],
        #                             measurement=measurements[k+1])

        # Filtering via the SIR particle filter
        sir_particles[k+1] = sir_filter(particles=sir_particles[k],
                                        measurement=measurements[k+1])

        # Compute the effective Sample Size (Neff) for the SIS filter
        Neff[k+1] = int(1 / np.sum(sis_weights[k+1] ** 2))
        print(f'Iteration: {k+1}\tSIS Neff: {Neff[k+1]:.2f}')

    # Create a figure for plotting several time steps together
    _, ax1 = plt.subplots(nrows=2, ncols=3, figsize=(14, 9))    # SIS
    _, ax2 = plt.subplots(nrows=2, ncols=3, figsize=(14, 9))    # GPF
    # _, ax3 = plt.subplots(nrows=2, ncols=3, figsize=(14, 9))    # SIR-PF

    time_steps = [1, 3, 5, 10, 15, 19]
    indices = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for step, ax_idx in zip(time_steps, indices):
        # Plot SIS data
        ax1[ax_idx].scatter(sis_particles[step], sis_weights[step], marker=".")
        ax1[ax_idx].vlines(sis_particles[step], 0, sis_weights[step], lw=1)
        ax1[ax_idx].set_title(f"SIS Posterior, time k={step}, Neff={Neff[step]}")
        ax1[ax_idx].set_xlabel(f"1D State, x")
        ax1[ax_idx].set_xlim(left=0.0, right=60)

        # Plot Generic PF data
        ax2[ax_idx].scatter(pf_particles[step], pf_weights[step], marker=".")
        ax2[ax_idx].vlines(pf_particles[step], 0, pf_weights[step], lw=1)
        ax2[ax_idx].set_title(f"PF Posterior, time k={step}")
        ax2[ax_idx].set_xlabel(f"1D State, x")
        ax2[ax_idx].set_xlim(left=0.0, right=60)

        # Plot the SIR-PF Data

    plt.show()
