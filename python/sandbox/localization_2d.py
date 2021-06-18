"""
This sandbox script will recreate the 2D Localization Demo
"""
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, uniform

from unicycle_model_2d import transition_fcn_2d, likelihood_fcn_2d, \
    truth_fcn_2d, range_sensor_model, get_landmarks
from particle_filter import systematic_resample


# Shorthand for type hints
nparr = np.ndarray

# Seed the uniform distribution used to create the initial condition
STATE_RNG = np.random.default_rng(117)
uniform.random_state = STATE_RNG

# Seed the multivariate normal distribution random number generator
INIT_RNG = np.random.default_rng(1234)


class Localization2D(object):
    """
    The animated 2D Localization demo
    """
    def __init__(self):
        pass

    @property
    def trajectory(self):
        pass

    @ property
    def cov(self):
        pass

    def __call__(self, frame: int):
        pass

    def update(self):
        pass



def localization_2d(times, measurements: nparr, landmarks: nparr,
                    linear_vel: nparr, angular_vel: nparr, dt: float,
                    n_particles: int = 300, skip: int = 100,
                    neff_thres: int = 20) -> Tuple[nparr, nparr]:
    """

    Args:
        times:
        measurements:
        landmarks:
        linear_vel: The true linear velocity profile, in m/s.
        angular_vel: The true angular velocity profile, in rad/s.
        dt: dt: The discrete time step, in seconds.
        n_particles: The number of particles used.
        skip: The number of time steps between measurement updates.

    Returns:
        particles:
        weights:
        trajectory:
    """
    Q_pf = np.diag([0.15, 0.15, 0.1])  # process noise tuning
    R_pf = 0.5  # measurement noise tuning

    # Shorthand for various state dimensions
    nx = 3  # dimension of the state vector, [x, y, theta]
    m = len(times)  # number of discrete time steps

    # Store the particle, weights and the estimated state trajectory and cov
    particles = np.zeros((m, n_particles, nx))
    weights = (1 / n_particles) * np.ones((m, n_particles))
    trajectory = np.nan * np.ones((m, 2))
    cov = np.zeros((m, 2, 2))

    # Create the particle's initial condition from a large uniform distribution
    # NOTE: This didn't work - 300 particles was not enough to adequately
    # describe the posterior without getting lucky.
    # for idx in range(nx-1):
    #     min_val = _state_truth[:, idx].min()
    #     max_val = _state_truth[:, idx].min()
    #     dist = uniform(loc=min_val, scale=max_val-min_val)
    #     particles[0, :, idx] = dist.rvs(n_particles)
    # # particles[0, :, 2] = uniform(loc=0, scale=(2*np.pi)).rvs(n_particles)
    # particles[0, :, 2] = np.pi * np.ones(n_particles)

    # Initialize the particles by adding Gaussian noise to the initial cond.
    init_dist = multivariate_normal(mean=_state_truth[0, :2], cov=1.0)
    init_dist.random_state = INIT_RNG
    particles[0, :, :2] = init_dist.rvs(n_particles)
    particles[0, :, 2] = np.zeros(n_particles)

    # Estimate the robot's state via particle filtering
    for k, particle_set in enumerate(particles):
        if k == 5000:
            break

        # Prediction step - propagate the particles through the process model
        particles[k+1] = transition_fcn_2d(particle_set, linear_vel[k],
                                           angular_vel[k], dt, Q_pf)

        # Compute the mean and covariance from the current set of particles, k+1
        trajectory[k+1] = np.mean(particles[k+1], axis=0)[:2]
        cov[k+1] = np.cov(particles[k+1, :, :2], rowvar=False)

        # Update the particles weights via the current measurement
        if np.mod(k+1, skip) == 0:
            print(f'Iteration {k+1}')
            # TODO: Attempt to vectorize the evaluation of the likelihood.
            # expected_meas_vec = range_sensor_model(particles[k+1][:, :2],
            #                                        landmarks)
            # print('Evaluating likelihood in vectorized form.')
            # w1 = likelihood_fcn_2d(expected_meas_vec[1:], measurements[k+1])

            # Update all weights from the current time step to the end such
            # that the "new_weights" are correctly set at the next measurement
            # update.
            weights[k+1:] = likelihood_fcn_2d(particles[k+1], landmarks,
                                              measurements[k+1])

            # Calculate N_eff using Arulampalam Eqn. (51)
            Neff = 1 / np.sum(weights[k+1] ** 2)
            if Neff < neff_thres:
                # Resample particles via "systematic resampling"
                print('RESAMPLING PARTICLES')
                particles[k+1], weights[k+1] = \
                    systematic_resample(particles[k+1], weights[k+1])

    return trajectory, cov


if __name__ == '__main__':
    """
    Setup time, noise and additional parameters.
    """
    # Define time parameters
    _dt = 0.02
    _final_time = 100.0
    _times = np.linspace(start=0, stop=100, num=int(_final_time / _dt) + 1)

    # Process noise variance
    _Q = np.diag([0.1, 0.1, 0.05])
    # True range  measurement noise variance
    _R = 1.5

    # Angular velocity profile, omega in [rad/s]
    omega_true = 0.2 * np.sin(0.1 * _times)
    # Linear velocity profile, s(?) in [m/s]
    v_true = 0.5 * np.ones(len(_times))

    """
    Simulate the truth data and a set of noisy (range, bearing) measurements.
    """
    # Simulate the forward truth data
    _state_truth = truth_fcn_2d(_times, np.array([0.1, 0.1, 0]), _Q, v_true,
                                omega_true, _dt)

    # Simulate the range/bearing measurements for each landmark
    _landmarks = get_landmarks()
    _measurements = range_sensor_model(_state_truth[:, :2], _landmarks, _R)

    """
    Run the 2D Localization Particle Filter.
    """
    _trajectory, _cov = localization_2d(_times, _measurements, _landmarks,
                                        v_true, omega_true, _dt)

    """
    Plot the results.
    """
    # Create figure for plotting
    fig, ax = plt.subplots()
    ax.plot(_state_truth[:, 0], _state_truth[:, 1])
    ax.scatter(_state_truth[0, 0], _state_truth[0, 1], c="g", marker="X")
    ax.scatter(_state_truth[-1, 0], _state_truth[-1, 1], c="red", marker="X")
    ax.scatter(_landmarks[:, 0], _landmarks[:, 1], c="m", marker="P")

    # Plot the estimated trajectory
    ax.plot(_trajectory[:, 0], _trajectory[:, 1], c="darkorange")
    ax.scatter(_trajectory[-1, 0], _trajectory[-1, 1], c="m", marker="X")
    plt.show()
