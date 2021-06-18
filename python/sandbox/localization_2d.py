"""
This sandbox script will recreate the 2D Localization Demo
"""
import time
from typing import Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import uniform

from unicycle_model_2d import transition_fcn_2d, likelihood_fcn_2d, \
    prior_fcn_2d, truth_fcn_2d, range_sensor_model, get_landmarks
from particle_filter import systematic_resample


# Shorthand for type hints
nparr = np.ndarray

# Seed the uniform distribution used to create the initial condition
STATE_RNG = np.random.default_rng(117)
uniform.random_state = STATE_RNG

# Set to true if visualizing the 2D localization animation
ANIMATE = True


def localization_2d_pf(particles: nparr, weights: nparr, measurement: nparr,
                       linear_vel: float, angular_vel: float, dt: float,
                       process_noise_cov: nparr, time_step: int,
                       neff_thres: int = 50, skip: int = 100,
                       transition_fcn: Callable = transition_fcn_2d,
                       likelihood_fcn: Callable = likelihood_fcn_2d) -> \
        Tuple[nparr, nparr, nparr]:
    """
    An implementation of the SIR-PR for the 2D Localization demo. This particle
    filter uses the transition density as the importance density from which to
    sample x_{k+1}^i, and resamples when the effective sample size drops below a
    threshold value.

    Args:
        particles: Particles containing state estimates at the previous time
            step, k.
        weights: The particle weights at the previous time step, k.
        measurement: The current measurement, y_{k+1}.
        linear_vel: The true linear velocity profile, in m/s.
        angular_vel: The true angular velocity profile, in rad/s.
        dt: The discrete time step, in seconds.
        process_noise_cov:
        time_step: the current time step, k+1.
        neff_thres: Minimum number of effective particles before resampling
            occurs.
        skip: The number of time steps between measurement updates.
        transition_fcn: p(x_{k+1} | x_k).
        likelihood_fcn: p(y_{k+1} | x_{k+1}).

    Returns:
        new_particles: Particles containing state estimates at the current time
            step, k+1.
        new_weights: The particle weights at the current time step, k+!.
        state: The 2D state at the current time step computed as an expected
            value of the current set of particles.
    """
    # Rename some variables
    Q = process_noise_cov

    # Prediction step - propagate the particles through the process model
    new_particles = transition_fcn(particles, linear_vel, angular_vel, dt, Q)

    # Compute the mean and covariance from the current set of particles, k+1
    state = np.average(new_particles, axis=0, weights=weights)
    # state = new_particles[weights.argmax()]  # MAP estimate (ugly)
    cov = np.cov(new_particles[:, :2], rowvar=False)

    # Update the particles weights via the current measurement
    if np.mod(time_step, skip) == 0:
        # TODO: Attempt to vectorize the evaluation of the likelihood.
        # expected_meas_vec = range_sensor_model(particles[k+1][:, :2],
        #                                        landmarks)
        # print('Evaluating likelihood in vectorized form.')
        # w1 = likelihood_fcn_2d(expected_meas_vec[1:], measurements[k+1])

        # Update all weights from the current time step to the end such
        # that the "new_weights" are correctly set at the next measurement
        # update.
        landmarks = get_landmarks()
        new_weights = likelihood_fcn(new_particles, landmarks, measurement)

        # Calculate N_eff using Arulampalam Eqn. (51)
        Neff = 1 / np.sum(new_weights ** 2)
        if Neff < neff_thres:
            # Resample particles via "systematic resampling"
            print('RESAMPLING PARTICLES')
            new_particles, new_weights = \
                systematic_resample(new_particles, new_weights)
    else:
        # If resampling does not occur, the current weights get passed through
        new_weights = weights

    return new_particles, new_weights, state


class Localization2D(object):
    """
    The animated 2D Localization demo
    """
    def __init__(self, name: str, algo: Callable, steps: int, n_particles: int,
                 prior: Callable, truth: nparr, measurements: nparr,
                 linear_vel: nparr, angular_vel: nparr, dt: float,
                 process_noise_cov: nparr):
        """
        Constructor.

        Args:
            name:
            algo:
            steps:
            n_particles:
            prior:
            truth:
            measurements:
        """
        self._name = name
        self._fig, self._ax = plt.subplots()
        self._fig.suptitle(self._name)

        # Create plots
        self._scat = self._ax.scatter([], [], c="tab:grey", marker=".",
                                      edgecolor="black", zorder=1)
        self._line1, = self._ax.plot([], [], c="tab:blue", zorder=0)
        self._line2, = self._ax.plot([], [], c="darkorange", zorder=2)
        self._lm = self._ax.scatter([], [], c="m", marker="P", zorder=3)
        self._start = self._ax.scatter([], [], c="g", marker="X", zorder=4)
        self._end = self._ax.scatter([], [], c="tab:red", marker="X", zorder=5)

        # Set plot attributes
        self._ax.set_xlim(left=truth[:, 0].min()-1, right=truth[:, 0].max()+1)
        self._ax.set_ylim(bottom=truth[:, 1].min()-1, top=truth[:, 1].max()+1)

        # Store the algorithm to be used for filtering
        self._algo = algo

        # Store the number of time steps and number of particles
        self._steps = steps
        self._n_particles = n_particles
        self._nx = truth.shape[1]

        # Store other relevant data
        self._linear_vel = linear_vel
        self._angular_vel = angular_vel
        self._dt = dt
        self._process_noise_cov = process_noise_cov

        # Store particle states and weights for plotting
        self._particles = np.zeros((self._steps, self._n_particles, self._nx))
        self._particles[0] = prior(truth[0], self._n_particles)
        self._weights = np.zeros((self._steps, self._n_particles))
        self._weights[0] = (1 / self._n_particles) * np.ones(self._n_particles)

        # Store the estimated trajectory
        self._trajectory = np.nan * np.ones((self._steps, 2))

        # Store the truth, measurements and landmarks
        self._truth = truth
        self._measurements = measurements
        self._landmarks = get_landmarks()

        # Store the current index
        self._idx = 0

        # If True, the posterior estimate at every time step has been calculated
        self._complete = False

        # Setup the FuncAnimation
        self._ani = animation.FuncAnimation(self._fig, self, interval=1,
                                            blit=True)

    @property
    def trajectory(self):
        return self._trajectory

    @ property
    def cov(self):
        pass

    def __call__(self, frame: int):
        if frame == 0:
            self._scat.set_offsets(self._particles[self._idx, :, :2])
            self._line1.set_data(self._truth[:, 0], self._truth[:, 1])
            self._line2.set_data([], [])
            self._lm.set_offsets(self._landmarks)
            self._start.set_offsets(self._truth[0, :2])
            self._end.set_offsets(self._truth[-1, :2])

            return [self._scat, self._line1, self._line2, self._lm, self._start,
                    self._end]

        # Update the posterior via the chosen algorithm
        self.update()

        # Update the particles with data at the current time step, k+1
        self._scat.set_offsets(self._particles[self._idx, :, :2])
        self._scat.set_sizes(10000 * self._weights[self._idx])

        # Update the static truth data
        self._line1.set_data(self._truth[:, 0], self._truth[:, 1])
        self._lm.set_offsets(self._landmarks)
        self._start.set_offsets(self._truth[0, :2])
        self._end.set_offsets(self._truth[-1, :2])

        # Update the trajectory plot
        self._line2.set_data(self._trajectory[:self._idx, 0],
                             self._trajectory[:self._idx, 1])

        # Update title (won't update of blit=True)
        # self._ax.set_title(f"{self._name}, time k={self._idx}")

        if self._idx == self._steps - 1:
            # Reset the index such that the animation starts from the beginning
            self._idx = 0
            self._complete = True

        return [self._scat, self._line1, self._line2, self._lm, self._start,
                self._end]

    def update(self):
        """
        Update the particle state estimates and weights via the chosen filtering
        algorithm.
        """
        if not self._complete:
            # Update the particle states and weights
            self._particles[self._idx+1], self._weights[self._idx+1], state = \
                self._algo(particles=self._particles[self._idx],
                           weights=self._weights[self._idx],
                           measurement=self._measurements[self._idx+1],
                           linear_vel=self._linear_vel[self._idx],
                           angular_vel=self._angular_vel[self._idx],
                           dt=self._dt,
                           process_noise_cov=self._process_noise_cov,
                           time_step=self._idx+1)

            # Updated the estimated trajectory
            self._trajectory[self._idx+1] = state[:2]

        # Update the internal index
        self._idx += 1


if __name__ == '__main__':
    """
    Setup time, noise and additional parameters.
    """
    # Define time parameters
    _dt = 0.02
    _final_time = 100.0
    _times = np.linspace(start=0, stop=100, num=int(_final_time / _dt) + 1)

    # Shorthand for various state dimensions
    _n_particles = 100
    _nx = 3  # dimension of the state vector, [x, y, theta]
    _m = len(_times)  # number of discrete time steps

    # True process noise and range measurement covariance
    _Q = np.diag([0.1, 0.1, 0.05])
    _R = 1.5

    # Angular velocity profile, omega in [rad/s]
    omega_true = 0.2 * np.sin(0.1 * _times)
    # Linear velocity profile, s(?) in [m/s]
    v_true = 0.5 * np.ones(len(_times))

    """
    Particle Filter parameters
    """
    # Process noise and measurement nise used by the particle filter
    _Q_pf = np.diag([0.15, 0.15, 0.1])  # process noise tuning
    _R_pf = 0.5  # measurement noise tuning

    # Store the particle, weights and the estimated state trajectory and cov
    _particles = np.zeros((_m, _n_particles, _nx))
    _weights = np.zeros((_m, _n_particles))
    _weights[0] = (1 / _n_particles) * np.ones(_n_particles)

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
    Create the initial condition for the particle set
    """
    # Initialize particles by adding Gaussian noise to the true initial cond.
    _particles[0] = prior_fcn_2d(_state_truth[0], _n_particles)

    """
    Run the 2D Localization Particle Filter.
    """
    if ANIMATE:
        # Animate the 2D Localization Demo
        anim = Localization2D(name="2D Localization Particle Filter",
                              algo=localization_2d_pf,
                              steps=_m, n_particles=_n_particles,
                              prior=prior_fcn_2d, truth=_state_truth,
                              measurements=_measurements,
                              linear_vel=v_true, angular_vel=omega_true,
                              dt=_dt, process_noise_cov=_Q_pf)
        plt.show()

        # Get the trajectory from the animation
        _trajectory = anim.trajectory

    else:
        # Store the estimate 2D trajectory and covariance
        _trajectory = np.nan * np.ones((_m, 2))
        _cov = np.zeros((_m, 2, 2))

        timing = np.nan * np.ones(_m)
        for k in range(_m-1):
            start = time.perf_counter()
            # Filtering via the 2D Localization particle filter
            _particles[k+1], _weights[k+1], state = \
                localization_2d_pf(_particles[k], _weights[k], _measurements[k+1],
                                   v_true[k], omega_true[k], _dt, _Q_pf, k+1)

            # Update the estimated trajectory and covariance
            _trajectory[k+1] = state[:2]

            end = time.perf_counter()
            timing[k+1] = (end - start) * 1e3

        # Display timing statistics:
        timing = timing[1:]
        print('2D Localization Particle Filter Profiling Statistics (ms):')
        print('\tmean\t std\t min\t max')
        print(f'pf:\t{timing.mean():.2f}\t {timing.std():.2f}\t '
              f'{timing.min():.2f}\t {timing.max():.2f}')

    """
    Plot the results.
    """
    # Create figure for plotting
    fig, ax = plt.subplots()
    ax.set_title("2D Localization Particle Filter")
    ax.plot(_state_truth[:, 0], _state_truth[:, 1], c="tab:blue")
    ax.scatter(_state_truth[0, 0], _state_truth[0, 1], c="g", marker="X")
    ax.scatter(_state_truth[-1, 0], _state_truth[-1, 1], c="tab:red", marker="X")
    ax.scatter(_landmarks[:, 0], _landmarks[:, 1], c="m", marker="P")

    # Plot the estimated trajectory
    ax.plot(_trajectory[:, 0], _trajectory[:, 1], c="darkorange")
    ax.scatter(_trajectory[1, 0], _trajectory[1, 1], c="lime", marker="X")
    ax.scatter(_trajectory[-1, 0], _trajectory[-1, 1], c="red", marker="X")
    plt.show()
