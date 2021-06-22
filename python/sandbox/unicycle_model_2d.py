"""
The sandbox file stores the following models for the 2D Unicycle model:

1. State transition density, p(x_{k+1} | x_k).
2. Measurement likelihood density, p(y_{k+1} | x_{k+1}).
3. The SIS Importance density, q(x_{k+1} | x_k, y_{k+1}).
"""
from typing import Optional, Union

import numpy as np
from scipy.stats import multivariate_normal as mvn


# Shorthand for type hints
nparr = np.ndarray

# Seed the multivariate normal distribution random number generator
RNG = np.random.default_rng(1234)
mvn.random_state = RNG

# Seed the multivariate normal distribution random number generator
INIT_RNG = np.random.default_rng(1234)


def transition_fcn_2d(particles: nparr, linear_vel: float, angular_vel: float,
                      dt: float,  process_noise_cov: Optional[nparr] = None,
                      vectorize: bool = True) -> nparr:
    """
    Propagates the set of particles through the nonlinear dynamics model f()
    with added white Gaussian noise (AWGN) with mean zero and covariance equal
    to the given process noise covariance, Q.

    x_{k+1}^i = f(x_k^i) + w_k^i, where w_k^i ~ N(0, Q)

    This is equivalent to drawing x_{k+1}^i from a Gaussian distribution with
    mean f(x_k^i) and covariance Q:

    x_{k+1}^i ~ N(f(x_k^i), Q), where i = [1, 2, ..., Nparticles].

    Note that the prediction step here relies on the true linear and angular
    velocity profiles. These values could be estimated in a later iteration of
    this demo.

    Args:
        particles: Particles containing state estimates at the previous time
            step, k. Particles can be either 1D of shape (stateDim,) or 2D of
            shape (nParticles, stateDim).
        process_noise_cov: The process noise covariance matrix, typically a
            diagonal matrix (zero correlation between states).
        linear_vel: The true linear velocity at time step k.
        angular_vel: The true angular velocity at time step k.
        dt: The discrete time step, in seconds.
        vectorize:

    Returns:
        new_particles:
    """
    # Reshape particles to be two dimensional
    particles = np.atleast_2d(particles)

    if process_noise_cov is None:
        # No process noise will be added
        w = np.zeros(particles.shape)
    else:
        # Create zero mean process noise, w
        w = np.atleast_2d(mvn(np.zeros(particles.shape[1]),
                              process_noise_cov).rvs(particles.shape[0]))

    new_particles = np.zeros_like(particles)

    if vectorize:
        # Propagate state through the nonlinear dynamics
        new_particles[:, 0] = particles[:, 0] + \
            linear_vel * np.cos(particles[:, 2]) * dt + w[:, 0] * dt
        new_particles[:, 1] = particles[:, 1] + \
            linear_vel * np.sin(particles[:, 2]) * dt + w[:, 1] * dt
        new_particles[:, 2] = particles[:, 2] + angular_vel * dt + w[:, 2] * dt
    else:
        for idx, p in enumerate(particles):
            # Propagate state through the nonlinear dynamics
            # TODO: Why is the process noise multiplied by the integration time?
            new_particles[idx, 0] = p[0] + linear_vel * np.cos(p[2]) * dt \
                                    + w[idx, 0] * dt
            new_particles[idx, 1] = p[1] + linear_vel * np.sin(p[2]) * dt \
                                    + w[idx, 1] * dt
            new_particles[idx, 2] = p[2] + angular_vel*dt + w[idx, 2] * dt

    return np.squeeze(new_particles)


def importance_fnc_2d():
    """
    The importance density function. In the case of the 2D Unicycle model,
    the transition density is used as the importance density.

    Returns:
        The predicted current state, x_{k+1}, sampled from the importance
        density, q(x_{k+1} | x_k, y_{k+1}).
    """
    pass


def likelihood_fcn_2d(state: nparr, landmarks: nparr, measurements: nparr,
                      meas_noise_cov: Union[nparr, float] = 0.5) -> nparr:
    """

    Args:
        state: The current state, x_{k+1}^i for all i in Nparticles.
        landmarks: The array of 2D landmarks, (Nlandmarks, 2).
        measurements: The current measurement vector, y_{k+1} of shape
            (Nlandmarks,).
        meas_noise_cov: The measurement noise covariance.

    Returns:
        likelihood: The likelihood of observing the current measurement given
        the current state.
    """
    # Initialize the weights
    n_particles = state.shape[0]
    weights = np.ones(n_particles)

    for l, (m, lm) in enumerate(zip(measurements, landmarks)):
        # Compute the expected measurement for each particle
        expected_meas = range_sensor_model(state[:, :2], lm)

        # Compute the likelihood that the current measurement is
        # explained by a given particle's current state estimate
        likelihood = np.array([mvn(expected, meas_noise_cov).pdf(m) for
                               expected in expected_meas[1:]])

        # Update the particle weight (and normalize to avoid underflow)
        weights = weights * likelihood
        weights = weights / np.sum(weights)

    return weights


def prior_fcn_2d(mean: nparr, dim: int) -> nparr:
    """
    The prior density for the 2D Localization demo, p(x_0).

    Args:
        mean: The mean value around which the Gaussian prior will be
            distributed.
        dim: The number of particles.

    Returns:
        prior: The set or particles at time step zero.
    """
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
    init_dist = mvn(mean=mean[:2], cov=1.5)
    init_dist.random_state = INIT_RNG

    prior = np.nan * np.ones((dim, 3))
    prior[:, :2] = init_dist.rvs(dim)
    prior[:, 2] = np.zeros(dim)

    return prior


def truth_fcn_2d(times: nparr, x0: nparr, process_noise_cov: nparr,
                 linear_vel: nparr, angular_vel: nparr, dt: float) -> nparr:
    """
    Simulates the forward truth data for the 2D unicycle model using Euler
    integration. The state vector for the 2D Unicycle model consists of
    [x, y, theta] where (x,y) is the 2D position in meters and theta is the
    heading in radians.

    Args:
        times: A vector of times at which to simulate the state vector, (K,)
        x0: The initial condition, a vector of shape (n,).
        process_noise_cov: The process noise covariance matrix, typically a
            diagonal matrix (zero correlation between states).
        linear_vel: The true linear velocity profile, in m/s.
        angular_vel: The true angular velocity profile, in rad/s.
        dt: The discrete time step, in seconds.

    Return:
        state_truth: The simulated state vector at every time step, (K,n).
    """
    # Initialize the state truth vector with the initial condition
    state_truth = np.zeros((len(times), 3))
    state_truth[0] = x0

    # Add process noise (w) as in lecture notes, Lec. 12 slide 12
    Q = process_noise_cov
    W = mvn(mean=np.zeros(3), cov=Q).rvs(len(times))

    for k in range(len(times) - 1):
        state_truth[k+1] = transition_fcn_2d(state_truth[k], linear_vel[k],
                                             angular_vel[k], dt, Q)

    return state_truth


def range_sensor_model(positions: nparr, landmarks: nparr,
                       meas_noise_cov: Optional[float] = None,
                       vectorize: bool = True) -> nparr:
    """
    Creates a set of measurements for the 2D Localization Demo.

    Args:
        positions: The true state vector at every time step, (K,n).
        meas_noise_cov: The 1D range measurement noise covariance.
        landmarks: The set of 2D landmarks, (nLandmarks, 2).
        vectorize: If True, calculation of the measurement vector is vectorized.

    Returns:
        measurements: The simulated range measurements for each landmark at
            every time step in the truth vector.
    """
    # Ensure that the landmarks vector is two dimensional
    landmarks = np.atleast_2d(landmarks)

    # Add measurement noise, v
    if meas_noise_cov is None:
        # NOTE: Does not account for the dimension of R
        r = np.zeros(landmarks.shape[0])
    else:
        R = meas_noise_cov
        r = mvn(mean=0.0, cov=R).rvs(size=landmarks.shape[0])

    # Initialize the measurements with a NaN value at index zero
    measurements = np.nan * np.ones((positions.shape[0]+1, landmarks.shape[0]))

    # Create simulated measurements for each landmark
    if vectorize:
        for l, lm in enumerate(landmarks):
            lm_vector = lm * np.ones((positions.shape[0], landmarks.shape[1]))
            # TODO: Noise is incorrectly added to each measurement (not added at
            #   all right now). For each position and each landmark, Gaussian
            #   noise should be added to the measurement. This is done below,
            #   but needs to be vectorized.
            measurements[1:, l] = np.sqrt(np.sum(
                (positions - lm_vector) ** 2, axis=1))
    else:
        for idx, pos in enumerate(positions):
            for l, lm in enumerate(landmarks):
                # Create the simulated measurement
                measurements[idx+1, l] = np.sqrt((lm[0] - pos[0]) ** 2 +
                                                 (lm[1] - pos[1]) ** 2) + r[l]

                # Enforce non-negativity (when would this ever be negative?)
                measurements[idx+1, l] = max(measurements[idx+1, l], 0.0)

    return np.squeeze(measurements)


def get_landmarks() -> nparr:
    """
    Returns the landmarks vector.
    """
    landmarks = np.array([
        [-1.2856, 5.8519],
        [1.5420, 2.1169],
        [-0.1104, 1.7926],
        [4.2603, 9.7480],
        [2.6365, 12.9204],
        [- 3.5036, 7.7518],
        [- 1.6228, 10.2106],
        [- 9.8876, 1.2568],
        [2.1522, 0.5491],
        [-7.3594, 11.9139]])

    # landmarks = np.array([
    #     [-1.2856-2.5, 6.4964],
    #     [1.5420-2.5, 8.3772],
    #     [-0.1104-2.5, 0.1124],
    #     [4.2603-2.5, 12.1522],
    #     [2.6365-2.5, 2.6406]])

    return landmarks
