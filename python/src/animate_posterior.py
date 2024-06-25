"""
Create a class to be used with matplotlib.animation.FuncAnimation to display
an "animated", auto-updating posterior distribution.
"""
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Shorthand for type hints
nparr = np.ndarray


class AnimatePosterior(object):
    def __init__(self, name: str, algo: Callable, steps: int, n_particles: int,
                 prior: Callable, truth: nparr, measurements: nparr):
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

        # Create plots
        self._scat = self._ax.scatter([], [], marker=".", zorder=0)
        self._lines = self._ax.vlines([], [], [], lw=1, zorder=1)
        self._line1 = self._ax.axvline(truth[0], color='black', zorder=3)
        self._line2 = self._ax.axvline(0, ls='--', color='tab:orange', zorder=4)

        # Assign names to each plot (displayed by the lengend)
        self._scat.set_label('Particles')
        self._line1.set_label('True 1D State')
        self._line2.set_label(f'{self._name} Filter 1D State')

        # Set plot attributes
        self._ax.set_xlabel(f"1D State, x")
        self._ax.set_ylabel(f"Posterior Density Value")
        self._ax.set_xlim(left=0.0, right=60)

        # Store the algorithm to be used for filtering
        self._algo = algo

        # Store the number of time steps and number of particles
        self._steps = steps
        self._n_particles = n_particles

        # Store particle states and weights for plotting
        self._particles = np.zeros((self._steps, self._n_particles))
        self._particles[0] = prior(self._n_particles)
        self._weights = np.zeros((self._steps, self._n_particles))
        self._weights[0] = (1 / self._n_particles) * np.ones(self._n_particles)

        # Store the effective sample size
        self._neff = np.zeros(self._steps, dtype=int)
        self._neff[0] = self._n_particles

        # Store the truth and measurements
        self._truth = truth
        self._measurements = measurements

        # Store the 1D MAP estimate
        self._estimate_1d = np.nan * np.ones(self._steps)

        # Store the current index
        self._idx = 0

        # If True, the posterior estimate at every time step has been calculated
        self._complete = False

        # Setup the FuncAnimation
        self._ani = animation.FuncAnimation(self._fig, self, interval=250,
                                            blit=False)

    @property
    def particles(self):
        return self._particles

    @property
    def weights(self):
        return self._weights

    @property
    def trajectory(self):
        return self._estimate_1d

    def __call__(self, frame: int):
        """

        Args:
            frame:
        """
        if frame == 0:
            self._scat.set_offsets(np.zeros((self._n_particles, 2)))
            self._lines.set_segments(np.zeros((1, 2, 2)))
            self._line1.set_xdata(self._truth[self._idx])
            self._line2.set_xdata(self._truth[self._idx])
            return [self._scat, self._lines, self._line1, self._line2]

        # Update the posterior via the chosen algorithm
        self.update()
        print(f'AnimatePosterior Iteration: frame: {frame}\tidx: {self._idx}\t '
              f'Weight Sum: {np.sum(self._weights[self._idx]):.2f}\t '
              f'Neff: {self._neff[self._idx]}')

        # Update the scatter plot with data at the current time step, k+1
        self._scat.set_offsets(np.column_stack((self._particles[self._idx],
                                                self._weights[self._idx])))
        # Update the vertica line segments (requires a 3D array)
        line_segs = np.array([[[x, 0.0], [x, y]] for x, y in
            zip(self._particles[self._idx], self._weights[self._idx])])
        self._lines.set_segments(line_segs)

        # Update the vertical dashed line representing the true 1D position
        self._ax.set_ylim(0, max(self._weights[self._idx]))
        self._line1.set_xdata(self._truth[self._idx])
        self._line2.set_xdata(self._estimate_1d[self._idx])

        self._ax.set_title(f"{self._name} Posterior, time k={self._idx}, "
                           f"Neff={self._neff[self._idx]}")
        self._ax.legend()

        if self._idx == self._steps - 1:
            # Reset the index such that the animation starts from the beginning
            self._idx = 0
            self._complete = True

        return [self._scat, self._lines, self._line1, self._line2]

    def update(self):
        """
        Update the particle state estimates and weights via the chosen filtering
        algorithm.
        """
        if not self._complete:
            # print(f'Updating particles at time step k = {self._idx+1}')
            # Update the particle states and weights
            self._particles[self._idx+1], self._weights[self._idx+1], state = \
                self._algo(particles=self._particles[self._idx],
                           weights=self._weights[self._idx],
                           measurement=self._measurements[self._idx+1])

            # Store the estimated 1D state (MAP estimate)
            self._estimate_1d[self._idx+1] = state

            # Compute the effective sample size
            self._neff[self._idx+1] = \
                int(1 / np.sum(self._weights[self._idx+1] ** 2))

        # Update the internal index
        self._idx += 1
