from typing import List, Tuple
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


class Phases:
    """Handles laser phase analysis and tracking."""
    def __init__(self):
        self.values = []
        self.times = []
        self.errors = []

    def set_laser_freq(self, laser_freq):
        self.laser_freq = laser_freq

    def process_likely_pulses(self, likely_pulses: List[float]):
        """Processes timestamps of likely laser pulses and updates phase tracking.
        Returns early if no pulses are provided.
        """

        if len(likely_pulses) == 0:
            return
        phases = np.array(likely_pulses) % (1 / self.laser_freq)
        mean_time = np.nanmean(likely_pulses)
        self.append(mean_time, phases)

    def append(self, time: float, phases: np.ndarray):
        """Adds a new phase measurement at the given time."""
        self.values.append(np.nanmean(phases))
        self.times.append(time)
        self.errors.append(np.nanstd(phases) / np.sqrt(phases.size))

    def filter_outliers(self, low_cut: float = 10, top_cut: float = 90):
        """Removes phase measurements outside the specified percentile range."""
        self.values, self.times, self.errors = (
            np.array(self.values),
            np.array(self.times),
            np.array(self.errors),
        )
        bottom_percentile, top_percentile = np.percentile(
            self.values, [low_cut, top_cut]
        )
        mask = np.logical_and(
            self.values > bottom_percentile, self.values < top_percentile
        )
        self.values = self.values[mask]
        self.times = self.times[mask]
        self.errors = self.errors[mask]

    def get_mean_phase(self) -> Tuple[float, float]:
        """Returns the mean phase and its standard error."""
        return np.mean(self.values), np.std(self.values) / np.sqrt(self.values.size)

    def get_interpolator(self) -> interp1d:
        """Returns a function that interpolates phases in time."""
        values_filteres = savgol_filter(self.values, 5, 1)
        return interp1d(self.times, values_filteres, fill_value="extrapolate")


