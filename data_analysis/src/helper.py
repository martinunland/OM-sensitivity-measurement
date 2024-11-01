from typing import List, Tuple
import numpy as np
from scipy.interpolate import interp1d
from src.constants import ADC_CLOCK


def create_laser_mask(
    timestamps: float, laser_phase: float, laser_frequency: float, time_window: float
) -> Tuple[List[bool], List[float]]:
    """Creates a mask identifying laser pulse events using a fixed phase.

    Args:
        timestamps: Event timestamps
        laser_phase: Fixed laser phase
        laser_frequency: Laser frequency in Hz
        time_window: Time window for signal detection

    Returns:
        (signal_mask, phase_differences, input_phase)
    """
    all_phases = timestamps % (1 / laser_frequency)
    rel_phase = np.abs(all_phases - laser_phase)
    return rel_phase <= time_window, all_phases - laser_phase, laser_phase


def create_laser_mask_interp(
    timestamps: float,
    laser_phase_interpolator: interp1d,
    laser_frequency,
    time_window: float,
) -> Tuple[List[bool], List[float], float]:
    """Creates a mask identifying laser pulse events using interpolated phases.

    Args:
        timestamps: Event timestamps
        laser_phase_interpolator: Phase interpolation function
        laser_frequency: Laser frequency in Hz
        time_window: Time window for signal detection

    Returns:
        (signal_mask, phase_differences, mean_phase)
    """
    all_phases = timestamps % (1 / laser_frequency)
    laser_phase = laser_phase_interpolator(timestamps)
    rel_phase = np.abs(all_phases - laser_phase)
    return rel_phase <= time_window, all_phases - laser_phase, np.mean(laser_phase)


def get_likely_laser_pulses(timestamps: List[float], laser_freq: int) -> List[float]:
    """Identifies timestamps likely corresponding to laser pulses.

    Args:
        timestamps: List of event timestamps
        laser_freq: Laser frequency in Hz

    Returns:
        List of timestamps likely from laser pulses
    """

    expected_period = 1 / laser_freq
    min_dt = 0.5 * expected_period
    max_dt = 20 * expected_period
    tolerance = 0.01
    clock_tolerance = 1 / ADC_CLOCK

    # Compute the time differences between consecutive timestamps
    dT = np.diff(timestamps)

    # Filter differences that fall within the valid range
    valid_indices = np.where((dT > min_dt) & (dT < max_dt))[0]

    # Compute the multiples and their rounded values
    multiples = dT[valid_indices] / expected_period
    rounded_multiples = np.round(multiples)

    # Calculate the condition mask
    condition_mask = (np.abs(rounded_multiples - multiples) < tolerance) & (
        np.abs(dT[valid_indices] - rounded_multiples * expected_period)
        <= clock_tolerance
    )

    # Filter the indices that meet the condition
    likely_indices = valid_indices[condition_mask]

    # Collect the corresponding timestamps
    likely_laser_pulses = np.unique(
        np.concatenate((timestamps[likely_indices], timestamps[likely_indices + 1]))
    )

    return likely_laser_pulses
