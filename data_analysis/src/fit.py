from typing import Dict, List
from iminuit import Minuit
from scipy.special import gammaln
import numpy as np
from src.analysis_mDOM_position import FilteredData
from src.constants import AF_PROB, DEAD_TIME, LASER_FREQ, TIME_WINDOW


# class Fit:
#     def __init__(self, x_l, y_l, x_b, y_b):
#         self.x_l = x_l
#         self.y_l = y_l
#         self.x_b = x_b
#         self.y_b = y_b

#         log_factorial = []
#         for y in self.y_l:
#             log_factorial.append(gammaln(y + 1))
#         self.log_factorial_l = np.array(log_factorial)

#         log_factorial = []
#         for y in self.y_b:
#             log_factorial.append(gammaln(y + 1))
#         self.log_factorial_b = np.array(log_factorial)
#         self.window = 2 * TIME_WINDOW
#         self.nr_fit_parameters = 4
#         self.after_pulse_prob = 0.0565 # mean after pulse probability from AT

#     def laser_model(self, mu, dark_rate, norm_l, norm_b):
#         mu_b = dark_rate * self.window * (1 + self.after_pulse_prob*0.5)
#         return norm_l * (np.exp(-(mu + mu_b) * self.x_l))

#     def background_model(self, mu, dark_rate, norm_l, norm_b):
#         mu = dark_rate * (1 - self.window * LASER_FREQ)
#         h = norm_b
#         return (
#             mu
#             * np.power(10, self.x_b)
#             * np.log(10)
#             * np.exp(-mu * np.power(10, self.x_b))
#             * h
#         )

#     def llh(self, mdl, y, factorial):
#         return np.nansum(-mdl + np.log(mdl) * y - factorial)

#     def poisson_llh(self, mu, dark_rate, norm_l, norm_b):
#         y_model = self.laser_model(mu, dark_rate, norm_l, norm_b)
#         llh = -self.llh(y_model, self.y_l, self.log_factorial_l)
#         y_model = self.background_model(mu, dark_rate, norm_l, norm_b)
#         llh += -self.llh(y_model, self.y_b, self.log_factorial_b)
#         return llh

#     def goodness_of_fit(self, mu, dark_rate, norm_l, norm_b):
#         llh_best_fit = self.poisson_llh(mu, dark_rate, norm_l, norm_b)
#         llh_n = -self.llh(self.y_l, self.y_l, self.log_factorial_l)
#         llh_n += -self.llh(self.y_b, self.y_b, self.log_factorial_b)
#         return -2 * (-llh_best_fit + llh_n) / (self.y_l.size - self.nr_fit_parameters)


class Fit:
    def __init__(self, x_l, y_l, x_b, y_b):
        self.x_l = x_l
        self.y_l = y_l
        self.x_b = x_b
        self.y_b = y_b
        self._nr_fit_parameters = 4

        self._calculate_factorials()

    def set_passing_fractions(self, passing_fractions: List[float]):
        self._passing_fractions = passing_fractions

    def _calculate_factorials(self):
        log_factorial = []
        for y in self.y_l:
            log_factorial.append(gammaln(y + 1))
        self._log_factorial_l = np.array(log_factorial)

        log_factorial = []
        for y in self.y_b:
            log_factorial.append(gammaln(y + 1))
        self._log_factorial_b = np.array(log_factorial)

    def set_time_window(self, window: float):
        self._window = window

    def set_laser_freq(self, laser_freq: float):
        self._laser_freq = laser_freq

    def _P_no_background_in_tw(self, dark_rate, t_w):
        mu_b = dark_rate * t_w
        return np.exp(-mu_b)

    def _P_dead(self, mu, dark_rate, norm_l, norm_b):
        P_previous_back = 1 - self._P_no_background_in_tw(
            dark_rate * (1 + AF_PROB), DEAD_TIME
        )
        return P_previous_back

    def _P_no_photons_from_laser(self, mu):

        P_0p = np.exp(-mu)
        P_ip = []
        for const in [mu, mu**2 / 2.0, mu**3 / 6.0]:
            P_ip.append(P_0p * const)

        P_gt3 = 1 - P_0p - sum(P_ip)  # probability of more than 3 photons

        for i, passing_fraction in enumerate(self._passing_fractions):
            P_ip[i] *= passing_fraction

        P_0p_after_trigger_loss = 1 - sum(P_ip) - P_gt3
        return P_0p_after_trigger_loss

    def _P_measure_nothing(self, mu, dark_rate, norm_l, norm_b):
        # After thinking about it, there should be no dead time in the laser window. The window is much larger than the dead time.
        # If there is background before the laser window, it will be counted as a single event (we measured 'something').
        # Anyways difference with and without dead time is around 3rd digit.

        # P_dead = self._P_dead(mu, dark_rate, norm_l, norm_b)
        P_no_laser = self._P_no_photons_from_laser(mu)
        # P_ng0_laser = 1-P_no_laser
        # P_undetected_laser = P_ng0_laser*P_dead
        # P_no_laser_contribution = P_undetected_laser+P_no_laser
        P_no_background = self._P_no_background_in_tw(dark_rate, self._window)
        return P_no_background * P_no_laser

    def time_distribution_model(self, mu, dark_rate, norm_l, norm_b):
        P_measure_nothing = self._P_measure_nothing(mu, dark_rate, norm_l, norm_b)
        P_measure_something = 1 - P_measure_nothing
        return (
            norm_l * np.power(P_measure_nothing, self.x_l - 1) * P_measure_something**2
        )

    def background_model(self, mu, dark_rate, norm_l, norm_b):
        mu = dark_rate * (1 - self._window * self._laser_freq)
        h = norm_b
        return (
            mu
            * np.power(10, self.x_b)
            * np.log(10)
            * np.exp(-mu * np.power(10, self.x_b))
            * h
        )

    def llh(self, mdl, y, factorial):
        return np.sum(-mdl + np.log(mdl) * y - factorial)

    def poisson_llh(self, mu, dark_rate, norm_l, norm_b):
        y_model = self.time_distribution_model(mu, dark_rate, norm_l, norm_b)
        llh = -self.llh(y_model, self.y_l, self._log_factorial_l)
        y_model = self.background_model(mu, dark_rate, norm_l, norm_b)
        llh += -self.llh(y_model, self.y_b, self._log_factorial_b)
        return llh

    def goodness_of_fit(self, mu, dark_rate, norm_l, norm_b):
        llh_best_fit = self.poisson_llh(mu, dark_rate, norm_l, norm_b)
        llh_n = -self.llh(self.y_l + 1e-9, self.y_l + 1e-9, self._log_factorial_l)
        llh_n += -self.llh(self.y_b + 1e-9, self.y_b + 1e-9, self._log_factorial_b)
        return -2 * (-llh_best_fit + llh_n) / (self.y_l.size - self._nr_fit_parameters)


def fit_filtered_data(
    data: FilteredData,
    passing_fraction: List[float],
    p0: Dict[str,float] = {"mu": 0.5, "dark_rate": 500, "norm_l": 50000.0, "norm_b": 300.0},
    time_window: float = TIME_WINDOW,
    laser_freq: float = LASER_FREQ,
):
    x = data.background_x
    y = data.background_h
    mask = np.logical_and(x > -3.5, x < -2)
    y = y[mask]
    x = x[mask]
    x_l = data.signal_x
    y_l = data.signal_h
    fit = Fit(x_l, y_l, x, y)
    fit.set_passing_fractions(passing_fraction)
    fit.set_time_window(2 * time_window)
    fit.set_laser_freq(laser_freq)

    m = Minuit(fit.poisson_llh, **p0)
    m.limits["mu"] = (0.00, 5.0)
    m.limits["dark_rate"] = (10, 5000.0)
    m.limits["norm_l"] = (0.00, 1e6)
    m.limits["norm_b"] = (0.00, 1e6)
    m.errors["mu"] = 0.5
    m.errors["dark_rate"] = 1000
    m.errors["norm_l"] = 5000
    m.errors["norm_b"] = 5000

    m.errordef = 0.5
    m.strategy = 2
    m.migrad()

    if m.valid:
        return (
            m.values[0],
            m.errors[0],
            m.values[1],
            m.errors[1],
            fit.goodness_of_fit(*m.values),
        )
    else:
        return 0, 0, 0, 0, -1
