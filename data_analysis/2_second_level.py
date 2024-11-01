from typing import Dict
from src.constants import LASER_FREQ
from src.fit import fit_filtered_data
from src.analysis_mDOM_position import PositionStage
from src.reference_diode import ReferenceDiode
import numpy as np
import pickle
from multiprocessing import Pool
import os
from uncertainties import ufloat

fname_calibration_constants = (
    "../light_transmission/ref_current_to_number_of_photons.dat"
)
fname_reference_currents = "/data/240823_effective_area_DVT09_Martin/measurement_data_scan_2/240904_theta_phi_scan__diode.dat"
fname_reference_currents = "/data/240823_effective_area_DVT09_Martin/measurement_data_scan/240828_theta_phi_scan__diode.dat"
passing_trigger_data_fname = "/HDD/backuped/Promotion_data/Postdoc/241011_mDOM_pass_threshold/241014_loss_per_phi.dat"


class PassingTriggerData:
    def __init__(self, fname: str):
        PMT, phi, passing_fraction, error = np.loadtxt(fname, unpack=True)
        self._data_dict = {}
        for pmt, phi, passing_fraction, error in zip(PMT, phi, passing_fraction, error):
            if (pmt, phi) not in self._data_dict:
                self._data_dict[(pmt, phi)] = ([], [])
            self._data_dict[(pmt, phi)][0].append(passing_fraction)
            self._data_dict[(pmt, phi)][1].append(error)
        self._sort_data()

    def _sort_data(self):
        for key in self._data_dict:
            fractions, errors = self._data_dict[key]
            argsort = np.argsort(fractions)
            self._data_dict[key] = (
                [fractions[i] for i in argsort],
                [errors[i] for i in argsort],
            )

    def get_trigger_loss(self, phi: int, PMT: int):
        return self._data_dict[(PMT, phi)]


def process_pickle(
    fname: str, diode: ReferenceDiode, passing_trigger_data: PassingTriggerData
):
    output_filename = fname.replace(".pickle", "_summary_data_with_trigger_loss.dat")

    with open(output_filename, "w") as f:
        f.write(
            "#PMT_nr \t phi \t wavelength \t N_signal \t mu \t mu_err \t dark_rate \t dark_rate_err \t N_photons_emitted \t N_photons_emitted_err \t meas_time(s) \t gof \n"
        )
    with open(fname, "rb") as handle:
        data_positions: Dict[float, PositionStage] = pickle.load(handle)

    for PMT in range(0, 24):
        for phi, data in data_positions.items():
            passing_fraction, passing_fraction_error = (
                passing_trigger_data.get_trigger_loss(phi, PMT)
            )
            for wavelength, pmt_data in data.PMT_data[PMT].items():
                try:
                    mu, err, dr, err_dr, gof = fit_filtered_data(
                        pmt_data, passing_fraction
                    )
                    N_photons = (
                        ufloat(0, 0)
                        if wavelength == 400
                        else -data.mean_ref_current[wavelength]
                        * diode.current_to_number[wavelength]
                        / LASER_FREQ
                    )
                    with open(output_filename, "a") as f:
                        for val in [PMT, phi, wavelength]:
                            f.write(f"{val}\t")
                        for val in [
                            np.sum(pmt_data.signal_h) + 1,
                            mu,
                            err,
                            dr,
                            err_dr,
                            N_photons.n,
                            N_photons.s,
                            pmt_data.mes_time,
                            gof,
                        ]:
                            f.write(f"{val:.5g}\t")
                        f.write("\n")

                except Exception as err:
                    print(err)
                    pass


def main():

    fnames = [
        "output_data/" + fname
        for fname in os.listdir("output_data/")
        if ".pickle" in fname
    ]
    ref_diode = ReferenceDiode(fname_reference_currents, fname_calibration_constants)
    passing_trigger_data = PassingTriggerData(passing_trigger_data_fname)
    with Pool(processes=3) as pool:
        pool.starmap(
            process_pickle,
            [(fname, ref_diode, passing_trigger_data) for fname in fnames],
        )


############################################################################################################
fname_fited = "/HDD/backuped/Promotion_data/Postdoc/240822_mDOM_efficiency/pre_analysis/output_data/241014_summary_data_with_trigger_loss.dat"


class UnfitedData:
    def __init__(self, fname: str):
        self._fname = fname
        self._load_data()

    def _find_nearest_fit(self, phi, phi_data, mu_data, dark_rate_data):
        idx = np.argmin(np.abs(phi_data - phi))
        return mu_data[idx], dark_rate_data[idx]

    def _load_data(self):
        theta, pmt, phi, wavelength, _, mu, _, dark_rate, _, _, _, _, gof = np.loadtxt(
            self._fname, unpack=True
        )
        self.to_fit_again = {}
        for theta_i, pmt_i, phi_i, wavelength_i, mu_i, dark_rate_i, gof_i in zip(
            theta, pmt, phi, wavelength, mu, dark_rate, gof
        ):
            if gof_i == -1:
                mask = np.logical_and(theta == theta_i, pmt == pmt_i)
                mask = np.logical_and(mask, wavelength == wavelength_i)
                mask = np.logical_and(mask, gof != -1)
                nearest_mu, nearest_dark_rate = self._find_nearest_fit(
                    phi_i, phi[mask], mu[mask], dark_rate[mask]
                )
                if theta_i not in self.to_fit_again:
                    self.to_fit_again[theta_i] = {}
                self.to_fit_again[theta_i][(pmt_i, phi_i, wavelength_i)] = (
                    nearest_mu,
                    nearest_dark_rate,
                )


def process_unfitted_data(
    data_positions: Dict[float, PositionStage],
    ref_diode: ReferenceDiode,
    passing_trigger_data: PassingTriggerData,
    theta: float,
    pmt: int,
    phi: float,
    wavelength: float,
    p0mu: float,
    p0darkrate: float,
    output_filename: str,
):
    passing_fraction, passing_fraction_error = passing_trigger_data.get_trigger_loss(
        phi, pmt
    )
    pmt_data = data_positions[phi].PMT_data[pmt][wavelength]

    try:
        mu, err, dr, err_dr, gof = fit_filtered_data(
            pmt_data, passing_fraction, p0={"mu": p0mu, "dark_rate": p0darkrate, "norm_l": 50000.0, "norm_b": 300.0}
        )
        N_photons = (
            ufloat(0, 0)
            if wavelength == 400
            else -data_positions[phi].mean_ref_current[wavelength]
            * ref_diode.current_to_number[wavelength]
            / LASER_FREQ
        )
        with open(output_filename, "a") as f:
            for val in [theta, pmt, phi, wavelength]:
                f.write(f"{val}\t")
            for val in [
                np.sum(pmt_data.signal_h) + 1,
                mu,
                err,
                dr,
                err_dr,
                N_photons.n,
                N_photons.s,
                pmt_data.mes_time,
                gof,
            ]:
                f.write(f"{val:.5g}\t")
            f.write("\n")

    except Exception as err:
        print(err)
        pass


def replace_unfitted_lines(fname_fited: str, refited_lines: str):

    def process_line(line: str, refited_data: Dict):
        theta, pmt, phi, wavelength, N, mu, muerr, darkrate, darkrate_err, Nphot, Nerr, mes, gof, *rest  = line.split("\t")
        if gof == "-1":
            return refited_data[(float(theta), int(pmt), float(phi), float(wavelength))]
        else:
            return line

    new_data= np.loadtxt(refited_lines)
    refited_data = {}

    for (theta, PMT, phi, wavelength, N, mu, muerr, darkrate, darkrate_err, Nphot, Nerr, mes, gof) in new_data:
        if (theta, PMT, phi, wavelength) not in refited_data:
            refited_data[(theta, PMT, phi, wavelength)] = f"{theta}\t{PMT}\t{phi}\t{wavelength}\t{N}\t{mu}\t{muerr}\t{darkrate}\t{darkrate_err}\t{Nphot}\t{Nerr}\t{mes}\t{gof}\n"                   
    
    with open(fname_fited.replace(".dat", "_with_refit.dat"), "w") as f_new:
        with open(fname_fited, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = process_line(line, refited_data)
                f_new.write(line)

def fit_unfitted_data():

    fnames = [
        "output_data/" + fname
        for fname in os.listdir("output_data/")
        if ".pickle" in fname
    ]
    output_filename = f"{np.random.random_integers(1000,9999)}tmp.dat"
    ref_diode = ReferenceDiode(fname_reference_currents, fname_calibration_constants)
    passing_trigger_data = PassingTriggerData(passing_trigger_data_fname)
    unfited_data = UnfitedData(fname_fited)
    for theta in unfited_data.to_fit_again.keys():
        for fname in fnames:
            if str(theta) in fname:
                with open(fname, "rb") as handle:
                    if "scan1" in fname:
                        data_positions_1: Dict[float, PositionStage] = pickle.load(handle)
                    else:
                        data_positions_2: Dict[float, PositionStage] = pickle.load(handle)

        for (pmt, phi, wavelength), (p0mu, p0darkrate) in unfited_data.to_fit_again[theta].items():
            print(f"Processing: {theta}, {pmt}, {phi}, {wavelength}")
            try:
                process_unfitted_data(
                    data_positions_1,
                    ref_diode,
                    passing_trigger_data,
                    theta,
                    pmt,
                    phi,
                    wavelength,
                    p0mu,
                    p0darkrate,
                    output_filename
                )
            except:
                process_unfitted_data(
                    data_positions_2,
                    ref_diode,
                    passing_trigger_data,
                    theta,
                    pmt,
                    phi,
                    wavelength,
                    p0mu,
                    p0darkrate,
                    output_filename
                )
    replace_unfitted_lines(fname_fited, output_filename)
    os.remove(output_filename)
############################################################################################################
if __name__ == "__main__":
    #main()
    fit_unfitted_data()








