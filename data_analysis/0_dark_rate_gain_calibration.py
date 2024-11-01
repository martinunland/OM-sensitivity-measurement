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
from scipy.special import gammaln
from iminuit import Minuit

def gaussian(x, mu, sigma, N):
    return N * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

class GaussianFit:
    def __init__(self, x, y):

        self.x = np.array(x)
        self.y = np.array(y)
        self._calculate_factorials()
        self.nr_fit_parameters = 3

    def _calculate_factorials(self):
        log_factorial = []
        for y in self.y:
            log_factorial.append(gammaln(y + 1))
        self.log_factorial = np.array(log_factorial)

    def llh(self, mdl, y, factorial):
        return np.nansum(-mdl + np.log(mdl) * y - factorial)

    def poisson_llh(self, mu, sigma, N):
        y_model = gaussian(self.x, mu, sigma, N)
        llh = -self.llh(y_model, self.y, self.log_factorial)
        return llh

    def goodness_of_fit(self, mu, sigma, N):
        llh_best_fit = self.poisson_llh(mu, sigma, N)
        llh_n = -self.llh(self.y, self.y, self.log_factorial)
        return -2 * (-llh_best_fit + llh_n) / (self.y.size - self.nr_fit_parameters)
    

def fit_gaussian(x, y, PMT):

    mup0 = 0.8 if PMT != 3 else 1.8
    mask = np.logical_and(x < 1.4, x > 0.2) if PMT != 3 else np.logical_and(x < 3.5, x > 1.0)
    fit = GaussianFit(x[mask], y[mask])
    m = Minuit(fit.poisson_llh, mu=mup0, sigma=0.4*mup0, N=np.amax(y))
    m.limits["mu"] = (0.00, 3)
    m.limits["sigma"] = (1e-9, 2)
    m.limits["N"] = (0.00, np.inf)
    m.errors["mu"] = 0.3
    m.errors["sigma"] = 0.1
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
    
def process_pickle(fname:str):
    output_filename = fname.replace(".pickle", "_gain_fits.dat")
    
    with open(output_filename, "w") as f:
        f.write("#PMT_nr \t phi \t theta \t wavelength \t mean_charge \t  mean_charge_err \t charge_width \t charge_width_err \n")
    with open(fname, 'rb') as handle:
        data_positions: Dict[float, PositionStage]  = pickle.load(handle)
        
    for PMT in range(0,24):
        for phi, data in data_positions.items():
            for wavelength, pmt_data in data.PMT_data[PMT].items():
                try:
                    mu, muerr, sigma, sigmaerr, gof = fit_gaussian(pmt_data.background_charge_x, pmt_data.background_charge_h, PMT)
                    np.savetxt(f"histogram_darkrate_{wavelength}_{PMT}_{phi}", np.array([pmt_data.background_charge_x, pmt_data.background_charge_h]).T)
                    print(mu, muerr, sigma, sigmaerr, gof)
                    # with open(output_filename, "a") as f:
                    #     for val in [PMT, phi, wavelength]:
                    #         f.write(f"{val}\t")
                    #     for val in [mu, muerr, sigma, sigmaerr, gof]:
                    #         f.write(f"{val:.5g}\t")
                    #     f.write("\n")
                    exit()
                except Exception as err:
                    print(err)
                    pass



def main():
    fnames = ["output_data/"+fname for fname in os.listdir("output_data/") if ".pickle" in fname]
    with Pool(processes=2) as pool:
        pool.starmap(process_pickle, [(fname, ) for fname in fnames])
    path = "/HDD/backuped/Promotion_data/Postdoc/240822_mDOM_efficiency/pre_analysis/output_data/"
    fnames = [path+fname for fname in os.listdir(path) if "_gain_fits.dat" in fname]

    # with open(path+"241014_gain_fits.dat", "a") as f:
    #     for fname in fnames:
    #             theta = fname.split("theta_")[1].split("_")[0]
    #             with open(fname, "r") as f2:
    #                 for line in f2:
    #                     if "#" not in line:
    #                         f.write(theta + "\t")
    #                         f.write(line)

if __name__ == "__main__":
    main()