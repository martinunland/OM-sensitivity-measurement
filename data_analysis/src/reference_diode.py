import numpy as np
from uncertainties import ufloat

class ReferenceDiode:
    def __init__(self, fname: str, calibration_constants_file: str):
        self.time,self.wavelengths, self.vals, self.error = np.loadtxt(fname, unpack=1)
        x,y,_,_,_ = np.loadtxt(calibration_constants_file)#"light_transmission/ref_current_to_number_of_photons.dat")
        self.current_to_number = {x[i]: y[i] for i in range(x.size)}
        #self.calculate_background()
        self.background = ufloat(0,0)
    def calculate_background(self):
        mask = self.wavelengths == 400
        currents = np.average(self.vals[mask], weights = 1/self.error[mask]**2)
        standard_deviation = np.sqrt(np.average((self.vals[mask]-currents)**2, weights = 1/self.error[mask]**2))
        self.background = ufloat(currents, standard_deviation/np.sqrt(self.vals[mask].size))
        print("background:", self.background)
        
    def get_mean(self, t1: float, t2: float, wavelength: float):   
        mask = np.logical_and(self.time>t1, self.time<t2)
        mask = np.logical_and(self.wavelengths == wavelength, mask)
        currents = np.average(self.vals[mask], weights = 1/self.error[mask]**2)
        standard_deviation = np.sqrt(np.average((self.vals[mask]-currents)**2, weights = 1/self.error[mask]**2))
        toreturn = ufloat(currents, standard_deviation/np.sqrt(self.vals[mask].size))-self.background
        if len(self.vals[mask])<3:
            print(f"Warning, small ref current set: {len(self.vals[mask]<3)}")
            
        return toreturn
    
    def get_number_photons(self, t1: float, t2: float, wavelength: float):
        current = self.get_mean(t1,t2,wavelength)
        return current*self.current_to_number[wavelength]
    