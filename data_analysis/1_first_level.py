from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Tuple
from pre_analysis.src.single_file_processor import SingleFileProcessor
from src.analysis_mDOM_position import PositionStage
from src.reference_diode import ReferenceDiode
import numpy as np
import pickle
from multiprocessing import Pool

fname_calibration_constants = "../light_transmission/ref_current_to_number_of_photons.dat"

fname_measurement_stages = "/data/240823_effective_area_DVT09_Martin/measurement_data_scan_2/240904_theta_phi_scan__state.dat"
fname_reference_currents = "/data/240823_effective_area_DVT09_Martin/measurement_data_scan_2/240904_theta_phi_scan__diode.dat"
data_fname_prefix = "/data/240823_effective_area_DVT09_Martin/measurement_data_scan_2/240904_theta_phi_scan_"

fname_measurement_stages = "/data/240823_effective_area_DVT09_Martin/measurement_data_scan/240828_theta_phi_scan__state.dat"
fname_reference_currents = "/data/240823_effective_area_DVT09_Martin/measurement_data_scan/240828_theta_phi_scan__diode.dat"
data_fname_prefix = "/data/240823_effective_area_DVT09_Martin/measurement_data_scan/240828_theta_phi_scan_"

output_path_and_prefix = "./output_data/241009_scan2_theta_"
output_path_and_prefix = "./output_data/241009_scan1_theta_"

@dataclass
class MeasurementStageInfo:
    """Dataclass representing measurement stage data."""
    time: List[float]
    phi: List[float]
    theta: List[float]
    wavelength: List[float]
    
    @classmethod
    def from_file(cls, fname: str) -> 'MeasurementStageInfo':
        """Load stage data from a text file and return a MeasurementStageInfo instance."""
        stages_data = np.loadtxt(fname, unpack=True)
        return cls(*stages_data)
    
def pickle_processed_data(data_positions: DefaultDict[float, PositionStage], current_theta: float):
    with open(f'{output_path_and_prefix}{current_theta}.pickle', 'wb') as handle:
        pickle.dump(data_positions, handle, protocol=pickle.HIGHEST_PROTOCOL)

def process_data_file(time: float, phi: float, wavelength: float, fname: str, ref_diode: ReferenceDiode) -> Tuple[SingleFileProcessor, float]:
    datafile = SingleFileProcessor(fname, time)  
    reference_current = ref_diode.get_mean(
        datafile.starting_time,
        datafile.final_time,
        wavelength
    )
    return datafile, reference_current

def process_theta(current_theta, stages: MeasurementStageInfo, ref_diode: ReferenceDiode):
    
    data_positions : Dict[float, PositionStage] = {}
    current_phi = -1
    print("Processing theta:", current_theta)
    for file_index, (time, phi, theta, wavelength) in enumerate(zip(stages.time, stages.phi, stages.theta, stages.wavelength)):

        if theta != current_theta:
            continue
        if phi != current_phi:
            if current_phi!= -1:
                data_positions[current_phi].process_stage() # Process the previous phi stage as it should be complete (all phis are measured before changing theta)
            current_phi = phi

        if phi not in data_positions:
            data_positions[phi] = PositionStage()

        fname = f"{data_fname_prefix}{file_index}.txt"
        datafile, reference_current = process_data_file(time, phi, wavelength, fname, ref_diode)
        data_positions[phi].add_single_file_data(wavelength, datafile)
        data_positions[phi].add_reference_current(wavelength, reference_current)

    data_positions[current_phi].process_stage() # Process the last phi stage
    pickle_processed_data(data_positions, current_theta)
        
def main():
    stages = MeasurementStageInfo.from_file(fname_measurement_stages)
    unique_thetas = np.unique(stages.theta)

    ref_diode = ReferenceDiode(fname_reference_currents, fname_calibration_constants)

    with Pool(processes=3) as pool:
        pool.starmap(process_theta, [(theta, stages, ref_diode) for theta in unique_thetas])

if __name__ == "__main__":
    main()