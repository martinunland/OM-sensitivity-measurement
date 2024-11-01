from src.constants import LASER_FREQ, TIME_WINDOW
from typing import List, Dict
import numpy as np
import gc
from src.single_file_processor import SingleFileProcessor, ChannelChunkData
from src.phase import Phases
from src.helper import create_laser_mask, create_laser_mask_interp, get_likely_laser_pulses  
import uncertainties

def edge_to_x(edge:np.ndarray)->np.ndarray:
    return 0.5*(edge[:-1]+edge[1:])

class FilteredData:
    """Class to store separated and processed signal/background data for a single PMT channel and wavelength"""
    def __init__(self):
        self.background_timestamps: List[float]=  []
        self.signal_timestamps: List[float] = []
        self.signal_charges: List[float] = []
        self.background_charges: List[float] = []
        self.deviation : List[float] =  []
        self.mean_phase : List[float] =  []
        self.mid_time : List[float] =  []
        self.mes_time = 0

    def set_laser_freq(self, laser_freq):
        self.laser_freq = laser_freq
        
    def set_time_window(self, time_window):
        self.time_window = time_window

    def append_separated_data(self, charges:List[float], timestamps:List[float], signal_mask: List[bool], deviation: List[float], mes_time: float, mean_phase: float, mid_time: float):
        """Appends separated signal and background data to the class"""
        self.deviation.extend(deviation)
        self.signal_timestamps.extend(timestamps[signal_mask])
        self.background_timestamps.extend(timestamps[~signal_mask])
        self.signal_charges.extend(charges[signal_mask])
        self.background_charges.extend(charges[~signal_mask])
        self.mean_phase.append(mean_phase)
        self.mid_time.append(mid_time)
        self.mes_time += mes_time
    
    def summarise_data_and_clean(self):
        """Summarises the data and cleans the raw data to avoid memory issues"""
        wait_time = np.diff(np.sort(self.signal_timestamps))/(1/self.laser_freq)
        self.signal_h, bins  = np.histogram(wait_time, bins = np.arange(0.5, 80+.5, 1) ) 
        self.signal_x = edge_to_x(bins)
        
        background_dt = np.diff(np.sort(self.background_timestamps))
        background_dt = background_dt[background_dt>0]
        self.background_h, bins = np.histogram(np.log10(background_dt), bins = "sqrt", range=(-4., -1.5))
        self.background_x = edge_to_x(bins)
        
        self.deviation_h, bins = np.histogram(self.deviation, bins = 200, range = (-10e-6, 10e-6))
        self.deviation_x = edge_to_x(bins)
        
        self.n_background = len(self.background_timestamps)
        self.n_signal = len(self.signal_timestamps)
        
        self.signal_charge_h, bins = np.histogram(self.signal_charges, bins = "sqrt", range = (0,5))
        self.signal_charge_x = edge_to_x(bins)

        self.background_charge_h, bins = np.histogram(self.background_charges, bins = "sqrt", range = (0,5))
        self.background_charge_x = edge_to_x(bins)  

        del self.background_timestamps, self.signal_timestamps, self.deviation, self.background_charges, self.signal_charges
        gc.collect()
        
class PositionStage: 
    """Class to process data (all wavelengths and PMTs) from a single mDOM/fibre position stage"""
    def __init__(self):
        self.wavelength_data_file : Dict[float, SingleFileProcessor] = dict() #All data from files, will be later filtered and cleaned
        self.mean_ref_current : Dict[float, uncertainties.core.Variable] = dict() #Mean reference current for each wavelength
        self.PMT_data : Dict[int, Dict[float, FilteredData]]= { i: {} for i in range(24)} #Filtered data for each channel and wavelength
        self.set_laser_freq()
        self.set_time_window()

    def set_laser_freq(self, laser_freq: float = LASER_FREQ):
        self.laser_freq = laser_freq
        
    def set_time_window(self, time_window: float = TIME_WINDOW):
        self.time_window = time_window

    def add_single_file_data(self, wavelength: float, data_file: SingleFileProcessor):
        self.wavelength_data_file[wavelength] = data_file

    def add_reference_current(self, wavelength: float, current: uncertainties.core.Variable):
        self.mean_ref_current[wavelength] = current

    def process_stage(self):
        self.calculate_laser_phase()
        self.filter_data_and_clean()
        
    def calculate_laser_phase(self):
        phases = Phases()
        phases.set_laser_freq(self.laser_freq)

        for wavelength, data_file in self.wavelength_data_file.items():
            if wavelength == 400: #There is no light from the laser at 400 nm, so we cannot use it for the laser phase
                continue
            for chunk in data_file.data_chunks:
                likely_pulses = []
                for channel_chunk in chunk.values():
                    likely_pulses.extend(get_likely_laser_pulses(channel_chunk.timestamp, self.laser_freq))
                phases.process_likely_pulses(likely_pulses)
        
        phases.filter_outliers()
        self.phase_interpolator = phases.get_interpolator()
        self.mean_phase, self.mean_phase_err = phases.get_mean_phase()

    def filter_data_and_clean(self):

        for wavelength, data_file in self.wavelength_data_file.items():
            for chunk in data_file.data_chunks:
                for channel, ch_chunk in chunk.items():
                    self._filter_channel_data(channel, wavelength, ch_chunk)
                        
        for channel in self.PMT_data.values():
            for wavelength_data in channel.values():
                wavelength_data.summarise_data_and_clean()
            
        self.wavelength_data_file.clear()
        del self.wavelength_data_file
        gc.collect()
        
    def _filter_channel_data(self, channel: int, wavelength: float, ch_chunk: ChannelChunkData):
        if wavelength not in self.PMT_data[channel]:
            self.PMT_data[channel][wavelength] = FilteredData()
            self.PMT_data[channel][wavelength].set_laser_freq(self.laser_freq)
            self.PMT_data[channel][wavelength].set_time_window(self.time_window)

        mask1, deviation1, mean_phase1 = create_laser_mask(ch_chunk.timestamp, self.mean_phase, self.laser_freq, self.time_window)
        mask2, deviation2, mean_phase2 = create_laser_mask_interp(ch_chunk.timestamp, self.phase_interpolator, self.laser_freq, self.time_window)
        bool_choose1 = np.mean(deviation1)<np.mean(deviation2)
        mask = mask1 if bool_choose1 else mask2
        deviation = deviation1 if bool_choose1 else deviation2
        mean_phase = mean_phase1 if bool_choose1 else mean_phase2

        phase = ch_chunk.t_utc_min % (1/self.laser_freq)
        mid_time = (ch_chunk.t_utc_min+ch_chunk.t_utc_max)/2

        if np.abs(phase-self.mean_phase) < self.time_window:
            t0 = ch_chunk.t_utc_min
        else:
            time_to_next = (1/self.laser_freq)-phase
            t0 = ch_chunk.t_utc_min+time_to_next
            
        mes_time = ch_chunk.t_utc_max-t0
        self.PMT_data[channel][wavelength].append_separated_data(
            ch_chunk.charge, ch_chunk.timestamp, mask, deviation, mes_time, mean_phase, mid_time
        )

