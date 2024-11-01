import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from src.constants import DISCRIMINATOR_CLOCK, CHUNK_SPACING_IN_SECONDS
from scipy.special import gammaln

@dataclass
class ChannelChunkData:
    """Container for processed PMT channel data within a time chunk."""
    charge: np.ndarray
    timestamp: np.ndarray
    timestamp_utc: np.ndarray
    t_utc_min: float
    t_utc_max: float
        
class SingleFileProcessor:
    """Processes single data files into analyzable chunks."""

    def __init__(self, fname:str, time:float):
        self.signal = []
        self.background = []
        self.data_chunks:List[Dict[int, ChannelChunkData]] = []
        self.starting_time = time
        self.process_file(fname)
        self.recalculate_mins_max()
        
        
    def process_appended_data(self):
        """Sorts and chunks the accumulated data for the current channel.
        Splits data into chunks when time gaps > 0.05s (CHUNK_SPACING_IN_SECONDS) are detected.
        """

        argsort = np.argsort(self.__timestamps)
        utc_times = np.array(self.__utc_times)[argsort]
        charges = np.array(self.__charges)[argsort]
        timestamps = np.array(self.__timestamps)[argsort]/DISCRIMINATOR_CLOCK # Converting to seconds

        ts2_diff = np.diff(utc_times)
        split_indices = np.where(ts2_diff > CHUNK_SPACING_IN_SECONDS)[0] + 1
        starts = np.insert(split_indices, 0, 0)
        ends = np.append(split_indices, utc_times.size)
        
        for i, (start, end) in enumerate(zip(starts, ends)):
            if len(self.data_chunks)<=i:
                self.data_chunks.append({})
            self.data_chunks[i][self.__current_channel] = ChannelChunkData(charges[start:end], 
                                                       timestamps[start:end], 
                                                       utc_times[start:end],
                                                       np.amin(utc_times[start:end]),
                                                       np.amax(utc_times[start:end]))
    
    def parse_initial_line(self, line: str):
        """Extracts initial timestamp and UTC time from file header."""
        # Getting first timestamp, saved timestamps are relative to the initial timestamp to save space
        self.__t0_timestamp = int(line.split()[0].split("#")[1]) 
        self.__t0_utc = np.float128(line.split()[1])

    
    def parse_data_line(self, line: List[str]):
        """Parses a single data line containing charge and timing info."""
        charge, timestamp, utc_time = line.split()
        self.__charges.append(float(charge))
        self.__timestamps.append(int(timestamp) + self.__t0_timestamp) # Saved timestamps are relative to the initial timestamp to save space
        self.__utc_times.append(np.float128(utc_time) + self.__t0_utc)
    
    def process_file(self, fname: str):
        """Reads and processes the data file line by line.
        Format expected:
        - First line: initial timestamp and UTC
        - Second line: first channel number
        - Following lines: charge, timestamp, UTC time
        """
        with open(fname, "r") as f:
            self.parse_initial_line(next(f)) # Initial line contains the first timestamp and the first UTC time
            self.__current_channel = int(next(f).split("#")[1]) #second line contains the first channel
            self.__timestamps, self.__utc_times, self.__charges = [], [], []
            for line in f:
                if "#" in line:
                    self.process_appended_data()
                    self.__current_channel = int(line.split("#")[1])
                    self.__timestamps, self.__utc_times, self.__charges = [], [], []
                else:
                    self.parse_data_line(line)

            self.process_appended_data()
            
    def recalculate_mins_max(self):
        ''' Recalculates the global minimum and maximum UTC times across all chunks and adjusts the chunk bounds accordingly.'''
        global_min = np.inf
        global_max = -np.inf
        for chunk in self.data_chunks:
            mint = np.min([channel_data.t_utc_min for channel_data in chunk.values()])
            maxt = np.max([channel_data.t_utc_max for channel_data in chunk.values()])
            global_max = max(maxt, global_max)
            global_min = min(mint, global_min)
            for channel_data in chunk.values():
                channel_data.t_utc_min = mint
                channel_data.t_utc_max = maxt
        self.final_time = self.starting_time+(global_max-global_min)