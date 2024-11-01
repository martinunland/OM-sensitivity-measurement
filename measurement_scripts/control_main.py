from src.FAS import FiveAxisSystem, Position
from src.serialcontroller import SerialController
from src.utils_constants import Units
from picoamp_control import PicoampControl
import logging
import numpy as np
import time
import os
from LLTF_Contrast_Module import LLTFContrast
import requests

WSL_MESSAGE_DIR = "./mssg_WSL"
READY_FILE = "ready"
ERROR_FILE = "error_tmdom"
NEXT_FILE = "siguiente"
TERMINATE_FILE = "terminate"
WINDOWS_ERROR_FILE = "error_in_windows"

MAX_TIME = 20*60

def send_telegram_message(message):
    chat_id = 6289386472
    with open("telegram_token.txt", "r") as f:
        TOKEN = f.read()
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    print(requests.get(url).json()) # this sends the message
        
        
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

controller = SerialController("COM5", timeout=600)
FAS = FiveAxisSystem(controller)
FAS.speed = int(200)
FAS.acceleration = int(50)

myLLTF = LLTFContrast()
myLLTF.Create(conffile=r'C:\Users\AGKappesAdmin\Documents\PHySpec Installer\configuration\Devices\M000010304.xml')
system_name = myLLTF.GetSystemName(index=0)[1]
myLLTF.Open(system_name)
myLLTF.SetWavelength(400)

pico = PicoampControl()
pico.connect(com="COM10")
pico.auto_config()

class CoordinateSystem:
    Y_POS = 61.07
    _mDOM_x = None
    _mDOM_z = None
    _R = None
    _tilt_shift = 3.0
    def set_fibre_distance(self, distance):
        self._R = distance
        
    def set_mDOM_position(self, x, z):
        self._mDOM_x = x
        self._mDOM_z = z
        
    def _theta_to_x_z(self, theta):
        x = self._R*np.sin(np.deg2rad(theta))
        z = self._R*np.cos(np.deg2rad(theta))
        return x,z
    
    def _mDOM_to_global_coordinates(self, x, z):
        x_g = self._mDOM_x-x
        z_g = self._mDOM_z-z #z=0 at top
        return x_g, z_g

    def _get_light_output_position_relative_to_axis(self, tilt: float):
        x_at_zero_tilt = 7.0
        z_at_zero_tilt = 4.43
        distance_to_rotation_axis = np.sqrt(x_at_zero_tilt**2 + z_at_zero_tilt**2)
        phi_at_zero_tilt = np.arctan2(z_at_zero_tilt, x_at_zero_tilt)
        end_phi = phi_at_zero_tilt+np.deg2rad(tilt)
        x_after_rotation = distance_to_rotation_axis*np.cos(end_phi)
        z_after_rotation = distance_to_rotation_axis*np.sin(end_phi)
        return -x_after_rotation, z_after_rotation
    
    def calculate_position(self, theta: float, phi: float)-> Position:
        assert self._mDOM_x is not None and self._mDOM_z is not None, "call set_mDOM_position() and set the global coordinates of the module first!"
        assert self._R is not None, "call set_fibre_distance() and set distance fibre-mdom centre"
         
        
        
        x,z = self._theta_to_x_z(theta) #Light output should be at these coordinates w.r.t. mDOM centre as 0,0
        x,z = self._mDOM_to_global_coordinates(x,z)

        x_shift, z_shift = self._get_light_output_position_relative_to_axis(theta) #In current setup, tilt value exactly the same as theta from spherical coordinates with mDOM at centre
        x -= x_shift
        z -= z_shift
        
        tilt = theta + self._tilt_shift # Shift in axis
        return Position(x, self.Y_POS, z, phi, tilt)
        
def log_current(log_file):
    start = time.time()
    (c1,c1e), (c2,c2e) = pico.get_mean_current(20)
    stop = time.time()
    mid_time = (stop+start)*0.5 
    with open(log_file, "a") as f:
        f.write(f"{mid_time}\t{c1}\t{c1e}\n")
        
def log_state(theta, phi, log_file):
    with open(log_file, "a") as f:
        f.write(f"{time.time()}\t{theta}\t{phi}\n")
    
    
def wait_measurement(pd_log_file):
    nr_errors = 0
    start = time.time()
    while True:
        if time.time()-start > MAX_TIME:
            files = os.listdir(WSL_MESSAGE_DIR)
            if NEXT_FILE in files:
                send_telegram_message(f"T-mDOM dead?, terminating...")
                open(os.path.join(WSL_MESSAGE_DIR, TERMINATE_FILE), "w").close()
                exit()
                
            send_telegram_message(f"Max time over, trying to kick next measurement")
            open(os.path.join(WSL_MESSAGE_DIR, NEXT_FILE), "w").close()
            
        print("Waiting for measurement to finish")
        log_current(pd_log_file)
        files = os.listdir(WSL_MESSAGE_DIR)
        if READY_FILE in files:
            time.sleep(0.1)
            os.remove(os.path.join(WSL_MESSAGE_DIR, READY_FILE))
            return
        elif ERROR_FILE in files:
            os.remove(os.path.join(WSL_MESSAGE_DIR, ERROR_FILE))
            open(os.path.join(WSL_MESSAGE_DIR, NEXT_FILE), "w").close()
            nr_errors += 1
            send_telegram_message(f"T-mDOM failed, consecutive fails: {nr_errors}")
            if nr_errors>2:
                send_telegram_message(f"T-mDOM dead, terminating...")
                open(os.path.join(WSL_MESSAGE_DIR, TERMINATE_FILE), "w").close()
                exit()
                
cs = CoordinateSystem()
cs.set_mDOM_position(301.78, 69.59)
cs.set_fibre_distance(100+17.8)

skip_to = -1#1550 #in case of script restart
SKIPPING = skip_to>0
BACKGROUND_INTERVAL = 150
counter = -1

send_telegram_message(f"Starting {counter}")
wavelengths = np.arange(500, 640, 40)

thetas, phis = np.loadtxt("angles_to_measure.txt", unpack=1) #In my measurement I sorted the angles such, that for a theta all phis are measured first before changing theta
total_iterations = thetas.size*wavelengths.size
preffix = "240904_theta_phi_scan_"
pd_log_file = f"{preffix}_diode.dat"
state_log_file = f"{preffix}_state.dat"
positions_file = f"{preffix}_log_positions.txt"

try:
    with tqdm(total=total_iterations, desc="Overall Progress", position=0) as pbar_overall:
        for theta, phi in zip(thetas, phis):
            theta = round(theta,2)
            phi = round(phi,2)
            position = cs.calculate_position(theta, phi)
            if not SKIPPING:
                FAS.move_absolute_position(position=position)
                FAS.log_current_position(positions_file)
                time.sleep(5)
            
            for wavelength in wavelengths:
                counter += 1
                if SKIPPING and counter >= skip_to:
                    FAS.move_absolute_position(position=position)
                    FAS.log_current_position(positions_file)
                    time.sleep(5)
                    SKIPPING = False

                if SKIPPING:
                    if counter % BACKGROUND_INTERVAL ==0:
                        counter += 1
                    pbar_overall.update(1)
                    continue

                myLLTF.SetWavelength(wavelength)
                log_state(wavelength, phi, theta, state_log_file)
                log_current(pd_log_file, wavelength)
                open(os.path.join(WSL_MESSAGE_DIR, NEXT_FILE), "w").close()
                wait_measurement(pd_log_file,wavelength, counter, pbar_overall)
                pbar_overall.update(1)
                
                if counter % BACKGROUND_INTERVAL ==0:
                    send_telegram_message(f"Measuring background, current index {counter}")
                    wavelength = 400
                    myLLTF.SetWavelength(wavelength)
                    log_state(wavelength, phi, theta, state_log_file)
                    log_current(pd_log_file, wavelength)
                    counter+=1
                    open(os.path.join(WSL_MESSAGE_DIR, NEXT_FILE), "w").close()
                    wait_measurement(pd_log_file,wavelength, counter, pbar_overall)
                    
        for wavelength in wavelengths:
            myLLTF.SetWavelength(wavelength)       
            log_current(pd_log_file, wavelength)

        open(os.path.join(WSL_MESSAGE_DIR, TERMINATE_FILE), "w").close()
    
except Exception as err:
    print(err)
    send_telegram_message(f"Error in windows, terminating...")
    open(os.path.join(WSL_MESSAGE_DIR, TERMINATE_FILE), "w").close()