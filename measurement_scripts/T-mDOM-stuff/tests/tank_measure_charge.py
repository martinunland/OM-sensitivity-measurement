#!/usr/bin/env python



import numpy as np
from numpy import histogram, array, min
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import time
from mDOM.util.general_configs import gen_config
from opticalFAT import tools
import os
from mDOM.tools.STM32Tools.python.RapCal import rapcal as rp
import threading
import gzip
import subprocess


DO_RUN = False
pC_to_pe_at_5e6_gain = 0.801088317
MDOM_ADC_RATE_MHZ = 120
ADC_SAMPLE_TIME_TO_NS = 1000. / (MDOM_ADC_RATE_MHZ)
ADC_SAMPLE_TIME_TO_S = ADC_SAMPLE_TIME_TO_NS * 1E-9
MDOM_DISC_RATE_MHZ = MDOM_ADC_RATE_MHZ * 8
PRECISE_TIME_TO_S = 1e-6 / (MDOM_DISC_RATE_MHZ)
SCALER_PERIOD = 1e6
DEADTIME_CYCLES = 25

EXT_OSCILLATOR_ENABLE = ["python3", "/home/martin/Documents/fh_server/scripts/external_oscillator_enable.py"]
WSL_MESSAGE_DIR = "/mnt/c/Users/AGKappesAdmin/Desktop/FiveAxisSystem-master/mssg_WSL"
NEXT_FILE = "siguiente"
TERMINATE_FILE = "terminate"
READY_FILE = "ready"
ERROR_FILE = "error_tmdom"

def mdom_request(mDOM, req):
    reply = mDOM.wirepair.icms.request(req)
    if reply['status'] != 'OK':
        raise RuntimeError(f'Received status "{reply["status"]}" on request "{req}"')
    if 'value' in reply:
        return int(reply['value'], 0)


def get_utc_icm_times(mDOM):
    utc_time = list()
    for reg in [0x2E, 0x2D, 0x2C]:
        utc_time.append(mdom_request(mDOM, f'read 8 {reg}'))
    icm_time = mdom_request(mDOM, 'get_icm_time 8')
    return utc_time, icm_time


def collect_rapcal(wirepair, rapcal_time):
    data = list()
    global DO_RUN
    while DO_RUN:
        for mDOM in wirepair.devices.values():
            mDOM.lock.acquire()
        for mDOM in wirepair.devices.values():
            wirepair.rapcal(mDOM._address)
        for mDOM in wirepair.devices.values():
            if len(mDOM.all_rapcals.rapcals) >= 2:
                rp_pair = rp.RapCalPair(mDOM.all_rapcals.rapcals[-2],
                                        mDOM.all_rapcals.rapcals[-1])
                timestamps = mDOM.timestamps_tmp.copy()
                
                mDOM.timestamps_tmp = dict()
                data.append((mDOM, rp_pair, timestamps))
        for mDOM in wirepair.devices.values():
            mDOM.lock.release()
        for d in data:
            for channel, t_dom in zip(d[2].keys(), d[2].values()):
                converted_time = d[1].dom2surface(t_dom,
                                                  device_type='MDOM_PRECISE')
                if channel not in d[0].timestamps_utc_sec.keys():
                    d[0].timestamps_utc_sec[channel] = converted_time
                else:
                    d[0].timestamps_utc_sec[channel] = np.concatenate(
                        (d[0].timestamps_utc_sec[channel], converted_time))

                d[0].rp1_Tc_dor.append(d[1].rc0.T_tx_dor)
                d[0].rp1_Tc_dom.append(d[1].rc0.T_tx_dom)
                d[0].rp2_Tc_dor.append(d[1].rc1.T_tx_dor)
                d[0].rp2_Tc_dom.append(d[1].rc1.T_tx_dom)

        data = list()
        time.sleep(rapcal_time)
    tff = time.time()
    
def collect_charges_and_timestamps(mDOM):
    global DO_RUN
    while DO_RUN:
        with mDOM.lock:
            channels = gen_config['channel_config']['all']['channels']
            cs_all, ts_all = mDOM.get_charge_spectrum(channels=channels,
                                                      trigger='disc',
                                                      nmeas=5, nsamples=64,
                                                      BlockWFMsize=int(65536/2),
                                                      timestamp=True,
                                                      use_hit_buffer= True)

            # Alternative: save all charges:
            for channel in sorted(cs_all.keys()):
                if channel not in mDOM.charges.keys():
                    mDOM.charges[channel] = cs_all[channel]
                    mDOM.timestamps[channel] = ts_all[channel]
                else:
                    mDOM.charges[channel] = np.concatenate((mDOM.charges[channel], cs_all[channel]))
                    mDOM.timestamps[channel] = np.concatenate((mDOM.timestamps[channel], ts_all[channel]))
                    
            for channel in sorted(cs_all.keys()):
                if channel not in mDOM.timestamps_tmp.keys():
                    mDOM.timestamps_tmp[channel] = ts_all[channel]
                else:
                    mDOM.timestamps_tmp[channel] = np.concatenate((mDOM.timestamps_tmp[channel], ts_all[channel]))
                    
        time.sleep(1.0e-3)
        
def write_measurement(mDOM, index):
    fname = f"/mnt/c/Users/AGKappesAdmin/Desktop/FiveAxisSystem-master/measurement_data/241013_scan_{index}"
    with open(fname+".txt", "a") as msfile:
        min_t = np.amin([np.amin(arr) for arr in mDOM.timestamps.values()])
        min_tp = np.amin([np.amin(arr) for arr in mDOM.timestamps_utc_sec.values()])
        msfile.write(f"#{min_t}\t{min_tp:.10f}\n")
        for channel in mDOM.charges.keys():
            msfile.write(f"#{channel}\n")
            ts = mDOM.timestamps[channel]-min_t
            ts_p = mDOM.timestamps_utc_sec[channel]-min_tp
            cs = mDOM.charges[channel]
            for t,tp, c in zip(ts, ts_p, cs):
                msfile.write(f"{c:.4f}\t{t}\t{tp:.10f}\n") 
                

def reset(mDOM):
    mDOM.charges = dict()
    mDOM.timestamps = dict()
    mDOM.timestamps_tmp = dict()
    mDOM.timestamps_utc_sec = dict()
    mDOM.transit_times = dict()
    mDOM.rp1_Tc_dor = list()
    mDOM.rp1_Tc_dom = list()
    mDOM.rp2_Tc_dor = list()
    mDOM.rp2_Tc_dom = list()

def single_measurement(mDOM, run_time, rapcal_time):
    global DO_RUN
    DO_RUN = True
    threads = list()
    subprocess.run(EXT_OSCILLATOR_ENABLE)
    time.sleep(2)
    threads.append(threading.Thread(target=collect_rapcal, args=[mDOM.wirepair, rapcal_time]))
    threads.append(threading.Thread(target=collect_charges_and_timestamps, args=[mDOM,]))

    for t in threads:
        t.start()
        
    time.sleep(run_time)
    DO_RUN = False
    for t in threads:
        t.join()
        

    for channel in mDOM.timestamps_utc_sec:
        print(f'{mDOM.getName()}, channel {channel}: collected {len(mDOM.all_rapcals.rapcals)} RapCals,'
                    f' {len(mDOM.timestamps[channel])} raw timestamps,'

                    f' {len(mDOM.timestamps_utc_sec[channel])} timestamps utc and {len(mDOM.charges[channel])} charges')
def configure(mDOM, baseline, trigger_config, channels, trigger_channel, discriminator_threshold):
    mDOM.iceboot_session.mDOMEnableHV()
    mDOM.iceboot_session.mDOMSetBaselines(baseline)
    if not channels:
        channels = gen_config['channel_config'][trigger_config]['channels']
    applied_HVs = mDOM.set_target_hv(channels=channels)

    if trigger_channel == "disc" and discriminator_threshold != "default":
        mDOM.set_discriminator_threshold(disc_threshold=discriminator_threshold)
    elif trigger_channel == "disc" and discriminator_threshold == "default":
        mDOM.set_discriminator_threshold(channels=channels)
    
def run(mDOM, nmeas, nsamples,starting_index, trigger_config, channels,
        baseline, laser_frequency, trigger_channel, discriminator_threshold, rapcal_time):
    t0 = time.time()
    mDOM.logger.info(f'Charge measurement running for channels: {channels}, discriminator setting {discriminator_threshold}')
    configure(mDOM, baseline, trigger_config, channels, trigger_channel, discriminator_threshold)
    reset(mDOM)
    not_finished = True
    file_index = starting_index
    run_time = 2*60
    
    while not_finished:
        while True:
            print("Waiting for next position")
            files = os.listdir(WSL_MESSAGE_DIR)
            if NEXT_FILE in files:
                time.sleep(0.1)
                os.remove(os.path.join(WSL_MESSAGE_DIR, NEXT_FILE))
                break
            elif TERMINATE_FILE in files:
                not_finished = False
                break
            time.sleep(1)
        print("Measuring")
        try:
            single_measurement(mDOM, run_time, rapcal_time)
            write_measurement(mDOM, file_index)
            reset(mDOM)
            file_index +=1
            open(os.path.join(WSL_MESSAGE_DIR, READY_FILE), "w").close()
            
        except Exception as err:
            print(err)
            open(os.path.join(WSL_MESSAGE_DIR, ERROR_FILE), "w").close()
            return #os._exit(0)
            

    mDOM.iceboot_session.mDOMDisableHV()
    t_final = time.time()


    mDOM.logger.info(f'Linearity measurement total program time = {(t_final - t0) / 60.} mn')


