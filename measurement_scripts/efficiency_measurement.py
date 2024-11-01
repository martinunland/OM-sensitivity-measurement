import subprocess
import os
import time
import json
import signal


WSL_MESSAGE_DIR = "/mnt/c/Users/AGKappesAdmin/Desktop/FiveAxisSystem-master/mssg_WSL"
RESTART_TMDOM_FILE = "restart_tmdom"
TERMINATE_FILE = "terminate"

RUN_TMDOM = ["python3", "run_test.py", "Tank_measure_charge"]

WP_ON = ["python3", "/home/martin/Documents/fh_server/scripts/wp_on.py"]
WP_OFF = ["python3", "/home/martin/Documents/fh_server/scripts/wp_off.py"]

EXT_OSCILLATOR_ENABLE = [
    "python3",
    "/home/martin/Documents/fh_server/scripts/external_oscillator_enable.py",
]


def update_index(index, file_path="testconfig/tank_measure_charge.json"):
    with open(file_path, "r") as f:
        data = json.load(f)
    data["args"]["starting_index"] = int(index)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def run_script(script):
    process = subprocess.Popen(script, preexec_fn=os.setsid)
    try:
        process.wait()
    except KeyboardInterrupt:
        print(f"Script {script} interrupted by user")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        raise  # Re-raise the KeyboardInterrupt
    finally:
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    return process.returncode


#update_index(1550) #in case of script restart
run_script(RUN_TMDOM)

while True:

    files = os.listdir(WSL_MESSAGE_DIR)
    if RESTART_TMDOM_FILE in files:
        with open(os.path.join(WSL_MESSAGE_DIR, RESTART_TMDOM_FILE), "r") as f:
            index = f.read()
        update_index(index)
        os.remove(os.path.join(WSL_MESSAGE_DIR, RESTART_TMDOM_FILE))
        for script in [WP_OFF, WP_ON, EXT_OSCILLATOR_ENABLE, RUN_TMDOM]:
            print("Running", script)
            run_script(script)
            time.sleep(2)

    if TERMINATE_FILE in files:
        run_script(WP_OFF)
        os.remove(os.path.join(WSL_MESSAGE_DIR, TERMINATE_FILE))
        break
