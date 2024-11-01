# mDOM Efficiency Measurement

The measurement system is split between two scripts due to hardware and software constraints. Since the NKT laser currently has an API only for Windows, and T-MDOM works best in linux, we do the measurement in a Windows PC with a Linux VM:

1. **Windows PC**
   - Run `control_main.py` to manage all components except the optical module:
     - NKT laser (Windows-only API)
     - Motion axes (uses this [GitHub repository](https://github.com/martinunland/FiveAxisSystem))
     - Picoamperemeter (uses this [GitHub repository](https://github.com/martinunland/picoamp_control))

2. **Linux VM**
   - Run `efficiency_measurement.py` to interface with T-mDOM (Check [GitHub repository](https://github.com/WIPACrepo/T-mDOM))
   - You have to add the tank_measure_charge test (see files in "T-mDOM-stuff/") to T-mDOM
   - Automatically handles communication timeouts by:
     - Powering off/on the wirepair
     - Reinitializing the mDOM
     - Continuing the measurement sequence

Both scripts run in parallel during measurements and communicate with each other by producing temporary files