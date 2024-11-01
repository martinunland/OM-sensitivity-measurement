# Analysis Pipeline

This project provides a data analysis pipeline for processing and analyzing measurements in the water tank setup.

## Overview

The pipeline processes raw data from mDOM measurements, handling:
- Dark rate and gain calibration
- PMT signal/background separation
- Laser pulse detection and timing analysis
- Reference diode measurements

## Project Structure

```
├── src/
│   ├── constants.py           # Global constants and parameters
│   ├── helper.py             # Utility functions for data processing
│   ├── phase.py              # Laser phase calculation and analysis
│   ├── fit.py                # Fitting algorithms and models
│   ├── single_file_processor.py  # Raw data file processing
│   ├── analysis_mDOM_position.py # module position-dependent analysis
│   └── reference_diode.py    # Reference diode data handling
│── 0_process_raw_files.py  # Initial data processing
│── 1_calibrate_gain.py      # Dark rate and gain analysis
│── 2_analyse_detection.py     # Further analysis and fitting
│── compile_summary_files.py  # Results compilation
```

## Key Components

### Data Processing Pipeline

1. **First Level Processing** (`0_process_raw_files.py`):
   - Reads raw measurement data
   - Processes measurement stages (theta-phi positions)
   - Outputs pickled data for each theta position

0. **Calibration** (`1_calibrate_gain.py`):
   - Performs PMT gain calibration with dark rate charge
   - Fits single Gaussian to charge distributions
    If this step is performed, check further analysis in "notebooks/mDOM_passing_threshold" notebook

2. **Second Level Processing** (`2_analyse_detection.py`):
   - Performs detailed signal analysis
   - Applies trigger loss corrections
   - Fits signal and background models
   - Generates summary data files

## Further analysis

The data processed by the pipeline, is further investigated in the notebooks inside the folder "notebooks".
Data_vs_MC.ipynb demonstrates how the data and simulation are compared.
