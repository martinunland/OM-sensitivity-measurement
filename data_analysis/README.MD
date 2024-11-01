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
│── 0_dark_rate_gain_calibration.py  # Dark rate and gain analysis
│── 1_first_level.py      # Initial data processing
│── 2_second_level.py     # Advanced analysis and fitting
│── compile_summary_files.py  # Results compilation
```

## Key Components

### Data Processing Pipeline

0. **Calibration** (`0_dark_rate_gain_calibration.py`):
   - Performs PMT gain calibration with dark rate charge
   - Fits single Gaussian to charge distributions
    If this step is performed, check further analysis in "mDOM_passing_threshold" notebook

1. **First Level Processing** (`1_first_level.py`):
   - Reads raw measurement data
   - Processes measurement stages (theta-phi positions)
   - Outputs pickled data for each theta position

2. **Second Level Processing** (`2_second_level.py`):
   - Performs detailed signal analysis
   - Applies trigger loss corrections
   - Fits signal and background models
   - Generates summary data files

## Constants and Parameters

Key parameters in `constants.py`:
- Laser frequency: 4000 Hz
- Discriminator clock: 960 MHz
- ADC clock: 120 MHz
- Time window: 10μs
