# Restful Knights

## Overview

 Restful Knights is a comprehensive sleep analysis and data collection platform. It integrates hardware sensor data (Plux Biosignals EEG module, MAX30102 blood oxygen sensor) with machine learning models and a modern PyQt5 user interface. The project supports both real-time data acquisition and post-hoc analysis, leveraging custom GPU kernels and LLM-based reporting. Data acquisition is performed using an Arduino Nano ESP32.

## Features

- **EEG/SpO₂ Data Collection**: Plux Biosignals EEG module and MAX30102 blood oxygen sensor, streamed via Arduino Nano ESP32 and logged to CSV.
- **Sleep Stage Classification**: PyTorch convolutional neural network (CNN) models for sleep stage prediction, compiled with Triton for better performance.
- **Custom GPU Kernels**: CUDA/ROCm kernels for efficient neural network operations.
- **PyQt5 GUI**: User-friendly interface for data visualization and analysis.
- **LLM Integration**: Automated sleep report generation using Ollama LLM.
- **Docker Support**: Easy setup for GPU-accelerated environments.

## Directory Structure

```
Model/              # ML models, UI, LLM client, Docker setup
DataCollection/     # Arduino code, serial acquisition, plotting
KernelExperiment/   # Custom CUDA/ROCm kernels and experiments
```

## Setup

### Prerequisites

- Python 3.8+
- Docker (with GPU support: ROCm)S
- Arduino IDE (for sensor firmware)
- PyQt5, PyTorch, MNE, OpenCV, scikit-learn, ollama, beautifulsoup4, markdown

### Quick Start (Docker)

1. Build and run the Docker container:
    ```sh
    ./Model/run.sh
    ```

2. The GUI will launch and connect to your hardware (if available).

### Manual Python Setup

1. Install dependencies:
    ```sh
    pip install -r Model/requirements.txt
    ```

2. Run the main application:
    ```sh
    python Model/main.py
    ```

## Usage

- **Data Collection**: Flash `Sensors.ino` to your Arduino, connect sensors, and start the DataCollection UI.
- **Sleep Analysis**: Use the Model UI to load data, run analysis, and view results.
- **Custom Kernels**: See `KernelExperiment/` for advanced GPU experiments.

## Custom Kernel Performance

The custom CUDA/ROCm kernel implementation for Conv2d weight gradients (`my_custom_wrw_kernel`) reduced the time per call by approximately 50% compared to the standard fallback kernel. However, replacing the function that normally autoroutes to the optimal kernel caused all calls to be routed through the fallback replacement instead of the optimal implementation. Normally, the deep learning framework automatically selects from many optimal kernels depending on the hardware and configuration, but with this replacement, every convolution weight gradient calculation was handled by the custom kernel. While this kernel was faster than the fallback, it was much slower than the optimal kernels, resulting in increased total run time.

### Stats Comparison

- **Custom Kernel (`new.stats.csv`)**:
  - Kernel: `my_custom_wrw_kernel`
  - Calls: 928
  - Total Time: 26,710,813,140 ns
  - Avg Time/Call: 28,783,203 ns
  - **Total Run Time (all kernels):** 42.97 seconds
  - **Test Hardware:** AMD Radeon 7900XT GPU, AMD Ryzen 7700X CPU, 32GB DDR5 RAM

- **Standard Kernel (`standard.stats.csv`)**:
  - Kernel: `naive_conv_ab_nonpacked_wrw_nchw_float_double_float.kd`
  - Calls: 48
  - Total Time: 2,618,132,317 ns
  - Avg Time/Call: 54,544,423 ns
  - **Total Run Time (all kernels):** 37.78 seconds
  - **Test Hardware:** AMD Radeon 7900XT GPU, AMD Ryzen 7700X CPU, 32GB DDR5 RAM

See `KernelExperiment/new.stats.csv` and `KernelExperiment/standard.stats.csv` for full profiling details.

## Future Plans

- **PyTorch Source Integration:**
  Future work includes investigating how to integrate the custom convolution weight gradient kernel directly into the PyTorch source code. This will require understanding PyTorch's kernel dispatch system, registering the kernel as a backend, and compiling PyTorch from source with these changes.
- **Compilation and Benchmarking:**
  After integration, the kernel will be benchmarked against PyTorch's built-in kernels in real training scenarios to evaluate performance and correctness.
- **Additional Biosignals**
  Integration of the biosignals, including ECG and breath rate, would significantly increase the accuracy of the screening. The ECG analysis could utilize a data algorithm similar to the existing EEG convolutional neural network, while the breathing rate one could be modeled after the blood oxygen algorithm.


## Sleep-EDF Database
- The following database was used for as training and testing our model
- Link: https://www.physionet.org/content/sleep-edfx/1.0.0/sleep-cassette/#files-panel

## Special Thanks to AMD
We extend our deepest gratitude to AMD for their exceptional support of the hackathon and for providing the critical resources that powered our project to win Best Overall Hack!
