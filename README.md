# Smarter Canes: Low-Cost Sensing for Step Counting and Self-Supervised Gait Assessment

A low-cost instrumented walking cane — an IMU and a downward-facing time-of-flight distance sensor,
logging to SD — paired with an analysis pipeline that counts steps and flags abnormal gait from an
autoencoder trained only on normal walking.

The project has three parts:

- **[Firmware](Firmware/README.md)** — an ESP32-based recorder (accelerometer, gyroscope, distance
  sensor) with button-driven session control and a WiFi dashboard for offloading data. See also
  [Firmware/POWER_ANALYSIS.md](Firmware/POWER_ANALYSIS.md) for the current-draw budget and estimated
  battery life.
- **[Dataset](Dataset/README.md)** — the recorded IMU sessions and their labels (train/test/walk split,
  per-recording annotation, anomaly type).
- **[Analyzer](Analyzer/README.md)** — the Python pipeline that consumes the dataset: a step counter
  scored against the distance sensor's ground truth, and a self-supervised (one-class) autoencoder that
  learns normal gait and scores deviations from it. See
  [Analyzer/EXPERIMENT.md](Analyzer/EXPERIMENT.md) for the full experimental design and rationale behind
  the anomaly-detection pipeline.

## How the pieces fit together

The cane (Firmware) records 6-axis IMU data and cane-tip height to SD as CSV. Recordings are collected
into the Dataset and split into `train` (normal gait only) and `test` (labelled, includes abnormal gait).
The Analyzer's step counter turns raw acceleration into a step count, cross-checked against the distance
sensor; its autoencoder is fitted on `train` and evaluated frozen on `test`, so no anomaly label ever
enters the pipeline that produces the score meant to catch it.

## Getting started

- To build and flash the firmware: see [Firmware/README.md](Firmware/README.md#build--deploy).
- To run the analysis on the existing dataset: see [Analyzer/README.md](Analyzer/README.md#commands)
  (`cd Analyzer && uv sync`, then `uv run main.py ...`).
