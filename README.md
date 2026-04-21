# Dynamical System Telemetry Forecasting

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Statsmodels-Time_Series-orange.svg" alt="Statsmodels">
</p>

## Overview
An industry-grade framework for **Multivariate Time Series Forecasting of Command-Driven Dynamical System Telemetry**. This repository processes high-frequency, massive-scale sensor telemetry (originally 5 million+ points) and applies a rigorous analytical pipeline spanning statistical methodologies, ensemble machine learning, and recurrent neural networks (PyTorch). 

It is designed to evaluate, analyze, and forecast complex temporal, command-driven dynamics for predictive modeling and anomaly detection.

## Key Capabilities

- **Deep Learning Forecasting:** Implements Gated Recurrent Units (GRU) leveraging PyTorch with MPS/GPU acceleration for sequence-to-sequence prediction.
- **Statistical & ML Modeling:** Evaluates classic parametric models (ARIMAX) against exogenous variables, non-linear empirical baselines (SETAR), and Random Forest algorithms.
- **Rigorous Hypotheses Tests:** Advanced implementations of Augmented Dickey-Fuller (ADF) and Phillips-Perron (PP) alongside Tsay F-tests and Teräsvirta NN tests for Non-Linearity Assessment.
- **Advanced Preprocessing:** Handles non-linear transformations, sequence structurization (lookback windows), missing data interpolation, and high-frequency resampling.
- **Automated Diagnostic EDA:** Generates detailed Autocorrelation (ACF), Partial Autocorrelation (PACF), seasonal decomposition, and Cross-Validation (5-fold expanding window).

## Project Architecture

```text
├── data/                   # Multi-sensor telemetry logs (ignored from git)
├── notebooks/              # Interactive environments for model experimentation
│   ├── TSA_Project.ipynb   # Advanced modeling (ARIMAX, SETAR, PyTorch GRU)
│   └── TSA_ISP_Traffic.ipynb # ISP Traffic flow and metrics analysis
├── src/                    # Core Python Modules
│   ├── TSA_Updated.py      # Core processing and baseline statistical algorithms
│   └── TSA_ISP_Updated.py  # Specialized network metric processing algorithm
├── requirements.txt        # Virtual environment dependency lockfile
└── README.md
```

## Getting Started

### 1. Environment Initialization

```bash
git clone https://github.com/arjungop/dynamical-system-telemetry-forecasting.git
cd dynamical-system-telemetry-forecasting

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Execution

To run the primary forecasting pipeline natively:
```bash
python3 src/TSA_Updated.py
```

To explore the PyTorch GRU architecture and statistical breakdowns interactively:
```bash
jupyter notebook notebooks/TSA_Project.ipynb
```
