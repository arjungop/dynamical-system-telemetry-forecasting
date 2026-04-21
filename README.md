# Time Series Analysis: Network Traffic Prediction

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Statsmodels-Time_Series-orange.svg" alt="Statsmodels">
  <img src="https://img.shields.io/badge/Machine_Learning-Scikit_Learn-yellow.svg" alt="Scikit-Learn">
</p>

## Overview
An industry-grade time series analysis framework designed for anomaly detection and forecasting of high-volume network and ISP traffic. This repository implements comprehensive data decomposition, rigorous stationarity hypothesis testing, and advanced predictive modeling to optimize network capacity planning and bandwidth allocation.

## Key Features

- **Robust Preprocessing:** Handles non-linear transformations, missing data interpolation, and feature scaling (Standard & Min-Max).
- **Comprehensive Stationarity Testing:** Implements Augmented Dickey-Fuller (ADF), Kwiatkowski-Phillips-Schmidt-Shin (KPSS), and Phillips-Perron (PP) tests.
- **Statistical & ML Modeling:** Evaluates classic parametric models (ARIMA) against non-linear ensemble methods (Random Forest Regressor).
- **Interactive EDA:** Jupyter Notebook environments for immediate visualization of rolling means, decomposition, Autocorrelation (ACF), and Partial Autocorrelation (PACF).

## Project Structure

```text
├── data/                   # Target data directory (ignored by git to prevent massive data leaks)
├── notebooks/              # Jupyter Notebooks for interactive EDA and modeling
│   ├── TSA_ISP_Traffic.ipynb
│   └── TSA_Project.ipynb
├── src/                    # Core Python modules
│   ├── TSA_ISP_Updated.py  # Specific processing module for specialized ISP traffic
│   └── TSA_Updated.py      # Core time series processing pipeline
├── requirements.txt        # Virtual environment dependencies
└── README.md
```

## Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/arjungop/TSA.git
cd TSA

# Establish a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 2. Execution

To run the full suite for ISP traffic natively from the command line:

```bash
python src/TSA_ISP_Updated.py
```

Alternatively, load the interactive environment:

```bash
jupyter notebook notebooks/TSA_ISP_Traffic.ipynb
```
