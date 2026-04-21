"""
Time Series Analysis Project - ISP Traffic Version
With proper scaling and comprehensive stationarity tests (ADF, KPSS, PP)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from arch.unitroot import PhillipsPerron

# Configuration
DATA_PATH = '../isp_traffic_ts.csv'
TARGET_COL = 'flowBytesPerSecond'
OUTPUT_DIR = 'plots_isp'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (15, 6)

# ============================================================================
# 1. Data Loading and Preprocessing
# ============================================================================

def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
    print(f"Data loaded. Shape: {df.shape}")
    
    # Resample to 1H for analysis
    print("Resampling to 1H...")
    df_resampled = df.resample('1h').mean().dropna()
    return df_resampled

def scale_data(df, method='standard'):
    """
    Scale all numeric columns in the dataframe.
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")
    
    scaled_values = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)
    
    print(f"\nData scaled using {method} scaling")
    print(f"Columns scaled: {list(df.columns)}")
    
    return df_scaled, scaler

# ============================================================================
# 2. Stationarity Tests (ADF, KPSS, PP)
# ============================================================================

def perform_stationarity_tests(ts, column_name):
    """Perform comprehensive stationarity tests: ADF, KPSS, and Phillips-Perron"""
    
    print("=" * 60)
    print(f"STATIONARITY TESTS for '{column_name}'")
    print("=" * 60)
    
    # Clean the series
    ts_clean = ts.dropna()
    
    # 1. ADF Test (Augmented Dickey-Fuller)
    print("\n1. Augmented Dickey-Fuller Test:")
    print("-" * 40)
    dftest = adfuller(ts_clean, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Observations'])
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)
    adf_stationary = dftest[1] <= 0.05
    print(f"ADF Conclusion: {'Stationary' if adf_stationary else 'Non-Stationary'}")
    
    # 2. KPSS Test
    print("\n2. KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin):")
    print("-" * 40)
    kpss_result = kpss(ts_clean, regression='c', nlags='auto')
    kpss_output = pd.Series([kpss_result[0], kpss_result[1], kpss_result[2]], 
                            index=['Test Statistic', 'p-value', '#Lags Used'])
    for key, value in kpss_result[3].items():
        kpss_output[f'Critical Value ({key})'] = value
    print(kpss_output)
    kpss_stationary = kpss_result[1] > 0.05
    print(f"KPSS Conclusion: {'Stationary' if kpss_stationary else 'Non-Stationary'}")

    # 3. Phillips-Perron Test
    print("\n3. Phillips-Perron Test:")
    print("-" * 40)
    pp = PhillipsPerron(ts_clean)
    print(pp.summary())
    pp_stationary = pp.pvalue <= 0.05
    print(f"PP Conclusion: {'Stationary' if pp_stationary else 'Non-Stationary'}")
    
    return adf_stationary, kpss_stationary, pp_stationary

# ============================================================================
# 3. Exploratory Data Analysis & Plotting
# ============================================================================

def run_eda_plots(ts, column_name):
    """Generate basic EDA plots"""
    print(f"\nGenerating EDA plots for {column_name}...")
    
    # Time series plot
    plt.figure()
    plt.plot(ts)
    plt.title(f'Time Series: {column_name}')
    plt.savefig(f"{OUTPUT_DIR}/{column_name}_timeseries.png")
    
    # Seasonal Decomposition
    print("Performing seasonal decomposition...")
    result = seasonal_decompose(ts.dropna(), model='additive', period=24) # Assuming daily seasonality
    result.plot()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{column_name}_decomposition.png")
    
    # ACF/PACF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    plot_acf(ts.dropna(), ax=ax1, lags=48)
    plot_pacf(ts.dropna(), ax=ax2, lags=48)
    plt.savefig(f"{OUTPUT_DIR}/{column_name}_acf_pacf.png")
    
    print(f"Plots saved to {OUTPUT_DIR}/")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check if data exists
    full_data_path = os.path.join(os.path.dirname(__file__), DATA_PATH)
    if not os.path.exists(full_data_path):
        print(f"Error: Data file not found at {full_data_path}")
    else:
        # Load and resample
        df_resampled = load_data(full_data_path)
        
        # Focus on target column
        print(f"\nAnalyzing column: {TARGET_COL}")
        ts_data = df_resampled[TARGET_COL]
        
        # Run stationarity tests
        perform_stationarity_tests(ts_data, TARGET_COL)
        
        # Run EDA
        run_eda_plots(ts_data, TARGET_COL)
        
        print("\nISP Analysis completed successfully.")
