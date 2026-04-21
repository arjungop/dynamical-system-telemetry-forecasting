"""
Time Series Analysis Project - Updated Version
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
DATA_PATH = '../data.csv'
OUTPUT_DIR = 'plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (15, 6)

# ============================================================================
# 1. Data Loading and Preprocessing
# ============================================================================

def load_data(path):
    print("Loading data...")
    df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
    print(f"Data loaded. Shape: {df.shape}")
    
    # Resample to 1H for analysis
    print("Resampling to 1H...")
    df_resampled = df.resample('1h').mean()
    return df_resampled

def scale_data(df, method='standard'):
    """
    Scale all numeric columns in the dataframe.
    
    Args:
        df: DataFrame with time series data
        method: 'standard' for StandardScaler (z-score), 'minmax' for MinMaxScaler (0-1 range)
    
    Returns:
        Scaled DataFrame, scaler object
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
    kpss_stationary = kpss_result[1] >= 0.05
    print(f"KPSS Conclusion: {'Stationary' if kpss_stationary else 'Non-Stationary'}")
    
    # 3. Phillips-Perron Test
    print("\n3. Phillips-Perron Test:")
    print("-" * 40)
    pp_test = PhillipsPerron(ts_clean)
    print(f"Test Statistic: {pp_test.stat:.4f}")
    print(f"p-value: {pp_test.pvalue:.4f}")
    print(f"#Lags Used: {pp_test.lags}")
    print("Critical Values:")
    for key, value in pp_test.critical_values.items():
        print(f"    {key}: {value:.4f}")
    pp_stationary = pp_test.pvalue <= 0.05
    print(f"PP Conclusion: {'Stationary' if pp_stationary else 'Non-Stationary'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("STATIONARITY SUMMARY:")
    print(f"  ADF Test:  {'Stationary' if adf_stationary else 'Non-Stationary'}")
    print(f"  KPSS Test: {'Stationary' if kpss_stationary else 'Non-Stationary'}")
    print(f"  PP Test:   {'Stationary' if pp_stationary else 'Non-Stationary'}")
    
    if adf_stationary and kpss_stationary and pp_stationary:
        print("\nOverall: All tests indicate STATIONARY data")
    elif not adf_stationary and not kpss_stationary and not pp_stationary:
        print("\nOverall: All tests indicate NON-STATIONARY data")
    else:
        print("\nOverall: Mixed results - further analysis recommended")
    print("=" * 60)
    
    return {'adf': adf_stationary, 'kpss': kpss_stationary, 'pp': pp_stationary}

# ============================================================================
# 3. EDA Function
# ============================================================================

def perform_eda(df, column):
    """Perform comprehensive EDA on a time series column"""
    
    print(f"\n--- Analyzing {column} ---")
    ts = df[column].dropna()
    
    # 1. Time Series Plot
    plt.figure(figsize=(15, 6))
    plt.plot(ts)
    plt.title(f'Time Series: {column}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{column}_timeseries.png', dpi=150)
    plt.show()
    
    # 2. Decomposition
    print("Performing Decomposition...")
    result = seasonal_decompose(ts, model='additive', period=24)
    result.plot()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{column}_decomposition.png', dpi=150)
    plt.show()
    
    # 3. Stationarity Tests
    perform_stationarity_tests(ts, column)
    
    # 4. ACF and PACF
    fig, ax = plt.subplots(2, 1, figsize=(15, 8))
    plot_acf(ts, ax=ax[0], lags=50)
    plot_pacf(ts, ax=ax[1], lags=50)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{column}_acf_pacf.png', dpi=150)
    plt.show()

# ============================================================================
# 4. Main Execution
# ============================================================================

if __name__ == "__main__":
    try:
        # Load and scale data
        df = load_data(DATA_PATH)
        print("Columns:", df.columns.tolist())
        
        # Scale the data
        df_scaled, scaler = scale_data(df, method='standard')
        print("\nScaled data statistics:")
        print(df_scaled.describe())
        
        # Perform EDA on bso1 (scaled)
        perform_eda(df_scaled, 'bso1')
        
    except FileNotFoundError:
        print(f"File not found at {DATA_PATH}. Please check the path.")
