import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Tuple, Optional
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from utils.error_handler import safe_execute, log_error

def check_stationarity(series: pd.Series) -> Dict:
    try:
        result = adfuller(series.dropna())
        
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    except Exception as e:
        log_error("Stationarity Test", e)
        return None

def fit_arima_model(
    series: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    forecast_steps: int = 10
) -> Dict:
    try:
        series_clean = series.dropna()
        
        model = ARIMA(series_clean, order=order)
        fitted_model = model.fit()
        
        forecast = fitted_model.forecast(steps=forecast_steps)
        
        return {
            'model': fitted_model,
            'forecast': forecast,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'summary': fitted_model.summary(),
            'residuals': fitted_model.resid
        }
    except Exception as e:
        log_error("ARIMA Model", e)
        return None

def fit_sarima_model(
    series: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
    forecast_steps: int = 10
) -> Dict:
    try:
        series_clean = series.dropna()
        
        model = SARIMAX(series_clean, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        
        forecast = fitted_model.forecast(steps=forecast_steps)
        
        return {
            'model': fitted_model,
            'forecast': forecast,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'summary': fitted_model.summary(),
            'residuals': fitted_model.resid
        }
    except Exception as e:
        log_error("SARIMA Model", e)
        return None

def decompose_time_series(
    series: pd.Series,
    period: int = 12,
    model: str = 'additive'
) -> Dict:
    try:
        series_clean = series.dropna()
        
        if len(series_clean) < 2 * period:
            return {
                'error': f'Time series too short for period {period}. Need at least {2*period} observations.'
            }
        
        decomposition = seasonal_decompose(series_clean, model=model, period=period)
        
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'observed': decomposition.observed
        }
    except Exception as e:
        log_error("Time Series Decomposition", e)
        return None

def auto_arima(series: pd.Series, max_p: int = 3, max_d: int = 2, max_q: int = 3) -> Dict:
    try:
        best_aic = np.inf
        best_order = None
        best_model = None
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series.dropna(), order=(p, d, q))
                        fitted = model.fit()
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                            best_model = fitted
                    except:
                        continue
        
        if best_model is None:
            return None
        
        forecast = best_model.forecast(steps=10)
        
        return {
            'model': best_model,
            'best_order': best_order,
            'aic': best_aic,
            'bic': best_model.bic,
            'forecast': forecast,
            'summary': best_model.summary()
        }
    except Exception as e:
        log_error("Auto ARIMA", e)
        return None

def plot_acf_pacf(series: pd.Series, lags: int = 40):
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        plot_acf(series.dropna(), lags=lags, ax=ax1)
        ax1.set_title('Autocorrelation Function (ACF)')
        
        plot_pacf(series.dropna(), lags=lags, ax=ax2)
        ax2.set_title('Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        log_error("ACF/PACF Plot", e)
        return None

def calculate_forecast_metrics(actual: pd.Series, predicted: pd.Series) -> Dict:
    try:
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }
    except Exception as e:
        log_error("Forecast Metrics", e)
        return None
