import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

HAS_TENSORFLOW = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    HAS_TENSORFLOW = True
except ImportError:
    pass

class AdvancedTimeSeriesModels:
    
    @staticmethod
    def check_stationarity(series: pd.Series, significance_level: float = 0.05) -> Dict:
        """Check if time series is stationary using ADF and KPSS tests"""
        try:
            adf_result = adfuller(series.dropna())
            kpss_result = kpss(series.dropna(), regression='c')
            
            is_stationary_adf = adf_result[1] < significance_level
            is_stationary_kpss = kpss_result[1] >= significance_level
            
            return {
                'adf_statistic': adf_result[0],
                'adf_pvalue': adf_result[1],
                'adf_critical_values': adf_result[4],
                'is_stationary_adf': is_stationary_adf,
                'kpss_statistic': kpss_result[0],
                'kpss_pvalue': kpss_result[1],
                'kpss_critical_values': kpss_result[3],
                'is_stationary_kpss': is_stationary_kpss,
                'overall_stationary': is_stationary_adf and is_stationary_kpss
            }
        except Exception as e:
            raise Exception(f"Error in stationarity test: {str(e)}")
    
    @staticmethod
    def plot_acf_pacf(series: pd.Series, lags: int = 40) -> Tuple[plt.Figure, plt.Figure]:
        """Plot ACF and PACF for autocorrelation analysis"""
        fig_acf, ax_acf = plt.subplots(figsize=(12, 4))
        plot_acf(series.dropna(), lags=lags, ax=ax_acf)
        ax_acf.set_title('Autocorrelation Function (ACF)')
        
        fig_pacf, ax_pacf = plt.subplots(figsize=(12, 4))
        plot_pacf(series.dropna(), lags=lags, ax=ax_pacf)
        ax_pacf.set_title('Partial Autocorrelation Function (PACF)')
        
        return fig_acf, fig_pacf
    
    @staticmethod
    def seasonal_decomposition(series: pd.Series, period: int = 12, model: str = 'additive') -> Dict:
        """Perform seasonal decomposition"""
        try:
            decomposition = seasonal_decompose(series.dropna(), model=model, period=period)
            
            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'observed': decomposition.observed
            }
        except Exception as e:
            raise Exception(f"Error in seasonal decomposition: {str(e)}")
    
    @staticmethod
    def plot_decomposition(decomposition: Dict) -> go.Figure:
        """Plot seasonal decomposition components"""
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.1
        )
        
        components = ['observed', 'trend', 'seasonal', 'residual']
        for idx, component in enumerate(components, 1):
            data = decomposition[component]
            fig.add_trace(
                go.Scatter(x=data.index, y=data.values, mode='lines', name=component.capitalize()),
                row=idx, col=1
            )
        
        fig.update_layout(height=800, showlegend=False, title_text="Time Series Decomposition")
        return fig
    
    @staticmethod
    def train_prophet(df: pd.DataFrame, date_column: str, value_column: str,
                      periods: int = 30, freq: str = 'D',
                      yearly_seasonality: bool = True,
                      weekly_seasonality: bool = True,
                      daily_seasonality: bool = False) -> Dict:
        """Train Facebook Prophet model for time series forecasting"""
        try:
            prophet_df = df[[date_column, value_column]].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            
            with st.spinner('Training Prophet model...'):
                model = Prophet(
                    yearly_seasonality=yearly_seasonality,
                    weekly_seasonality=weekly_seasonality,
                    daily_seasonality=daily_seasonality
                )
                model.fit(prophet_df)
            
            future = model.make_future_dataframe(periods=periods, freq=freq)
            forecast = model.predict(future)
            
            train_size = len(prophet_df)
            train_forecast = forecast.iloc[:train_size]
            
            y_true = prophet_df['y'].values
            y_pred = train_forecast['yhat'].values
            
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            return {
                'model': model,
                'forecast': forecast,
                'components': model.predict(future),
                'mae': mae,
                'rmse': rmse,
                'mse': mse,
                'train_df': prophet_df,
                'periods': periods
            }
            
        except Exception as e:
            raise Exception(f"Error in Prophet training: {str(e)}")
    
    @staticmethod
    def plot_prophet_forecast(results: Dict) -> go.Figure:
        """Plot Prophet forecast"""
        forecast = results['forecast']
        train_df = results['train_df']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=train_df['ds'],
            y=train_df['y'],
            mode='markers',
            name='Actual',
            marker=dict(color='blue', size=4)
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            name='Lower Bound',
            line=dict(width=0),
            fillcolor='rgba(255,0,0,0.2)',
            fill='tonexty',
            showlegend=False
        ))
        
        fig.update_layout(
            title='Prophet Forecast',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_lstm_model(input_shape: Tuple, units: List[int] = [50, 25],
                          dropout: float = 0.2) -> Any:
        """Create LSTM model for time series forecasting"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow tidak terinstall untuk LSTM models")
        
        model = Sequential()
        
        model.add(LSTM(units[0], return_sequences=len(units) > 1, input_shape=input_shape))
        model.add(Dropout(dropout))
        
        for i in range(1, len(units)):
            return_seq = i < len(units) - 1
            model.add(LSTM(units[i], return_sequences=return_seq))
            model.add(Dropout(dropout))
        
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    @staticmethod
    def prepare_lstm_data(series: pd.Series, lookback: int = 10, 
                          train_split: float = 0.8) -> Tuple:
        """Prepare data for LSTM training"""
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        split_idx = int(len(X) * train_split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test, scaler
    
    @staticmethod
    def train_lstm(series: pd.Series, lookback: int = 10, units: List[int] = [50, 25],
                   epochs: int = 50, batch_size: int = 32) -> Dict:
        """Train LSTM model for time series forecasting"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow tidak terinstall untuk LSTM models")
        
        try:
            X_train, X_test, y_train, y_test, scaler = AdvancedTimeSeriesModels.prepare_lstm_data(
                series, lookback=lookback
            )
            
            model = AdvancedTimeSeriesModels.create_lstm_model(
                input_shape=(lookback, 1),
                units=units
            )
            
            with st.spinner('Training LSTM model...'):
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    verbose=0
                )
            
            y_pred_scaled = model.predict(X_test)
            y_pred = scaler.inverse_transform(y_pred_scaled)
            y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            mae = mean_absolute_error(y_test_original, y_pred)
            mse = mean_squared_error(y_test_original, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_original, y_pred)
            
            return {
                'model': model,
                'scaler': scaler,
                'history': history.history,
                'y_test': y_test_original.flatten(),
                'y_pred': y_pred.flatten(),
                'mae': mae,
                'rmse': rmse,
                'mse': mse,
                'r2_score': r2,
                'lookback': lookback,
                'architecture': units
            }
            
        except Exception as e:
            raise Exception(f"Error in LSTM training: {str(e)}")
    
    @staticmethod
    def create_time_series_dashboard(df: pd.DataFrame):
        """Create comprehensive time series analysis dashboard"""
        st.subheader("📈 Advanced Time Series Analysis")
        
        date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        if len(date_columns) == 0:
            for col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_columns.append(col)
                    break
                except:
                    continue
        
        if len(date_columns) == 0:
            st.error("No date column found in dataset")
            return
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            st.error("No numeric columns found")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox("Date Column", date_columns)
        with col2:
            value_col = st.selectbox("Value Column", numeric_cols)
        
        df_sorted = df.sort_values(date_col)
        series = df_sorted[value_col]
        
        tabs = st.tabs([
            "📊 Stationarity Tests",
            "🔄 Seasonal Decomposition",
            "📉 ACF/PACF",
            "🔮 Prophet Forecast",
            "🧠 LSTM Forecast" if HAS_TENSORFLOW else "🧠 LSTM (Not Available)"
        ])
        
        with tabs[0]:
            st.markdown("### Stationarity Tests (ADF & KPSS)")
            if st.button("Run Stationarity Tests"):
                with st.spinner("Running tests..."):
                    results = AdvancedTimeSeriesModels.check_stationarity(series)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Augmented Dickey-Fuller Test**")
                        st.metric("ADF Statistic", f"{results['adf_statistic']:.4f}")
                        st.metric("P-value", f"{results['adf_pvalue']:.4f}")
                        st.write("Critical Values:", results['adf_critical_values'])
                        
                        if results['is_stationary_adf']:
                            st.success("✅ Series is stationary (ADF)")
                        else:
                            st.warning("⚠️ Series is non-stationary (ADF)")
                    
                    with col2:
                        st.markdown("**KPSS Test**")
                        st.metric("KPSS Statistic", f"{results['kpss_statistic']:.4f}")
                        st.metric("P-value", f"{results['kpss_pvalue']:.4f}")
                        st.write("Critical Values:", results['kpss_critical_values'])
                        
                        if results['is_stationary_kpss']:
                            st.success("✅ Series is stationary (KPSS)")
                        else:
                            st.warning("⚠️ Series is non-stationary (KPSS)")
                    
                    if results['overall_stationary']:
                        st.success("✅ Overall: Series is STATIONARY")
                    else:
                        st.warning("⚠️ Overall: Series is NON-STATIONARY - Consider differencing")
        
        with tabs[1]:
            st.markdown("### Seasonal Decomposition")
            period = st.number_input("Period", min_value=2, value=12, help="Seasonality period")
            model_type = st.selectbox("Model", ["additive", "multiplicative"])
            
            if st.button("Decompose"):
                with st.spinner("Decomposing..."):
                    decomp = AdvancedTimeSeriesModels.seasonal_decomposition(series, period=period, model=model_type)
                    fig = AdvancedTimeSeriesModels.plot_decomposition(decomp)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            st.markdown("### ACF & PACF Plots")
            lags = st.slider("Number of Lags", 10, 100, 40)
            
            if st.button("Plot ACF/PACF"):
                with st.spinner("Generating plots..."):
                    fig_acf, fig_pacf = AdvancedTimeSeriesModels.plot_acf_pacf(series, lags=lags)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(fig_acf)
                    with col2:
                        st.pyplot(fig_pacf)
        
        with tabs[3]:
            st.markdown("### Prophet Forecasting")
            
            col1, col2 = st.columns(2)
            with col1:
                forecast_periods = st.number_input("Forecast Periods", min_value=1, max_value=365, value=30)
            with col2:
                freq = st.selectbox("Frequency", ['D', 'W', 'M', 'Y'])
            
            if st.button("🚀 Train Prophet Model", type="primary"):
                try:
                    results = AdvancedTimeSeriesModels.train_prophet(
                        df_sorted, date_col, value_col,
                        periods=forecast_periods, freq=freq
                    )
                    
                    st.success(f"✅ Model trained! RMSE: {results['rmse']:.4f}")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("MAE", f"{results['mae']:.4f}")
                    col2.metric("RMSE", f"{results['rmse']:.4f}")
                    col3.metric("MSE", f"{results['mse']:.4f}")
                    
                    fig = AdvancedTimeSeriesModels.plot_prophet_forecast(results)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.session_state.trained_models['prophet'] = results
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with tabs[4]:
            if HAS_TENSORFLOW:
                st.markdown("### LSTM Forecasting")
                
                col1, col2 = st.columns(2)
                with col1:
                    lookback = st.number_input("Lookback Window", min_value=5, max_value=100, value=10)
                with col2:
                    units_str = st.text_input("LSTM Units", value="50,25")
                
                if st.button("🚀 Train LSTM Model", type="primary"):
                    try:
                        units = [int(x.strip()) for x in units_str.split(',')]
                        
                        results = AdvancedTimeSeriesModels.train_lstm(
                            series, lookback=lookback, units=units
                        )
                        
                        st.success(f"✅ LSTM trained! R²: {results['r2_score']:.4f}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("MAE", f"{results['mae']:.4f}")
                        col2.metric("RMSE", f"{results['rmse']:.4f}")
                        col3.metric("MSE", f"{results['mse']:.4f}")
                        col4.metric("R² Score", f"{results['r2_score']:.4f}")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=results['y_test'], name='Actual', mode='lines'))
                        fig.add_trace(go.Scatter(y=results['y_pred'], name='Predicted', mode='lines'))
                        fig.update_layout(title='LSTM Predictions', template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.session_state.trained_models['lstm'] = results
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ TensorFlow tidak terinstall. LSTM models tidak tersedia.")
                st.info("Install TensorFlow untuk menggunakan LSTM: `pip install tensorflow`")

advanced_timeseries = AdvancedTimeSeriesModels()
