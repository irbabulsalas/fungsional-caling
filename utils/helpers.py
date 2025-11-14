import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

def format_number(num: float, decimals: int = 2) -> str:
    if num >= 1_000_000:
        return f"{num/1_000_000:.{decimals}f}M"
    elif num >= 1_000:
        return f"{num/1_000:.{decimals}f}K"
    else:
        return f"{num:.{decimals}f}"

def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def get_datetime_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=['datetime64']).columns.tolist()

def calculate_data_quality_score(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 0.0
    
    missing_score = (1 - df.isnull().sum().sum() / total_cells) * 0.3
    
    duplicates = df.duplicated().sum()
    duplicate_score = (1 - duplicates / len(df)) * 0.2
    
    numeric_cols = get_numeric_columns(df)
    outlier_score = 0.25
    if numeric_cols:
        outlier_count = 0
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_count += outliers
        outlier_score = (1 - min(outlier_count / len(df), 1)) * 0.25
    
    consistency_score = 0.25
    
    total_score = (missing_score + duplicate_score + outlier_score + consistency_score) * 100
    return round(total_score, 1)

def safe_get_session_state(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)

def set_session_state(key: str, value: Any):
    st.session_state[key] = value

def initialize_session_state():
    defaults = {
        'uploaded_data': None,
        'cleaned_data': None,
        'chat_history': [],
        'analysis_results': {},
        'trained_models': {},
        'current_page': '📈 Overview',
        'show_onboarding': True,
        'theme': 'light'
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def get_column_info(df: pd.DataFrame) -> Dict[str, Dict]:
    info = {}
    for col in df.columns:
        info[col] = {
            'dtype': str(df[col].dtype),
            'missing': df[col].isnull().sum(),
            'missing_pct': (df[col].isnull().sum() / len(df)) * 100,
            'unique': df[col].nunique(),
            'sample_values': df[col].dropna().head(3).tolist()
        }
    return info

def download_button(data, filename: str, label: str, mime_type: str = "text/csv"):
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime_type
    )

def show_data_preview(df: pd.DataFrame, title: str = "Data Preview", rows: int = 5):
    st.subheader(title)
    st.dataframe(df.head(rows), use_container_width=True)
    st.caption(f"Showing {min(rows, len(df))} of {len(df):,} rows")
