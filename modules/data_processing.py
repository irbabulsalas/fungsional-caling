import pandas as pd
import numpy as np
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from typing import Tuple, Dict, List, Optional
import sys
sys.path.append('..')
from utils.error_handler import safe_execute, validate_dataframe
from utils.helpers import get_numeric_columns, get_categorical_columns, calculate_data_quality_score

def load_data_from_file(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 200:
            st.error(f"❌ File too large: {file_size_mb:.1f}MB (Max: 200MB)")
            return None
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        elif file_extension == 'parquet':
            df = pd.read_parquet(uploaded_file)
        elif file_extension == 'tsv' or file_extension == 'txt':
            df = pd.read_csv(uploaded_file, sep='\t')
        else:
            st.error(f"❌ Unsupported file type: .{file_extension}")
            return None
        
        if df.empty:
            st.warning("📭 File is empty")
            return None
        
        st.success(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df
        
    except pd.errors.EmptyDataError:
        st.error("❌ Empty file")
        return None
    except pd.errors.ParserError as e:
        st.error(f"❌ File format error: {str(e)[:100]}")
        return None
    except Exception as e:
        st.error(f"⚠️ Upload failed: {type(e).__name__}")
        return None

def profile_data(df: pd.DataFrame) -> Dict:
    if not validate_dataframe(df, min_rows=1, operation="profiling"):
        return {}
    
    profile = {
        'basic_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
            'duplicates': df.duplicated().sum()
        },
        'column_info': {},
        'missing_values': {},
        'data_types': {},
        'quality_score': calculate_data_quality_score(df)
    }
    
    for col in df.columns:
        profile['column_info'][col] = {
            'dtype': str(df[col].dtype),
            'missing': int(df[col].isnull().sum()),
            'missing_pct': round((df[col].isnull().sum() / len(df)) * 100, 2),
            'unique': int(df[col].nunique()),
            'unique_pct': round((df[col].nunique() / len(df)) * 100, 2)
        }
        
        if df[col].dtype in ['int64', 'float64']:
            profile['column_info'][col].update({
                'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                'median': float(df[col].median()) if not df[col].isnull().all() else None,
                'std': float(df[col].std()) if not df[col].isnull().all() else None,
                'min': float(df[col].min()) if not df[col].isnull().all() else None,
                'max': float(df[col].max()) if not df[col].isnull().all() else None
            })
    
    profile['missing_values'] = {
        col: int(df[col].isnull().sum()) 
        for col in df.columns if df[col].isnull().sum() > 0
    }
    
    profile['data_types'] = {
        'numeric': len(get_numeric_columns(df)),
        'categorical': len(get_categorical_columns(df)),
        'datetime': len(df.select_dtypes(include=['datetime64']).columns)
    }
    
    return profile

def clean_data(
    df: pd.DataFrame,
    handle_missing: str = "mean",
    remove_duplicates: bool = True,
    handle_outliers: str = "none"
) -> Tuple[pd.DataFrame, Dict]:
    if not validate_dataframe(df, min_rows=1, operation="cleaning"):
        return df, {}
    
    df_clean = df.copy()
    report = {
        'missing_handled': 0,
        'duplicates_removed': 0,
        'outliers_handled': 0,
        'actions': []
    }
    
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        report['duplicates_removed'] = removed
        if removed > 0:
            report['actions'].append(f"Removed {removed} duplicate rows")
    
    numeric_cols = get_numeric_columns(df_clean)
    
    if handle_missing != "drop":
        if handle_missing == "mean" and numeric_cols:
            for col in numeric_cols:
                if df_clean[col].isnull().sum() > 0:
                    mean_val = df_clean[col].mean()
                    df_clean[col].fillna(mean_val, inplace=True)
                    report['missing_handled'] += 1
            report['actions'].append(f"Filled missing values with mean for {report['missing_handled']} columns")
        
        elif handle_missing == "median" and numeric_cols:
            for col in numeric_cols:
                if df_clean[col].isnull().sum() > 0:
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
                    report['missing_handled'] += 1
            report['actions'].append(f"Filled missing values with median for {report['missing_handled']} columns")
        
        elif handle_missing == "mode":
            for col in df_clean.columns:
                if df_clean[col].isnull().sum() > 0:
                    mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else None
                    if mode_val is not None:
                        df_clean[col].fillna(mode_val, inplace=True)
                        report['missing_handled'] += 1
            report['actions'].append(f"Filled missing values with mode for {report['missing_handled']} columns")
    else:
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna()
        removed = initial_rows - len(df_clean)
        report['missing_handled'] = removed
        if removed > 0:
            report['actions'].append(f"Dropped {removed} rows with missing values")
    
    if handle_outliers == "iqr" and numeric_cols:
        outliers_removed = 0
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            initial_len = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            outliers_removed += initial_len - len(df_clean)
        
        report['outliers_handled'] = outliers_removed
        if outliers_removed > 0:
            report['actions'].append(f"Removed {outliers_removed} outlier rows using IQR method")
    
    return df_clean, report

def encode_categorical(df: pd.DataFrame, method: str = "onehot") -> pd.DataFrame:
    df_encoded = df.copy()
    categorical_cols = get_categorical_columns(df_encoded)
    
    if not categorical_cols:
        return df_encoded
    
    try:
        if method == "onehot":
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
        elif method == "label":
            le = LabelEncoder()
            for col in categorical_cols:
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        return df_encoded
    except Exception as e:
        st.warning(f"⚠️ Encoding failed: {str(e)}")
        return df

def scale_features(df: pd.DataFrame, method: str = "standard", columns: Optional[List[str]] = None) -> pd.DataFrame:
    df_scaled = df.copy()
    
    if columns is None:
        columns = get_numeric_columns(df_scaled)
    
    if not columns:
        return df_scaled
    
    try:
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            return df_scaled
        
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        return df_scaled
    except Exception as e:
        st.warning(f"⚠️ Scaling failed: {str(e)}")
        return df
