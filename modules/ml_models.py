import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import joblib
try:
    import xgboost as xgb
    import lightgbm as lgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

import sys
sys.path.append('..')
from utils.error_handler import safe_execute, validate_dataframe
from utils.helpers import get_numeric_columns

def train_classification_models(
    df: pd.DataFrame,
    target_column: str,
    models: List[str] = None,
    test_size: float = 0.2,
    tune_hyperparameters: bool = False
) -> Dict:
    
    if not validate_dataframe(df, min_rows=50, operation="classification"):
        return {}
    
    if target_column not in df.columns:
        st.error(f"❌ Target column '{target_column}' not found")
        return {}
    
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        numeric_cols = get_numeric_columns(X)
        if not numeric_cols:
            st.error("❌ No numeric columns for features")
            return {}
        
        X = X[numeric_cols]
        X = X.fillna(X.mean())
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        if models is None:
            models = ['random_forest', 'logistic']
            if XGBOOST_AVAILABLE:
                models.append('xgboost')
        
        results = {}
        
        for model_name in models:
            try:
                if model_name == 'random_forest':
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                    model = xgb.XGBClassifier(n_estimators=100, random_state=42)
                elif model_name == 'logistic':
                    model = LogisticRegression(max_iter=1000, random_state=42)
                elif model_name == 'svm':
                    model = SVC(probability=True, random_state=42)
                elif model_name == 'lightgbm' and XGBOOST_AVAILABLE:
                    model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
                else:
                    continue
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }
                
                try:
                    y_pred_proba = model.predict_proba(X_test)
                    if len(np.unique(y_test)) == 2:
                        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                except:
                    pass
                
                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred,
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
                
            except Exception as e:
                st.warning(f"⚠️ {model_name} training failed: {str(e)[:50]}")
                continue
        
        return results
        
    except Exception as e:
        st.error(f"❌ Classification failed: {str(e)}")
        return {}

def train_regression_models(
    df: pd.DataFrame,
    target_column: str,
    models: List[str] = None,
    test_size: float = 0.2
) -> Dict:
    
    if not validate_dataframe(df, min_rows=50, operation="regression"):
        return {}
    
    if target_column not in df.columns:
        st.error(f"❌ Target column '{target_column}' not found")
        return {}
    
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        numeric_cols = get_numeric_columns(X)
        if not numeric_cols:
            st.error("❌ No numeric columns for features")
            return {}
        
        X = X[numeric_cols]
        X = X.fillna(X.mean())
        
        if not pd.api.types.is_numeric_dtype(y):
            st.error(f"❌ Target column must be numeric for regression")
            return {}
        
        y = y.fillna(y.mean())
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        if models is None:
            models = ['random_forest', 'ridge']
            if XGBOOST_AVAILABLE:
                models.append('xgboost')
        
        results = {}
        
        for model_name in models:
            try:
                if model_name == 'random_forest':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
                elif model_name == 'ridge':
                    model = Ridge(alpha=1.0, random_state=42)
                elif model_name == 'lasso':
                    model = Lasso(alpha=1.0, random_state=42)
                else:
                    continue
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                metrics = {
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred)
                }
                
                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred.tolist()
                }
                
            except Exception as e:
                st.warning(f"⚠️ {model_name} training failed: {str(e)[:50]}")
                continue
        
        return results
        
    except Exception as e:
        st.error(f"❌ Regression failed: {str(e)}")
        return {}

def perform_clustering(
    df: pd.DataFrame,
    n_clusters: int = 3,
    method: str = "kmeans"
) -> Dict:
    
    if not validate_dataframe(df, min_rows=20, operation="clustering"):
        return {}
    
    try:
        numeric_cols = get_numeric_columns(df)
        if not numeric_cols:
            st.error("❌ No numeric columns for clustering")
            return {}
        
        X = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=5)
        else:
            st.error(f"❌ Unknown clustering method: {method}")
            return {}
        
        labels = model.fit_predict(X)
        
        return {
            'labels': labels.tolist(),
            'n_clusters': len(np.unique(labels)),
            'model': model
        }
        
    except Exception as e:
        st.error(f"❌ Clustering failed: {str(e)}")
        return {}

def perform_pca(df: pd.DataFrame, n_components: int = 2) -> Dict:
    if not validate_dataframe(df, min_rows=10, operation="PCA"):
        return {}
    
    try:
        numeric_cols = get_numeric_columns(df)
        if not numeric_cols:
            st.error("❌ No numeric columns for PCA")
            return {}
        
        X = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        pca = PCA(n_components=min(n_components, len(numeric_cols)))
        X_pca = pca.fit_transform(X)
        
        return {
            'transformed_data': X_pca,
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'n_components': pca.n_components_
        }
        
    except Exception as e:
        st.error(f"❌ PCA failed: {str(e)}")
        return {}

def get_feature_importance(model, feature_names: List[str]) -> Dict:
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
            importance_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            return {}
    except Exception as e:
        return {}
