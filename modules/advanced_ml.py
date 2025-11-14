import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import streamlit as st
from utils.error_handler import log_error

def train_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str = 'classification',
    hidden_layers: Tuple[int] = (100, 50),
    activation: str = 'relu',
    max_iter: int = 500
) -> Dict:
    try:
        if task == 'classification':
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                max_iter=max_iter,
                random_state=42,
                early_stopping=True
            )
        else:
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                max_iter=max_iter,
                random_state=42,
                early_stopping=True
            )
        
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        if task == 'classification':
            train_score = accuracy_score(y_train, train_pred)
            test_score = accuracy_score(y_test, test_pred)
            f1 = f1_score(y_test, test_pred, average='weighted')
            
            metrics = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'f1_score': f1
            }
        else:
            train_score = r2_score(y_train, train_pred)
            test_score = r2_score(y_test, test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            metrics = {
                'train_r2': train_score,
                'test_r2': test_score,
                'rmse': rmse
            }
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': test_pred,
            'feature_names': X_train.columns.tolist()
        }
    except Exception as e:
        log_error("Neural Network Training", e)
        return None

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str = 'classification',
    params: Optional[Dict] = None
) -> Dict:
    try:
        if params is None:
            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'random_state': 42
            }
        
        if task == 'classification':
            model = XGBClassifier(**params)
        else:
            model = XGBRegressor(**params)
        
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        if task == 'classification':
            train_score = accuracy_score(y_train, train_pred)
            test_score = accuracy_score(y_test, test_pred)
            f1 = f1_score(y_test, test_pred, average='weighted')
            
            metrics = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'f1_score': f1
            }
        else:
            train_score = r2_score(y_train, train_pred)
            test_score = r2_score(y_test, test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            metrics = {
                'train_r2': train_score,
                'test_r2': test_score,
                'rmse': rmse
            }
        
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'predictions': test_pred
        }
    except Exception as e:
        log_error("XGBoost Training", e)
        return None

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str = 'classification',
    params: Optional[Dict] = None
) -> Dict:
    try:
        if params is None:
            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'random_state': 42,
                'verbosity': -1
            }
        
        if task == 'classification':
            model = LGBMClassifier(**params)
        else:
            model = LGBMRegressor(**params)
        
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        if task == 'classification':
            train_score = accuracy_score(y_train, train_pred)
            test_score = accuracy_score(y_test, test_pred)
            f1 = f1_score(y_test, test_pred, average='weighted')
            
            metrics = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'f1_score': f1
            }
        else:
            train_score = r2_score(y_train, train_pred)
            test_score = r2_score(y_test, test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            metrics = {
                'train_r2': train_score,
                'test_r2': test_score,
                'rmse': rmse
            }
        
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'predictions': test_pred
        }
    except Exception as e:
        log_error("LightGBM Training", e)
        return None

def ensemble_models(
    models: List[any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str = 'classification',
    method: str = 'voting'
) -> Dict:
    try:
        predictions = [model.predict(X_test) for model in models]
        
        if method == 'voting':
            if task == 'classification':
                ensemble_pred = np.round(np.mean(predictions, axis=0)).astype(int)
                score = accuracy_score(y_test, ensemble_pred)
                f1 = f1_score(y_test, ensemble_pred, average='weighted')
                
                metrics = {
                    'accuracy': score,
                    'f1_score': f1
                }
            else:
                ensemble_pred = np.mean(predictions, axis=0)
                score = r2_score(y_test, ensemble_pred)
                rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
                
                metrics = {
                    'r2_score': score,
                    'rmse': rmse
                }
        
        return {
            'predictions': ensemble_pred,
            'metrics': metrics,
            'individual_predictions': predictions
        }
    except Exception as e:
        log_error("Ensemble Models", e)
        return None
