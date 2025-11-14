import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px

HAS_TENSORFLOW = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    HAS_TENSORFLOW = True
except ImportError:
    pass

class DeepLearningModels:
    
    @staticmethod
    def create_keras_model(input_dim: int, output_dim: int, task_type: str = 'classification',
                           hidden_layers: List[int] = [128, 64, 32],
                           dropout_rate: float = 0.3, learning_rate: float = 0.001) -> Any:
        """Create Keras neural network model"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow tidak terinstall. Gunakan sklearn_neural_network sebagai alternatif.")
        
        model = models.Sequential()
        
        model.add(layers.Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
        model.add(layers.Dropout(dropout_rate))
        
        for units in hidden_layers[1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(dropout_rate))
        
        if task_type == 'classification':
            if output_dim == 2:
                model.add(layers.Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                model.add(layers.Dense(output_dim, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:
            model.add(layers.Dense(1))
            loss = 'mse'
            metrics = ['mae']
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
    @staticmethod
    def train_keras_classification(X: pd.DataFrame, y: pd.Series, 
                                   hidden_layers: List[int] = [128, 64, 32],
                                   epochs: int = 50, batch_size: int = 32,
                                   validation_split: float = 0.2) -> Dict:
        """Train Keras neural network for classification"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)
            
            n_classes = len(le.classes_)
            
            model = DeepLearningModels.create_keras_model(
                input_dim=X_train_scaled.shape[1],
                output_dim=n_classes,
                task_type='classification',
                hidden_layers=hidden_layers
            )
            
            early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            
            with st.spinner('Training neural network...'):
                history = model.fit(
                    X_train_scaled, y_train_encoded,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=[early_stop, reduce_lr],
                    verbose=0
                )
            
            y_pred = model.predict(X_test_scaled)
            if n_classes == 2:
                y_pred_classes = (y_pred > 0.5).astype(int).flatten()
            else:
                y_pred_classes = np.argmax(y_pred, axis=1)
            
            y_pred_labels = le.inverse_transform(y_pred_classes)
            
            accuracy = accuracy_score(y_test, y_pred_labels)
            precision = precision_score(y_test, y_pred_labels, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred_labels, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred_labels, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred_labels)
            
            return {
                'model': model,
                'scaler': scaler,
                'label_encoder': le,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm.tolist(),
                'history': history.history,
                'y_test': y_test.tolist(),
                'y_pred': y_pred_labels.tolist(),
                'architecture': hidden_layers,
                'epochs_trained': len(history.history['loss'])
            }
            
        except Exception as e:
            raise Exception(f"Error in Keras classification: {str(e)}")
    
    @staticmethod
    def train_keras_regression(X: pd.DataFrame, y: pd.Series,
                               hidden_layers: List[int] = [128, 64, 32],
                               epochs: int = 50, batch_size: int = 32,
                               validation_split: float = 0.2) -> Dict:
        """Train Keras neural network for regression"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            
            y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
            
            model = DeepLearningModels.create_keras_model(
                input_dim=X_train_scaled.shape[1],
                output_dim=1,
                task_type='regression',
                hidden_layers=hidden_layers
            )
            
            early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            
            with st.spinner('Training neural network...'):
                history = model.fit(
                    X_train_scaled, y_train_scaled,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=[early_stop, reduce_lr],
                    verbose=0
                )
            
            y_pred_scaled = model.predict(X_test_scaled).flatten()
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return {
                'model': model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'history': history.history,
                'y_test': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'architecture': hidden_layers,
                'epochs_trained': len(history.history['loss'])
            }
            
        except Exception as e:
            raise Exception(f"Error in Keras regression: {str(e)}")
    
    @staticmethod
    def train_sklearn_neural_network(X: pd.DataFrame, y: pd.Series, task_type: str = 'classification',
                                     hidden_layers: Tuple[int, ...] = (100, 50, 25),
                                     max_iter: int = 300) -> Dict:
        """Train sklearn MLP neural network (fallback jika TensorFlow tidak available)"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            if task_type == 'classification':
                model = MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    activation='relu',
                    solver='adam',
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.2,
                    random_state=42
                )
                
                with st.spinner('Training sklearn neural network...'):
                    model.fit(X_train_scaled, y_train)
                
                y_pred = model.predict(X_test_scaled)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                
                return {
                    'model': model,
                    'scaler': scaler,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'confusion_matrix': cm.tolist(),
                    'y_test': y_test.tolist(),
                    'y_pred': y_pred.tolist(),
                    'architecture': hidden_layers,
                    'iterations': model.n_iter_,
                    'loss_curve': model.loss_curve_
                }
            
            else:
                model = MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    activation='relu',
                    solver='adam',
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.2,
                    random_state=42
                )
                
                with st.spinner('Training sklearn neural network...'):
                    model.fit(X_train_scaled, y_train)
                
                y_pred = model.predict(X_test_scaled)
                
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                return {
                    'model': model,
                    'scaler': scaler,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2_score': r2,
                    'y_test': y_test.tolist(),
                    'y_pred': y_pred.tolist(),
                    'architecture': hidden_layers,
                    'iterations': model.n_iter_,
                    'loss_curve': model.loss_curve_
                }
                
        except Exception as e:
            raise Exception(f"Error in sklearn neural network: {str(e)}")
    
    @staticmethod
    def plot_training_history(history: Dict, metric: str = 'loss') -> go.Figure:
        """Plot training history untuk Keras models"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=history[metric],
            name=f'Training {metric}',
            mode='lines',
            line=dict(color='blue')
        ))
        
        if f'val_{metric}' in history:
            fig.add_trace(go.Scatter(
                y=history[f'val_{metric}'],
                name=f'Validation {metric}',
                mode='lines',
                line=dict(color='red')
            ))
        
        fig.update_layout(
            title=f'Model {metric.capitalize()} Over Epochs',
            xaxis_title='Epoch',
            yaxis_title=metric.capitalize(),
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_sklearn_loss_curve(loss_curve: List[float]) -> go.Figure:
        """Plot loss curve untuk sklearn MLP"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=loss_curve,
            mode='lines',
            line=dict(color='blue'),
            name='Training Loss'
        ))
        
        fig.update_layout(
            title='Training Loss Over Iterations',
            xaxis_title='Iteration',
            yaxis_title='Loss',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_deep_learning_dashboard(df: pd.DataFrame, target_column: str):
        """Create dashboard untuk deep learning models"""
        st.subheader("🧠 Deep Learning Models")
        
        if target_column not in df.columns:
            st.error("Target column tidak ditemukan")
            return
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        if len(numeric_cols) == 0:
            st.error("Tidak ada numeric features untuk training")
            return
        
        X = df[numeric_cols]
        y = df[target_column]
        
        task_type = st.selectbox(
            "Task Type",
            ["classification", "regression"],
            help="Pilih classification untuk categorical target, regression untuk numeric target"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_tensorflow = st.checkbox(
                "Use TensorFlow/Keras" if HAS_TENSORFLOW else "TensorFlow not available - Using sklearn",
                value=HAS_TENSORFLOW,
                disabled=not HAS_TENSORFLOW,
                help="TensorFlow provides more advanced features but sklearn MLP is faster"
            )
        
        with col2:
            architecture = st.text_input(
                "Hidden Layers Architecture",
                value="128,64,32",
                help="Comma-separated layer sizes, e.g., 128,64,32"
            )
        
        if st.button("🚀 Train Neural Network", type="primary"):
            try:
                layers = [int(x.strip()) for x in architecture.split(',')]
                
                if use_tensorflow and HAS_TENSORFLOW:
                    if task_type == 'classification':
                        results = DeepLearningModels.train_keras_classification(X, y, hidden_layers=layers)
                        
                        st.success(f"✅ Model trained! Accuracy: {results['accuracy']:.4f}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Accuracy", f"{results['accuracy']:.4f}")
                        col2.metric("Precision", f"{results['precision']:.4f}")
                        col3.metric("Recall", f"{results['recall']:.4f}")
                        col4.metric("F1-Score", f"{results['f1_score']:.4f}")
                        
                        if 'history' in results:
                            st.plotly_chart(DeepLearningModels.plot_training_history(results['history'], 'loss'), use_container_width=True)
                            st.plotly_chart(DeepLearningModels.plot_training_history(results['history'], 'accuracy'), use_container_width=True)
                        
                        st.session_state.trained_models['deep_learning'] = results
                        
                    else:
                        results = DeepLearningModels.train_keras_regression(X, y, hidden_layers=layers)
                        
                        st.success(f"✅ Model trained! R² Score: {results['r2_score']:.4f}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("RMSE", f"{results['rmse']:.4f}")
                        col2.metric("MAE", f"{results['mae']:.4f}")
                        col3.metric("MSE", f"{results['mse']:.4f}")
                        col4.metric("R² Score", f"{results['r2_score']:.4f}")
                        
                        if 'history' in results:
                            st.plotly_chart(DeepLearningModels.plot_training_history(results['history'], 'loss'), use_container_width=True)
                        
                        st.session_state.trained_models['deep_learning'] = results
                
                else:
                    results = DeepLearningModels.train_sklearn_neural_network(
                        X, y, 
                        task_type=task_type,
                        hidden_layers=tuple(layers)
                    )
                    
                    if task_type == 'classification':
                        st.success(f"✅ Model trained! Accuracy: {results['accuracy']:.4f}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Accuracy", f"{results['accuracy']:.4f}")
                        col2.metric("Precision", f"{results['precision']:.4f}")
                        col3.metric("Recall", f"{results['recall']:.4f}")
                        col4.metric("F1-Score", f"{results['f1_score']:.4f}")
                    else:
                        st.success(f"✅ Model trained! R² Score: {results['r2_score']:.4f}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("RMSE", f"{results['rmse']:.4f}")
                        col2.metric("MAE", f"{results['mae']:.4f}")
                        col3.metric("MSE", f"{results['mse']:.4f}")
                        col4.metric("R² Score", f"{results['r2_score']:.4f}")
                    
                    if 'loss_curve' in results:
                        st.plotly_chart(DeepLearningModels.plot_sklearn_loss_curve(results['loss_curve']), use_container_width=True)
                    
                    st.session_state.trained_models['deep_learning'] = results
                
                st.info(f"ℹ️ Architecture: {results['architecture']}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

deep_learning = DeepLearningModels()
