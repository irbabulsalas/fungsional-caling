import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, List
import sys
sys.path.append('..')
from utils.helpers import get_numeric_columns, get_categorical_columns

def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, color_by: Optional[str] = None):
    try:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_by if color_by else None,
            title=f"{y_col} vs {x_col}",
            template="plotly_white",
            height=500
        )
        
        fig.update_layout(
            hovermode='closest',
            showlegend=True if color_by else False
        )
        
        return fig
    except Exception as e:
        st.error(f"❌ Scatter plot failed: {str(e)}")
        return None

def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: Optional[str] = None):
    try:
        if y_col:
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                title=f"{y_col} by {x_col}",
                template="plotly_white",
                height=500
            )
        else:
            value_counts = df[x_col].value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution of {x_col}",
                template="plotly_white",
                height=500,
                labels={'x': x_col, 'y': 'Count'}
            )
        
        return fig
    except Exception as e:
        st.error(f"❌ Bar chart failed: {str(e)}")
        return None

def create_histogram(df: pd.DataFrame, column: str, bins: int = 30):
    try:
        fig = px.histogram(
            df,
            x=column,
            nbins=bins,
            title=f"Distribution of {column}",
            template="plotly_white",
            height=500
        )
        
        fig.update_layout(showlegend=False)
        
        return fig
    except Exception as e:
        st.error(f"❌ Histogram failed: {str(e)}")
        return None

def create_box_plot(df: pd.DataFrame, y_col: str, x_col: Optional[str] = None):
    try:
        fig = px.box(
            df,
            y=y_col,
            x=x_col if x_col else None,
            title=f"Box Plot of {y_col}",
            template="plotly_white",
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"❌ Box plot failed: {str(e)}")
        return None

def create_correlation_heatmap(df: pd.DataFrame):
    try:
        numeric_cols = get_numeric_columns(df)
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ Need at least 2 numeric columns for correlation")
            return None
        
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Correlation Heatmap",
            template="plotly_white",
            height=600,
            xaxis_showgrid=False,
            yaxis_showgrid=False
        )
        
        return fig
    except Exception as e:
        st.error(f"❌ Correlation heatmap failed: {str(e)}")
        return None

def create_line_chart(df: pd.DataFrame, x_col: str, y_col: str, color_by: Optional[str] = None):
    try:
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            color=color_by if color_by else None,
            title=f"{y_col} over {x_col}",
            template="plotly_white",
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"❌ Line chart failed: {str(e)}")
        return None

def create_violin_plot(df: pd.DataFrame, y_col: str, x_col: Optional[str] = None):
    try:
        fig = px.violin(
            df,
            y=y_col,
            x=x_col if x_col else None,
            title=f"Violin Plot of {y_col}",
            template="plotly_white",
            height=500,
            box=True
        )
        
        return fig
    except Exception as e:
        st.error(f"❌ Violin plot failed: {str(e)}")
        return None

def create_confusion_matrix_plot(cm: np.ndarray, labels: Optional[List] = None):
    try:
        if labels is None:
            labels = [f"Class {i}" for i in range(len(cm))]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Count")
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            template="plotly_white",
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"❌ Confusion matrix plot failed: {str(e)}")
        return None

def create_feature_importance_plot(importance_dict: dict, top_n: int = 10):
    try:
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        features, importances = zip(*sorted_features)
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(importances),
                y=list(features),
                orientation='h',
                marker_color='indianred'
            )
        ])
        
        fig.update_layout(
            title=f"Top {top_n} Feature Importances",
            xaxis_title="Importance",
            yaxis_title="Features",
            template="plotly_white",
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    except Exception as e:
        st.error(f"❌ Feature importance plot failed: {str(e)}")
        return None

def download_plotly_chart(fig, filename: str, format: str = "png"):
    try:
        if format == "png":
            img_bytes = fig.to_image(format="png", width=1200, height=800)
            st.download_button(
                label="📥 Download PNG",
                data=img_bytes,
                file_name=filename,
                mime="image/png"
            )
        elif format == "html":
            html_bytes = fig.to_html().encode()
            st.download_button(
                label="📥 Download HTML",
                data=html_bytes,
                file_name=filename.replace('.png', '.html'),
                mime="text/html"
            )
    except Exception as e:
        st.warning(f"⚠️ Download failed: {str(e)}")
