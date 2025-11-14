import os
import json
import streamlit as st
from google import genai
from google.genai import types
from typing import Optional, List, Dict, Any
import sys
sys.path.append('..')
from utils.error_handler import log_error

api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    client = genai.Client(api_key=api_key)
else:
    client = None

AVAILABLE_FUNCTIONS = [
    {
        "name": "profile_data",
        "description": "Generate comprehensive data profile including statistics, data types, missing values, and quality assessment",
        "parameters": {
            "type": "object",
            "properties": {
                "include_correlations": {
                    "type": "boolean",
                    "description": "Include correlation analysis"
                },
                "include_distributions": {
                    "type": "boolean",
                    "description": "Include distribution analysis"
                }
            }
        }
    },
    {
        "name": "clean_data",
        "description": "Clean dataset with various strategies for missing values, duplicates, and outliers",
        "parameters": {
            "type": "object",
            "properties": {
                "handle_missing": {
                    "type": "string",
                    "enum": ["drop", "mean", "median", "mode", "knn"],
                    "description": "Strategy for handling missing values"
                },
                "remove_duplicates": {
                    "type": "boolean",
                    "description": "Remove duplicate rows"
                },
                "handle_outliers": {
                    "type": "string",
                    "enum": ["none", "iqr", "zscore", "isolation_forest"],
                    "description": "Method for detecting/removing outliers"
                }
            },
            "required": ["handle_missing"]
        }
    },
    {
        "name": "train_classification_model",
        "description": "Train classification models (Random Forest, XGBoost, Logistic Regression, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "target_column": {
                    "type": "string",
                    "description": "Name of the target column for classification"
                },
                "models": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["random_forest", "xgboost", "logistic", "svm", "lightgbm"]
                    },
                    "description": "List of models to train"
                },
                "tune_hyperparameters": {
                    "type": "boolean",
                    "description": "Perform hyperparameter tuning"
                }
            },
            "required": ["target_column"]
        }
    },
    {
        "name": "create_visualization",
        "description": "Create interactive visualizations (scatter, bar, line, heatmap, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["scatter", "bar", "line", "histogram", "box", "heatmap", "violin"],
                    "description": "Type of chart to create"
                },
                "x_column": {
                    "type": "string",
                    "description": "Column for x-axis"
                },
                "y_column": {
                    "type": "string",
                    "description": "Column for y-axis (optional for some charts)"
                },
                "color_by": {
                    "type": "string",
                    "description": "Column to use for color grouping"
                }
            },
            "required": ["chart_type"]
        }
    },
    {
        "name": "analyze_text",
        "description": "Perform text analytics: sentiment analysis, topic modeling, word clouds",
        "parameters": {
            "type": "object",
            "properties": {
                "text_column": {
                    "type": "string",
                    "description": "Column containing text data"
                },
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["sentiment", "topic_modeling", "wordcloud", "ngrams"]
                    },
                    "description": "Text analysis tasks to perform"
                }
            },
            "required": ["text_column", "tasks"]
        }
    }
]

def chat_with_gemini(
    user_message: str,
    chat_history: Optional[List[Dict]] = None,
    context: Optional[Dict] = None,
    max_retries: int = 3
) -> Optional[str]:
    if client is None:
        return "🔑 Gemini API key not configured. Please add GEMINI_API_KEY to your secrets."
    
    try:
        system_prompt = """You are an expert data scientist and AI assistant helping users analyze their data.
        
You can help with:
- Data profiling and quality assessment
- Data cleaning and preprocessing
- Exploratory data analysis
- Machine learning (classification, regression, clustering)
- Text analytics and NLP
- Data visualization
- Statistical analysis

When users ask for analysis, use the available functions to help them.
Be concise, clear, and provide actionable insights.
Always explain results in simple, non-technical language.
"""
        
        if context:
            system_prompt += f"\n\nCurrent data context: {json.dumps(context, indent=2)}"
        
        messages = []
        if chat_history:
            for msg in chat_history[-10:]:
                role = "user" if msg.get("role") == "user" else "model"
                messages.append(types.Content(role=role, parts=[types.Part(text=msg.get("content", ""))]))
        
        messages.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
                top_p=0.95,
                max_output_tokens=2048
            )
        )
        
        return response.text if response.text else "I apologize, but I couldn't generate a response. Please try again."
        
    except Exception as e:
        error_type = type(e).__name__
        
        if "429" in str(e) or "quota" in str(e).lower():
            return "🚫 API rate limit reached. Please wait for the cooldown period."
        
        elif "network" in str(e).lower() or "connection" in str(e).lower():
            if max_retries > 0:
                return chat_with_gemini(user_message, chat_history, context, max_retries - 1)
            return "❌ Connection error. Please check your internet connection."
        
        elif "api_key" in str(e).lower() or "unauthorized" in str(e).lower():
            return "🔑 API key issue. Please check your Gemini API key in settings."
        
        else:
            log_error("Gemini Chat", e)
            return f"⚠️ AI assistant temporarily unavailable. Error: {error_type}"

def get_data_context(df) -> Dict:
    if df is None or df.empty:
        return {}
    
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist()[:20],
        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist()[:10],
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()[:10],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum())
    }

def generate_insights(df, analysis_type: str = "general") -> str:
    try:
        context = get_data_context(df)
        
        if analysis_type == "general":
            prompt = f"""Analyze this dataset and provide 3-5 key insights:
            
Dataset info:
- {context['rows']} rows, {context['columns']} columns
- Columns: {', '.join(context['column_names'][:10])}
- Missing values: {context['missing_values']}
- Duplicates: {context['duplicates']}

Provide actionable insights in bullet points."""
        
        elif analysis_type == "quality":
            prompt = f"""Assess the data quality of this dataset:
            
- Missing values: {context['missing_values']} ({(context['missing_values']/(context['rows']*context['columns'])*100):.1f}%)
- Duplicates: {context['duplicates']}

Provide quality assessment and recommendations."""
        
        else:
            prompt = f"Analyze this dataset with {context['rows']} rows and {context['columns']} columns."
        
        response = chat_with_gemini(prompt, context=context)
        return response or "Unable to generate insights at this time."
        
    except Exception as e:
        log_error("Generate Insights", e)
        return "Unable to generate insights. Please try manual analysis."
