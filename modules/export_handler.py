import pandas as pd
import streamlit as st
from fpdf import FPDF
import joblib
import json
import io
from typing import Dict, Any, Optional
import sys
sys.path.append('..')
from utils.error_handler import safe_execute

def export_dataframe_csv(df: pd.DataFrame, filename: str = "data.csv"):
    try:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )
        return True
    except Exception as e:
        st.error(f"❌ CSV export failed: {str(e)}")
        return False

def export_dataframe_excel(df: pd.DataFrame, filename: str = "data.xlsx"):
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Data')
        output.seek(0)
        
        st.download_button(
            label="📥 Download Excel",
            data=output,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        return True
    except Exception as e:
        st.error(f"❌ Excel export failed: {str(e)}")
        return False

def export_dataframe_json(df: pd.DataFrame, filename: str = "data.json"):
    try:
        json_str = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=filename,
            mime="application/json"
        )
        return True
    except Exception as e:
        st.error(f"❌ JSON export failed: {str(e)}")
        return False

def export_model(model: Any, filename: str = "model.pkl"):
    try:
        model_bytes = io.BytesIO()
        joblib.dump(model, model_bytes)
        model_bytes.seek(0)
        
        st.download_button(
            label="📥 Download Model (.pkl)",
            data=model_bytes,
            file_name=filename,
            mime="application/octet-stream"
        )
        return True
    except Exception as e:
        st.error(f"❌ Model export failed: {str(e)}")
        return False

def export_metrics_report(metrics: Dict, filename: str = "metrics.json"):
    try:
        json_str = json.dumps(metrics, indent=2)
        st.download_button(
            label="📥 Download Metrics (JSON)",
            data=json_str,
            file_name=filename,
            mime="application/json"
        )
        return True
    except Exception as e:
        st.error(f"❌ Metrics export failed: {str(e)}")
        return False

def generate_pdf_report(
    title: str,
    sections: Dict[str, str],
    filename: str = "report.pdf"
):
    try:
        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, title, ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font("Arial", "", 12)
        for section_title, content in sections.items():
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, section_title, ln=True)
            pdf.set_font("Arial", "", 11)
            pdf.multi_cell(0, 5, content)
            pdf.ln(5)
        
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf"
        )
        return True
    except Exception as e:
        st.error(f"❌ PDF generation failed: {str(e)}")
        return False

def create_export_center(df: Optional[pd.DataFrame] = None, models: Optional[Dict] = None):
    st.subheader("📦 Export Center")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Data Exports**")
        if df is not None and not df.empty:
            export_dataframe_csv(df, "cleaned_data.csv")
            export_dataframe_excel(df, "data_analysis.xlsx")
            export_dataframe_json(df, "data.json")
        else:
            st.info("Upload data to enable exports")
    
    with col2:
        st.write("**🤖 Model Exports**")
        if models and len(models) > 0:
            for model_name, model_data in models.items():
                if 'model' in model_data:
                    export_model(model_data['model'], f"{model_name}_model.pkl")
                if 'metrics' in model_data:
                    export_metrics_report(model_data['metrics'], f"{model_name}_metrics.json")
        else:
            st.info("Train models to enable exports")
