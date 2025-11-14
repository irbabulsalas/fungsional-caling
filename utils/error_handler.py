import logging
import streamlit as st
from datetime import datetime
from typing import Callable, Any, Optional

logging.basicConfig(
    filename='app_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def log_error(feature_name: str, error: Exception, user_context: Optional[dict] = None) -> str:
    error_info = {
        'timestamp': datetime.now().isoformat(),
        'feature': feature_name,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'user_context': user_context or {}
    }
    
    logging.error(f"Feature Error: {error_info}")
    error_id = f"ERR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return error_id

def safe_execute(feature_name: str, func: Callable, *args, **kwargs) -> Any:
    try:
        return func(*args, **kwargs)
    except ValueError as e:
        st.error(f"""
        ❌ **{feature_name} - Data Format Issue**
        
        {str(e)}
        
        **Solutions:**
        • Check data types are correct
        • Remove special characters
        • Ensure numeric columns contain only numbers
        
        [🔧 Try Auto-fix] [📖 Learn More]
        """)
        log_error(feature_name, e)
        return None
    except KeyError as e:
        st.error(f"""
        ❌ **{feature_name} - Column Not Found**
        
        Column {str(e)} doesn't exist in the dataset.
        
        **Solutions:**
        • Check column name spelling
        • View available columns
        • Reload your data
        
        [📋 Show Columns]
        """)
        log_error(feature_name, e)
        return None
    except MemoryError as e:
        st.error(f"""
        ❌ **{feature_name} - Memory Limit**
        
        Dataset is too large for this operation.
        
        **Solutions:**
        • Use data sampling (analyze subset)
        • Remove unnecessary columns
        • Export to Parquet format
        
        [📉 Sample 10% of Data]
        """)
        log_error(feature_name, e)
        return None
    except Exception as e:
        error_id = log_error(feature_name, e)
        st.error(f"""
        ⚠️ **{feature_name} Unavailable**
        
        This feature encountered an issue, but other features still work!
        
        Error ID: `{error_id}`
        Error type: {type(e).__name__}
        
        **You can still:**
        • Try other features
        • Reload the page
        • Contact support with error ID
        
        [🔄 Retry] [📧 Report Issue]
        """)
        return None

def validate_dataframe(df, min_rows: int = 10, operation: str = "analysis") -> bool:
    if df is None or df.empty:
        st.warning("""
        📭 **No data loaded**
        
        Please upload a dataset first!
        
        [📤 Upload Data] [📝 Try Sample Dataset]
        """)
        return False
    
    if len(df) < min_rows:
        st.warning(f"""
        ⚠️ **Dataset too small**
        
        You need at least {min_rows} rows for {operation}.
        Current: {len(df)} rows
        
        [📤 Upload larger dataset]
        """)
        return False
    
    return True

def show_error_recovery(error_type: str):
    recovery_guide = {
        'ValueError': {
            'message': 'Data format issue',
            'steps': [
                '1. Check your data has correct types',
                '2. Remove special characters',
                '3. Ensure numeric columns contain only numbers'
            ]
        },
        'KeyError': {
            'message': 'Column not found',
            'steps': [
                '1. Verify column name spelling',
                '2. Check for extra spaces',
                '3. Reload your data'
            ]
        },
        'MemoryError': {
            'message': 'Dataset too large',
            'steps': [
                '1. Use data sampling',
                '2. Remove unnecessary columns',
                '3. Export to Parquet format'
            ]
        }
    }
    
    guide = recovery_guide.get(error_type, {
        'message': 'Unexpected error',
        'steps': [
            '1. Refresh the page',
            '2. Re-upload your data',
            '3. Contact support if persists'
        ]
    })
    
    st.warning(f"**{guide['message']}**")
    for step in guide['steps']:
        st.write(step)
