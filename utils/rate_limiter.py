import streamlit as st
import time
from datetime import datetime, timedelta
from typing import Tuple, Union

RATE_LIMIT_SECONDS = 60
MAX_REQUESTS_PER_HOUR = 15

def initialize_rate_limiter():
    if 'last_request_time' not in st.session_state:
        st.session_state.last_request_time = None
    if 'request_timestamps' not in st.session_state:
        st.session_state.request_timestamps = []
    if 'question_queue' not in st.session_state:
        st.session_state.question_queue = []

def can_make_request() -> Tuple[bool, Union[int, str]]:
    current_time = datetime.now()
    
    st.session_state.request_timestamps = [
        ts for ts in st.session_state.request_timestamps 
        if current_time - ts < timedelta(hours=1)
    ]
    
    if len(st.session_state.request_timestamps) >= MAX_REQUESTS_PER_HOUR:
        return False, "hourly_limit"
    
    if st.session_state.last_request_time:
        time_since_last = (current_time - st.session_state.last_request_time).total_seconds()
        if time_since_last < RATE_LIMIT_SECONDS:
            return False, int(RATE_LIMIT_SECONDS - time_since_last)
    
    return True, 0

def get_remaining_time() -> int:
    if not st.session_state.last_request_time:
        return 0
    
    current_time = datetime.now()
    time_since_last = (current_time - st.session_state.last_request_time).total_seconds()
    remaining = max(0, RATE_LIMIT_SECONDS - time_since_last)
    return int(remaining)

def update_rate_limit():
    current_time = datetime.now()
    st.session_state.last_request_time = current_time
    st.session_state.request_timestamps.append(current_time)

def get_remaining_requests() -> int:
    return MAX_REQUESTS_PER_HOUR - len(st.session_state.request_timestamps)

def show_rate_limit_status():
    remaining_requests = get_remaining_requests()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"📊 Remaining requests this hour: **{remaining_requests}/{MAX_REQUESTS_PER_HOUR}**")
    
    with col2:
        if st.button("ℹ️", key="api_limits_info"):
            st.info("""
            **Free API Limits:**
            - 1 minute between questions
            - 15 requests per hour
            - Resets every hour
            """)

def show_cooldown_timer():
    can_request, wait_time = can_make_request()
    
    if not can_request and wait_time != "hourly_limit":
        st.warning(f"⏳ **Please wait {wait_time} seconds before asking next question**")
        
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        for remaining in range(wait_time, 0, -1):
            progress = 1 - (remaining / RATE_LIMIT_SECONDS)
            progress_placeholder.progress(progress)
            status_placeholder.text(f"⏱️ {remaining}s remaining...")
            time.sleep(1)
            
            if get_remaining_time() == 0:
                break
        
        progress_placeholder.progress(1.0)
        status_placeholder.success("✅ Ready! You can ask now.")
        time.sleep(1)
        st.rerun()
        
    elif not can_request and wait_time == "hourly_limit":
        st.error(f"""
        🚫 **Hourly limit reached!**
        
        You've used all {MAX_REQUESTS_PER_HOUR} requests this hour.
        
        **Meanwhile, you can:**
        • Explore dashboards
        • Download results
        • Try sample datasets
        • Use manual features
        
        💡 Limit resets in approximately {get_reset_time()} minutes
        """)
        return False
    
    return True

def get_reset_time() -> int:
    if not st.session_state.request_timestamps:
        return 0
    oldest_request = min(st.session_state.request_timestamps)
    reset_time = oldest_request + timedelta(hours=1)
    minutes_until_reset = int((reset_time - datetime.now()).total_seconds() / 60)
    return max(0, minutes_until_reset)
