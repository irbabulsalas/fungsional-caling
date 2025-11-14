import streamlit as st
from typing import Optional, Dict
from database.db_manager import db_manager, User
from datetime import datetime

class AuthManager:
    @staticmethod
    def init_session():
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'current_user' not in st.session_state:
            st.session_state.current_user = None
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        if 'user_role' not in st.session_state:
            st.session_state.user_role = None
    
    @staticmethod
    def login(username: str, password: str) -> bool:
        user = db_manager.authenticate_user(username, password)
        if user:
            st.session_state.authenticated = True
            st.session_state.current_user = user.username
            st.session_state.user_id = user.id
            st.session_state.user_role = user.role
            
            db_manager.log_audit(
                user_id=user.id,
                action='login',
                resource_type='user',
                resource_id=user.id
            )
            return True
        return False
    
    @staticmethod
    def logout():
        if st.session_state.get('user_id'):
            db_manager.log_audit(
                user_id=st.session_state.user_id,
                action='logout',
                resource_type='user',
                resource_id=st.session_state.user_id
            )
        
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.user_id = None
        st.session_state.user_role = None
    
    @staticmethod
    def register(username: str, email: str, password: str, role: str = 'analyst') -> Optional[str]:
        try:
            user = db_manager.create_user(username, email, password, role)
            if user:
                db_manager.log_audit(
                    user_id=user.id,
                    action='register',
                    resource_type='user',
                    resource_id=user.id
                )
                return None
            else:
                return "Username atau email sudah digunakan"
        except Exception as e:
            return f"Error saat registrasi: {str(e)}"
    
    @staticmethod
    def is_authenticated() -> bool:
        return st.session_state.get('authenticated', False)
    
    @staticmethod
    def get_current_user() -> Optional[str]:
        return st.session_state.get('current_user')
    
    @staticmethod
    def get_user_id() -> Optional[int]:
        return st.session_state.get('user_id')
    
    @staticmethod
    def get_user_role() -> Optional[str]:
        return st.session_state.get('user_role')
    
    @staticmethod
    def has_permission(required_role: str) -> bool:
        role_hierarchy = {
            'viewer': 1,
            'analyst': 2,
            'admin': 3
        }
        
        current_role = st.session_state.get('user_role', 'viewer')
        return role_hierarchy.get(current_role, 0) >= role_hierarchy.get(required_role, 0)
    
    @staticmethod
    def render_login_page():
        st.markdown("### 🔐 Login")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if username and password:
                    if AuthManager.login(username, password):
                        st.success("✅ Login berhasil!")
                        st.rerun()
                    else:
                        st.error("❌ Username atau password salah")
                else:
                    st.warning("⚠️ Mohon isi username dan password")
    
    @staticmethod
    def render_register_page():
        st.markdown("### 📝 Register")
        
        with st.form("register_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            password_confirm = st.text_input("Konfirmasi Password", type="password")
            role = st.selectbox("Role", ["analyst", "viewer"])
            submit = st.form_submit_button("Register")
            
            if submit:
                if not all([username, email, password, password_confirm]):
                    st.warning("⚠️ Mohon isi semua field")
                elif password != password_confirm:
                    st.error("❌ Password tidak cocok")
                elif len(password) < 8:
                    st.error("❌ Password minimal 8 karakter")
                else:
                    error = AuthManager.register(username, email, password, role)
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        st.success("✅ Registrasi berhasil! Silakan login.")
    
    @staticmethod
    def render_auth_sidebar():
        if AuthManager.is_authenticated():
            with st.sidebar:
                st.markdown("---")
                st.markdown(f"**👤 User:** {AuthManager.get_current_user()}")
                st.markdown(f"**🎭 Role:** {AuthManager.get_user_role()}")
                if st.button("🚪 Logout"):
                    AuthManager.logout()
                    st.rerun()
        else:
            with st.sidebar:
                st.markdown("---")
                auth_mode = st.radio("Mode", ["Login", "Register"], horizontal=True)
                
                if auth_mode == "Login":
                    AuthManager.render_login_page()
                else:
                    AuthManager.render_register_page()

auth_manager = AuthManager()
