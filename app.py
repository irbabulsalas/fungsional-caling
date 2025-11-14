import streamlit as st
import google.generativeai as genai
import os
import pandas as pd

st.set_page_config(
    page_title="Gemini Data Pilot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Gemini Data Pilot")
st.markdown("### AI-Powered Data Analysis Assistant")

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.warning("⚠️ Gemini API Key belum dikonfigurasi")
    st.info("""
    **Cara setup:**
    1. Dapatkan API key dari [Google AI Studio](https://makersuite.google.com/app/apikey)
    2. Klik tombol "Secrets" di panel kiri Replit
    3. Tambahkan secret dengan nama `GEMINI_API_KEY` dan masukkan API key Anda
    4. Restart aplikasi
    """)
    st.stop()

genai.configure(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel('gemini-pro')

col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("⚙️ Pengaturan")
    
    if st.button("🗑️ Hapus Riwayat Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Upload File Data**")
    uploaded_file = st.file_uploader(
        "Upload CSV untuk analisis",
        type=['csv'],
        help="Upload file CSV untuk dianalisis dengan AI"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded: {len(df)} baris, {len(df.columns)} kolom")
            
            with st.expander("👀 Preview Data"):
                st.dataframe(df.head(10))
            
            if st.button("📊 Analisis Data"):
                data_summary = f"""
                Dataset memiliki {len(df)} baris dan {len(df.columns)} kolom.
                
                Kolom: {', '.join(df.columns.tolist())}
                
                Info tipe data:
                {df.dtypes.to_string()}
                
                Statistik deskriptif:
                {df.describe().to_string()}
                """
                
                prompt = f"Tolong analisis dataset berikut dan berikan insight yang berguna:\n\n{data_summary}"
                
                with st.spinner("Menganalisis data..."):
                    response = st.session_state.model.generate_content(prompt)
                    
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"Analisis data dari file: {uploaded_file.name}"
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.text
                    })
                    st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")

with col1:
    st.subheader("💬 Chat dengan Gemini AI")
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    if prompt := st.chat_input("Tanya sesuatu tentang data atau AI..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Berpikir..."):
                try:
                    response = st.session_state.model.generate_content(prompt)
                    st.markdown(response.text)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.text
                    })
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Tentang Gemini Data Pilot

Aplikasi ini menggunakan Google Gemini AI untuk membantu Anda:
- 💬 Chat dengan AI yang canggih
- 📊 Menganalisis data CSV
- 💡 Mendapatkan insight dari data
- 🤔 Menjawab pertanyaan tentang data

**Powered by Google Gemini AI**
""")
