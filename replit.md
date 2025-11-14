# Gemini Data Pilot

## Overview
Aplikasi web berbasis Streamlit yang mengintegrasikan Google Gemini AI untuk analisis data dan chat interaktif. Aplikasi ini memungkinkan pengguna untuk:
- Chat dengan AI Gemini yang canggih
- Upload dan analisis file CSV
- Mendapatkan insight otomatis dari data
- Mengajukan pertanyaan tentang data

## Tech Stack
- **Frontend**: Streamlit 1.51.0
- **AI**: Google Generative AI (Gemini Pro)
- **Data Processing**: Pandas
- **Python**: 3.11

## Setup
1. **API Key**: Aplikasi memerlukan `GEMINI_API_KEY` untuk berfungsi
   - Dapatkan dari [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Tambahkan ke Secrets dengan nama `GEMINI_API_KEY`
   
2. **Dependencies**: Terinstall otomatis via uv
   - streamlit
   - google-generativeai
   - pandas

## Project Structure
```
.
├── app.py                 # Main application
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── .gitignore            # Python gitignore
├── pyproject.toml        # Python dependencies
└── replit.md             # Documentation
```

## Features
### Current Features
- ✅ Chat interface dengan Gemini AI
- ✅ Upload dan preview CSV files
- ✅ Auto-analysis data dengan AI
- ✅ Chat history management
- ✅ Responsive UI dengan sidebar

### Future Enhancements (dari user)
Daftar fitur yang direncanakan:
- Export hasil analisis (cleaned data, model, report)
- Deep learning integration (TensorFlow/Keras)
- Advanced ML algorithms dan hyperparameter tuning
- Time series analysis (ARIMA, Prophet)
- Database integration (PostgreSQL)
- Real-time collaboration
- REST API endpoints
- Cloud storage integration
- Enhanced security features

## Configuration
- **Port**: 5000 (required untuk Replit)
- **Host**: 0.0.0.0 (required untuk proxy)
- **CORS**: Disabled untuk kompatibilitas Replit
- **XSRF Protection**: Disabled untuk development

## Running Locally
```bash
streamlit run app.py --server.port=5000 --server.address=0.0.0.0
```

## Deployment
Configured untuk Replit Autoscale deployment. Klik tombol "Deploy" untuk publish.

## Recent Changes
- 2025-11-14: Initial setup dari GitHub import
- 2025-11-14: Created Streamlit app dengan Gemini AI integration
- 2025-11-14: Configured untuk Replit environment

## User Preferences
- Bahasa: Indonesia
- Framework: Streamlit untuk simplicity
