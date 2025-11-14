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
### Current Features (Updated: November 14, 2025)
- ✅ **Overview Dashboard** - Data quality metrics, quick insights dengan Gemini AI
- ✅ **Data Profiling** - Comprehensive data analysis, correlation heatmaps, automated cleaning
- ✅ **EDA (Exploratory Data Analysis)** - Distributions, relationships, comparisons dengan visualizations
- ✅ **ML Models** - Classification, Regression, Clustering dengan Random Forest, Logistic Regression, SVM
- ✅ **Advanced ML** - XGBoost, LightGBM, Neural Networks (MLP), Ensemble methods
- ✅ **Time Series Analysis** - ARIMA, SARIMA, seasonal decomposition, Auto ARIMA, stationarity tests
- ✅ **Text Analytics** - Sentiment analysis, WordCloud, bigrams, NLP features
- ✅ **Export Center** - Export cleaned data, trained models, comprehensive reports
- ✅ **Gemini AI Integration** - Chat assistant untuk data insights and analysis guidance
- ✅ **Rate Limiting** - Smart API rate management
- ✅ **Responsive UI** - Mobile-friendly interface

### Newly Added Advanced Features
- 🎯 **XGBoost & LightGBM** - State-of-the-art gradient boosting models
- 🧠 **Neural Networks** - Multi-layer perceptron untuk classification/regression
- 📈 **Time Series Forecasting** - ARIMA, SARIMA dengan automatic parameter selection
- 📊 **Seasonal Decomposition** - Trend, seasonal, residual analysis
- 🤝 **Ensemble Methods** - Model combination untuk better performance
- 🎨 **Feature Importance** - Understanding model decisions

### Future Enhancements
Fitur yang masih dalam rencana:
- Deep Learning (TensorFlow/Keras for CNN, LSTM)
- Prophet time series forecasting
- PostgreSQL database integration untuk project persistence
- Bayesian Optimization (Optuna) untuk hyperparameter tuning
- REST API endpoints untuk model deployment
- Cloud storage integration (Google Drive, AWS S3)
- Real-time collaboration features
- Enhanced security dan authentication

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
