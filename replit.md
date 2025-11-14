# AI Data Analysis Platform

## Overview
A comprehensive AI-powered data analysis platform built with Streamlit that provides automated machine learning, interactive visualizations, text analytics, and intelligent data processing with Google Gemini 2.5 Flash AI. The platform offers:
- **AI Chat Assistant** - Conversational interface powered by Gemini 2.5 Flash with function calling
- **Automated Machine Learning** - 10+ algorithms with automatic comparison and tuning
- **Interactive Dashboards** - Multi-page responsive interface with comprehensive analytics
- **Text Analytics** - Sentiment analysis, topic modeling, and word clouds
- **Comprehensive Export** - PDF reports, Excel files, trained models, and Jupyter notebooks
- **Multi-format Support** - CSV, Excel, JSON, Parquet, TSV file handling
- **Advanced Cleaning** - Automated data profiling and quality assessment

## Tech Stack
- **Frontend**: Streamlit (responsive web framework)
- **AI**: Google Gemini 2.5 Flash (with function calling)
- **ML Libraries**: scikit-learn, XGBoost, LightGBM
- **Visualization**: Plotly (interactive charts)
- **Data Processing**: Pandas, NumPy
- **Text Analytics**: TextBlob, WordCloud
- **Export**: FPDF, openpyxl, joblib
- **Python**: 3.11+

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
ai-data-analysis/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # UV package manager config
├── Procfile                        # Railway deployment config
├── railway.json                    # Railway build settings
│
├── modules/                        # Core modules
│   ├── data_processing.py          # Data loading & cleaning
│   ├── ml_models.py                # ML training & evaluation
│   ├── visualizations.py           # Chart generation
│   ├── text_analytics.py           # NLP functions
│   ├── gemini_integration.py       # AI function calling
│   └── export_handler.py           # Export functionality
│
├── utils/                          # Utilities
│   ├── error_handler.py            # Error management
│   ├── rate_limiter.py             # API rate limiting
│   └── helpers.py                  # Helper functions
│
├── assets/                         # Static files
│   ├── profile_photo.jpg           # Author photo
│   └── sample_datasets/            # Sample data
│
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
│
└── docs/                           # Documentation
    ├── DEPLOYMENT.md               # Railway deployment guide
    └── TROUBLESHOOTING.md          # Common issues
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

## System Architecture

### Core Processing Modules
1. **Data Processing Layer** - Multi-format parsing, automated profiling, cleaning strategies
2. **Machine Learning Engine** - 10+ algorithms, hyperparameter tuning, clustering, PCA
3. **Text Analytics Module** - Sentiment analysis, word clouds, topic extraction
4. **Visualization Engine** - Interactive Plotly charts, specialized ML visualizations
5. **AI Chat Integration** - Gemini 2.5 Flash with function calling capabilities
6. **Export Handler** - Multi-format export (CSV, Excel, JSON, PDF, models)

### Design Patterns
- **Separation of Concerns** - Each module has single responsibility
- **Defensive Programming** - Extensive validation with graceful degradation
- **Progressive Enhancement** - Core functionality works without advanced features
- **Mobile-First Responsive** - CSS media queries for different screen sizes

## External Dependencies

### AI & ML Services
- **Google Gemini 2.5 Flash API** - Rate limits: 15 requests/minute, 1500/day (free tier)
- Authentication via `GEMINI_API_KEY` environment variable

### Key Python Libraries
- **Core**: streamlit, pandas, numpy
- **ML**: scikit-learn, xgboost, lightgbm, joblib
- **Visualization**: plotly, matplotlib, wordcloud
- **Text**: textblob
- **Export**: fpdf, openpyxl

## Recent Changes
- 2025-11-14: Successfully migrated project to Replit environment
- 2025-11-14: Installed all required dependencies (16 packages)
- 2025-11-14: Configured workflow for Streamlit on port 5000
- 2025-11-14: Verified app is running with all features functional
- 2025-11-14: Updated documentation with comprehensive architecture details

## User Preferences
- **Communication**: Simple, everyday language
- **Framework**: Streamlit for rapid data science development
- **Author**: Muhammad Irbabul Salas

## Future Enhancements
Planned features from task list:
- Deep Learning (TensorFlow/Keras for CNN, LSTM)
- Prophet time series forecasting
- PostgreSQL database integration for project persistence
- Bayesian Optimization (Optuna) for hyperparameter tuning
- REST API endpoints for model deployment
- Cloud storage integration (Google Drive, AWS S3)
- Real-time collaboration features
- Enhanced security and authentication
- Performance optimization for big data (caching, lazy loading, parallel processing)
