# 🤖 AI Data Analysis Platform
### By Muhammad Irbabul Salas

Comprehensive AI-powered data analysis platform with Gemini 2.5, automated machine learning, interactive dashboards, and advanced analytics.

![Platform](https://img.shields.io/badge/Platform-Streamlit-red)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![AI](https://img.shields.io/badge/AI-Gemini_2.5-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

### 🎯 Core Capabilities
- **AI Chat Assistant** - Powered by Gemini 2.5 Flash with function calling
- **Automated Machine Learning** - 10+ algorithms with auto-comparison
- **Interactive Dashboards** - Multi-page responsive interface
- **Text Analytics** - Sentiment analysis, topic modeling, word clouds
- **Comprehensive Export** - PDF, Excel, models, Jupyter notebooks

### 📊 Data Analysis
- Multi-format upload (CSV, Excel, JSON, Parquet, TSV)
- Automatic data profiling & quality assessment  
- Advanced cleaning with multiple strategies
- Statistical tests & correlation analysis
- Feature importance & SHAP values

### 🎨 User Experience
- Responsive design (mobile/tablet/desktop)
- Dark/Light mode toggle
- Interactive onboarding & help system
- Sample datasets for instant testing
- Rate limiting for free API tier

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Gemini API Key ([Get Free Key](https://aistudio.google.com/app/apikey))

### Installation

1. **Clone or download this project**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Gemini API Key**
   - Get free API key from: https://aistudio.google.com/app/apikey
   - Add to Replit Secrets with key: `GEMINI_API_KEY`
   - Or set environment variable locally

4. **Run application**
   ```bash
   streamlit run app.py --server.port 5000
   ```

5. **Open browser**
   ```
   http://localhost:5000
   ```

---

## 📖 User Guide

### Uploading Data
1. Click sidebar "Upload Data"
2. Select file (CSV, Excel, JSON, Parquet)
3. Or load sample datasets to try features

### AI Chat Assistant
- Ask natural language questions about your data
- **Rate Limit**: 1 minute between questions, 15/hour (free tier)
- **Examples**:
  - "Show correlation between age and salary"
  - "Train classification models to predict churn"
  - "Analyze sentiment of customer reviews"

### Machine Learning
1. Go to "🤖 ML Models" tab
2. Select target column
3. Choose models to train
4. Click "Train Models"
5. View metrics, confusion matrix, feature importance

### Exporting Results
- Navigate to "📥 Export Center"
- Download cleaned data (CSV/Excel/JSON)
- Export trained models (.pkl format)
- Generate PDF reports

---

## 🏗️ Project Structure

```
ai-data-analysis/
├── app.py                          # Main application
├── requirements.txt                # Python dependencies
├── Procfile                       # Railway deployment
├── railway.json                   # Railway config
│
├── modules/                       # Core modules
│   ├── data_processing.py         # Data loading & cleaning
│   ├── ml_models.py               # ML training & evaluation
│   ├── visualizations.py          # Chart generation
│   ├── text_analytics.py          # NLP functions
│   ├── gemini_integration.py      # AI function calling
│   └── export_handler.py          # Export functionality
│
├── utils/                         # Utilities
│   ├── error_handler.py           # Error management
│   ├── rate_limiter.py            # API rate limiting
│   └── helpers.py                 # Helper functions
│
├── assets/                        # Static files
│   ├── profile_photo.jpg          # User photo
│   └── sample_datasets/           # Sample data
│
└── docs/                          # Documentation
    ├── DEPLOYMENT.md              # Railway deployment guide
    └── TROUBLESHOOTING.md         # Common issues
```

---

## 🌐 Deployment to Railway

See detailed guide in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

### Quick Steps:
1. Push code to GitHub
2. Connect Railway to your repo
3. Add `GEMINI_API_KEY` to environment variables
4. Deploy!

**Estimated Cost**: ~$5/month with Railway Hobby plan

---

## 💰 Cost Breakdown

| Service | Free Tier | Monthly Cost |
|---------|-----------|--------------|
| Gemini API (Flash) | 15 req/min, 1.5K/day | **FREE** |
| Railway | $5 credit trial | ~$5 after trial |
| GitHub | Unlimited repos | **FREE** |
| **Total** | | **~$5/month** |

---

## 🔑 Getting API Keys

### Gemini API (Required)
1. Visit: https://aistudio.google.com/app/apikey
2. Login with Google account
3. Click "Create API Key"
4. Copy and save to Replit Secrets

---

## 🎯 Features by Dashboard

### 📈 Overview Dashboard
- Total rows, columns, missing values
- Data quality score
- Column type breakdown
- AI-generated insights

### 🔍 Data Profiling
- Detailed column statistics
- Missing values analysis
- Correlation heatmap
- Data cleaning interface

### 📊 EDA (Exploratory Data Analysis)
- Distribution plots (histogram, box, violin)
- Relationship analysis (scatter, line)
- Statistical comparisons

### 🤖 ML Models
- Classification (Random Forest, XGBoost, Logistic Regression, etc.)
- Regression (Ridge, Lasso, Random Forest)
- Clustering (K-Means, DBSCAN)
- Feature importance & SHAP values

### 📝 Text Analytics
- Sentiment analysis
- Word clouds
- N-gram analysis (bigrams, trigrams)
- Text statistics

### 📥 Export Center
- Data exports (CSV, Excel, JSON, Parquet)
- Model exports (.pkl, .joblib)
- PDF reports
- Jupyter notebooks

---

## ⚙️ Tech Stack

**Frontend/UI:**
- Streamlit (web framework)
- Plotly (interactive visualizations)
- Custom CSS (responsive design)

**AI/ML:**
- Google Gemini 2.5 (AI chat & function calling)
- scikit-learn (traditional ML)
- XGBoost, LightGBM (gradient boosting)
- SHAP (model interpretability)

**Data Processing:**
- pandas (data manipulation)
- NumPy (numerical computing)
- NLTK, TextBlob (NLP)

**Export:**
- FPDF, ReportLab (PDF generation)
- Joblib (model serialization)
- NBFormat (Jupyter notebooks)

---

## 🐛 Troubleshooting

### Common Issues

**Q: "API rate limit reached"**
A: Wait 1 minute between questions. Free tier allows 15 requests/hour.

**Q: "File upload failed"**
A: Check file size (max 200MB) and format. Try converting to CSV.

**Q: "Model training failed"**
A: Ensure you have enough data (min 50 rows) and numeric features.

**Q: "GEMINI_API_KEY not found"**
A: Add API key to Replit Secrets or environment variables.

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more.

---

## 📝 License

MIT License - Free to use, modify, and distribute.

---

## 👨‍💻 Author

**Muhammad Irbabul Salas**

Platform for automated data analysis with AI assistance.

---

## 🙏 Acknowledgments

- Google Gemini AI for powerful LLM capabilities
- Streamlit for amazing web framework
- Open source ML libraries (scikit-learn, XGBoost, etc.)

---

## 📊 Version

**Version 1.0.0** - Initial Release (November 2025)

---

**Made with ❤️ by Muhammad Irbabul Salas**

*Powered by Gemini 2.5 Flash | Built with Streamlit*
