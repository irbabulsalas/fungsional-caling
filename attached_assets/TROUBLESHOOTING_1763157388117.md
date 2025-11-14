# 🔧 Troubleshooting Guide

**AI Data Analysis Platform - Common Issues & Solutions**

---

## 🚨 Installation Issues

### Issue: Package Installation Fails

**Error**: `ERROR: Could not find a version that satisfies the requirement...`

**Solution**:
```bash
# Update pip first
pip install --upgrade pip

# Install packages one by one to identify issue
pip install streamlit
pip install google-generativeai
# etc.

# Or use specific versions
pip install streamlit==1.31.0
```

---

### Issue: Python Version Mismatch

**Error**: `Python 3.11+ required`

**Solution**:
```bash
# Check Python version
python --version

# If < 3.11, upgrade Python
# Download from: https://www.python.org/downloads/

# Or use pyenv
pyenv install 3.11.7
pyenv global 3.11.7
```

---

## 🔑 API Key Issues

### Issue: GEMINI_API_KEY Not Found

**Error**: `GEMINI_API_KEY not found in environment`

**Solutions**:

**On Replit:**
1. Click Secrets (🔒 icon in sidebar)
2. Add new secret: `GEMINI_API_KEY`
3. Paste your API key
4. Restart app

**Locally:**
```bash
# Add to .bashrc or .zshrc
export GEMINI_API_KEY="your_key_here"

# Or use .env file
echo "GEMINI_API_KEY=your_key_here" > .env

# Restart terminal
```

---

### Issue: API Key Invalid

**Error**: `401 Unauthorized` or `Invalid API key`

**Solution**:
1. Get **NEW** API key from https://aistudio.google.com/app/apikey
2. Ensure you copy the ENTIRE key (no spaces)
3. Update in Replit Secrets
4. Restart application

---

### Issue: API Rate Limit

**Error**: `429 Too Many Requests` or `Quota exceeded`

**Expected Behavior**: The app shows countdown timer

**Solutions**:
- **Wait** for cooldown period (1 minute)
- **Free Tier Limits**: 15 requests/hour
- Use **manual features** during cooldown
- **Upgrade** to paid tier if needed

---

## 📂 Data Upload Issues

### Issue: Upload Failed - File Too Large

**Error**: `File size exceeds limit`

**Solution**:
- **Max size**: 200MB
- **Compress** data or **sample** it
- Try **Parquet** format (more efficient)

```python
# Sample large dataset
df_sample = df.sample(frac=0.1)  # 10% sample
df_sample.to_csv('sampled_data.csv', index=False)
```

---

### Issue: Upload Failed - Format Error

**Error**: `ParserError` or `Cannot read file`

**Solutions**:
1. **Check file format**:
   - CSV: Must be comma-separated
   - Excel: Must be .xlsx or .xls
   - JSON: Must be valid JSON

2. **Fix encoding**:
   ```bash
   # Convert to UTF-8
   iconv -f ISO-8859-1 -t UTF-8 input.csv > output.csv
   ```

3. **Re-export from Excel**:
   - Open in Excel
   - Save As → CSV (UTF-8)

---

### Issue: Missing Columns After Upload

**Error**: Some columns not showing

**Solutions**:
- Check if file has **headers**
- Verify **delimiter** (comma vs tab)
- Ensure **no special characters** in column names

---

## 🤖 Machine Learning Issues

### Issue: Model Training Failed

**Error**: `ValueError` or `Not enough data`

**Solutions**:
- **Minimum rows**: 50 for classification/regression
- **Check target column**: Must exist and have valid values
- **Numeric features**: ML needs numeric columns
- **Clean data first**: Remove missing values

---

### Issue: All Models Failed

**Error**: Every model returns error

**Solutions**:
1. **Data Quality Check**:
   - Run data profiling first
   - Clean data (handle missing values)
   - Ensure target column is correct type

2. **Feature Check**:
   - Classification: Need categorical or binary target
   - Regression: Need numeric target
   - At least 2-3 numeric features

3. **Data Size**:
   - Too small: Add more data
   - Too large: Sample it

---

### Issue: Low Model Accuracy

**Not an error, but common question**

**Solutions**:
- **Clean data better**: Remove outliers, handle missing values
- **Feature engineering**: Create new features
- **Try different models**: XGBoost often performs better
- **Tune hyperparameters**: Enable tuning option
- **More data**: Collect more training samples

---

## 📊 Visualization Issues

### Issue: Charts Not Rendering

**Error**: Blank space where chart should be

**Solutions**:
1. **Check data**:
   - Need numeric columns for most charts
   - Need at least 2 values

2. **Browser**:
   - Refresh page
   - Clear cache
   - Try different browser (Chrome recommended)

3. **Plotly**:
   ```bash
   # Reinstall plotly
   pip uninstall plotly
   pip install plotly==5.18.0
   ```

---

### Issue: Correlation Heatmap Empty

**Error**: "Need at least 2 numeric columns"

**Solution**:
- Dataset must have **2+ numeric columns**
- Convert categorical to numeric if needed
- Check data types in Data Profiling dashboard

---

## 📝 Text Analytics Issues

### Issue: Sentiment Analysis Failed

**Error**: `TextBlob error` or `NLTK data not found`

**Solution**:
```python
# Download NLTK data
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')
```

Or run in terminal:
```bash
python -c "import nltk; nltk.download('all')"
```

---

### Issue: Word Cloud Empty

**Error**: Blank word cloud or "No text data"

**Solutions**:
- **Check column has text**: Not empty
- **Remove stopwords**: Common words filter out
- **Verify encoding**: UTF-8 recommended

---

## 💾 Export Issues

### Issue: Download Failed

**Error**: Download button not working

**Solutions**:
1. **Browser settings**:
   - Allow downloads from this site
   - Check download folder permissions

2. **File size**:
   - Large files may timeout
   - Try smaller datasets

3. **Format**:
   - Try different export format
   - CSV usually most reliable

---

### Issue: PDF Report Blank

**Error**: PDF downloads but is empty

**Solution**:
- Run analysis first (PDF needs content)
- Check browser PDF viewer
- Try opening in Adobe Reader

---

## 🌐 Deployment Issues (Railway)

### Issue: Build Failed on Railway

**Error**: Build logs show error

**Solutions**:
1. **Check requirements**:
   - All dependencies in requirements.txt
   - Correct versions specified

2. **Procfile**:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
   ```

3. **Runtime**:
   - Ensure Python 3.11+ available
   - Check Railway build logs

---

### Issue: App Crashes After Deploy

**Error**: 503 or 502 error

**Solutions**:
1. **Check logs** in Railway dashboard
2. **Verify environment variables**:
   - `GEMINI_API_KEY` is set
   - No typos in variable names

3. **Port binding**:
   - Must use `$PORT` from Railway
   - Not hardcoded port

---

### Issue: App Very Slow

**Error**: Long loading times

**Solutions**:
1. **Upgrade Railway plan**: Free tier has limits
2. **Optimize code**: Add caching
3. **Reduce data size**: Sample large datasets
4. **Check region**: Use region closest to users

---

## 💬 AI Chat Issues

### Issue: Chat Not Responding

**Error**: No response from AI

**Solutions**:
1. **Check API key**: Valid and set correctly
2. **Rate limiting**: Wait for cooldown
3. **Internet**: Verify connection
4. **Logs**: Check for specific errors

---

### Issue: Chat Gives Wrong Answers

**Not an error, AI limitation**

**Solutions**:
- **Be specific**: Clear, detailed questions
- **Provide context**: "Using column X, analyze Y"
- **Try rephrasing**: Different wording might help
- **Use manual features**: For precise control

---

## 🖥️ Performance Issues

### Issue: App Runs Slow

**Solutions**:
1. **Restart app**: Fresh start
2. **Clear session**: Reload page
3. **Reduce data**: Work with sample
4. **Close other tabs**: Free up RAM

---

### Issue: Out of Memory

**Error**: `MemoryError`

**Solutions**:
1. **Sample data**:
   ```python
   df_sample = df.sample(n=10000)  # Use 10K rows
   ```

2. **Reduce columns**: Remove unnecessary
3. **Use Parquet**: More memory efficient
4. **Batch processing**: Analyze in chunks

---

## 🔍 Debugging Tips

### Enable Debug Mode

```python
# In app.py, add at top:
import streamlit as st
st.write("Debug info:", st.session_state)
```

### Check Logs

**On Replit:**
- View console for error messages

**On Railway:**
- Dashboard → Deployments → View Logs

### Common Error Patterns

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| KeyError | Column not found | Check column names |
| ValueError | Wrong data type | Convert/clean data |
| MemoryError | Dataset too large | Sample data |
| 429 Error | Rate limit | Wait for cooldown |
| 401 Error | Invalid API key | Update key |

---

## 📞 Getting Help

If you're still stuck:

1. **Check error message**: Often tells you what's wrong
2. **Search error**: Google the specific error
3. **Review documentation**: README.md, this guide
4. **Simplify**: Try with sample dataset first
5. **Restart**: Sometimes fixes mysterious issues

---

## ✅ Prevention Tips

**Best Practices:**
- ✅ Start with sample datasets to learn
- ✅ Clean data before ML
- ✅ Save work frequently (download results)
- ✅ Monitor API usage
- ✅ Keep Gemini API key secure
- ✅ Test on small data first
- ✅ Read error messages carefully

---

**Still Having Issues?**

Most problems are solved by:
1. Restarting the application
2. Checking the API key
3. Verifying data quality
4. Reading error messages

---

**Muhammad Irbabul Salas**
*AI Data Analysis Platform*
