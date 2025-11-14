import pandas as pd
import numpy as np
import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append('..')
from utils.error_handler import safe_execute, validate_dataframe

def analyze_sentiment(text_series: pd.Series) -> pd.DataFrame:
    try:
        sentiments = []
        for text in text_series:
            if pd.isna(text) or not isinstance(text, str):
                sentiments.append({'polarity': 0, 'subjectivity': 0, 'label': 'Neutral'})
                continue
            
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                label = 'Positive'
            elif polarity < -0.1:
                label = 'Negative'
            else:
                label = 'Neutral'
            
            sentiments.append({
                'polarity': polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'label': label
            })
        
        return pd.DataFrame(sentiments)
        
    except Exception as e:
        st.error(f"❌ Sentiment analysis failed: {str(e)}")
        return pd.DataFrame()

def generate_wordcloud(text_series: pd.Series, max_words: int = 100) -> Optional[WordCloud]:
    try:
        text = ' '.join([str(t) for t in text_series.dropna() if isinstance(t, str)])
        
        if not text.strip():
            st.warning("⚠️ No text data available for word cloud")
            return None
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=max_words,
            colormap='viridis'
        ).generate(text)
        
        return wordcloud
        
    except Exception as e:
        st.error(f"❌ Word cloud generation failed: {str(e)}")
        return None

def extract_ngrams(text_series: pd.Series, n: int = 2, top_k: int = 10) -> List[Tuple[str, int]]:
    try:
        from nltk import ngrams
        from nltk.tokenize import word_tokenize
        import nltk
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        
        all_ngrams = []
        for text in text_series.dropna():
            if not isinstance(text, str):
                continue
            
            tokens = word_tokenize(str(text).lower())
            tokens = [t for t in tokens if t.isalnum()]
            
            text_ngrams = list(ngrams(tokens, n))
            all_ngrams.extend([' '.join(ng) for ng in text_ngrams])
        
        if not all_ngrams:
            return []
        
        counter = Counter(all_ngrams)
        return counter.most_common(top_k)
        
    except Exception as e:
        st.warning(f"⚠️ N-gram extraction failed: {str(e)}")
        return []

def get_text_statistics(text_series: pd.Series) -> Dict:
    try:
        stats = {
            'total_texts': len(text_series),
            'non_empty': text_series.notna().sum(),
            'avg_length': 0,
            'max_length': 0,
            'min_length': 0,
            'total_words': 0,
            'avg_words': 0
        }
        
        valid_texts = [str(t) for t in text_series.dropna() if isinstance(t, str)]
        
        if valid_texts:
            lengths = [len(t) for t in valid_texts]
            stats['avg_length'] = np.mean(lengths)
            stats['max_length'] = max(lengths)
            stats['min_length'] = min(lengths)
            
            word_counts = [len(t.split()) for t in valid_texts]
            stats['total_words'] = sum(word_counts)
            stats['avg_words'] = np.mean(word_counts)
        
        return stats
        
    except Exception as e:
        st.warning(f"⚠️ Text statistics failed: {str(e)}")
        return {}

def analyze_text_column(df: pd.DataFrame, column: str) -> Dict:
    if column not in df.columns:
        st.error(f"❌ Column '{column}' not found")
        return {}
    
    try:
        text_col = df[column]
        
        results = {
            'statistics': get_text_statistics(text_col),
            'sentiment': analyze_sentiment(text_col),
            'wordcloud': generate_wordcloud(text_col),
            'bigrams': extract_ngrams(text_col, n=2, top_k=10),
            'trigrams': extract_ngrams(text_col, n=3, top_k=10)
        }
        
        return results
        
    except Exception as e:
        st.error(f"❌ Text analysis failed: {str(e)}")
        return {}
