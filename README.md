# 📌 Financial News Sentiment Analysis

A comprehensive analysis of financial news sentiment and its correlation with stock price movements using NLP and technical indicators.

---

## 📊 Project Overview

This project analyzes how financial news sentiment impacts stock performance through three key phases:

1. **EDA & Infrastructure:** Data processing and exploratory analysis  
2. **Technical Analysis:** Financial indicators and trading signals  
3. **Sentiment Correlation:** NLP sentiment analysis and stock return correlations  

---

## 🚀 Features

- **Data Processing:** Automated cleaning of 4,776 financial news articles  
- **Technical Indicators:** RSI, MACD, Moving Averages, Bollinger Bands  
- **Sentiment Analysis:** NLP-powered scoring using TextBlob  
- **Correlation Studies:** Statistical analysis between sentiment and stock returns  
- **Visualization:** Professional charts, plots, and reports  

---

## 📁 Project Structure

```
financial-news-analysis/
├── src/                    # Core modules
├── scripts/                # Execution scripts
├── data/                   # Raw and processed data
├── reports/                # Analysis outputs
└── notebooks/              # Exploratory analysis
```

---

## 🛠️ Quick Start

```bash
# Clone and setup
git clone https://github.com/FEBEN-G/financial-news-analysis.git
cd financial-news-analysis

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run analysis
python scripts/run_task2_technical.py
python scripts/run_task3_sentiment_correlation.py
```

---

## 📈 Key Results

### **Technical Analysis**
- **Stock A:** +13.60% return with strong bullish signals  
- **Stock AA:** -18.91% return, higher volatility  
- **Stock AAL:** -13.76% return, mixed signals  

### **Sentiment Analysis**
- **81.2%** neutral sentiment in news articles  
- Weak positive correlation (**0.209**) between sentiment and returns  
- Limited statistical significance in current dataset  

---

## 📊 Outputs

- Technical analysis reports and visualizations  
- Sentiment correlation summaries  
- Trading signals and recommendations  
- Comparative stock performance charts  

---

## 🔧 Dependencies

- `pandas`, `numpy` — Data analysis  
- `yfinance` — Financial data  
- `textblob` — NLP sentiment analysis  
- `matplotlib` — Visualization  
- `pandas_ta` — Technical indicators  

