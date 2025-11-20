"""
Configuration settings for the financial news analysis project.
"""
import os
from datetime import datetime

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed')

# Analysis parameters
DEFAULT_START_DATE = '2020-01-01'
DEFAULT_END_DATE = '2023-12-31'

# Sentiment analysis thresholds
SENTIMENT_THRESHOLDS = {
    'positive': 0.05,
    'negative': -0.05,
    'neutral': (-0.05, 0.05)
}

# Technical analysis parameters
TECHNICAL_INDICATORS = {
    'rsi_period': 14,
    'sma_short': 20,
    'sma_long': 50,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9
}

# Check available technical analysis libraries
try:
    import talib
    TA_LIB_AVAILABLE = True
    print("✓ TA-Lib is available")
except ImportError:
    try:
        import pandas_ta as ta
        TA_LIB_AVAILABLE = False
        PANDAS_TA_AVAILABLE = True
        print("✓ pandas_ta is available (TA-Lib alternative)")
    except ImportError:
        TA_LIB_AVAILABLE = False
        PANDAS_TA_AVAILABLE = False
        print("⚠ No technical analysis library available")

def create_directories():
    """Create necessary directories if they don't exist"""
    for path in [DATA_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH]:
        os.makedirs(path, exist_ok=True)

# Create directories when module is imported
create_directories()