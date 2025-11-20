"""
Test script to verify all modules are working
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

print("Testing imports...")

try:
    from src.data_processing import load_financial_news_data
    print("✅ data_processing import successful")
    
    # Test the function
    data = load_financial_news_data(sample_size=100)
    print(f"✅ Data loaded: {len(data)} rows")
    
except Exception as e:
    print(f"❌ data_processing error: {e}")

try:
    from src.technical_analysis import StockTechnicalAnalyzer
    print("✅ technical_analysis import successful")
    
    # Test the class
    analyzer = StockTechnicalAnalyzer('AAPL', period='1mo')
    print("✅ StockTechnicalAnalyzer created")
    
except Exception as e:
    print(f"❌ technical_analysis error: {e}")

print("\nTesting yfinance...")
try:
    import yfinance as yf
    data = yf.download('AAPL', period='1mo', progress=False)
    print(f"✅ yfinance working: {len(data)} rows downloaded")
except Exception as e:
    print(f"❌ yfinance error: {e}")

print("\nTesting pandas_ta...")
try:
    import pandas_ta
    print("✅ pandas_ta import successful")
except Exception as e:
    print(f"❌ pandas_ta error: {e}")