"""
Test script with virtual environment and fixed dependencies.
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_environment():
    """Test if the environment is properly set up."""
    print("=" * 60)
    print("TESTING ENVIRONMENT SETUP")
    print("=" * 60)
    
    # Test basic imports
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        print("✓ Basic data science libraries imported successfully")
    except ImportError as e:
        print(f"✗ Error importing basic libraries: {e}")
        return False
    
    # Test financial libraries
    try:
        import yfinance as yf
        print("✓ yfinance imported successfully")
    except ImportError as e:
        print(f"✗ Error importing yfinance: {e}")
        return False
    
    # Test technical analysis libraries
    from src.config import TA_LIB_AVAILABLE, PANDAS_TA_AVAILABLE
    if TA_LIB_AVAILABLE:
        print("✓ TA-Lib is available")
    elif PANDAS_TA_AVAILABLE:
        print("✓ pandas_ta is available")
    else:
        print("⚠ No technical analysis library available (will use basic indicators)")
    
    # Test NLP libraries
    try:
        import nltk
        from textblob import TextBlob
        print("✓ NLP libraries imported successfully")
    except ImportError as e:
        print(f"✗ Error importing NLP libraries: {e}")
        return False
    
    return True

def test_data_loading():
    """Test data loading functionality."""
    print("\n" + "=" * 60)
    print("TESTING DATA LOADING")
    print("=" * 60)
    
    try:
        from data_loader import FinancialNewsLoader
        
        loader = FinancialNewsLoader()
        data = loader.load_data()
        
        if data is not None:
            print("✓ Data loading successful!")
            stats = loader.get_basic_stats()
            print(f"  Loaded {stats['total_articles']} articles")
            print(f"  Covering {stats['unique_stocks']} stocks")
            return data
        else:
            print("✗ Data loading failed")
            return None
            
    except Exception as e:
        print(f"✗ Error in data loading: {e}")
        return None

def test_technical_analysis():
    """Test technical analysis functionality."""
    print("\n" + "=" * 60)
    print("TESTING TECHNICAL ANALYSIS")
    print("=" * 60)
    
    try:
        from technical_analysis import StockTechnicalAnalyzer
        
        # Test with Apple stock
        analyzer = StockTechnicalAnalyzer('AAPL', period='3mo')
        analyzer.calculate_indicators()
        
        signals = analyzer.generate_signals()
        stats = analyzer.get_summary_stats()
        
        print("✓ Technical analysis successful!")
        print(f"  Analyzed {stats['total_days']} trading days")
        print(f"  Total return: {stats['price_stats']['total_return']:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in technical analysis: {e}")
        return False

if __name__ == "__main__":
    # Test environment
    env_ok = test_environment()
    
    if env_ok:
        # Test data loading
        data = test_data_loading()
        
        # Test technical analysis
        ta_ok = test_technical_analysis()
        
        if env_ok and data is not None and ta_ok:
            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED! Environment is ready for development.")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("⚠ Some tests failed. Please check the errors above.")
            print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Environment setup failed. Please check your installation.")
        print("=" * 60)