"""
Basic unit tests for financial news analysis.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all required packages can be imported."""
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from src.data_loader import FinancialNewsLoader
        from src.eda import FinancialNewsEDA
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    try:
        loader = FinancialNewsLoader()
        data = loader.load_data(sample_size=100)
        assert data is not None, "Data loading failed"
        assert len(data) > 0, "No data loaded"
        print("✅ Data loading test passed")
        return True
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running basic tests...")
    test_imports()
    test_data_loading()
    print("Tests completed!")