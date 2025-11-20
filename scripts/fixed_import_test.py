"""
Fixed import test that handles path issues.
"""
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing imports with fixed paths...")

try:
    # Test importing our modules directly
    from src.data_loader import FinancialNewsLoader
    print("✅ data_loader imported successfully")
    
    from src.eda import FinancialNewsEDA
    print("✅ eda imported successfully")
    
    from src.technical_analysis import StockTechnicalAnalyzer
    print("✅ technical_analysis imported successfully")
    
    from src.config import TECHNICAL_INDICATORS, SENTIMENT_THRESHOLDS
    print("✅ config imported successfully")
    
    print("\n🎉 All custom modules imported successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you're in the project root directory")
    print("2. Check that src/ directory exists with Python files")
    print("3. Verify the file structure")