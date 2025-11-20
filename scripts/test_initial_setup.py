"""
Test script to verify the initial setup and data loading functionality.
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import FinancialNewsLoader
from eda import FinancialNewsEDA

def test_data_loading():
    """Test data loading functionality."""
    print("=" * 60)
    print("TESTING DATA LOADING FUNCTIONALITY")
    print("=" * 60)
    
    # Initialize loader
    loader = FinancialNewsLoader()
    
    # Try to load data
    data = loader.load_data()
    
    if data is not None:
        print("✓ Data loading successful!")
        
        # Get basic stats
        stats = loader.get_basic_stats()
        print("\nBasic Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test EDA
        print("\n" + "=" * 60)
        print("TESTING EDA FUNCTIONALITY")
        print("=" * 60)
        
        eda = FinancialNewsEDA(data)
        comprehensive_stats = eda.comprehensive_analysis()
        
        print("✓ EDA completed successfully!")
        
        # Save processed data
        loader.save_processed_data()
        print("✓ Processed data saved!")
        
    else:
        print("✗ Data loading failed. Please check your data file.")
        
        # Create sample data for testing if no real data exists
        print("\nCreating sample data for testing...")
        create_sample_data()

def create_sample_data():
    """Create sample data for testing if no real data exists."""
    import pandas as pd
    from datetime import datetime, timedelta
    import numpy as np
    
    # Create sample financial news data
    sample_size = 1000
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'AMD', 'INTC']
    publishers = ['Reuters', 'Bloomberg', 'CNBC', 'Wall Street Journal', 'Financial Times', 'Yahoo Finance']
    
    # Generate sample dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 365), 
                                   hours=np.random.randint(0, 24)) 
            for _ in range(sample_size)]
    
    # Generate sample headlines
    base_headlines = [
        "Stock hits new high as earnings beat expectations",
        "Analyst upgrades price target amid strong growth",
        "Company announces dividend increase",
        "New product launch drives investor optimism",
        "Regulatory concerns weigh on stock performance",
        "Merger talks boost share prices",
        "CEO comments on future outlook at conference",
        "Market reacts to economic data release",
        "Institutional investors increase positions",
        "Short seller report impacts trading volume"
    ]
    
    data = {
        'headline': [f"{np.random.choice(base_headlines)} - {stock}" 
                    for stock in np.random.choice(stocks, sample_size)],
        'url': [f"https://example.com/news/{i}" for i in range(sample_size)],
        'publisher': np.random.choice(publishers, sample_size),
        'date': dates,
        'stock': np.random.choice(stocks, sample_size)
    }
    
    df = pd.DataFrame(data)
    
    # Save sample data
    from src.config import RAW_DATA_PATH
    sample_path = os.path.join(RAW_DATA_PATH, 'sample_financial_news.csv')
    df.to_csv(sample_path, index=False)
    
    print(f"✓ Sample data created at: {sample_path}")
    print("You can now run the analysis with the sample data.")
    print("Replace sample_financial_news.csv with your actual data when available.")

if __name__ == "__main__":
    test_data_loading()