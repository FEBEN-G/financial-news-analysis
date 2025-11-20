"""
Check stock coverage in the full dataset.
"""
import pandas as pd
import os

def check_stock_coverage():
    """Check how many stocks are covered in the data."""
    file_path = 'data/raw/raw_analyst_ratings.csv'
    
    print("📊 CHECKING STOCK COVERAGE IN FULL DATASET")
    print("=" * 50)
    
    # Load just the stock column from the full dataset
    print("Loading stock data from full dataset...")
    stocks = pd.read_csv(file_path, usecols=['stock'])
    
    print(f"✅ Loaded {len(stocks):,} articles")
    
    # Analyze stock coverage
    stock_counts = stocks['stock'].value_counts()
    
    print(f"\n📈 STOCK COVERAGE SUMMARY:")
    print(f"Total unique stocks: {len(stock_counts)}")
    print(f"Articles per stock statistics:")
    print(f"  Mean: {stock_counts.mean():.1f}")
    print(f"  Median: {stock_counts.median():.1f}")
    print(f"  Max: {stock_counts.max():,}")
    print(f"  Min: {stock_counts.min()}")
    
    print(f"\n🏆 TOP 20 MOST COVERED STOCKS:")
    for i, (stock, count) in enumerate(stock_counts.head(20).items()):
        print(f"  {i+1:2d}. {stock}: {count:,} articles")
    
    print(f"\n📉 STOCKS WITH FEW ARTICLES:")
    few_articles = stock_counts[stock_counts <= 5]
    print(f"  Stocks with ≤5 articles: {len(few_articles)}")
    
    # Coverage distribution
    print(f"\n📊 COVERAGE DISTRIBUTION:")
    bins = [1, 5, 10, 50, 100, 500, 1000, float('inf')]
    labels = ['1-5', '6-10', '11-50', '51-100', '101-500', '501-1000', '1000+']
    
    for i in range(len(bins)-1):
        count = ((stock_counts >= bins[i]) & (stock_counts < bins[i+1])).sum()
        print(f"  {labels[i]}: {count} stocks")

if __name__ == "__main__":
    check_stock_coverage()