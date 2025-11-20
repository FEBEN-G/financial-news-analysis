"""
Data processing utilities for financial news analysis
"""
import pandas as pd
import numpy as np
from datetime import datetime
import re
import os

def load_financial_news_data(file_path=None, sample_size=None):
    """
    Load and process financial news data from raw_analyst_ratings.csv
    
    Args:
        file_path (str): Path to the data file
        sample_size (int): Number of rows to sample for testing
    
    Returns:
        pandas.DataFrame: Processed financial news data
    """
    print("============================================================")
    print("LOADING FINANCIAL NEWS DATA")
    print("============================================================")
    
    # Use the correct data file path
    if file_path is None:
        file_path = "data/raw/raw_analyst_ratings.csv"
    
    try:
        if sample_size:
            print(f"Loading sample of {sample_size} rows for testing...")
            data = pd.read_csv(file_path, nrows=sample_size)
        else:
            data = pd.read_csv(file_path)
        
        print(f"✅ Raw data loaded from: {file_path}")
        print(f"✅ Raw data loaded: {data.shape[0]} rows × {data.shape[1]} columns")
        
    except FileNotFoundError:
        print(f"❌ Data file not found: {file_path}")
        print("Creating sample data for testing...")
        return create_sample_data(sample_size)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return create_sample_data(sample_size)
    
    # Process the data
    processed_data = process_financial_data(data)
    
    return processed_data

def process_financial_data(data):
    """
    Process raw financial news data from raw_analyst_ratings.csv
    
    Args:
        data (pandas.DataFrame): Raw financial news data
    
    Returns:
        pandas.DataFrame: Processed data
    """
    print("\n📊 ORIGINAL COLUMNS:")
    for i, col in enumerate(data.columns, 1):
        print(f"  {i}. {col}")
    
    print("\n🔄 PROCESSING DATA...")
    
    # 1. Remove unnecessary columns
    print("  1. Removing unnecessary columns...")
    columns_to_drop = ['Unnamed: 0', 'unnamed: 0', 'Unnamed: 0.1']
    for col in columns_to_drop:
        if col in data.columns:
            data = data.drop(columns=[col])
            print(f"     ✅ Removed '{col}' column")
    
    print(f"     Columns after cleaning: {list(data.columns)}")
    
    # 2. Process dates - handle the date column in raw_analyst_ratings.csv
    print("  2. Processing dates...")
    date_columns = ['date', 'Date', 'DATE', 'published', 'timestamp']
    date_col = None
    
    for col in date_columns:
        if col in data.columns:
            date_col = col
            break
    
    if date_col:
        try:
            # Try different date formats - FIXED: Ensure we get a Series, not DataFrame
            date_series = pd.to_datetime(data[date_col], errors='coerce', utc=True)
            
            # If we get a DataFrame instead of Series, extract the first column
            if isinstance(date_series, pd.DataFrame):
                date_series = date_series.iloc[:, 0]
            
            if date_series.isna().all():
                date_series = pd.to_datetime(data[date_col], errors='coerce')
                if isinstance(date_series, pd.DataFrame):
                    date_series = date_series.iloc[:, 0]
            
            # Check if we have valid dates
            valid_dates = date_series.notna().sum()
            print(f"     ✅ Parsed {valid_dates} valid dates from '{date_col}'")
            
            # Assign the date series to the 'date' column
            data = data.copy()
            data['date'] = date_series
            
            if date_col != 'date':
                data = data.drop(columns=[date_col])
            
            # FIXED: Properly format date range
            min_date = data['date'].min()
            max_date = data['date'].max()
            date_range = f"{min_date} to {max_date}"
            print(f"     Date range: {date_range}")
        except Exception as e:
            print(f"     ⚠ Error parsing dates: {e}")
            print("     Using default dates...")
            data['date'] = pd.to_datetime('2023-01-01')  # Default date
    else:
        print("     ⚠ No date column found. Using default dates.")
        data['date'] = pd.to_datetime('2023-01-01')
    
    # 3. Clean text data - look for headline or similar columns
    print("  3. Cleaning text data...")
    text_columns = ['headline', 'title', 'text', 'content', 'Headline', 'news', 'article']
    text_col = None
    
    for col in text_columns:
        if col in data.columns:
            text_col = col
            break
    
    if text_col:
        if text_col != 'headline':
            data = data.rename(columns={text_col: 'headline'})
        data['headline'] = data['headline'].fillna('').astype(str)
        data['headline_clean'] = data['headline'].apply(clean_text)
        print(f"     ✅ Using '{text_col}' as headline")
        print(f"     Sample headline: '{data['headline'].iloc[0][:80]}...'")
    else:
        print("     ⚠ No headline column found. Creating placeholder.")
        data['headline'] = "Financial News Article"
    
    # 4. Clean stock data - look for stock ticker columns
    print("  4. Cleaning stock data...")
    stock_columns = ['stock', 'ticker', 'symbol', 'Stock', 'Ticker', 'company', 'firm']
    stock_col = None
    
    for col in stock_columns:
        if col in data.columns:
            stock_col = col
            break
    
    if stock_col:
        if stock_col != 'stock':
            data = data.rename(columns={stock_col: 'stock'})
        data['stock'] = data['stock'].fillna('UNKNOWN').astype(str)
        # Clean stock symbols (remove non-alphanumeric, take first 5 chars)
        data['stock'] = data['stock'].str.upper().str.replace(r'[^A-Z]', '', regex=True)
        data['stock'] = data['stock'].str[:5]  # Limit to reasonable length
        
        unique_stocks = data['stock'].nunique()
        top_stocks = data['stock'].value_counts().head(5).to_dict()
        print(f"     ✅ Unique stocks: {unique_stocks}")
        print(f"     Top 5 stocks: {top_stocks}")
    else:
        print("     ⚠ No stock column found. Creating placeholder.")
        data['stock'] = "AAPL"
    
    # 5. Remove duplicates based on content
    print("  5. Removing duplicates...")
    initial_count = len(data)
    
    # Try to identify duplicates based on available columns
    duplicate_cols = []
    if 'headline' in data.columns:
        duplicate_cols.append('headline')
    if 'stock' in data.columns:
        duplicate_cols.append('stock')
    if 'date' in data.columns:
        duplicate_cols.append('date')
    
    if duplicate_cols:
        data = data.drop_duplicates(subset=duplicate_cols, keep='first')
        final_count = len(data)
        duplicates_removed = initial_count - final_count
        print(f"     Removed {duplicates_removed} duplicate entries")
        print(f"     Final count: {final_count} articles")
    else:
        print("     ⚠ No columns available for duplicate detection")
    
    # 6. Create derived features - FIXED: Ensure 'date' is a Series before using .dt
    print("  6. Creating derived features...")
    
    # Headline features
    if 'headline' in data.columns:
        data['headline_length'] = data['headline'].str.len()
        data['word_count'] = data['headline'].str.split().str.len()
        print("     ✅ Created headline features")
    
    # Date features - FIXED: Check if 'date' is a datetime Series
    if 'date' in data.columns and pd.api.types.is_datetime64_any_dtype(data['date']):
        data['is_weekend'] = data['date'].dt.dayofweek >= 5
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year
        data['day_of_week'] = data['date'].dt.day_name()
        print("     ✅ Created date features")
    else:
        print("     ⚠ Date column not available or not datetime type for feature creation")
    
    # Publisher features (if available)
    if 'publisher' in data.columns:
        data['publisher'] = data['publisher'].fillna('Unknown')
        major_publishers = data['publisher'].value_counts().head(10).index
        data['is_major_publisher'] = data['publisher'].isin(major_publishers)
        print(f"     ✅ Created publisher features")
    
    # URL features (if available)
    if 'url' in data.columns:
        data['has_url'] = data['url'].notna()
        print(f"     ✅ Created URL features")
    
    # Analyst rating features (common in analyst ratings data)
    rating_columns = ['rating', 'Rating', 'action', 'Action', 'recommendation']
    for col in rating_columns:
        if col in data.columns:
            data['analyst_rating'] = data[col].fillna('Unknown')
            print(f"     ✅ Using '{col}' as analyst rating")
            break
    
    print("\n🎉 DATA PROCESSING COMPLETE!")
    print(f"   Final dataset: {len(data)} articles")
    if 'date' in data.columns and pd.api.types.is_datetime64_any_dtype(data['date']):
        min_date = data['date'].min()
        max_date = data['date'].max()
        print(f"   Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    if 'stock' in data.columns:
        print(f"   Unique stocks: {data['stock'].nunique()}")
    if 'publisher' in data.columns:
        print(f"   Unique publishers: {data['publisher'].nunique()}")
    
    return data

def clean_text(text):
    """
    Clean text data
    
    Args:
        text (str): Raw text
    
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
    
    return text.strip()

def create_sample_data(sample_size=1000):
    """
    Create sample financial news data for testing
    
    Args:
        sample_size (int): Number of sample rows to create
    
    Returns:
        pandas.DataFrame: Sample financial news data
    """
    print("Creating sample data for testing...")
    
    np.random.seed(42)
    
    # Sample stocks
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'AMD', 'INTC']
    
    # Sample headlines
    base_headlines = [
        "Stock hits new high amid strong earnings",
        "Company announces dividend increase", 
        "Market reacts to economic data",
        "Analysts raise price target",
        "CEO comments on future growth",
        "Quarterly results exceed expectations",
        "New product launch announced",
        "Merger and acquisition news",
        "Regulatory approval received",
        "Market volatility continues"
    ]
    
    # Generate sample data
    data = []
    for i in range(sample_size):
        stock = np.random.choice(stocks)
        headline = f"{stock} {np.random.choice(base_headlines)}"
        
        row = {
            'headline': headline,
            'stock': stock,
            'date': pd.to_datetime('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365)),
            'publisher': np.random.choice(['Reuters', 'Bloomberg', 'CNBC', 'WSJ', 'Financial Times']),
            'url': f"https://example.com/news/{i}",
            'headline_length': len(headline),
            'word_count': len(headline.split())
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    print(f"✅ Created sample data with {len(df)} rows")
    
    return df

# Test the functions with the actual data file
if __name__ == "__main__":
    print("Testing data processing with actual data file...")
    
    # Test with the actual file
    data = load_financial_news_data("data/raw/raw_analyst_ratings.csv", sample_size=100)
    
    print(f"\nSample data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    if not data.empty:
        print(f"\nFirst few rows:")
        display_columns = ['headline', 'stock', 'date']
        available_columns = [col for col in display_columns if col in data.columns]
        if available_columns:
            print(data[available_columns].head())