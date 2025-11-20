"""
Analyze the structure of your raw_analyst_ratings.csv file.
"""
import pandas as pd
import os

def analyze_data_file():
    """Analyze the data file structure."""
    file_path = 'data/raw/raw_analyst_ratings.csv'
    
    print("=" * 60)
    print("ANALYZING YOUR DATA FILE")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    try:
        # Load the data
        print("Loading data file...")
        df = pd.read_csv(file_path)
        
        print(f"✅ File loaded successfully!")
        print(f"   Shape: {df.shape} (rows: {df.shape[0]:,}, columns: {df.shape[1]})")
        
        # Display column information
        print(f"\n📊 COLUMNS ANALYSIS:")
        print("-" * 40)
        for i, col in enumerate(df.columns):
            sample_value = str(df[col].iloc[0])[:50] if not df.empty else "N/A"
            null_count = df[col].isnull().sum()
            null_percent = (null_count / len(df)) * 100
            print(f"  {i+1:2d}. {col:20} | Nulls: {null_count:6,} ({null_percent:5.1f}%) | Sample: {sample_value}...")
        
        # Data types
        print(f"\n🔧 DATA TYPES:")
        print("-" * 40)
        print(df.dtypes)
        
        # Sample data
        print(f"\n👀 SAMPLE DATA (first 3 rows):")
        print("-" * 40)
        print(df.head(3).to_string())
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            print(f"\n📈 NUMERIC COLUMNS SUMMARY:")
            print("-" * 40)
            print(df[numeric_cols].describe())
        
        # Check for date-like columns
        date_candidates = []
        for col in df.columns:
            col_lower = col.lower()
            if any(date_word in col_lower for date_word in ['date', 'time', 'published', 'timestamp']):
                date_candidates.append(col)
        
        if date_candidates:
            print(f"\n📅 POTENTIAL DATE COLUMNS: {date_candidates}")
        
        # Check for text columns (headline-like)
        text_candidates = []
        for col in df.columns:
            col_lower = col.lower()
            if any(text_word in col_lower for text_word in ['headline', 'title', 'text', 'content', 'news', 'summary']):
                text_candidates.append(col)
        
        if text_candidates:
            print(f"\n📝 POTENTIAL TEXT COLUMNS: {text_candidates}")
        
        # Check for stock columns
        stock_candidates = []
        for col in df.columns:
            col_lower = col.lower()
            if any(stock_word in col_lower for stock_word in ['stock', 'ticker', 'symbol', 'company']):
                stock_candidates.append(col)
        
        if stock_candidates:
            print(f"\n📈 POTENTIAL STOCK COLUMNS: {stock_candidates}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        return None

if __name__ == "__main__":
    analyze_data_file()