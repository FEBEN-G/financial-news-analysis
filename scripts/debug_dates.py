"""
Debug script to understand the date formats in your data.
"""
import pandas as pd
import os

def debug_dates():
    """Analyze the date formats in your data file."""
    file_path = 'data/raw/raw_analyst_ratings.csv'
    
    print("🔍 DEBUGGING DATE FORMATS")
    print("=" * 50)
    
    # Load first 20 rows to examine date formats
    df_sample = pd.read_csv(file_path, nrows=20)
    
    print("First 20 date values:")
    print("-" * 40)
    for i, date_val in enumerate(df_sample['date']):
        print(f"{i+1:2d}. {date_val} (type: {type(date_val)})")
    
    # Check for different date patterns
    print(f"\n📊 DATE PATTERN ANALYSIS:")
    print("-" * 40)
    
    patterns = {
        'has_timezone': lambda x: '-' in str(x)[-6:] and ':' in str(x),  # Ends with -04:00
        'has_time': lambda x: ':' in str(x) and len(str(x)) > 10,  # Contains time
        'date_only': lambda x: len(str(x)) <= 10,  # Just date
        'iso_format': lambda x: 'T' in str(x),  # ISO format with T
    }
    
    for pattern_name, pattern_func in patterns.items():
        count = sum(1 for date_val in df_sample['date'] if pattern_func(date_val))
        print(f"  {pattern_name}: {count}/20 rows")
    
    # Test different parsing methods
    print(f"\n🧪 TESTING PARSING METHODS:")
    print("-" * 40)
    
    test_dates = df_sample['date'].head(5).tolist()
    
    # Method 1: Default pandas
    try:
        parsed1 = pd.to_datetime(test_dates)
        print("✅ Method 1 (default): Success")
        for orig, parsed in zip(test_dates, parsed1):
            print(f"   {orig} -> {parsed}")
    except Exception as e:
        print(f"❌ Method 1 (default): {e}")
    
    print()
    
    # Method 2: With UTC
    try:
        parsed2 = pd.to_datetime(test_dates, utc=True)
        print("✅ Method 2 (utc=True): Success")
        for orig, parsed in zip(test_dates, parsed2):
            print(f"   {orig} -> {parsed}")
    except Exception as e:
        print(f"❌ Method 2 (utc=True): {e}")
    
    print()
    
    # Method 3: With errors='coerce'
    try:
        parsed3 = pd.to_datetime(test_dates, errors='coerce')
        print("✅ Method 3 (errors='coerce'): Success")
        for orig, parsed in zip(test_dates, parsed3):
            print(f"   {orig} -> {parsed}")
    except Exception as e:
        print(f"❌ Method 3 (errors='coerce'): {e}")
    
    print()
    
    # Method 4: Mixed format
    try:
        parsed4 = pd.to_datetime(test_dates, format='mixed')
        print("✅ Method 4 (format='mixed'): Success")
        for orig, parsed in zip(test_dates, parsed4):
            print(f"   {orig} -> {parsed}")
    except Exception as e:
        print(f"❌ Method 4 (format='mixed'): {e}")

if __name__ == "__main__":
    debug_dates()