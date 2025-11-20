
"""
Examine the structure of the raw_analyst_ratings.csv file
"""
import pandas as pd
import os

def examine_data():
    file_path = "data/raw/raw_analyst_ratings.csv"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    try:
        # Load a sample to examine
        data = pd.read_csv(file_path, nrows=10)
        
        print("📊 DATA FILE STRUCTURE ANALYSIS")
        print("=" * 50)
        print(f"File: {file_path}")
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print("\nFirst 5 rows:")
        print(data.head())
        print("\nColumn info:")
        print(data.info())
        
        # Check for specific column patterns
        print("\n🔍 COLUMN ANALYSIS:")
        for col in data.columns:
            sample_values = data[col].head(3).tolist()
            print(f"  {col}: {sample_values}")
            
    except Exception as e:
        print(f"❌ Error examining data: {e}")

if __name__ == "__main__":
    examine_data()
