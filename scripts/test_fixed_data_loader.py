"""
Test the fixed data loader with proper indentation.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("🧪 TESTING FIXED DATA LOADER")
    print("=" * 50)
    
    try:
        from src.data_loader import FinancialNewsLoader
        
        # Test with a small sample first
        print("Loading sample data (1,000 rows) for testing...")
        loader = FinancialNewsLoader()
        data = loader.load_data(sample_size=1000)
        
        if data is not None:
            print(f"\n✅ SUCCESS! Data loaded and processed.")
            print(f"Shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            print(f"Date range: {data['date'].min()} to {data['date'].max()}")
            
            # Display sample
            print(f"\n📊 SAMPLE DATA:")
            print(data[['headline', 'stock', 'publisher', 'date']].head(2))
            
            # Get insights
            print(f"\n{loader.get_sample_insights()}")
            
            # Save processed data
            loader.save_processed_data('test_fixed_loader.csv')
            
        else:
            print("❌ Failed to load data")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()