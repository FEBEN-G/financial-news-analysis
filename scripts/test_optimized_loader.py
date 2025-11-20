"""
Test the optimized data loader with your actual data.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("🧪 TESTING OPTIMIZED DATA LOADER")
    print("=" * 50)
    
    from src.data_loader import FinancialNewsLoader
    
    # Test with a small sample first
    print("Loading sample data (10,000 rows) for testing...")
    loader = FinancialNewsLoader()
    data = loader.load_data(sample_size=10000)
    
    if data is not None:
        print("\n✅ SUCCESS! Data loaded and processed.")
        
        # Display basic info
        print(f"\n📊 DATA OVERVIEW:")
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"Memory usage: {data.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
        # Display sample
        print(f"\n👀 SAMPLE DATA:")
        print(data[['headline', 'stock', 'publisher', 'date']].head(3).to_string())
        
        # Get statistics
        stats = loader.get_basic_stats()
        print(f"\n📈 KEY STATISTICS:")
        print(f"• Articles: {stats['total_articles']:,}")
        print(f"• Stocks: {stats['stocks']['unique_count']}")
        print(f"• Publishers: {stats['publishers']['unique_count']}")
        print(f"• Date range: {stats['date_range']['start'].date()} to {stats['date_range']['end'].date()}")
        
        # Top stocks
        print(f"\n🏆 TOP 5 STOCKS:")
        for stock, count in list(stats['stocks']['top_10'].items())[:5]:
            print(f"  {stock}: {count:,} articles")
        
        # Save processed data
        loader.save_processed_data('test_processed_data.csv')
        
    else:
        print("❌ Failed to load data.")

if __name__ == "__main__":
    main()