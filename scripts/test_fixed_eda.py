"""
Test the fixed EDA module with proper indentation.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("🧪 TESTING FIXED EDA MODULE")
    print("=" * 50)
    
    try:
        # First load some data
        from src.data_loader import FinancialNewsLoader
        
        print("Loading sample data...")
        loader = FinancialNewsLoader()
        data = loader.load_data(sample_size=1000)
        
        if data is None:
            print("❌ Failed to load data")
            return
        
        # Test EDA
        from src.eda import FinancialNewsEDA
        
        print("Testing EDA module...")
        eda = FinancialNewsEDA(data)
        
        # Test basic analysis
        print("\n1. Testing basic statistics...")
        basic_stats = eda.basic_statistics()
        print(f"   ✅ Basic stats: {basic_stats['total_articles']} articles")
        
        print("\n2. Testing temporal analysis...")
        temporal_stats = eda.temporal_analysis()
        print(f"   ✅ Temporal analysis: {temporal_stats['busiest_day']}")
        
        print("\n3. Testing publisher analysis...")
        publisher_stats = eda.publisher_analysis()
        print(f"   ✅ Publisher analysis: {publisher_stats['total_publishers']} publishers")
        
        print("\n4. Testing stock analysis...")
        stock_stats = eda.stock_analysis()
        print(f"   ✅ Stock analysis: {stock_stats['total_stocks']} stocks")
        
        print("\n5. Testing text analysis...")
        text_stats = eda.text_analysis()
        print(f"   ✅ Text analysis: {text_stats['total_words']} total words")
        
        print("\n6. Testing comprehensive analysis (non-blocking)...")
        comprehensive_stats = eda.comprehensive_analysis(save_visualizations=True)
        print("   ✅ Comprehensive analysis completed!")
        
        print("\n🎉 ALL EDA TESTS PASSED!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()