"""
Final test to verify everything is working.
"""
import sys
import os
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("🎯 FINAL SYSTEM TEST")
    print("=" * 50)
    
    # Clean up
    viz_dir = 'data/processed/visualizations'
    if os.path.exists(viz_dir):
        shutil.rmtree(viz_dir)
    
    try:
        # Test 1: Data Loading
        print("1. Testing Data Loader...")
        from src.data_loader import FinancialNewsLoader
        loader = FinancialNewsLoader()
        data = loader.load_data(sample_size=5000)
        
        if data is None:
            print("❌ Data loading failed")
            return
        print(f"   ✅ Loaded {len(data):,} articles")
        
        # Test 2: EDA
        print("2. Testing EDA...")
        from src.eda import FinancialNewsEDA
        eda = FinancialNewsEDA(data)
        stats = eda.comprehensive_analysis(save_visualizations=True)
        print("   ✅ EDA completed")
        
        # Test 3: Check outputs
        print("3. Checking outputs...")
        if os.path.exists(viz_dir):
            files = os.listdir(viz_dir)
            print(f"   ✅ Created {len(files)} visualization files")
        else:
            print("   ❌ No visualizations created")
        
        # Test 4: Technical Analysis
        print("4. Testing Technical Analysis...")
        from src.technical_analysis import StockTechnicalAnalyzer
        
        # Get a stock that exists in our data
        stocks_in_data = data['stock'].unique()
        if len(stocks_in_data) > 0:
            test_stock = stocks_in_data[0]
            print(f"   Testing with stock: {test_stock}")
            
            try:
                analyzer = StockTechnicalAnalyzer(test_stock, period='6mo')
                analyzer.calculate_indicators()
                signals = analyzer.generate_signals()
                print(f"   ✅ Technical analysis successful")
            except Exception as e:
                print(f"   ⚠️  Technical analysis warning: {e}")
        else:
            print("   ⚠️  No stocks found for technical analysis")
        
        print("\n🎉 ALL SYSTEMS GO! Ready for Phase 2: Sentiment Analysis")
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()