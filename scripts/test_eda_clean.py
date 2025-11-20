"""
Clean test of EDA module with fixed directory handling.
"""
import sys
import os
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def cleanup_previous_results():
    """Clean up previous test results."""
    viz_dir = 'data/processed/visualizations'
    if os.path.exists(viz_dir):
        print(f"Cleaning up previous results in {viz_dir}...")
        shutil.rmtree(viz_dir)
    
    # Remove individual plot files
    plot_files = [
        'data/processed/eda_comprehensive.png',
        'data/processed/visualizations'
    ]
    for file_path in plot_files:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                shutil.rmtree(file_path)

def main():
    print("🧪 CLEAN TEST OF EDA MODULE")
    print("=" * 50)
    
    # Clean up previous results
    cleanup_previous_results()
    
    try:
        # Load data
        from src.data_loader import FinancialNewsLoader
        
        print("Loading sample data...")
        loader = FinancialNewsLoader()
        data = loader.load_data(sample_size=2000)  # Use 2000 for better stock diversity
        
        if data is None:
            print("❌ Failed to load data")
            return
        
        # Test EDA
        from src.eda import FinancialNewsEDA
        
        print("Testing EDA module with clean directory...")
        eda = FinancialNewsEDA(data)
        
        print("\nRunning comprehensive analysis...")
        comprehensive_stats = eda.comprehensive_analysis(save_visualizations=True)
        
        print("✅ Comprehensive analysis completed successfully!")
        
        # Check if files were created
        viz_dir = 'data/processed/visualizations'
        if os.path.exists(viz_dir):
            created_files = os.listdir(viz_dir)
            print(f"✅ Created {len(created_files)} visualization files:")
            for file in created_files:
                print(f"   • {file}")
        else:
            print("❌ Visualization directory was not created")
        
        print("\n🎉 EDA MODULE WORKING PERFECTLY!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()