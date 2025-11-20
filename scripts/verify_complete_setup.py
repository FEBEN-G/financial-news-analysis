"""
Complete setup verification script that tests all components.
"""
import sys
import os
import subprocess

def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ Success")
            return True
        else:
            print(f"   ❌ Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_import(package_name, import_name=None):
    """Test if a package can be imported."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"   ✅ {package_name}")
        return True
    except ImportError as e:
        print(f"   ❌ {package_name}: {e}")
        return False

def main():
    print("=" * 70)
    print("COMPLETE SETUP VERIFICATION")
    print("=" * 70)
    
    # Add src to path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    # Test critical imports
    print("\n📦 TESTING PACKAGE IMPORTS")
    print("-" * 40)
    
    critical_packages = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'sklearn',
        'nltk',
        'textblob',
        'yfinance',
        'scipy',
        'plotly',
        'wordcloud',
        'pandas_ta'
    ]
    
    critical_success = True
    for package in critical_packages:
        if package == 'sklearn':
            success = test_import('scikit-learn', 'sklearn')
        else:
            success = test_import(package)
        critical_success = critical_success and success
    
    # Test our custom modules
    print("\n🔧 TESTING CUSTOM MODULES")
    print("-" * 40)
    
    custom_modules = [
        'data_loader',
        'eda',
        'technical_analysis',
        'config'
    ]
    
    custom_success = True
    for module in custom_modules:
        try:
            __import__(f'src.{module}')
            print(f"   ✅ {module}")
        except Exception as e:
            print(f"   ❌ {module}: {e}")
            custom_success = False
    
    # Test data directory structure
    print("\n📁 TESTING DATA DIRECTORY STRUCTURE")
    print("-" * 40)
    
    required_dirs = ['data/raw', 'data/processed']
    dir_success = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}")
            # List files in raw directory
            if dir_path == 'data/raw':
                files = os.listdir(dir_path)
                if files:
                    print(f"      Files: {', '.join(files)}")
                else:
                    print("      ⚠️  No data files found")
        else:
            print(f"   ❌ {dir_path} - Directory missing")
            dir_success = False
    
    # Test data loading if possible
    if custom_success and dir_success:
        print("\n📊 TESTING DATA LOADING")
        print("-" * 40)
        
        try:
            from src.data_loader import FinancialNewsLoader
            
            # Check for data files
            raw_files = os.listdir('data/raw') if os.path.exists('data/raw') else []
            if raw_files:
                print(f"   Found data files: {raw_files}")
                
                # Try to load data
                loader = FinancialNewsLoader()
                data = loader.load_data()
                
                if data is not None:
                    stats = loader.get_basic_stats()
                    print(f"   ✅ Data loaded successfully!")
                    print(f"      Articles: {stats['total_articles']:,}")
                    print(f"      Stocks: {stats['unique_stocks']}")
                    print(f"      Date range: {stats['date_range'][0].date()} to {stats['date_range'][1].date()}")
                else:
                    print("   ❌ Data loading failed")
            else:
                print("   ⚠️  No data files found in data/raw/")
                print("      Please add your financial_news.csv file")
                
        except Exception as e:
            print(f"   ❌ Data loading test failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SETUP SUMMARY")
    print("=" * 70)
    
    if critical_success and custom_success:
        print("🎉 EXCELLENT! All critical components are working!")
        print("\nNext steps:")
        print("1. Add your financial_news.csv file to data/raw/ directory")
        print("2. Run: python scripts/run_complete_analysis.py")
        print("3. Check the generated visualizations and reports")
    else:
        print("⚠️  Some components need attention:")
        if not critical_success:
            print("   - Some Python packages failed to import")
        if not custom_success:
            print("   - Some custom modules have issues")
        if not dir_success:
            print("   - Directory structure needs setup")
        
        print("\nTo fix issues:")
        print("1. Install missing packages: pip install <package_name>")
        print("2. Check the error messages above for specific issues")
    
    print("=" * 70)

if __name__ == "__main__":
    main()