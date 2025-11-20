"""
Complete analysis with the optimized data loader.
"""
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def step(step_number, description):
    """Print step header."""
    print(f"\n{'='*70}")
    print(f"🚀 STEP {step_number}: {description}")
    print(f"{'='*70}")

def main():
    print("🎯 FINANCIAL NEWS ANALYSIS - COMPLETE WORKFLOW")
    print("=" * 70)
    
    # STEP 1: Load Data
    step(1, "LOADING AND PROCESSING YOUR DATA")
    
    from src.data_loader import FinancialNewsLoader
    
    # For initial testing, use sample_size=50000 to test with 50,000 rows
    # Remove sample_size parameter to load full dataset (1.4M rows)
    loader = FinancialNewsLoader()
    data = loader.load_data(sample_size=50000)  # Change this number as needed
    
    if data is None:
        print("❌ Failed to load data. Stopping analysis.")
        return
    
    # Display quick insights
    print("\n" + loader.get_sample_insights())
    
    # STEP 2: Exploratory Data Analysis
    step(2, "EXPLORATORY DATA ANALYSIS (EDA)")
    
    from src.eda import FinancialNewsEDA
    
    print("Performing comprehensive EDA...")
    eda = FinancialNewsEDA(data)
    eda_stats = eda.comprehensive_analysis()
    
    # STEP 3: Technical Analysis
    step(3, "TECHNICAL ANALYSIS OF TOP STOCKS")
    
    from src.technical_analysis import StockTechnicalAnalyzer
    
    # Get top 5 most mentioned stocks
    top_stocks = data['stock'].value_counts().head(5).index.tolist()
    print(f"Analyzing technical indicators for: {', '.join(top_stocks)}")
    
    technical_results = {}
    for stock in top_stocks:
        try:
            print(f"\n📈 Analyzing {stock}...")
            # Use 6 months period for recent analysis
            analyzer = StockTechnicalAnalyzer(stock, period='6mo')
            analyzer.calculate_indicators()
            
            signals = analyzer.generate_signals()
            stock_stats = analyzer.get_summary_stats()
            
            technical_results[stock] = {
                'signals': signals,
                'stats': stock_stats
            }
            
            # Print key metrics
            price_stats = stock_stats.get('price_stats', {})
            if price_stats:
                print(f"  📊 Return: {price_stats.get('total_return', 0):.2f}%")
                print(f"  📊 Volatility: {price_stats.get('volatility', 0):.2f}%")
                
                # Show bullish signals
                bullish_count = sum(1 for sig, count in signals['counts'].items() 
                                  if count > 0 and any(word in sig for word in ['bull', 'golden', 'oversold']))
                print(f"  🔔 {bullish_count} bullish signals detected")
            
        except Exception as e:
            print(f"  ❌ Error analyzing {stock}: {e}")
    
    # STEP 4: Save Results
    step(4, "SAVING ANALYSIS RESULTS")
    
    # Save processed data
    loader.save_processed_data()
    
    # Save technical analysis summary
    save_technical_summary(technical_results)
    
    # STEP 5: Generate Final Report
    step(5, "GENERATING FINAL REPORT")
    
    generate_comprehensive_report(loader, technical_results)
    
    print(f"\n{'='*70}")
    print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    print("\n📋 WHAT WE'VE ACCOMPLISHED:")
    print("✅ Loaded and cleaned financial news data")
    print("✅ Performed comprehensive exploratory analysis") 
    print("✅ Analyzed technical indicators for top stocks")
    print("✅ Generated visualizations and insights")
    print("✅ Saved processed data for future analysis")
    print("\n🎯 NEXT STEPS: Sentiment Analysis & Correlation Study")

def save_technical_summary(technical_results):
    """Save technical analysis results."""
    if not technical_results:
        return
    
    summary = ["TECHNICAL ANALYSIS SUMMARY", "=" * 40, ""]
    
    for stock, results in technical_results.items():
        stats = results['stats']
        signals = results['signals']
        
        summary.append(f"STOCK: {stock}")
        summary.append(f"  Period: {stats.get('period', 'N/A')}")
        
        price_stats = stats.get('price_stats', {})
        if price_stats:
            summary.append(f"  Total Return: {price_stats.get('total_return', 0):.2f}%")
            summary.append(f"  Volatility: {price_stats.get('volatility', 0):.2f}%")
        
        # Add signals
        signal_counts = signals.get('counts', {})
        if signal_counts:
            summary.append("  Signals:")
            for signal, count in signal_counts.items():
                if count > 0:
                    summary.append(f"    • {signal}: {count}")
        
        summary.append("")
    
    # Save to file
    report_path = os.path.join('data', 'processed', 'technical_analysis_summary.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(summary))
    
    print(f"✅ Technical analysis summary saved to: {report_path}")

def generate_comprehensive_report(loader, technical_results):
    """Generate a comprehensive project report."""
    stats = loader.get_basic_stats()
    
    report = []
    report.append("FINANCIAL NEWS ANALYSIS - COMPREHENSIVE REPORT")
    report.append("=" * 50)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Data Overview
    report.append("DATA OVERVIEW")
    report.append("-" * 30)
    report.append(f"Total Articles Analyzed: {stats['total_articles']:,}")
    report.append(f"Original Dataset Size: {stats['original_size']:,}")
    report.append(f"Data Retention Rate: {stats['data_retention_rate']}")
    report.append(f"Analysis Period: {stats['date_range']['start'].date()} to {stats['date_range']['end'].date()}")
    report.append(f"Unique Stocks: {stats['stocks']['unique_count']}")
    report.append(f"Unique Publishers: {stats['publishers']['unique_count']}")
    report.append("")
    
    # Stock Coverage
    report.append("STOCK COVERAGE INSIGHTS")
    report.append("-" * 30)
    report.append(f"Average Articles per Stock: {stats['stocks']['coverage_stats']['mean_articles_per_stock']:.1f}")
    report.append(f"Most Covered Stock: {list(stats['stocks']['top_10'].keys())[0]} "
                 f"({stats['stocks']['top_10'][list(stats['stocks']['top_10'].keys())[0]]:,} articles)")
    report.append("")
    
    # Technical Analysis Summary
    if technical_results:
        report.append("TECHNICAL ANALYSIS RESULTS")
        report.append("-" * 30)
        for stock, results in technical_results.items():
            price_stats = results['stats'].get('price_stats', {})
            if price_stats:
                return_val = price_stats.get('total_return', 0)
                volatility = price_stats.get('volatility', 0)
                report.append(f"{stock}: {return_val:+.2f}% return, {volatility:.2f}% volatility")
        report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS FOR NEXT PHASE")
    report.append("-" * 30)
    report.append("1. Perform sentiment analysis on headlines")
    report.append("2. Correlate sentiment scores with stock returns") 
    report.append("3. Build predictive models")
    report.append("4. Analyze publisher bias and impact")
    report.append("")
    
    # Save report
    report_path = os.path.join('data', 'processed', 'comprehensive_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✅ Comprehensive report saved to: {report_path}")

if __name__ == "__main__":
    main()