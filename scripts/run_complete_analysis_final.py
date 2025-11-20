"""
Complete analysis with larger data sample for better insights.
"""
import sys
import os
import pandas as pd
from datetime import datetime

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
    
    # STEP 1: Load Larger Data Sample
    step(1, "LOADING LARGER DATA SAMPLE FOR BETTER INSIGHTS")
    
    from src.data_loader import FinancialNewsLoader
    
    # Load 50,000 rows to get more stocks and better analysis
    print("Loading 50,000 rows for comprehensive analysis...")
    loader = FinancialNewsLoader()
    data = loader.load_data(sample_size=50000)
    
    if data is None:
        print("❌ Failed to load data. Stopping analysis.")
        return
    
    # Display comprehensive insights
    stats = loader.get_basic_stats()
    print(f"\n📊 COMPREHENSIVE DATA OVERVIEW:")
    print(f"   • Total Articles: {stats['total_articles']:,}")
    print(f"   • Date Range: {stats['date_range']['start'].date()} to {stats['date_range']['end'].date()}")
    print(f"   • Unique Stocks: {stats['stocks']['unique_count']}")
    print(f"   • Unique Publishers: {stats['publishers']['unique_count']}")
    
    # Show top stocks if we have multiple
    if stats['stocks']['unique_count'] > 1:
        print(f"\n🏆 TOP 5 MOST COVERED STOCKS:")
        top_stocks = list(stats['stocks']['top_10'].items())[:5]
        for stock, count in top_stocks:
            print(f"   • {stock}: {count:,} articles")
    
    print(f"\n{loader.get_sample_insights()}")
    
    # STEP 2: Exploratory Data Analysis
    step(2, "EXPLORATORY DATA ANALYSIS (EDA)")
    
    from src.eda import FinancialNewsEDA
    
    print("Performing comprehensive EDA... This may take a moment...")
    eda = FinancialNewsEDA(data)
    eda_stats = eda.comprehensive_analysis()
    
    # STEP 3: Technical Analysis (Only if we have multiple stocks)
    step(3, "TECHNICAL ANALYSIS OF TOP STOCKS")
    
    from src.technical_analysis import StockTechnicalAnalyzer
    
    # Get top stocks (minimum 2 articles to be considered)
    stock_counts = data['stock'].value_counts()
    top_stocks = stock_counts[stock_counts >= 2].head(5).index.tolist()
    
    if len(top_stocks) > 0:
        print(f"Analyzing technical indicators for: {', '.join(top_stocks)}")
        
        technical_results = {}
        for stock in top_stocks:
            try:
                print(f"\n📈 Analyzing {stock}...")
                # Use 1 year period for comprehensive analysis
                analyzer = StockTechnicalAnalyzer(stock, period='1y')
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
                    return_val = price_stats.get('total_return', 0)
                    volatility = price_stats.get('volatility', 0)
                    print(f"  📊 Return: {return_val:+.2f}%")
                    print(f"  📊 Volatility: {volatility:.2f}%")
                    
                    # Show active signals
                    active_signals = []
                    for signal, count in signals['counts'].items():
                        if count > 0:
                            signal_name = signal.replace('_', ' ').title()
                            active_signals.append(f"{signal_name}({count})")
                    
                    if active_signals:
                        print(f"  🔔 Active signals: {', '.join(active_signals)}")
                
            except Exception as e:
                print(f"  ❌ Error analyzing {stock}: {e}")
    else:
        print("⚠️  Not enough stock data for technical analysis")
        technical_results = {}
    
    # STEP 4: Save Results
    step(4, "SAVING ANALYSIS RESULTS")
    
    # Save processed data
    loader.save_processed_data('final_processed_data.csv')
    
    # Save technical analysis summary if we have results
    if technical_results:
        save_technical_summary(technical_results)
    
    # STEP 5: Generate Final Report
    step(5, "GENERATING COMPREHENSIVE REPORT")
    
    generate_final_report(loader, technical_results)
    
    print(f"\n{'='*70}")
    print("🎉 PHASE 1: DATA ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    
    print(f"\n📋 ACCOMPLISHED IN THIS PHASE:")
    print("✅ Data loading and cleaning")
    print("✅ Exploratory Data Analysis (EDA) with visualizations") 
    print("✅ Technical analysis of stock indicators")
    print("✅ Data quality assessment")
    print("✅ Processed data saved for next phase")
    
    print(f"\n🎯 READY FOR PHASE 2: SENTIMENT ANALYSIS")
    print("   Next steps:")
    print("   1. Sentiment analysis on news headlines")
    print("   2. Correlation between sentiment and stock prices")
    print("   3. Predictive modeling")

def save_technical_summary(technical_results):
    """Save technical analysis results."""
    summary = ["TECHNICAL ANALYSIS SUMMARY", "=" * 50, ""]
    
    for stock, results in technical_results.items():
        stats = results['stats']
        signals = results['signals']
        
        summary.append(f"STOCK: {stock}")
        summary.append(f"  Analysis Period: {stats.get('period', 'N/A')}")
        
        price_stats = stats.get('price_stats', {})
        if price_stats:
            return_val = price_stats.get('total_return', 0)
            volatility = price_stats.get('volatility', 0)
            summary.append(f"  Total Return: {return_val:+.2f}%")
            summary.append(f"  Volatility: {volatility:.2f}%")
        
        # Add significant signals
        signal_counts = signals.get('counts', {})
        significant_signals = {k: v for k, v in signal_counts.items() if v > 0}
        if significant_signals:
            summary.append("  Active Signals:")
            for signal, count in significant_signals.items():
                signal_name = signal.replace('_', ' ').title()
                summary.append(f"    • {signal_name}: {count}")
        
        summary.append("")
    
    # Save to file
    report_path = os.path.join('data', 'processed', 'technical_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(summary))
    
    print(f"✅ Technical analysis report saved to: {report_path}")

def generate_final_report(loader, technical_results):
    """Generate comprehensive project report."""
    stats = loader.get_basic_stats()
    
    report = []
    report.append("FINANCIAL NEWS ANALYSIS - PHASE 1 COMPLETION REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Data Overview
    report.append("DATA OVERVIEW")
    report.append("-" * 30)
    report.append(f"Total Articles Analyzed: {stats['total_articles']:,}")
    report.append(f"Original Sample Size: {stats['original_size']:,}")
    report.append(f"Data Quality: {stats['data_retention_rate']} retention")
    report.append(f"Analysis Period: {stats['date_range']['start'].date()} to {stats['date_range']['end'].date()}")
    report.append(f"Time Span: {stats['date_range']['days']} days")
    report.append(f"Unique Stocks Covered: {stats['stocks']['unique_count']}")
    report.append(f"News Publishers: {stats['publishers']['unique_count']}")
    report.append("")
    
    # Stock Coverage Insights
    report.append("STOCK COVERAGE ANALYSIS")
    report.append("-" * 30)
    if stats['stocks']['unique_count'] > 1:
        report.append(f"Average Articles per Stock: {stats['stocks']['coverage_stats']['mean_articles_per_stock']:.1f}")
        report.append(f"Most Covered Stock: {list(stats['stocks']['top_10'].keys())[0]} "
                     f"({stats['stocks']['top_10'][list(stats['stocks']['top_10'].keys())[0]]:,} articles)")
    else:
        report.append("Single stock detected in sample - consider larger dataset")
    report.append("")
    
    # Text Analysis
    report.append("TEXT ANALYSIS SUMMARY")
    report.append("-" * 30)
    report.append(f"Average Headline Length: {stats['text_analysis']['avg_headline_length']:.1f} characters")
    report.append(f"Average Words per Headline: {stats['text_analysis']['avg_word_count']:.1f} words")
    report.append(f"Headline Length Range: {stats['text_analysis']['headline_length_range'][0]} to {stats['text_analysis']['headline_length_range'][1]} chars")
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
    
    # Phase 1 Completion
    report.append("PHASE 1: DATA ANALYSIS COMPLETED")
    report.append("-" * 30)
    report.append("✓ Data loading and validation")
    report.append("✓ Data cleaning and preprocessing") 
    report.append("✓ Exploratory Data Analysis (EDA)")
    report.append("✓ Technical indicator calculation")
    report.append("✓ Data quality assessment")
    report.append("")
    
    # Recommendations for Phase 2
    report.append("NEXT PHASE: SENTIMENT ANALYSIS & CORRELATION")
    report.append("-" * 30)
    report.append("1. Perform sentiment analysis on all headlines")
    report.append("2. Calculate daily sentiment scores per stock")
    report.append("3. Correlate sentiment with price movements")
    report.append("4. Build sentiment-based trading signals")
    report.append("5. Create predictive models")
    report.append("")
    
    # Save report
    report_path = os.path.join('data', 'processed', 'phase1_completion_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✅ Phase 1 completion report saved to: {report_path}")

if __name__ == "__main__":
    main()