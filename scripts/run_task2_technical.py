"""
TASK 2: Quantitative Analysis with Technical Indicators
"""
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.data_processing import load_financial_news_data
    from src.technical_analysis import StockTechnicalAnalyzer
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def main():
    print("🚀 TASK 2: QUANTITATIVE ANALYSIS WITH TECHNICAL INDICATORS")
    print("=" * 70)
    
    # Check for technical analysis libraries
    try:
        import pandas_ta
        print("✓ pandas_ta is available (TA-Lib alternative)")
        PANDAS_TA_AVAILABLE = True
    except ImportError:
        print("❌ pandas_ta not available. Please install: pip install pandas_ta")
        PANDAS_TA_AVAILABLE = False
        return

    # Step 1: Load financial news data from the correct file
    print("\n" + "=" * 70)
    print("📊 STEP 1: LOADING FINANCIAL NEWS DATA")
    print("=" * 70)
    
    # Use the actual data file
    news_data = load_financial_news_data("data/raw/raw_analyst_ratings.csv", sample_size=5000)
    
    if news_data is None or len(news_data) == 0:
        print("❌ Failed to load financial news data")
        # Try with sample data as fallback
        print("Trying with sample data...")
        news_data = load_financial_news_data(sample_size=1000)
    
    if news_data is None or len(news_data) == 0:
        print("❌ No data available for analysis")
        return
    
    print(f"✅ Loaded {len(news_data)} articles")
    
    # Get top stocks from the data - ensure we have the stock column
    if 'stock' not in news_data.columns:
        print("❌ No 'stock' column found in data")
        return
    
    top_stocks = news_data['stock'].value_counts().head(5).index.tolist()
    print(f"📈 Top {len(top_stocks)} stocks to analyze: {', '.join(top_stocks)}")

    # Step 2: Perform technical analysis on each stock
    print("\n" + "=" * 70)
    print("📊 STEP 2: PERFORMING TECHNICAL ANALYSIS")
    print("=" * 70)
    
    technical_results = {}
    
    for symbol in top_stocks:
        print(f"\n🔍 Analyzing {symbol}...")
        try:
            # Use the StockTechnicalAnalyzer class
            analyzer = StockTechnicalAnalyzer(symbol, period="1y")
            analyzer.calculate_indicators()
            
            # Get summary statistics
            stats = analyzer.get_summary_stats()
            signals = analyzer.generate_signals()
            
            technical_results[symbol] = {
                'analyzer': analyzer,
                'stats': stats,
                'signals': signals,
                'data': analyzer.data
            }
            
            print(f"✅ Technical analysis completed for {symbol}")
            
            # Safely display price information
            if stats and 'price_stats' in stats:
                price_stats = stats['price_stats']
                try:
                    end_price = price_stats.get('end_price', 0)
                    total_return = price_stats.get('total_return', 0)
                    print(f"   Current Price: ${end_price:.2f}")
                    print(f"   Total Return: {total_return:.2f}%")
                except (TypeError, ValueError) as e:
                    print(f"   Current Price: ${price_stats.get('end_price', 0)}")
                    print(f"   Total Return: {price_stats.get('total_return', 0)}%")
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            technical_results[symbol] = None

    # Step 3: Generate technical analysis report
    print("\n" + "=" * 70)
    print("📊 STEP 3: GENERATING TECHNICAL ANALYSIS REPORT")
    print("=" * 70)
    
    generate_technical_report(technical_results)

    # Step 4: Create visualizations
    print("\n" + "=" * 70)
    print("📊 STEP 4: CREATING TECHNICAL ANALYSIS VISUALIZATIONS")
    print("=" * 70)
    
    create_technical_visualizations(technical_results)

    # Step 5: Comparative analysis
    print("\n" + "=" * 70)
    print("📊 STEP 5: COMPARATIVE TECHNICAL ANALYSIS")
    print("=" * 70)
    
    comparative_analysis(technical_results)

    # Step 6: Key findings and insights
    print("\n" + "=" * 70)
    print("📊 STEP 6: KEY FINDINGS & INSIGHTS")
    print("=" * 70)
    
    generate_insights(technical_results)
    
    print("\n🎉 TASK 2 COMPLETED SUCCESSFULLY!")

def generate_technical_report(technical_results, output_dir="reports"):
    """Generate a comprehensive technical analysis report"""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "technical_analysis_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("TECHNICAL ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        successful_analysis = {k: v for k, v in technical_results.items() if v is not None}
        
        if not successful_analysis:
            f.write("❌ No successful technical analysis results.\n")
            return
        
        f.write(f"Stocks Analyzed: {len(successful_analysis)}\n\n")
        
        for symbol, result in successful_analysis.items():
            f.write(f"STOCK: {symbol}\n")
            f.write("-" * 40 + "\n")
            
            stats = result['stats']
            signals = result['signals']
            
            # Price statistics
            f.write("PRICE STATISTICS:\n")
            price_stats = stats.get('price_stats', {})
            f.write(f"  Current Price: ${price_stats.get('end_price', 0):.2f}\n")
            f.write(f"  Total Return: {price_stats.get('total_return', 0):.2f}%\n")
            f.write(f"  Volatility: {price_stats.get('volatility', 0):.2f}%\n")
            f.write(f"  Max Price: ${price_stats.get('max_price', 0):.2f}\n")
            f.write(f"  Min Price: ${price_stats.get('min_price', 0):.2f}\n\n")
            
            # Trading signals
            f.write("TRADING SIGNALS:\n")
            signal_counts = signals.get('counts', {})
            signal_descriptions = signals.get('descriptions', {})
            
            for signal, count in signal_counts.items():
                if count > 0:
                    desc = signal_descriptions.get(signal, 'No description available')
                    f.write(f"  {signal}: {count} occurrences\n")
                    f.write(f"    - {desc}\n")
            
            f.write("\n")
    
    print(f"✅ Technical analysis report saved to: {report_path}")

def create_technical_visualizations(technical_results, output_dir="reports/figures"):
    """Create technical analysis charts for each stock"""
    os.makedirs(output_dir, exist_ok=True)
    
    successful_analysis = {k: v for k, v in technical_results.items() if v is not None}
    
    if not successful_analysis:
        print("❌ No successful analysis results to visualize")
        return
    
    print(f"Creating charts for {len(successful_analysis)} stocks...")
    
    for symbol, result in successful_analysis.items():
        try:
            analyzer = result['analyzer']
            
            # Create the plot using the analyzer's method
            chart_path = os.path.join(output_dir, f"technical_analysis_{symbol}.png")
            analyzer.plot_technical_analysis(save_path=chart_path)
            
            print(f"✅ Chart saved for {symbol}")
            
        except Exception as e:
            print(f"❌ Error creating chart for {symbol}: {e}")

def comparative_analysis(technical_results, output_dir="data/processed"):
    """Perform comparative analysis across all stocks"""
    os.makedirs(output_dir, exist_ok=True)
    
    successful_analysis = {k: v for k, v in technical_results.items() if v is not None}
    
    if not successful_analysis:
        print("❌ No data for comparative analysis")
        return
    
    # Create comparative DataFrame
    comparative_data = []
    
    for symbol, result in successful_analysis.items():
        stats = result['stats']
        signals = result['signals']
        
        price_stats = stats.get('price_stats', {})
        signal_counts = signals.get('counts', {})
        
        comparative_data.append({
            'Symbol': symbol,
            'Current_Price': price_stats.get('end_price', 0),
            'Total_Return_%': price_stats.get('total_return', 0),
            'Volatility_%': price_stats.get('volatility', 0),
            'Bullish_Signals': sum(1 for count in signal_counts.values() if count > 0),
            'Total_Signals': len([count for count in signal_counts.values() if count > 0])
        })
    
    comparative_df = pd.DataFrame(comparative_data)
    
    # Display results
    print("\n📊 COMPARATIVE ANALYSIS SUMMARY:")
    print("-" * 60)
    print(comparative_df.to_string(index=False))
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "comparative_technical_analysis.csv")
    comparative_df.to_csv(csv_path, index=False)
    print(f"✅ Comparative analysis saved to: {csv_path}")
    
    # Create comparative visualization
    try:
        plt.figure(figsize=(12, 8))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comparative Technical Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Current Prices
        axes[0, 0].bar(comparative_df['Symbol'], comparative_df['Current_Price'], color='skyblue')
        axes[0, 0].set_title('Current Stock Prices')
        axes[0, 0].set_ylabel('Price ($)')
        
        # Plot 2: Total Returns
        colors = ['green' if x >= 0 else 'red' for x in comparative_df['Total_Return_%']]
        axes[0, 1].bar(comparative_df['Symbol'], comparative_df['Total_Return_%'], color=colors)
        axes[0, 1].set_title('Total Returns (%)')
        axes[0, 1].set_ylabel('Return (%)')
        
        # Plot 3: Volatility
        axes[1, 0].bar(comparative_df['Symbol'], comparative_df['Volatility_%'], color='orange')
        axes[1, 0].set_title('Volatility (%)')
        axes[1, 0].set_ylabel('Volatility (%)')
        
        # Plot 4: Bullish Signals
        axes[1, 1].bar(comparative_df['Symbol'], comparative_df['Bullish_Signals'], color='lightgreen')
        axes[1, 1].set_title('Number of Bullish Signals')
        axes[1, 1].set_ylabel('Signal Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join("reports/figures", "comparative_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Comparative analysis chart saved")
        
    except Exception as e:
        print(f"❌ Error creating comparative chart: {e}")

def generate_insights(technical_results):
    """Generate key insights from technical analysis"""
    successful_analysis = {k: v for k, v in technical_results.items() if v is not None}
    
    if not successful_analysis:
        print("❌ No insights available - no successful analysis results")
        return
    
    print("\n🔍 KEY TECHNICAL ANALYSIS INSIGHTS:")
    print("-" * 50)
    
    # Basic statistics
    total_stocks = len(successful_analysis)
    positive_returns = sum(1 for result in successful_analysis.values() 
                          if result['stats'].get('price_stats', {}).get('total_return', 0) > 0)
    
    print(f"• Stocks Analyzed: {total_stocks}")
    print(f"• Stocks with Positive Returns: {positive_returns}/{total_stocks}")
    
    # Find best and worst performers
    returns_data = []
    for symbol, result in successful_analysis.items():
        price_stats = result['stats'].get('price_stats', {})
        returns_data.append({
            'symbol': symbol,
            'return': price_stats.get('total_return', 0),
            'volatility': price_stats.get('volatility', 0)
        })
    
    returns_df = pd.DataFrame(returns_data)
    
    if not returns_df.empty:
        best_performer = returns_df.loc[returns_df['return'].idxmax()]
        worst_performer = returns_df.loc[returns_df['return'].idxmin()]
        
        print(f"• Best Performer: {best_performer['symbol']} ({best_performer['return']:.2f}%)")
        print(f"• Worst Performer: {worst_performer['symbol']} ({worst_performer['return']:.2f}%)")
        
        # Volatility insights
        avg_volatility = returns_df['volatility'].mean()
        high_vol_stocks = returns_df[returns_df['volatility'] > avg_volatility]['symbol'].tolist()
        
        print(f"• Average Volatility: {avg_volatility:.2f}%")
        if high_vol_stocks:
            print(f"• High Volatility Stocks: {', '.join(high_vol_stocks)}")
    
    # Signal insights
    total_bullish_signals = 0
    for symbol, result in successful_analysis.items():
        signals = result['signals'].get('counts', {})
        bullish_signals = sum(count for signal, count in signals.items() 
                             if 'bullish' in signal.lower() or 'oversold' in signal.lower())
        total_bullish_signals += bullish_signals
    
    print(f"• Total Bullish Signals Detected: {total_bullish_signals}")
    print(f"• Market Outlook: {'Mostly Bullish' if total_bullish_signals > total_stocks * 2 else 'Mixed'}")
    
    # Trading recommendations
    print("\n💡 TRADING RECOMMENDATIONS:")
    print("-" * 30)
    
    for symbol, result in successful_analysis.items():
        price_stats = result['stats'].get('price_stats', {})
        current_return = price_stats.get('total_return', 0)
        volatility = price_stats.get('volatility', 0)
        
        if current_return > 10 and volatility < 20:
            recommendation = "STRONG BUY"
        elif current_return > 5:
            recommendation = "BUY"
        elif current_return < -10:
            recommendation = "AVOID"
        else:
            recommendation = "HOLD"
        
        print(f"• {symbol}: {recommendation} (Return: {current_return:.2f}%, Vol: {volatility:.2f}%)")

if __name__ == "__main__":
    main()