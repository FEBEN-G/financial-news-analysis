"""
Phase 2: Sentiment Analysis Execution
"""
import sys
import os
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def step(step_number, description):
    """Print step header."""
    print(f"\n{'='*70}")
    print(f"🎯 PHASE 2 - STEP {step_number}: {description}")
    print(f"{'='*70}")

def main():
    print("🚀 FINANCIAL NEWS ANALYSIS - PHASE 2: SENTIMENT ANALYSIS")
    print("=" * 70)
    
    # STEP 1: Load Processed Data
    step(1, "LOADING PROCESSED DATA FROM PHASE 1")
    
    processed_file = 'data/processed/phase1_analysis_data.csv'
    
    if not os.path.exists(processed_file):
        print("❌ Processed data not found. Please run Phase 1 first.")
        return
    
    print("Loading processed data...")
    data = pd.read_csv(processed_file, parse_dates=['date'])
    print(f"✅ Loaded {len(data):,} articles for sentiment analysis")
    print(f"   Stocks: {data['stock'].nunique()}")
    print(f"   Date range: {data['date'].min().date()} to {data['date'].max().date()}")
    
    # STEP 2: Initialize Sentiment Analyzer
    step(2, "INITIALIZING SENTIMENT ANALYSIS")
    
    from src.sentiment_analysis import NewsSentimentAnalyzer
    
    print("Initializing sentiment analyzer...")
    sentiment_analyzer = NewsSentimentAnalyzer()
    
    # STEP 3: Perform Sentiment Analysis
    step(3, "PERFORMING SENTIMENT ANALYSIS ON HEADLINES")
    
    # Analyze a sample first for testing (remove sample_size for full analysis)
    print("Analyzing sentiment for all articles...")
    sentiment_results = sentiment_analyzer.analyze_news_data(data, sample_size=2000)
    
    print(f"✅ Sentiment analysis completed for {len(sentiment_results):,} articles")
    
    # STEP 4: Analyze Sentiment Results
    step(4, "ANALYZING SENTIMENT RESULTS")
    
    sentiment_summary = sentiment_analyzer.get_sentiment_summary(sentiment_results)
    
    print(f"\n📊 SENTIMENT SUMMARY:")
    print(f"   Total Articles: {sentiment_summary['total_articles']:,}")
    print(f"   Sentiment Distribution:")
    for sentiment, count in sentiment_summary['sentiment_distribution'].items():
        percentage = (count / sentiment_summary['total_articles']) * 100
        print(f"     • {sentiment}: {count} ({percentage:.1f}%)")
    
    print(f"\n   Average Scores:")
    for score_name, value in sentiment_summary['average_scores'].items():
        print(f"     • {score_name}: {value:.3f}")
    
    # STEP 5: Save Sentiment Results
    step(5, "SAVING SENTIMENT ANALYSIS RESULTS")
    
    # Save full sentiment data
    sentiment_file = 'data/processed/sentiment_analysis_results.csv'
    sentiment_results.to_csv(sentiment_file, index=False)
    print(f"✅ Full sentiment results saved to: {sentiment_file}")
    
    # Save sentiment summary
    summary_file = 'data/processed/sentiment_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("SENTIMENT ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERVIEW:\n")
        f.write(f"  Articles Analyzed: {sentiment_summary['total_articles']:,}\n")
        f.write(f"  Stocks Covered: {data['stock'].nunique()}\n\n")
        
        f.write("SENTIMENT DISTRIBUTION:\n")
        for sentiment, count in sentiment_summary['sentiment_distribution'].items():
            percentage = (count / sentiment_summary['total_articles']) * 100
            f.write(f"  {sentiment}: {count} ({percentage:.1f}%)\n")
        
        f.write("\nAVERAGE SENTIMENT SCORES:\n")
        for score_name, value in sentiment_summary['average_scores'].items():
            f.write(f"  {score_name}: {value:.3f}\n")
        
        f.write("\nMOST POSITIVE ARTICLES:\n")
        for article in sentiment_summary['most_positive_articles']:
            f.write(f"  Score {article['combined_sentiment_score']:.3f}: {article['headline'][:80]}...\n")
        
        f.write("\nMOST NEGATIVE ARTICLES:\n")
        for article in sentiment_summary['most_negative_articles']:
            f.write(f"  Score {article['combined_sentiment_score']:.3f}: {article['headline'][:80]}...\n")
    
    print(f"✅ Sentiment summary saved to: {summary_file}")
    
    # STEP 6: Prepare for Correlation Analysis
    step(6, "PREPARING FOR CORRELATION ANALYSIS")
    
    print("Phase 2 completed successfully! Ready for correlation analysis.")
    print("\nNext steps:")
    print("1. Correlate sentiment scores with stock price movements")
    print("2. Analyze sentiment impact on different timeframes")
    print("3. Build sentiment-based trading signals")
    print("4. Create predictive models")
    
    print(f"\n{'='*70}")
    print("🎉 PHASE 2: SENTIMENT ANALYSIS COMPLETED!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()