"""
Sentiment Analysis module for financial news headlines.
"""
import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import warnings
warnings.filterwarnings('ignore')

class NewsSentimentAnalyzer:
    """
    A class to perform sentiment analysis on financial news headlines.
    
    This class uses:
    - TextBlob for general sentiment analysis
    - VADER for financial text sentiment analysis
    - Custom financial lexicon enhancement
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        # Initialize VADER sentiment analyzer
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
            self.vader_available = True
        except:
            self.vader_available = False
            print("⚠️  VADER sentiment analyzer not available")
        
        # Financial sentiment words enhancement
        self.financial_positive = {
            'beat', 'surge', 'rally', 'gain', 'profit', 'growth', 'bullish', 
            'upgrade', 'buy', 'outperform', 'strong', 'positive', 'record',
            'high', 'rise', 'jump', 'soar', 'boost', 'optimistic', 'recovery'
        }
        
        self.financial_negative = {
            'fall', 'drop', 'decline', 'loss', 'bearish', 'downgrade', 'sell',
            'underperform', 'weak', 'negative', 'low', 'plunge', 'crash',
            'slide', 'tumble', 'warning', 'cut', 'miss', 'disappoint', 'risk'
        }
    
    def analyze_sentiment_textblob(self, text):
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores
        """
        try:
            analysis = TextBlob(str(text))
            polarity = analysis.sentiment.polarity  # -1 to 1
            subjectivity = analysis.sentiment.subjectivity  # 0 to 1
            
            # Categorize sentiment
            if polarity > 0.1:
                sentiment_category = 'positive'
            elif polarity < -0.1:
                sentiment_category = 'negative'
            else:
                sentiment_category = 'neutral'
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'category': sentiment_category,
                'method': 'textblob'
            }
        except:
            return {
                'polarity': 0,
                'subjectivity': 0,
                'category': 'neutral',
                'method': 'textblob'
            }
    
    def analyze_sentiment_vader(self, text):
        """
        Analyze sentiment using VADER (specifically for social media/text).
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores
        """
        if not self.vader_available:
            return {
                'compound': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'category': 'neutral',
                'method': 'vader'
            }
        
        try:
            scores = self.sia.polarity_scores(str(text))
            
            # Categorize sentiment
            compound = scores['compound']
            if compound >= 0.05:
                sentiment_category = 'positive'
            elif compound <= -0.05:
                sentiment_category = 'negative'
            else:
                sentiment_category = 'neutral'
            
            return {
                'compound': compound,
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'category': sentiment_category,
                'method': 'vader'
            }
        except:
            return {
                'compound': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'category': 'neutral',
                'method': 'vader'
            }
    
    def analyze_sentiment_financial(self, text):
        """
        Analyze sentiment with financial context.
        
        Args:
            text (str): Financial text to analyze
            
        Returns:
            dict: Financial sentiment scores
        """
        text_lower = str(text).lower()
        
        # Count financial sentiment words
        positive_count = sum(1 for word in self.financial_positive if word in text_lower)
        negative_count = sum(1 for word in self.financial_negative if word in text_lower)
        
        # Calculate financial sentiment score
        total_financial_words = positive_count + negative_count
        if total_financial_words > 0:
            financial_score = (positive_count - negative_count) / total_financial_words
        else:
            financial_score = 0
        
        # Categorize financial sentiment
        if financial_score > 0.1:
            financial_category = 'bullish'
        elif financial_score < -0.1:
            financial_category = 'bearish'
        else:
            financial_category = 'neutral'
        
        return {
            'financial_score': financial_score,
            'positive_words': positive_count,
            'negative_words': negative_count,
            'financial_category': financial_category,
            'method': 'financial'
        }
    
    def analyze_headline(self, headline):
        """
        Perform comprehensive sentiment analysis on a headline.
        
        Args:
            headline (str): News headline to analyze
            
        Returns:
            dict: Comprehensive sentiment analysis
        """
        # Get all sentiment scores
        textblob_result = self.analyze_sentiment_textblob(headline)
        vader_result = self.analyze_sentiment_vader(headline)
        financial_result = self.analyze_sentiment_financial(headline)
        
        # Combined sentiment score (weighted average)
        combined_score = (
            textblob_result['polarity'] * 0.3 +
            vader_result['compound'] * 0.4 +
            financial_result['financial_score'] * 0.3
        )
        
        # Final sentiment category
        if combined_score > 0.05:
            final_sentiment = 'positive'
        elif combined_score < -0.05:
            final_sentiment = 'negative'
        else:
            final_sentiment = 'neutral'
        
        return {
            'headline': headline,
            'textblob_polarity': textblob_result['polarity'],
            'textblob_subjectivity': textblob_result['subjectivity'],
            'textblob_category': textblob_result['category'],
            'vader_compound': vader_result['compound'],
            'vader_positive': vader_result['positive'],
            'vader_negative': vader_result['negative'],
            'vader_neutral': vader_result['neutral'],
            'vader_category': vader_result['category'],
            'financial_score': financial_result['financial_score'],
            'positive_words': financial_result['positive_words'],
            'negative_words': financial_result['negative_words'],
            'financial_category': financial_result['financial_category'],
            'combined_sentiment_score': combined_score,
            'final_sentiment': final_sentiment
        }
    
    def analyze_news_data(self, news_data, sample_size=None):
        """
        Perform sentiment analysis on a dataset of news articles.
        
        Args:
            news_data (pd.DataFrame): News data with 'headline' column
            sample_size (int): Number of articles to analyze (None for all)
            
        Returns:
            pd.DataFrame: News data with sentiment scores
        """
        print("Starting sentiment analysis...")
        
        # Use sample if specified
        if sample_size and len(news_data) > sample_size:
            analysis_data = news_data.head(sample_size).copy()
            print(f"Analyzing sample of {sample_size} articles...")
        else:
            analysis_data = news_data.copy()
            print(f"Analyzing all {len(analysis_data)} articles...")
        
        # Analyze each headline
        sentiment_results = []
        
        for idx, row in analysis_data.iterrows():
            if idx % 500 == 0:  # Progress indicator
                print(f"  Processed {idx}/{len(analysis_data)} articles...")
            
            sentiment_analysis = self.analyze_headline(row['headline'])
            sentiment_analysis.update({
                'stock': row['stock'],
                'date': row['date'],
                'publisher': row.get('publisher', 'Unknown')
            })
            sentiment_results.append(sentiment_analysis)
        
        # Convert to DataFrame
        sentiment_df = pd.DataFrame(sentiment_results)
        
        print("✅ Sentiment analysis completed!")
        return sentiment_df
    
    def get_sentiment_summary(self, sentiment_data):
        """
        Get summary statistics for sentiment analysis.
        
        Args:
            sentiment_data (pd.DataFrame): Data with sentiment scores
            
        Returns:
            dict: Sentiment summary statistics
        """
        summary = {
            'total_articles': len(sentiment_data),
            'sentiment_distribution': sentiment_data['final_sentiment'].value_counts().to_dict(),
            'average_scores': {
                'textblob_polarity': sentiment_data['textblob_polarity'].mean(),
                'vader_compound': sentiment_data['vader_compound'].mean(),
                'financial_score': sentiment_data['financial_score'].mean(),
                'combined_score': sentiment_data['combined_sentiment_score'].mean()
            },
            'sentiment_by_stock': sentiment_data.groupby('stock')['final_sentiment'].value_counts().unstack(fill_value=0).to_dict(),
            'most_positive_articles': sentiment_data.nlargest(5, 'combined_sentiment_score')[['headline', 'combined_sentiment_score']].to_dict('records'),
            'most_negative_articles': sentiment_data.nsmallest(5, 'combined_sentiment_score')[['headline', 'combined_sentiment_score']].to_dict('records')
        }
        
        return summary

# Example usage
if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = NewsSentimentAnalyzer()
    
    # Test with sample headlines
    test_headlines = [
        "Stocks surge to record high as earnings beat expectations",
        "Company faces losses amid market downturn and weak outlook",
        "Quarterly results show steady growth and positive guidance"
    ]
    
    print("Testing Sentiment Analyzer...")
    for headline in test_headlines:
        result = analyzer.analyze_headline(headline)
        print(f"\nHeadline: {headline}")
        print(f"Sentiment: {result['final_sentiment']} (Score: {result['combined_sentiment_score']:.3f})")