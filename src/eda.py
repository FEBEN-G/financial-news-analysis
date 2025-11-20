"""
Exploratory Data Analysis module for financial news data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinancialNewsEDA:
    """
    A class to perform comprehensive exploratory data analysis on financial news data.
    
    This class provides methods for:
    - Descriptive statistics
    - Temporal analysis
    - Text analysis
    - Publisher analysis
    - Visualization
    """
    
    def __init__(self, data):
        """
        Initialize the EDA class with data.
        
        Args:
            data (pandas.DataFrame): The financial news data
        """
        self.df = data.copy()
        self.stats = {}
        
        # Create derived features for analysis
        self._create_features()
    
    def _create_features(self):
        """Create derived features for analysis."""
        print("Creating derived features...")
        
        # Text-based features
        self.df['headline_length'] = self.df['headline'].str.len()
        self.df['word_count'] = self.df['headline'].str.split().str.len()
        
        # Temporal features
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        self.df['hour'] = self.df['date'].dt.hour
        self.df['date_only'] = self.df['date'].dt.date
        
        print("  Derived features created successfully!")
    
    def comprehensive_analysis(self, save_visualizations=True):
        """
        Perform comprehensive EDA and generate all analyses.
        
        Returns:
            dict: Comprehensive statistics and insights
        """
        print("Starting comprehensive EDA...")
        
        # Perform all analyses
        self.stats['basic'] = self.basic_statistics()
        self.stats['temporal'] = self.temporal_analysis()
        self.stats['publisher'] = self.publisher_analysis()
        self.stats['stock'] = self.stock_analysis()
        self.stats['text'] = self.text_analysis()
        
        # Generate all visualizations (non-blocking)
        if save_visualizations:
            # Create visualizations directory first
            import os
            viz_dir = 'data/processed/visualizations'
            os.makedirs(viz_dir, exist_ok=True)
            
            self.generate_all_visualizations(
                save_path=os.path.join(viz_dir, 'eda_comprehensive.png'),
                show_plots=False  # Don't show plots to avoid blocking
            )
        
        print("Comprehensive EDA completed!")
        return self.stats
    
    def basic_statistics(self):
        """
        Calculate basic descriptive statistics.
        
        Returns:
            dict: Basic statistics
        """
        print("Calculating basic statistics...")
        
        stats = {
            'total_articles': len(self.df),
            'date_range': {
                'start': self.df['date'].min(),
                'end': self.df['date'].max(),
                'days': (self.df['date'].max() - self.df['date'].min()).days
            },
            'unique_counts': {
                'stocks': self.df['stock'].nunique(),
                'publishers': self.df['publisher'].nunique(),
                'days': self.df['date_only'].nunique()
            },
            'text_statistics': {
                'avg_headline_length': self.df['headline_length'].mean(),
                'median_headline_length': self.df['headline_length'].median(),
                'avg_word_count': self.df['word_count'].mean(),
                'median_word_count': self.df['word_count'].median()
            }
        }
        
        # Display basic stats
        print(f"  Total articles: {stats['total_articles']:,}")
        print(f"  Date range: {stats['date_range']['start'].date()} to {stats['date_range']['end'].date()}")
        print(f"  Unique stocks: {stats['unique_counts']['stocks']}")
        print(f"  Unique publishers: {stats['unique_counts']['publishers']}")
        print(f"  Average headline length: {stats['text_statistics']['avg_headline_length']:.1f} characters")
        print(f"  Average word count: {stats['text_statistics']['avg_word_count']:.1f} words")
        
        return stats
    
    def temporal_analysis(self):
        """
        Analyze temporal patterns in the data.
        
        Returns:
            dict: Temporal analysis results
        """
        print("Performing temporal analysis...")
        
        # Articles by year
        articles_by_year = self.df['year'].value_counts().sort_index()
        
        # Articles by month (across all years)
        articles_by_month = self.df['month'].value_counts().sort_index()
        
        # Articles by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        articles_by_dow = self.df['day_of_week'].value_counts()
        articles_by_dow = articles_by_dow.reindex(day_order)
        
        # Articles by hour
        articles_by_hour = self.df['hour'].value_counts().sort_index()
        
        # Daily article count
        daily_counts = self.df.groupby('date_only').size()
        
        temporal_stats = {
            'articles_by_year': articles_by_year,
            'articles_by_month': articles_by_month,
            'articles_by_dow': articles_by_dow,
            'articles_by_hour': articles_by_hour,
            'daily_counts': daily_counts,
            'busiest_day': articles_by_dow.index[0] if len(articles_by_dow) > 0 else 'N/A',
            'busiest_hour': articles_by_hour.index[0] if len(articles_by_hour) > 0 else 'N/A'
        }
        
        print(f"  Busiest day: {temporal_stats['busiest_day']}")
        print(f"  Busiest hour: {temporal_stats['busiest_hour']}:00")
        
        return temporal_stats
    
    def publisher_analysis(self):
        """
        Analyze publisher patterns and distributions.
        
        Returns:
            dict: Publisher analysis results
        """
        print("Analyzing publisher data...")
        
        # Top publishers
        top_publishers = self.df['publisher'].value_counts().head(20)
        
        # Publisher activity over time
        publisher_dates = self.df.groupby('publisher')['date'].agg(['min', 'max', 'count'])
        publisher_dates['span_days'] = (publisher_dates['max'] - publisher_dates['min']).dt.days
        
        publisher_stats = {
            'top_publishers': top_publishers,
            'publisher_timeline': publisher_dates,
            'total_publishers': self.df['publisher'].nunique(),
            'top_publisher': top_publishers.index[0] if len(top_publishers) > 0 else 'N/A',
            'articles_top_publisher': top_publishers.iloc[0] if len(top_publishers) > 0 else 0
        }
        
        print(f"  Total publishers: {publisher_stats['total_publishers']}")
        print(f"  Top publisher: {publisher_stats['top_publisher']} ({publisher_stats['articles_top_publisher']} articles)")
        
        return publisher_stats
    
    def stock_analysis(self):
        """
        Analyze stock coverage and patterns.
        
        Returns:
            dict: Stock analysis results
        """
        print("Analyzing stock coverage...")
        
        # Top stocks by coverage
        top_stocks = self.df['stock'].value_counts().head(20)
        
        # Stocks with most consistent coverage
        stock_coverage = self.df.groupby('stock').agg({
            'date': ['min', 'max', 'count'],
            'publisher': 'nunique'
        }).round(2)
        
        stock_coverage.columns = ['first_article', 'last_article', 'article_count', 'unique_publishers']
        stock_coverage['coverage_span'] = (stock_coverage['last_article'] - stock_coverage['first_article']).dt.days
        
        stock_stats = {
            'top_stocks': top_stocks,
            'stock_coverage': stock_coverage,
            'total_stocks': self.df['stock'].nunique(),
            'most_covered_stock': top_stocks.index[0] if len(top_stocks) > 0 else 'N/A',
            'articles_most_covered': top_stocks.iloc[0] if len(top_stocks) > 0 else 0
        }
        
        print(f"  Total stocks covered: {stock_stats['total_stocks']}")
        print(f"  Most covered stock: {stock_stats['most_covered_stock']} ({stock_stats['articles_most_covered']} articles)")
        
        return stock_stats
    
    def text_analysis(self):
        """
        Perform text analysis on headlines.
        
        Returns:
            dict: Text analysis results
        """
        print("Performing text analysis...")
        
        # Common words analysis (basic)
        all_headlines = ' '.join(self.df['headline'].astype(str))
        words = all_headlines.lower().split()
        
        from collections import Counter
        word_freq = Counter(words)
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                     'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'this', 'that'}
        
        filtered_words = {word: count for word, count in word_freq.items() 
                         if word not in stop_words and len(word) > 2}
        
        top_words = dict(sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:20])
        
        text_stats = {
            'total_words': len(words),
            'unique_words': len(set(words)),
            'top_words': top_words,
            'headline_length_stats': {
                'min': self.df['headline_length'].min(),
                'max': self.df['headline_length'].max(),
                'mean': self.df['headline_length'].mean(),
                'median': self.df['headline_length'].median()
            }
        }
        
        print(f"  Total words in all headlines: {text_stats['total_words']:,}")
        print(f"  Unique words: {text_stats['unique_words']:,}")
        print(f"  Top 5 words: {list(text_stats['top_words'].keys())[:5]}")
        
        return text_stats
    
    def generate_all_visualizations(self, save_path=None, show_plots=True):
        """
        Generate all EDA visualizations.
        
        Args:
            save_path (str): Path to save visualizations. If None, displays them.
            show_plots (bool): Whether to display plots (can block execution)
        """
        print("Generating visualizations...")
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Articles over time
        plt.subplot(3, 3, 1)
        self._plot_articles_over_time()
        
        # 2. Articles by day of week
        plt.subplot(3, 3, 2)
        self._plot_articles_by_dow()
        
        # 3. Articles by hour
        plt.subplot(3, 3, 3)
        self._plot_articles_by_hour()
        
        # 4. Top publishers
        plt.subplot(3, 3, 4)
        self._plot_top_publishers()
        
        # 5. Top stocks
        plt.subplot(3, 3, 5)
        self._plot_top_stocks()
        
        # 6. Headline length distribution
        plt.subplot(3, 3, 6)
        self._plot_headline_length_dist()
        
        # 7. Word count distribution
        plt.subplot(3, 3, 7)
        self._plot_word_count_dist()
        
        # 8. Articles by month
        plt.subplot(3, 3, 8)
        self._plot_articles_by_month()
        
        # 9. Word cloud
        plt.subplot(3, 3, 9)
        self._generate_wordcloud()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualizations saved to {save_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()  # Close the figure to avoid blocking
        
        # Also create individual plots for better analysis
        self.create_individual_plots()
    
    def create_individual_plots(self):
        """Create individual plots for better analysis."""
        import os
        
        save_dir = 'data/processed/visualizations'
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # 1. Articles over time (individual)
            plt.figure(figsize=(12, 6))
            self._plot_articles_over_time()
            plt.title('Articles Published Over Time', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'articles_over_time.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # 2. Top stocks coverage
            plt.figure(figsize=(10, 8))
            self._plot_top_stocks()
            plt.title('Top 10 Stocks by News Coverage', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'top_stocks_coverage.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # 3. Headline length distribution
            plt.figure(figsize=(10, 6))
            self._plot_headline_length_dist()
            plt.title('Headline Length Distribution', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'headline_length_dist.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Individual plots saved to: {save_dir}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save individual plots: {e}")
    
    def _plot_articles_over_time(self):
        """Plot articles published over time."""
        daily_counts = self.stats['temporal']['daily_counts']
        plt.plot(daily_counts.index, daily_counts.values, linewidth=1)
        plt.title('Articles Published Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    def _plot_articles_by_dow(self):
        """Plot articles by day of week."""
        dow_data = self.stats['temporal']['articles_by_dow']
        plt.bar(dow_data.index, dow_data.values, color='skyblue')
        plt.title('Articles by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
    
    def _plot_articles_by_hour(self):
        """Plot articles by hour of day."""
        hour_data = self.stats['temporal']['articles_by_hour']
        plt.bar(hour_data.index, hour_data.values, color='lightcoral')
        plt.title('Articles by Hour of Day')
        plt.xlabel('Hour')
        plt.ylabel('Number of Articles')
        plt.xticks(range(0, 24, 2))
    
    def _plot_top_publishers(self):
        """Plot top publishers."""
        top_pubs = self.stats['publisher']['top_publishers'].head(10)
        plt.barh(range(len(top_pubs)), top_pubs.values, color='lightgreen')
        plt.title('Top 10 Publishers')
        plt.xlabel('Number of Articles')
        plt.yticks(range(len(top_pubs)), top_pubs.index)
        plt.gca().invert_yaxis()
    
    def _plot_top_stocks(self):
        """Plot top stocks by coverage."""
        top_stocks = self.stats['stock']['top_stocks'].head(10)
        plt.barh(range(len(top_stocks)), top_stocks.values, color='gold')
        plt.title('Top 10 Stocks by Coverage')
        plt.xlabel('Number of Articles')
        plt.yticks(range(len(top_stocks)), top_stocks.index)
        plt.gca().invert_yaxis()
    
    def _plot_headline_length_dist(self):
        """Plot headline length distribution."""
        plt.hist(self.df['headline_length'], bins=50, alpha=0.7, color='purple')
        plt.title('Headline Length Distribution')
        plt.xlabel('Headline Length (characters)')
        plt.ylabel('Frequency')
        plt.axvline(self.df['headline_length'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.df["headline_length"].mean():.1f}')
        plt.legend()
    
    def _plot_word_count_dist(self):
        """Plot word count distribution."""
        plt.hist(self.df['word_count'], bins=30, alpha=0.7, color='orange')
        plt.title('Word Count Distribution')
        plt.xlabel('Words per Headline')
        plt.ylabel('Frequency')
        plt.axvline(self.df['word_count'].mean(), color='red', linestyle='--',
                   label=f'Mean: {self.df["word_count"].mean():.1f}')
        plt.legend()
    
    def _plot_articles_by_month(self):
        """Plot articles by month."""
        month_data = self.stats['temporal']['articles_by_month']
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.bar(month_data.index, month_data.values, color='lightblue')
        plt.title('Articles by Month')
        plt.xlabel('Month')
        plt.ylabel('Number of Articles')
        plt.xticks(range(1, 13), months[:len(month_data)])
    
    def _generate_wordcloud(self):
        """Generate and display word cloud."""
        try:
            from wordcloud import WordCloud
            
            all_text = ' '.join(self.df['headline'].astype(str))
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white',
                                max_words=100,
                                colormap='viridis').generate(all_text)
            
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title('Word Cloud of Headlines')
            plt.axis('off')
        except ImportError:
            plt.text(0.5, 0.5, 'WordCloud not available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Word Cloud (Library not installed)')
            plt.axis('off')

# Example usage
if __name__ == "__main__":
    # This would be used with actual data
    print("EDA Module - Test functionality")