"""
Optimized Data Loader for Financial News Analysis
Specifically designed for raw_analyst_ratings.csv structure
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
import re
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

class FinancialNewsLoader:
    """
    Optimized data loader for the raw_analyst_ratings.csv file structure.
    
    Your data structure:
    - Unnamed: 0: Index column
    - headline: News headlines (perfect!)
    - url: Article URLs
    - publisher: News publishers (perfect!)
    - date: Publication dates with timezone (perfect!)
    - stock: Stock tickers (perfect!)
    """
    
    def __init__(self, data_file='raw_analyst_ratings.csv'):
        """
        Initialize the data loader with your specific file.
        
        Args:
            data_file (str): Name of your data file in data/raw directory
        """
        self.data_file = os.path.join(RAW_DATA_PATH, data_file)
        self.df = None
        self.loaded = False
        self.original_shape = None
        
    def load_data(self, sample_size=None):
        """
        Load and optimize the data for analysis.
        
        Args:
            sample_size (int): If provided, load only a sample of data for testing
            
        Returns:
            pandas.DataFrame: Loaded and processed data
        """
        print("=" * 60)
        print("LOADING FINANCIAL NEWS DATA")
        print("=" * 60)
        
        try:
            # Load the data
            if sample_size:
                print(f"Loading sample of {sample_size:,} rows for testing...")
                self.df = pd.read_csv(self.data_file, nrows=sample_size)
            else:
                print("Loading full dataset...")
                self.df = pd.read_csv(self.data_file)
            
            self.original_shape = self.df.shape
            print(f"✅ Raw data loaded: {self.original_shape[0]:,} rows × {self.original_shape[1]} columns")
            
            # Display initial column info
            print(f"\n📊 ORIGINAL COLUMNS:")
            for i, col in enumerate(self.df.columns):
                print(f"  {i+1}. {col}")
            
            # Process the data
            self._process_data()
            
            self.loaded = True
            print(f"\n🎉 DATA PROCESSING COMPLETE!")
            print(f"   Final dataset: {len(self.df):,} articles")
            print(f"   Date range: {self.df['date'].min().date()} to {self.df['date'].max().date()}")
            print(f"   Unique stocks: {self.df['stock'].nunique()}")
            print(f"   Unique publishers: {self.df['publisher'].nunique()}")
            
            return self.df
            
        except FileNotFoundError:
            print(f"❌ Error: File {self.data_file} not found.")
            print("Please make sure your data file is in the data/raw/ directory")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def _process_data(self):
        """
        Process and clean the data step by step.
        """
        print("\n🔄 PROCESSING DATA...")
        
        # Step 1: Remove unnecessary columns
        self._remove_unnecessary_columns()
        
        # Step 2: Parse and clean dates
        self._process_dates()
        
        # Step 3: Clean text data
        self._clean_text_data()
        
        # Step 4: Clean stock symbols
        self._clean_stock_data()
        
        # Step 5: Remove duplicates
        self._remove_duplicates()
        
        # Step 6: Create derived features
        self._create_features()
    
    def _remove_unnecessary_columns(self):
        """Remove columns that are not needed for analysis."""
        print("  1. Removing unnecessary columns...")
        
        # Remove the index column if it exists
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop('Unnamed: 0', axis=1)
            print("     ✅ Removed 'Unnamed: 0' column")
        
        print(f"     Columns after cleaning: {list(self.df.columns)}")
    
    def _process_dates(self):
        """Parse and clean date column with robust error handling."""
        print("  2. Processing dates...")
        
        # Use mixed format parsing which handles various date formats
        try:
            self.df['date'] = pd.to_datetime(self.df['date'], format='mixed', utc=True)
            print("     ✅ Used 'mixed' format parsing")
        except (ValueError, TypeError):
            # Fallback: use coerce to handle problematic dates
            print("     ⚠️  Mixed format failed, using error coercion")
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce', utc=True)
            
            # Remove rows with invalid dates
            invalid_dates = self.df['date'].isnull().sum()
            if invalid_dates > 0:
                print(f"     ⚠️  Removing {invalid_dates} rows with invalid dates")
                self.df = self.df.dropna(subset=['date'])
        
        # Extract date components
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        self.df['hour'] = self.df['date'].dt.hour
        self.df['date_only'] = self.df['date'].dt.date
        
        print(f"     Date range: {self.df['date'].min()} to {self.df['date'].max()}")
    
    def _clean_text_data(self):
        """Clean and preprocess text columns."""
        print("  3. Cleaning text data...")
        
        # Clean headlines
        self.df['headline'] = self.df['headline'].str.strip()
        
        # Remove very short headlines (likely errors)
        initial_count = len(self.df)
        self.df = self.df[self.df['headline'].str.len() > 10]
        removed_count = initial_count - len(self.df)
        if removed_count > 0:
            print(f"     Removed {removed_count} very short headlines")
        
        # Clean publisher names
        self.df['publisher'] = self.df['publisher'].str.strip()
        
        print(f"     Sample headline: '{self.df['headline'].iloc[0][:80]}...'")
    
    def _clean_stock_data(self):
        """Clean and standardize stock symbols."""
        print("  4. Cleaning stock data...")
        
        # Convert to uppercase and strip whitespace
        self.df['stock'] = self.df['stock'].str.upper().str.strip()
        
        # Remove any stock symbols that are too long (likely errors)
        initial_count = len(self.df)
        self.df = self.df[self.df['stock'].str.len() <= 5]
        removed_count = initial_count - len(self.df)
        if removed_count > 0:
            print(f"     Removed {removed_count} rows with invalid stock symbols")
        
        print(f"     Unique stocks: {self.df['stock'].nunique()}")
        print(f"     Top 5 stocks: {self.df['stock'].value_counts().head(5).to_dict()}")
    
    def _remove_duplicates(self):
        """Remove duplicate articles."""
        print("  5. Removing duplicates...")
        
        initial_count = len(self.df)
        
        # Remove exact duplicates
        self.df = self.df.drop_duplicates()
        
        # Remove duplicates based on headline and date (same news on same day)
        self.df = self.df.drop_duplicates(subset=['headline', 'date_only', 'stock'])
        
        final_count = len(self.df)
        duplicates_removed = initial_count - final_count
        
        print(f"     Removed {duplicates_removed:,} duplicate entries")
        print(f"     Final count: {final_count:,} articles")
    
    def _create_features(self):
        """Create derived features for analysis."""
        print("  6. Creating derived features...")
        
        # Text-based features
        self.df['headline_length'] = self.df['headline'].str.len()
        self.df['word_count'] = self.df['headline'].str.split().str.len()
        
        # Time-based features
        self.df['is_weekend'] = self.df['day_of_week'].isin(['Saturday', 'Sunday'])
        
        # Publisher categories (major vs minor)
        publisher_counts = self.df['publisher'].value_counts()
        major_publishers = publisher_counts[publisher_counts > 1000].index
        self.df['is_major_publisher'] = self.df['publisher'].isin(major_publishers)
        
        print(f"     Created features: headline_length, word_count, is_weekend, is_major_publisher")
    
    def get_basic_stats(self):
        """
        Get comprehensive statistics about the loaded data.
        
        Returns:
            dict: Dictionary containing various statistics
        """
        if not self.loaded:
            return {"error": "Data not loaded. Please call load_data() first."}
        
        stats = {
            'data_file': os.path.basename(self.data_file),
            'total_articles': len(self.df),
            'original_size': self.original_shape[0] if self.original_shape else len(self.df),
            'data_retention_rate': f"{(len(self.df) / self.original_shape[0] * 100):.1f}%" if self.original_shape else "N/A",
            'date_range': {
                'start': self.df['date'].min(),
                'end': self.df['date'].max(),
                'days': (self.df['date'].max() - self.df['date'].min()).days
            },
            'stocks': {
                'unique_count': self.df['stock'].nunique(),
                'top_10': self.df['stock'].value_counts().head(10).to_dict(),
                'coverage_stats': {
                    'mean_articles_per_stock': self.df['stock'].value_counts().mean(),
                    'median_articles_per_stock': self.df['stock'].value_counts().median(),
                    'max_articles_one_stock': self.df['stock'].value_counts().max()
                }
            },
            'publishers': {
                'unique_count': self.df['publisher'].nunique(),
                'top_5': self.df['publisher'].value_counts().head(5).to_dict(),
                'major_publishers_count': self.df['is_major_publisher'].sum()
            },
            'temporal': {
                'articles_by_year': self.df['year'].value_counts().to_dict(),
                'articles_by_day': self.df['day_of_week'].value_counts().to_dict(),
                'articles_by_hour': self.df['hour'].value_counts().head().to_dict()
            },
            'text_analysis': {
                'avg_headline_length': self.df['headline_length'].mean(),
                'avg_word_count': self.df['word_count'].mean(),
                'headline_length_range': (self.df['headline_length'].min(), self.df['headline_length'].max())
            }
        }
        
        return stats
    
    def get_sample_insights(self):
        """
        Get quick insights about the data.
        """
        if not self.loaded:
            return "Data not loaded."
        
        insights = []
        insights.append("📈 QUICK INSIGHTS:")
        insights.append(f"   • Most active publisher: {self.df['publisher'].value_counts().index[0]}")
        insights.append(f"   • Most covered stock: {self.df['stock'].value_counts().index[0]}")
        insights.append(f"   • Busiest day: {self.df['day_of_week'].value_counts().index[0]}")
        insights.append(f"   • Average headline length: {self.df['headline_length'].mean():.1f} characters")
        insights.append(f"   • Average words per headline: {self.df['word_count'].mean():.1f} words")
        
        return "\n".join(insights)
    
    def save_processed_data(self, filename='processed_financial_news.csv'):
        """
        Save the processed data to CSV file.
        
        Args:
            filename (str): Name for the processed data file
        """
        if not self.loaded:
            print("❌ No data loaded. Please call load_data() first.")
            return
        
        output_path = os.path.join(PROCESSED_DATA_PATH, filename)
        self.df.to_csv(output_path, index=False)
        print(f"✅ Processed data saved to: {output_path}")
        print(f"   File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

    def get_data_for_analysis(self, stocks=None, date_range=None):
        """
        Get filtered data for specific analysis.
        
        Args:
            stocks (list): List of stock symbols to filter by
            date_range (tuple): (start_date, end_date) as datetime objects
            
        Returns:
            pandas.DataFrame: Filtered data
        """
        if not self.loaded:
            print("❌ No data loaded.")
            return None
        
        filtered_df = self.df.copy()
        
        # Filter by stocks
        if stocks:
            filtered_df = filtered_df[filtered_df['stock'].isin(stocks)]
            print(f"Filtered to {len(filtered_df)} articles for stocks: {stocks}")
        
        # Filter by date range
        if date_range:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df['date'] >= start_date) & 
                (filtered_df['date'] <= end_date)
            ]
            print(f"Filtered to {len(filtered_df)} articles from {start_date.date()} to {end_date.date()}")
        
        return filtered_df

# Example usage and testing
if __name__ == "__main__":
    # Test the data loader
    print("Testing FinancialNewsLoader with your data...")
    
    # Load a sample for testing (remove sample_size parameter for full dataset)
    loader = FinancialNewsLoader()
    data = loader.load_data(sample_size=10000)  # Test with 10,000 rows
    
    if data is not None:
        # Get basic statistics
        stats = loader.get_basic_stats()
        
        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        
        print(f"Total Articles: {stats['total_articles']:,}")
        print(f"Date Range: {stats['date_range']['start'].date()} to {stats['date_range']['end'].date()}")
        print(f"Unique Stocks: {stats['stocks']['unique_count']}")
        print(f"Unique Publishers: {stats['publishers']['unique_count']}")
        
        print("\n" + loader.get_sample_insights())
        
        # Save processed data
        loader.save_processed_data()