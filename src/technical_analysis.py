"""
Technical analysis module for stock data using TA-Lib or pandas_ta as fallback.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import config - make sure this exists
try:
    from src.config import TA_LIB_AVAILABLE, PANDAS_TA_AVAILABLE, TECHNICAL_INDICATORS
except ImportError:
    # Default values if config doesn't exist
    TA_LIB_AVAILABLE = False
    PANDAS_TA_AVAILABLE = True
    TECHNICAL_INDICATORS = ['RSI', 'MACD', 'BBANDS', 'SMA', 'EMA']

class StockTechnicalAnalyzer:
    """
    A class to perform technical analysis on stock data.
    
    This class can use either TA-Lib or pandas_ta for calculations.
    """
    
    def __init__(self, symbol, period="1y", start_date=None, end_date=None):
        """
        Initialize the technical analyzer.
        
        Args:
            symbol (str): Stock ticker symbol
            period (str): Period for data (1y, 6mo, 3mo, etc.)
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
        """
        self.symbol = symbol
        self.period = period
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.indicators_calculated = False
        
        # Load data immediately
        self.load_data()
    
    def load_data(self):
        """
        Load stock data from Yahoo Finance.
        """
        print(f"Loading data for {self.symbol}...")
        
        try:
            if self.start_date and self.end_date:
                self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
            else:
                self.data = yf.download(self.symbol, period=self.period)
            
            if self.data.empty:
                print(f"Warning: No data found for {self.symbol}")
                # Create dummy data for testing
                self._create_dummy_data()
                return
            
            print(f"✓ Loaded {len(self.data)} trading days of data for {self.symbol}")
            
        except Exception as e:
            print(f"Error loading data for {self.symbol}: {e}")
            # Create dummy data for testing
            self._create_dummy_data()
    
    def _create_dummy_data(self):
        """
        Create dummy data for testing when real data is unavailable.
        """
        print("Creating dummy data for testing...")
        
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)  # For reproducible results
        
        # Create realistic-looking stock data
        price = 100
        data = []
        
        for date in dates:
            # Random walk for price
            change = np.random.normal(0, 2)
            price = max(1, price + change)
            
            # Create OHLC data
            open_price = price * (1 + np.random.normal(0, 0.01))
            high = max(open_price, price) * (1 + abs(np.random.normal(0, 0.02)))
            low = min(open_price, price) * (1 - abs(np.random.normal(0, 0.02)))
            close = price
            volume = np.random.randint(1000000, 10000000)
            
            data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        self.data = pd.DataFrame(data, index=dates)
        print("✓ Dummy data created for testing")
    
    def calculate_indicators(self):
        """
        Calculate all technical indicators.
        """
        if self.data is None or self.data.empty:
            print("No data available to calculate indicators")
            return
        
        print(f"Calculating technical indicators for {self.symbol}...")
        
        # Calculate basic price-based indicators
        self._calculate_basic_indicators()
        
        # Calculate technical indicators using available library
        if TA_LIB_AVAILABLE:
            self._calculate_indicators_talib()
        elif PANDAS_TA_AVAILABLE:
            self._calculate_indicators_pandas_ta()
        else:
            print("⚠ No technical analysis library available. Using basic indicators only.")
        
        self.indicators_calculated = True
        print("✓ Technical indicators calculated")
    
    def _calculate_basic_indicators(self):
        """
        Calculate basic price-based indicators that don't require special libraries.
        SIMPLIFIED VERSION: Avoids all DataFrame assignment issues
        """
        # Create a new DataFrame to avoid any reference issues
        new_data = self.data.copy()
        
        # Basic calculations that are guaranteed to work
        new_data['daily_return'] = new_data['Close'].pct_change()
        
        # Calculate SMAs using simple rolling mean
        new_data['SMA_20'] = new_data['Close'].rolling(window=20, min_periods=1).mean()
        new_data['SMA_50'] = new_data['Close'].rolling(window=50, min_periods=1).mean()
        
        # Calculate EMAs
        new_data['EMA_12'] = new_data['Close'].ewm(span=12, min_periods=1).mean()
        new_data['EMA_26'] = new_data['Close'].ewm(span=26, min_periods=1).mean()
        
        # Volatility
        new_data['volatility_20d'] = new_data['daily_return'].rolling(window=20, min_periods=1).std()
        
        # Price vs SMA ratios - calculate safely
        sma_20_valid = new_data['SMA_20'].notna() & (new_data['SMA_20'] != 0)
        sma_50_valid = new_data['SMA_50'].notna() & (new_data['SMA_50'] != 0)
        
        new_data.loc[sma_20_valid, 'price_vs_sma20'] = (
            (new_data.loc[sma_20_valid, 'Close'] - new_data.loc[sma_20_valid, 'SMA_20']) / 
            new_data.loc[sma_20_valid, 'SMA_20']
        )
        
        new_data.loc[sma_50_valid, 'price_vs_sma50'] = (
            (new_data.loc[sma_50_valid, 'Close'] - new_data.loc[sma_50_valid, 'SMA_50']) / 
            new_data.loc[sma_50_valid, 'SMA_50']
        )
        
        # Volume indicators
        new_data['volume_sma_20'] = new_data['Volume'].rolling(window=20, min_periods=1).mean()
        
        # Volume ratio - calculate safely
        volume_sma_valid = new_data['volume_sma_20'].notna() & (new_data['volume_sma_20'] != 0)
        new_data.loc[volume_sma_valid, 'volume_ratio'] = (
            new_data.loc[volume_sma_valid, 'Volume'] / new_data.loc[volume_sma_valid, 'volume_sma_20']
        )
        
        # Price ranges
        new_data['daily_range'] = (new_data['High'] - new_data['Low']) / new_data['Close']
        
        # True range
        high_low = new_data['High'] - new_data['Low']
        high_close_prev = abs(new_data['High'] - new_data['Close'].shift(1))
        low_close_prev = abs(new_data['Low'] - new_data['Close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1)
        new_data['true_range'] = true_range.max(axis=1)
        
        # Update the original data
        self.data = new_data
    
    def _calculate_indicators_talib(self):
        """
        Calculate indicators using TA-Lib.
        """
        try:
            import talib
            
            # RSI
            self.data['RSI'] = talib.RSI(self.data['Close'], timeperiod=14)
            
            # MACD
            self.data['MACD'], self.data['MACD_signal'], self.data['MACD_hist'] = talib.MACD(
                self.data['Close'], 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            
            # Bollinger Bands
            self.data['BB_upper'], self.data['BB_middle'], self.data['BB_lower'] = talib.BBANDS(
                self.data['Close'], 
                timeperiod=20, 
                nbdevup=2, 
                nbdevdn=2
            )
            
            # Stochastic
            self.data['slowk'], self.data['slowd'] = talib.STOCH(
                self.data['High'], 
                self.data['Low'], 
                self.data['Close'],
                fastk_period=14,
                slowk_period=3,
                slowd_period=3
            )
            
            # Average Directional Index (ADX)
            self.data['ADX'] = talib.ADX(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=14)
            
            # Commodity Channel Index (CCI)
            self.data['CCI'] = talib.CCI(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=14)
            
        except ImportError:
            print("TA-Lib not available. Install with: pip install TA-Lib")
        except Exception as e:
            print(f"Error calculating TA-Lib indicators: {e}")
    
    def _calculate_indicators_pandas_ta(self):
        """
        Calculate indicators using pandas_ta.
        """
        try:
            import pandas_ta as ta
            
            # RSI
            rsi_result = ta.rsi(self.data['Close'], length=14)
            if rsi_result is not None:
                if isinstance(rsi_result, pd.DataFrame):
                    self.data['RSI'] = rsi_result.iloc[:, 0]
                else:
                    self.data['RSI'] = rsi_result
            
            # MACD
            macd_result = ta.macd(self.data['Close'], fast=12, slow=26, signal=9)
            if macd_result is not None:
                if isinstance(macd_result, pd.DataFrame):
                    self.data['MACD'] = macd_result.iloc[:, 0]  # MACD line
                    self.data['MACD_signal'] = macd_result.iloc[:, 1]  # Signal line
                    self.data['MACD_hist'] = macd_result.iloc[:, 2]  # Histogram
                else:
                    self.data['MACD'] = macd_result
            
            # Bollinger Bands
            bb_result = ta.bbands(self.data['Close'], length=20, std=2)
            if bb_result is not None:
                if isinstance(bb_result, pd.DataFrame):
                    self.data['BB_upper'] = bb_result.iloc[:, 0]
                    self.data['BB_middle'] = bb_result.iloc[:, 1]
                    self.data['BB_lower'] = bb_result.iloc[:, 2]
            
            # Stochastic
            stoch_result = ta.stoch(self.data['High'], self.data['Low'], self.data['Close'], k=14, d=3)
            if stoch_result is not None:
                if isinstance(stoch_result, pd.DataFrame):
                    self.data['slowk'] = stoch_result.iloc[:, 0]
                    self.data['slowd'] = stoch_result.iloc[:, 1]
            
            # Additional indicators
            adx_result = ta.adx(self.data['High'], self.data['Low'], self.data['Close'], length=14)
            if adx_result is not None and isinstance(adx_result, pd.DataFrame):
                self.data['ADX'] = adx_result.iloc[:, 0]
            
            cci_result = ta.cci(self.data['High'], self.data['Low'], self.data['Close'], length=14)
            if cci_result is not None:
                if isinstance(cci_result, pd.DataFrame):
                    self.data['CCI'] = cci_result.iloc[:, 0]
                else:
                    self.data['CCI'] = cci_result
                
        except ImportError:
            print("pandas_ta not available. Install with: pip install pandas_ta")
        except Exception as e:
            print(f"Error calculating pandas_ta indicators: {e}")
    
    def generate_signals(self):
        """
        Generate trading signals based on technical indicators.
        
        Returns:
            dict: Dictionary containing signal counts and descriptions
        """
        if not self.indicators_calculated:
            self.calculate_indicators()
        
        signals = {}
        
        # Moving Average Signals
        if all(col in self.data.columns for col in ['SMA_20', 'SMA_50']):
            signals['golden_cross'] = (self.data['SMA_20'] > self.data['SMA_50']) & \
                                     (self.data['SMA_20'].shift(1) <= self.data['SMA_50'].shift(1))
            signals['death_cross'] = (self.data['SMA_20'] < self.data['SMA_50']) & \
                                    (self.data['SMA_20'].shift(1) >= self.data['SMA_50'].shift(1))
        
        # RSI Signals
        if 'RSI' in self.data.columns:
            signals['rsi_oversold'] = self.data['RSI'] < 30
            signals['rsi_overbought'] = self.data['RSI'] > 70
        
        # MACD Signals
        if all(col in self.data.columns for col in ['MACD', 'MACD_signal']):
            signals['macd_bullish'] = (self.data['MACD'] > self.data['MACD_signal'])
            signals['macd_bearish'] = (self.data['MACD'] < self.data['MACD_signal'])
            signals['macd_crossover'] = (self.data['MACD'] > self.data['MACD_signal']) & \
                                       (self.data['MACD'].shift(1) <= self.data['MACD_signal'].shift(1))
        
        # Bollinger Bands Signals
        if all(col in self.data.columns for col in ['BB_upper', 'BB_lower']):
            signals['bb_oversold'] = self.data['Close'] < self.data['BB_lower']
            signals['bb_overbought'] = self.data['Close'] > self.data['BB_upper']
        
        # Stochastic Signals
        if all(col in self.data.columns for col in ['slowk', 'slowd']):
            signals['stoch_oversold'] = (self.data['slowk'] < 20) & (self.data['slowd'] < 20)
            signals['stoch_overbought'] = (self.data['slowk'] > 80) & (self.data['slowd'] > 80)
        
        # Convert boolean series to signal counts
        signal_counts = {key: value.sum() for key, value in signals.items()}
        
        # Add signal descriptions
        signal_descriptions = {
            'golden_cross': '20-day SMA crosses above 50-day SMA (Bullish)',
            'death_cross': '20-day SMA crosses below 50-day SMA (Bearish)',
            'rsi_oversold': 'RSI below 30 (Oversold, potential buy)',
            'rsi_overbought': 'RSI above 70 (Overbought, potential sell)',
            'macd_bullish': 'MACD above signal line (Bullish momentum)',
            'macd_bearish': 'MACD below signal line (Bearish momentum)',
            'macd_crossover': 'MACD crosses above signal line (Buy signal)',
            'bb_oversold': 'Price below lower Bollinger Band (Oversold)',
            'bb_overbought': 'Price above upper Bollinger Band (Overbought)',
            'stoch_oversold': 'Stochastic below 20 (Oversold)',
            'stoch_overbought': 'Stochastic above 80 (Overbought)'
        }
        
        return {
            'counts': signal_counts,
            'descriptions': signal_descriptions,
            'signals_df': pd.DataFrame(signals, index=self.data.index)
        }
    
    def get_summary_stats(self):
        """
        Get summary statistics for the stock and its indicators.
        
        Returns:
            dict: Summary statistics
        """
        if self.data is None or self.data.empty:
            return {}
        
        # Ensure we extract scalar values from Series
        def get_scalar_value(series):
            """Extract scalar value from series, handling different cases"""
            if hasattr(series, 'iloc'):
                return series.iloc[-1] if len(series) > 0 else None
            elif hasattr(series, 'item'):
                return series.item()
            else:
                return series
        
        try:
            start_price = get_scalar_value(self.data['Close'].iloc[0])
            end_price = get_scalar_value(self.data['Close'].iloc[-1])
            
            # Calculate total return safely
            if start_price and start_price != 0:
                total_return = (end_price - start_price) / start_price * 100
            else:
                total_return = 0
            
            max_price = get_scalar_value(self.data['Close'].max())
            min_price = get_scalar_value(self.data['Close'].min())
            
            # Handle daily return calculation
            daily_return_mean = get_scalar_value(self.data['daily_return'].mean()) if 'daily_return' in self.data.columns else 0
            daily_return_std = get_scalar_value(self.data['daily_return'].std()) if 'daily_return' in self.data.columns else 0
            
            stats = {
                'symbol': self.symbol,
                'period': f"{self.data.index.min().date()} to {self.data.index.max().date()}",
                'total_days': len(self.data),
                'price_stats': {
                    'start_price': start_price,
                    'end_price': end_price,
                    'total_return': total_return,
                    'max_price': max_price,
                    'min_price': min_price,
                    'avg_daily_return': daily_return_mean * 100,
                    'volatility': daily_return_std * 100
                }
            }
            
            # Add indicator stats if calculated
            if self.indicators_calculated:
                indicator_stats = {}
                
                if 'RSI' in self.data.columns:
                    rsi_current = get_scalar_value(self.data['RSI'].iloc[-1]) if len(self.data) > 0 else None
                    indicator_stats['rsi'] = {
                        'mean': get_scalar_value(self.data['RSI'].mean()),
                        'min': get_scalar_value(self.data['RSI'].min()),
                        'max': get_scalar_value(self.data['RSI'].max()),
                        'current': rsi_current
                    }
                
                if 'MACD' in self.data.columns:
                    macd_current = get_scalar_value(self.data['MACD'].iloc[-1]) if len(self.data) > 0 else None
                    signal_current = get_scalar_value(self.data['MACD_signal'].iloc[-1]) if 'MACD_signal' in self.data.columns and len(self.data) > 0 else None
                    
                    indicator_stats['macd'] = {
                        'current_macd': macd_current,
                        'current_signal': signal_current
                    }
                
                stats['indicator_stats'] = indicator_stats
            
            return stats
            
        except Exception as e:
            print(f"Error calculating stats for {self.symbol}: {e}")
            # Return basic stats without calculations
            return {
                'symbol': self.symbol,
                'period': "Unknown",
                'total_days': len(self.data),
                'price_stats': {
                    'start_price': 0,
                    'end_price': 0,
                    'total_return': 0,
                    'max_price': 0,
                    'min_price': 0,
                    'avg_daily_return': 0,
                    'volatility': 0
                }
            }

    def plot_technical_analysis(self, save_path=None):
        """
        Create comprehensive technical analysis visualization.
        
        Args:
            save_path (str): Path to save the visualization
        """
        if not self.indicators_calculated:
            self.calculate_indicators()
        
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
        
        # 1. Price and Moving Averages
        ax1 = plt.subplot(gs[0])
        ax1.plot(self.data.index, self.data['Close'], label='Close Price', linewidth=2, color='black')
        
        if 'SMA_20' in self.data.columns:
            ax1.plot(self.data.index, self.data['SMA_20'], label='20-day SMA', alpha=0.7)
        if 'SMA_50' in self.data.columns:
            ax1.plot(self.data.index, self.data['SMA_50'], label='50-day SMA', alpha=0.7)
        if all(col in self.data.columns for col in ['BB_upper', 'BB_lower']):
            ax1.fill_between(self.data.index, self.data['BB_upper'], self.data['BB_lower'], 
                           alpha=0.2, label='Bollinger Bands')
        
        ax1.set_title(f'{self.symbol} - Technical Analysis', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. RSI
        ax2 = plt.subplot(gs[1])
        if 'RSI' in self.data.columns:
            ax2.plot(self.data.index, self.data['RSI'], label='RSI', color='purple')
            ax2.axhline(70, linestyle='--', alpha=0.7, color='red')
            ax2.axhline(30, linestyle='--', alpha=0.7, color='green')
            ax2.axhline(50, linestyle='--', alpha=0.3, color='gray')
            ax2.set_ylabel('RSI')
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. MACD
        ax3 = plt.subplot(gs[2])
        if all(col in self.data.columns for col in ['MACD', 'MACD_signal']):
            ax3.plot(self.data.index, self.data['MACD'], label='MACD', color='blue')
            ax3.plot(self.data.index, self.data['MACD_signal'], label='Signal', color='red')
            # Check if MACD_hist exists and is not all zeros/NaNs
            if 'MACD_hist' in self.data.columns and not self.data['MACD_hist'].isna().all():
                ax3.bar(self.data.index, self.data['MACD_hist'], label='Histogram', alpha=0.3, color='gray')
            ax3.axhline(0, linestyle='-', alpha=0.5, color='black')
            ax3.set_ylabel('MACD')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Volume
        ax4 = plt.subplot(gs[3])
        ax4.bar(self.data.index, self.data['Volume'], alpha=0.7, color='orange')
        ax4.set_ylabel('Volume')
        ax4.set_xlabel('Date')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Technical analysis chart saved to: {save_path}")
        
        plt.show()
    
    def get_technical_summary(self):
        """
        Get a comprehensive technical summary of the stock.
        
        Returns:
            dict: Technical summary
        """
        if not self.indicators_calculated:
            self.calculate_indicators()
        
        summary = {
            'symbol': self.symbol,
            'current_price': self.data['Close'].iloc[-1],
            'trend': self._get_trend_direction(),
            'momentum': self._get_momentum_status(),
            'volatility': self._get_volatility_status(),
            'signals': self.generate_signals()['counts']
        }
        
        return summary
    
    def _get_trend_direction(self):
        """Determine the current trend direction."""
        if 'SMA_20' not in self.data.columns or 'SMA_50' not in self.data.columns:
            return "Unknown"
        
        current_price = self.data['Close'].iloc[-1]
        sma_20 = self.data['SMA_20'].iloc[-1]
        sma_50 = self.data['SMA_50'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            return "Strong Uptrend"
        elif current_price > sma_20 and sma_20 > sma_50:
            return "Uptrend"
        elif current_price < sma_20 < sma_50:
            return "Strong Downtrend"
        elif current_price < sma_20 and sma_20 < sma_50:
            return "Downtrend"
        else:
            return "Sideways/Consolidating"
    
    def _get_momentum_status(self):
        """Determine the current momentum status."""
        momentum_indicators = []
        
        if 'RSI' in self.data.columns:
            rsi = self.data['RSI'].iloc[-1]
            if not pd.isna(rsi):
                if rsi < 30:
                    momentum_indicators.append("Oversold")
                elif rsi > 70:
                    momentum_indicators.append("Overbought")
        
        if 'MACD' in self.data.columns and 'MACD_signal' in self.data.columns:
            macd = self.data['MACD'].iloc[-1]
            signal = self.data['MACD_signal'].iloc[-1]
            if not pd.isna(macd) and not pd.isna(signal):
                if macd > signal:
                    momentum_indicators.append("Bullish MACD")
                else:
                    momentum_indicators.append("Bearish MACD")
        
        return ", ".join(momentum_indicators) if momentum_indicators else "Neutral"
    
    def _get_volatility_status(self):
        """Determine the current volatility status."""
        if 'volatility_20d' not in self.data.columns:
            return "Unknown"
        
        vol = self.data['volatility_20d'].iloc[-1]
        if pd.isna(vol):
            return "Unknown"
        
        # Simple volatility classification
        if vol > 0.03:  # 3% daily volatility
            return "High"
        elif vol > 0.015:  # 1.5% daily volatility
            return "Medium"
        else:
            return "Low"

# Example usage and testing
if __name__ == "__main__":
    # Test the technical analyzer
    print("Testing Technical Analysis...")
    
    # Test with a popular stock
    analyzer = StockTechnicalAnalyzer('AAPL', period='6mo')
    analyzer.calculate_indicators()
    
    # Get signals
    signals = analyzer.generate_signals()
    print("\nTrading Signals:")
    for signal, count in signals['counts'].items():
        desc = signals['descriptions'][signal]
        print(f"  {signal}: {count} occurrences - {desc}")
    
    # Get summary statistics
    stats = analyzer.get_summary_stats()
    print(f"\nSummary for {stats['symbol']}:")
    print(f"  Period: {stats['period']}")
    print(f"  Total Return: {stats['price_stats']['total_return']:.2f}%")
    print(f"  Volatility: {stats['price_stats']['volatility']:.2f}%")
    
    # Get technical summary
    tech_summary = analyzer.get_technical_summary()
    print(f"\nTechnical Summary:")
    print(f"  Trend: {tech_summary['trend']}")
    print(f"  Momentum: {tech_summary['momentum']}")
    print(f"  Volatility: {tech_summary['volatility']}")
    
    # Plot the analysis
    analyzer.plot_technical_analysis()