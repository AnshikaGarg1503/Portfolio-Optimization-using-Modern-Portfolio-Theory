import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
import os

class MLBacktester:
    """Class for machine learning and walk-forward backtesting."""
    
    def __init__(self, data_file='portfolio_data.csv', ticker=None, cache_dir='ml_models_cache'):
        """
        Initialize the ML backtester with portfolio data.
        
        Args:
            data_file (str): Path to CSV file with stock prices
            ticker (str, optional): Specific ticker to analyze. If None, will use all tickers.
            cache_dir (str): Directory to cache trained models
        """
        # Add code to filter out future dates when loading the portfolio data
        
        # Load the portfolio data
        self.portfolio_data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
        
        # Filter out future dates
        current_date = datetime.now().date()
        print(f"Current date: {current_date}")
        print(f"Min date in DataFrame: {self.portfolio_data.index.min().date()}")
        print(f"Max date in DataFrame: {self.portfolio_data.index.max().date()}")
        print(f"Number of rows before filtering: {len(self.portfolio_data)}")
        
        # Filter out future dates
        self.portfolio_data = self.portfolio_data[self.portfolio_data.index.date <= current_date]
        
        print(f"Number of rows after filtering: {len(self.portfolio_data)}")
        print(f"Min date in filtered DataFrame: {self.portfolio_data.index.min().date() if not self.portfolio_data.empty else 'N/A'}")
        print(f"Max date in filtered DataFrame: {self.portfolio_data.index.max().date() if not self.portfolio_data.empty else 'N/A'}")
        
        # Sort by date
        self.portfolio_data = self.portfolio_data.sort_index()
        
        # Filter for specific ticker if provided
        self.ticker = ticker
        if ticker is not None and ticker in self.portfolio_data.columns:
            self.data = self.portfolio_data[[ticker]]
            print(f"Analyzing {ticker}")
        else:
            self.data = self.portfolio_data
            print(f"Analyzing all {len(self.portfolio_data.columns)} tickers")
        
        # Create cache directory if it doesn't exist
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Initialize results storage
        self.results = {
            'predictions': [],
            'actual': [],
            'dates': [],
            'accuracy': [],
            'mse': [],
            'window_start': [],
            'window_end': [],
            'test_start': [],
            'test_end': []
        }
        
        # Initialize model cache
        self.model_cache = {}
    
    def _create_features(self, df):
        """
        Create features for machine learning models.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            
        Returns:
            pd.DataFrame: DataFrame with features
        """
        # Make a copy to avoid modifying the original
        features = df.copy()
        
        print(f"Input data shape: {features.shape}")
        print(f"Input data date range: {features.index.min().strftime('%Y-%m-%d')} to {features.index.max().strftime('%Y-%m-%d')}")
        print(f"Input data columns: {features.columns.tolist()}")
        
        # Check for NaN values in input data
        nan_count = features.isna().sum().sum()
        print(f"NaN values in input data: {nan_count}")
        
        # Calculate returns
        features['returns'] = features.pct_change()
        
        # Calculate lagged returns (1, 2, 3, 5, 10 days)
        for lag in [1, 2, 3, 5, 10]:
            features[f'lag_{lag}_return'] = features['returns'].shift(lag)
        
        # Calculate moving averages (5, 10, 20, 50 days)
        for window in [5, 10, 20, 50]:
            features[f'ma_{window}'] = features.iloc[:, 0].rolling(window=window).mean()
            # Calculate price relative to moving average
            features[f'price_to_ma_{window}'] = features.iloc[:, 0] / features[f'ma_{window}']
        
        # Calculate volatility (5, 10, 20, 50 days)
        for window in [5, 10, 20, 50]:
            features[f'volatility_{window}'] = features['returns'].rolling(window=window).std()
        
        # Calculate RSI (14 days)
        delta = features.iloc[:, 0].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        features['ema_12'] = features.iloc[:, 0].ewm(span=12, adjust=False).mean()
        features['ema_26'] = features.iloc[:, 0].ewm(span=26, adjust=False).mean()
        features['macd'] = features['ema_12'] - features['ema_26']
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Calculate target variables
        # Direction (1 if price goes up, 0 if down)
        features['direction'] = (features['returns'].shift(-1) > 0).astype(int)
        # Next day return
        features['next_return'] = features['returns'].shift(-1)
        
        # Check for NaN values before dropping
        nan_count_before_drop = features.isna().sum().sum()
        print(f"NaN values before dropping: {nan_count_before_drop}")
        
        # Drop NaN values
        features = features.dropna()
        
        # Check final feature set
        print(f"Final features shape: {features.shape}")
        print(f"Final features date range: {features.index.min().strftime('%Y-%m-%d')} to {features.index.max().strftime('%Y-%m-%d')}")
        
        return features
    
    def _normalize_features(self, X_train, X_test):
        """
        Normalize features using StandardScaler.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Testing features
            
        Returns:
            tuple: (normalized X_train, normalized X_test, scaler)
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns), \
               pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns), \
               scaler
    
    def _train_models(self, X_train, y_train_direction, y_train_return, model_key):
        """
        Train classification and regression models.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train_direction (pd.Series): Training target for direction
            y_train_return (pd.Series): Training target for return
            model_key (str): Key for model caching
            
        Returns:
            tuple: (direction_model, return_model)
        """
        # Check if models are already cached
        if model_key in self.model_cache:
            print(f"Using cached models for {model_key}")
            return self.model_cache[model_key]['direction'], self.model_cache[model_key]['return']
        
        # Train direction model (classification)
        print(f"Training direction model for {model_key}...")
        direction_models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Train return model (regression)
        print(f"Training return model for {model_key}...")
        return_models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression()
        }
        
        # Train models
        for name, model in direction_models.items():
            model.fit(X_train, y_train_direction)
        
        for name, model in return_models.items():
            model.fit(X_train, y_train_return)
        
        # Use RandomForest as default
        direction_model = direction_models['RandomForest']
        return_model = return_models['RandomForest']
        
        # Cache models
        self.model_cache[model_key] = {
            'direction': direction_model,
            'return': return_model
        }
        
        # Save models to disk
        joblib.dump(direction_model, os.path.join(self.cache_dir, f"{model_key}_direction.joblib"))
        joblib.dump(return_model, os.path.join(self.cache_dir, f"{model_key}_return.joblib"))
        
        return direction_model, return_model
    
    # Add more debugging information to the run_walk_forward_backtest method
    
    def run_walk_forward_backtest(self, initial_window_years=2, test_window_months=6):
        """
        Run walk-forward backtest using machine learning models.
        
        Args:
            initial_window_years (int): Number of years for initial training window
            test_window_months (int): Number of months for each test window
            
        Returns:
            pd.DataFrame: DataFrame with backtest results
        """
        # Process each ticker
        if self.ticker is not None:
            tickers = [self.ticker]
        else:
            tickers = self.portfolio_data.columns
        
        all_results = []
        
        for ticker in tickers:
            print(f"\nRunning walk-forward backtest for {ticker}...")
            
            # Get data for this ticker
            ticker_data = self.portfolio_data[[ticker]]
            
            # Create features
            print(f"Creating features for {ticker}...")
            features = self._create_features(ticker_data)
            print(f"Features created. Shape: {features.shape}")
            
            # Define feature columns and target columns
            feature_cols = [col for col in features.columns if col not in ['direction', 'next_return', 'returns']]
            print(f"Number of features: {len(feature_cols)}")
            
            # Get the date range
            start_date = features.index.min()
            end_date = features.index.max()
            print(f"Data date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Calculate initial window end date
            initial_window_end = start_date + pd.DateOffset(years=initial_window_years)
            print(f"Initial window end date: {initial_window_end.strftime('%Y-%m-%d')}")
            
            # Check if we have enough data
            if initial_window_end >= end_date:
                print(f"Not enough data for {ticker}. Need at least {initial_window_years} years of data.")
                print(f"Available data: {(end_date - start_date).days / 365.25:.2f} years")
                continue
            
            # Initialize current window
            current_window_start = start_date
            current_window_end = initial_window_end
            
            # Initialize results for this ticker
            ticker_results = {
                'ticker': ticker,
                'predictions_direction': [],
                'actual_direction': [],
                'predictions_return': [],
                'actual_return': [],
                'dates': [],
                'accuracy': [],
                'mse': [],
                'window_start': [],
                'window_end': [],
                'test_start': [],
                'test_end': []
            }
            
            # Flag to track if we've successfully processed any windows
            windows_processed = False
            
            # Run walk-forward backtest
            window_count = 0
            while current_window_end < end_date:
                window_count += 1
                # Define test window
                test_window_start = current_window_end
                test_window_end = test_window_start + pd.DateOffset(months=test_window_months)
                
                # Ensure test window doesn't exceed available data
                if test_window_end > end_date:
                    test_window_end = end_date
                
                print(f"\nWindow {window_count}:")
                print(f"Training window: {current_window_start.strftime('%Y-%m-%d')} to {current_window_end.strftime('%Y-%m-%d')}")
                print(f"Testing window: {test_window_start.strftime('%Y-%m-%d')} to {test_window_end.strftime('%Y-%m-%d')}")
                
                # Split data into training and testing sets
                train_data = features[(features.index >= current_window_start) & (features.index < current_window_end)]
                test_data = features[(features.index >= test_window_start) & (features.index < test_window_end)]
                
                print(f"Training data shape: {train_data.shape}")
                print(f"Testing data shape: {test_data.shape}")
                
                # Skip if not enough data
                if len(train_data) < 252 or len(test_data) < 20:  # At least 1 year of training data and 1 month of testing data
                    print(f"Not enough data for window. Train: {len(train_data)}, Test: {len(test_data)}")
                    print(f"Minimum required - Train: 252, Test: 20")
                    current_window_end = test_window_end
                    current_window_start = current_window_start + pd.DateOffset(months=test_window_months)
                    continue
                
                # Prepare training data
                X_train = train_data[feature_cols]
                y_train_direction = train_data['direction']
                y_train_return = train_data['next_return']
                
                # Prepare testing data
                X_test = test_data[feature_cols]
                y_test_direction = test_data['direction']
                y_test_return = test_data['next_return']
                
                # Normalize features
                X_train_scaled, X_test_scaled, _ = self._normalize_features(X_train, X_test)
                
                # Train models
                model_key = f"{ticker}_{current_window_start.strftime('%Y%m%d')}_{current_window_end.strftime('%Y%m%d')}"
                direction_model, return_model = self._train_models(X_train_scaled, y_train_direction, y_train_return, model_key)
                
                # Make predictions
                y_pred_direction = direction_model.predict(X_test_scaled)
                y_pred_return = return_model.predict(X_test_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test_direction, y_pred_direction)
                mse = mean_squared_error(y_test_return, y_pred_return)
                
                print(f"Accuracy: {accuracy:.4f}, MSE: {mse:.6f}")
                
                # Store results
                ticker_results['predictions_direction'].extend(y_pred_direction)
                ticker_results['actual_direction'].extend(y_test_direction)
                ticker_results['predictions_return'].extend(y_pred_return)
                ticker_results['actual_return'].extend(y_test_return)
                ticker_results['dates'].extend(test_data.index)
                ticker_results['accuracy'].append(accuracy)
                ticker_results['mse'].append(mse)
                ticker_results['window_start'].append(current_window_start)
                ticker_results['window_end'].append(current_window_end)
                ticker_results['test_start'].append(test_window_start)
                ticker_results['test_end'].append(test_window_end)
                
                # Set flag to indicate we've processed at least one window
                windows_processed = True
                
                # Move to next window
                current_window_end = test_window_end
                current_window_start = current_window_start + pd.DateOffset(months=test_window_months)
            
            print(f"\nTotal windows processed for {ticker}: {window_count}")
            print(f"Windows with sufficient data: {len(ticker_results['accuracy'])}")
            
            # Only add results if we've processed at least one window
            if windows_processed:
                # Convert results to DataFrame
                results_df = pd.DataFrame({
                    'date': ticker_results['dates'],
                    'ticker': ticker,
                    'predicted_direction': ticker_results['predictions_direction'],
                    'actual_direction': ticker_results['actual_direction'],
                    'predicted_return': ticker_results['predictions_return'],
                    'actual_return': ticker_results['actual_return']
                })
                
                print(f"Results DataFrame shape: {results_df.shape}")
                
                # Add to all results
                all_results.append(results_df)
                
                # Store ticker results
                self.results[ticker] = ticker_results
            else:
                print(f"No valid windows processed for {ticker}. Skipping.")
        
        # Combine all results
        if all_results:
            self.combined_results = pd.concat(all_results)
            self.combined_results.set_index('date', inplace=True)
            self.combined_results.sort_index(inplace=True)
            
            print(f"\nCombined results shape: {self.combined_results.shape}")
            
            # Save results to CSV
            self.combined_results.to_csv('ml_backtest_results.csv')
            print("Backtest results saved to ml_backtest_results.csv")
            
            return self.combined_results
        else:
            print("No results generated. Please check if you have enough historical data.")
            # Create an empty DataFrame with the correct columns to avoid errors
            empty_df = pd.DataFrame(columns=['date', 'ticker', 'predicted_direction', 'actual_direction', 'predicted_return', 'actual_return'])
            empty_df.set_index('date', inplace=True)
            self.combined_results = empty_df
            return empty_df
    
    def calculate_strategy_performance(self, initial_investment=10000):
        """
        Calculate performance metrics for the ML-based trading strategy.
        
        Args:
            initial_investment (float): Initial investment amount
            
        Returns:
            dict: Dictionary with performance metrics
        """
        if not hasattr(self, 'combined_results') or self.combined_results.empty:
            print("No backtest results available or results are empty. Cannot calculate performance metrics.")
            # Return default metrics to avoid errors
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'strategy_returns': pd.DataFrame(columns=['return', 'cumulative_return', 'portfolio_value'])
            }
        
        # Calculate strategy returns
        strategy_returns = []
        dates = []
        
        # Group by date
        daily_results = self.combined_results.groupby(self.combined_results.index)
        
        for date, group in daily_results:
            # Calculate strategy return for this day
            # For each stock, invest if predicted direction is up (1)
            invested_stocks = group[group['predicted_direction'] == 1]
            
            if len(invested_stocks) > 0:
                # Equal weight allocation to stocks predicted to go up
                weight = 1.0 / len(invested_stocks)
                day_return = (invested_stocks['actual_return'] * weight).sum()
            else:
                # If no stocks predicted to go up, return is 0 (cash position)
                day_return = 0.0
            
            strategy_returns.append(day_return)
            dates.append(date)
        
        # Create DataFrame with strategy returns
        strategy_df = pd.DataFrame({
            'return': strategy_returns
        }, index=dates)
        
        # Calculate cumulative returns
        strategy_df['cumulative_return'] = (1 + strategy_df['return']).cumprod()
        
        # Calculate portfolio value
        strategy_df['portfolio_value'] = initial_investment * strategy_df['cumulative_return']
        
        # Calculate performance metrics
        total_return = (strategy_df['cumulative_return'].iloc[-1] - 1) if len(strategy_df) > 0 else 0
        annualized_return = ((1 + total_return) ** (252 / len(strategy_df)) - 1) if len(strategy_df) > 0 else 0
        
        # Calculate volatility
        volatility = strategy_df['return'].std() * np.sqrt(252)
        
        # Calculate Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility != 0 else 0
        
        # Calculate maximum drawdown
        cumulative_returns = strategy_df['cumulative_return']
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Create a dictionary with the results
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'strategy_returns': strategy_df
        }
        
        return metrics
    
    def plot_results(self, metrics=None):
        """
        Plot the results of the ML-based trading strategy.
        
        Args:
            metrics (dict, optional): Performance metrics from calculate_strategy_performance
            
        Returns:
            tuple: (fig1, fig2, fig3, fig4) - Figure objects for the plots
        """
        if not hasattr(self, 'combined_results') or self.combined_results.empty:
            print("No backtest results available or results are empty. Cannot generate plots.")
            return None
        
        if metrics is None:
            metrics = self.calculate_strategy_performance()
        
        # Check if we have valid metrics and strategy returns
        if metrics is None or 'strategy_returns' not in metrics or metrics['strategy_returns'].empty:
            print("No valid metrics available for plotting.")
            return None
        
        # 1. Plot predicted vs actual returns
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.scatter(self.combined_results['actual_return'], self.combined_results['predicted_return'], alpha=0.5)
        ax1.set_xlabel('Actual Return')
        ax1.set_ylabel('Predicted Return')
        ax1.set_title('Predicted vs Actual Returns')
        ax1.grid(True, alpha=0.3)
        
        # Add diagonal line (perfect prediction)
        min_val = min(self.combined_results['actual_return'].min(), self.combined_results['predicted_return'].min())
        max_val = max(self.combined_results['actual_return'].max(), self.combined_results['predicted_return'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.tight_layout()
        plt.savefig('predicted_vs_actual_returns.png')
        
        # 2. Plot strategy performance
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        strategy_df = metrics['strategy_returns']
        
        # Check if strategy_df has portfolio_value column and is not empty
        if 'portfolio_value' in strategy_df.columns and not strategy_df.empty:
            ax2.plot(strategy_df.index, strategy_df['portfolio_value'], label='ML Strategy')
            
            # Add buy and hold benchmark if single ticker
            if self.ticker is not None:
                try:
                    # Calculate buy and hold returns
                    ticker_data = self.portfolio_data[[self.ticker]]
                    ticker_returns = ticker_data.pct_change().dropna()
                    
                    # Make sure we have data for the same period as strategy_df
                    if not strategy_df.empty and len(strategy_df.index) > 0:
                        start_date = strategy_df.index[0]
                        end_date = strategy_df.index[-1]
                        ticker_returns = ticker_returns.loc[start_date:end_date]
                        
                        if not ticker_returns.empty:
                            ticker_cumulative = (1 + ticker_returns).cumprod()
                            ticker_value = 10000 * ticker_cumulative
                            
                            ax2.plot(ticker_value.index, ticker_value.values, label=f'Buy & Hold {self.ticker}', alpha=0.7)
                except Exception as e:
                    print(f"Error plotting buy & hold benchmark: {e}")
            
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Portfolio Value')
            ax2.set_title('ML Strategy Performance')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Format y-axis as currency
            ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f'₹{int(x):,}'))
        else:
            ax2.text(0.5, 0.5, 'No portfolio value data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=14)
        
        plt.tight_layout()
        plt.savefig('ml_strategy_performance.png')
        
        # 3. Plot confusion matrix for direction prediction
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        
        if not self.combined_results.empty and len(self.combined_results) > 0:
            cm = confusion_matrix(self.combined_results['actual_direction'], self.combined_results['predicted_direction'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
            ax3.set_xlabel('Predicted Direction')
            ax3.set_ylabel('Actual Direction')
            ax3.set_title('Confusion Matrix for Direction Prediction')
            ax3.set_xticklabels(['Down', 'Up'])
            ax3.set_yticklabels(['Down', 'Up'])
        else:
            ax3.text(0.5, 0.5, 'No direction prediction data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax3.transAxes, fontsize=14)
        
        plt.tight_layout()
        plt.savefig('direction_confusion_matrix.png')
        
        # 4. Plot accuracy and MSE over time
        fig4 = None
        if self.ticker is not None and self.ticker in self.results:
            ticker_results = self.results[self.ticker]
            
            # Check if we have accuracy and MSE data
            if len(ticker_results['accuracy']) > 0 and len(ticker_results['mse']) > 0:
                fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                
                # Plot accuracy
                test_starts = [d.strftime('%Y-%m-%d') for d in ticker_results['test_start']]
                ax4a.plot(test_starts, ticker_results['accuracy'], marker='o')
                ax4a.set_ylabel('Direction Accuracy')
                ax4a.set_title('Direction Prediction Accuracy Over Time')
                ax4a.grid(True, alpha=0.3)
                
                # Plot MSE
                ax4b.plot(test_starts, ticker_results['mse'], marker='o', color='orange')
                ax4b.set_xlabel('Test Window Start')
                ax4b.set_ylabel('Return MSE')
                ax4b.set_title('Return Prediction MSE Over Time')
                ax4b.grid(True, alpha=0.3)
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('prediction_metrics_over_time.png')
        
        print("Results plots saved to disk.")
        
        return fig1, fig2, fig3, fig4
    
    def print_performance_report(self, metrics=None):
        """
        Print a performance report for the ML-based trading strategy.
        
        Args:
            metrics (dict, optional): Performance metrics from calculate_strategy_performance
        """
        if not hasattr(self, 'combined_results') or self.combined_results.empty:
            print("No backtest results available or results are empty. Cannot generate performance report.")
            return
        
        if metrics is None:
            metrics = self.calculate_strategy_performance()
        
        # Check if we have valid metrics
        if metrics is None or 'strategy_returns' not in metrics or metrics['strategy_returns'].empty:
            print("No valid metrics available for performance reporting.")
            return
        
        strategy_df = metrics['strategy_returns']
        
        # Check if we have enough data points for a meaningful report
        if len(strategy_df) < 5:  # Arbitrary threshold, adjust as needed
            print("Not enough data points for a meaningful performance report.")
            return
        
        # Calculate investment period
        try:
            start_date = strategy_df.index[0]
            end_date = strategy_df.index[-1]
            investment_period = (end_date - start_date).days / 365.25  # in years
            
            # Print report
            print("\n" + "=" * 50)
            print("ML STRATEGY PERFORMANCE REPORT".center(50))
            print("=" * 50)
            
            print(f"\nInvestment Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({investment_period:.2f} years)")
            print(f"Initial Investment: ₹10,000.00")
            print(f"Final Value: ₹{strategy_df['portfolio_value'].iloc[-1]:,.2f}")
            print(f"Total Return: {metrics['total_return']:.2%}")
            print(f"Annualized Return: {metrics['annualized_return']:.2%}")
            
            print("\nRISK METRICS:")
            print(f"Volatility: {metrics['volatility']:.2%}")
            print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            
            print("\nDIRECTION PREDICTION METRICS:")
            if not self.combined_results.empty and 'actual_direction' in self.combined_results.columns and 'predicted_direction' in self.combined_results.columns:
                accuracy = accuracy_score(self.combined_results['actual_direction'], self.combined_results['predicted_direction'])
                print(f"Overall Accuracy: {accuracy:.2%}")
                
                # Calculate accuracy by ticker if multiple tickers
                if self.ticker is None and 'ticker' in self.combined_results.columns:
                    print("\nACCURACY BY TICKER:")
                    for ticker in self.combined_results['ticker'].unique():
                        ticker_data = self.combined_results[self.combined_results['ticker'] == ticker]
                        if not ticker_data.empty and len(ticker_data) > 0:
                            ticker_accuracy = accuracy_score(ticker_data['actual_direction'], ticker_data['predicted_direction'])
                            print(f"{ticker}: {ticker_accuracy:.2%}")
            else:
                print("No direction prediction data available.")
            
            print("\nRETURN PREDICTION METRICS:")
            if not self.combined_results.empty and 'actual_return' in self.combined_results.columns and 'predicted_return' in self.combined_results.columns:
                mse = mean_squared_error(self.combined_results['actual_return'], self.combined_results['predicted_return'])
                print(f"Overall MSE: {mse:.6f}")
                
                # Calculate MSE by ticker if multiple tickers
                if self.ticker is None and 'ticker' in self.combined_results.columns:
                    print("\nMSE BY TICKER:")
                    for ticker in self.combined_results['ticker'].unique():
                        ticker_data = self.combined_results[self.combined_results['ticker'] == ticker]
                        if not ticker_data.empty and len(ticker_data) > 0:
                            ticker_mse = mean_squared_error(ticker_data['actual_return'], ticker_data['predicted_return'])
                            print(f"{ticker}: {ticker_mse:.6f}")
            else:
                print("No return prediction data available.")
            
            print("\n" + "=" * 50)
            
            # Save report to file
            with open('ml_strategy_performance_report.txt', 'w') as f:
                f.write("ML STRATEGY PERFORMANCE REPORT\n\n")
                f.write(f"Investment Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({investment_period:.2f} years)\n")
                f.write(f"Initial Investment: ₹10,000.00\n")
                f.write(f"Final Value: ₹{strategy_df['portfolio_value'].iloc[-1]:,.2f}\n")
                f.write(f"Total Return: {metrics['total_return']:.2%}\n")
                f.write(f"Annualized Return: {metrics['annualized_return']:.2%}\n\n")
                
                f.write("RISK METRICS:\n")
                f.write(f"Volatility: {metrics['volatility']:.2%}\n")
                f.write(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}\n")
                f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n\n")
                
                f.write("DIRECTION PREDICTION METRICS:\n")
                if not self.combined_results.empty and 'actual_direction' in self.combined_results.columns and 'predicted_direction' in self.combined_results.columns:
                    accuracy = accuracy_score(self.combined_results['actual_direction'], self.combined_results['predicted_direction'])
                    f.write(f"Overall Accuracy: {accuracy:.2%}\n\n")
                    
                    if self.ticker is None and 'ticker' in self.combined_results.columns:
                        f.write("ACCURACY BY TICKER:\n")
                        for ticker in self.combined_results['ticker'].unique():
                            ticker_data = self.combined_results[self.combined_results['ticker'] == ticker]
                            if not ticker_data.empty and len(ticker_data) > 0:
                                ticker_accuracy = accuracy_score(ticker_data['actual_direction'], ticker_data['predicted_direction'])
                                f.write(f"{ticker}: {ticker_accuracy:.2%}\n")
                        f.write("\n")
                else:
                    f.write("No direction prediction data available.\n\n")
                
                f.write("RETURN PREDICTION METRICS:\n")
                if not self.combined_results.empty and 'actual_return' in self.combined_results.columns and 'predicted_return' in self.combined_results.columns:
                    mse = mean_squared_error(self.combined_results['actual_return'], self.combined_results['predicted_return'])
                    f.write(f"Overall MSE: {mse:.6f}\n\n")
                    
                    if self.ticker is None and 'ticker' in self.combined_results.columns:
                        f.write("MSE BY TICKER:\n")
                        for ticker in self.combined_results['ticker'].unique():
                            ticker_data = self.combined_results[self.combined_results['ticker'] == ticker]
                            if not ticker_data.empty and len(ticker_data) > 0:
                                ticker_mse = mean_squared_error(ticker_data['actual_return'], ticker_data['predicted_return'])
                                f.write(f"{ticker}: {ticker_mse:.6f}\n")
                else:
                    f.write("No return prediction data available.\n")
            
            print("Performance report saved to ml_strategy_performance_report.txt")
        except Exception as e:
            print(f"Error generating performance report: {e}")

def main():
    """Main function to demonstrate the ML backtester."""
    try:
        # Check if portfolio_data.csv exists
        if not os.path.exists('portfolio_data.csv'):
            print("Error: portfolio_data.csv not found. Please run stock_data_processor.py first.")
            return
        
        # Ask user for ticker
        print("\nML Backtester for Stock Price Prediction")
        print("----------------------------------------")
        print("1. Analyze a specific stock")
        print("2. Analyze all stocks in the portfolio")
        choice = input("Enter your choice (1/2): ")
        
        ticker = None
        if choice == '1':
            # Load portfolio data to get available tickers
            portfolio_data = pd.read_csv('portfolio_data.csv', index_col='Date', parse_dates=True)
            available_tickers = portfolio_data.columns.tolist()
            
            print("\nAvailable tickers:")
            for i, ticker_name in enumerate(available_tickers):
                print(f"{i+1}. {ticker_name}")
            
            ticker_idx = int(input("\nEnter the number of the ticker to analyze: ")) - 1
            if 0 <= ticker_idx < len(available_tickers):
                ticker = available_tickers[ticker_idx]
            else:
                print("Invalid ticker number. Analyzing all stocks.")
        
        # Initialize the ML backtester
        backtester = MLBacktester(ticker=ticker)
        
        # Run walk-forward backtest
        results = backtester.run_walk_forward_backtest(initial_window_years=5, test_window_years=1)
        
        if results is not None:
            # Calculate strategy performance
            metrics = backtester.calculate_strategy_performance()
            
            # Plot results
            backtester.plot_results(metrics)
            
            # Print performance report
            backtester.print_performance_report(metrics)
            
            # Ask if user wants to save results
            save_choice = input("\nDo you want to save the prediction results to CSV? (y/n): ")
            if save_choice.lower() == 'y':
                filename = input("Enter filename (default: ml_predictions.csv): ") or "ml_predictions.csv"
                results.to_csv(filename)
                print(f"Results saved to {filename}")
            
            # Ask if user wants to run for a different asset
            run_again = input("\nDo you want to run the model for a different asset? (y/n): ")
            if run_again.lower() == 'y':
                main()  # Restart the main function
        
        print("\nML backtesting completed successfully!")
    except Exception as e:
        print(f"Error in ML backtesting: {e}")

if __name__ == "__main__":
    main()