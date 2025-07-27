#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Portfolio Assistant

A comprehensive tool that combines:
- Machine Learning return prediction
- Modern Portfolio Theory (PyPortfolioOpt) optimization
- Rolling walk-forward backtesting
- Investment amount calculator

Author: AI Assistant
Date: June 2025
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

# Handle potential import errors gracefully
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingRegressor
    # Try to import XGBoost if available
    try:
        import xgboost as xgb
        from xgboost import XGBRegressor
        XGBOOST_AVAILABLE = True
    except ImportError:
        XGBOOST_AVAILABLE = False
except ImportError:
    print("Error: scikit-learn is not installed. Please install it using:")
    print("pip install scikit-learn")
    sys.exit(1)

try:
    import ta
except ImportError:
    print("Error: ta (Technical Analysis library) is not installed. Please install it using:")
    print("pip install ta")
    sys.exit(1)

try:
    import pypfopt
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.discrete_allocation import DiscreteAllocation
except ImportError:
    print("Error: PyPortfolioOpt is not installed. Please install it using:")
    print("pip install PyPortfolioOpt")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings('ignore')

# Global variables
DATA_DIR = "closing_prices_only"
PLOTS_DIR = "plots"

### =====================================================================
### DATA LOADING
### =====================================================================

def load_data():
    """
    Load all closing price CSV files from the data directory,
    merge them into a single DataFrame, and clean the data.
    
    Returns:
        pd.DataFrame: Cleaned and merged price data indexed by date
    """
    print("\n" + "=" * 80)
    print("LOADING AND PREPARING DATA".center(80))
    print("=" * 80)
    
    # Create plots directory if it doesn't exist
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        print(f"Created directory: {PLOTS_DIR}")
    
    # Get all CSV files in the data directory
    csv_files = glob.glob(os.path.join(DATA_DIR, "*_closing.csv"))
    
    if not csv_files:
        print(f"Error: No closing price CSV files found in {DATA_DIR}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} stock price files")
    
    # Initialize an empty DataFrame to store the merged data
    all_data = pd.DataFrame()
    
    # Process each CSV file
    for file in csv_files:
        try:
            # Extract ticker from filename
            ticker = os.path.basename(file).replace("_closing.csv", "")
            
            # Read the CSV file
            df = pd.read_csv(file, index_col='Date', parse_dates=True)
            
            # Rename the column to the ticker
            if 'Close' in df.columns:
                df = df[['Close']].rename(columns={'Close': ticker})
            elif df.shape[1] == 1:  # If there's only one column, assume it's the closing price
                df.columns = [ticker]
            else:
                print(f"Warning: Unexpected columns in {file}. Skipping.")
                continue
            
            # Merge with the main DataFrame
            if all_data.empty:
                all_data = df
            else:
                all_data = all_data.join(df, how='outer')
                
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    # Forward-fill missing values (gaps)
    all_data = all_data.fillna(method='ffill')
    
    # Drop rows where all prices are NaN
    all_data = all_data.dropna(how='all')
    
    # Sort by date
    all_data = all_data.sort_index()
    
    # Print summary
    print(f"\nData loaded successfully:")
    print(f"- Time period: {all_data.index.min().strftime('%Y-%m-%d')} to {all_data.index.max().strftime('%Y-%m-%d')}")
    print(f"- Number of stocks: {all_data.shape[1]}")
    print(f"- Number of trading days: {all_data.shape[0]}")
    print(f"- Stocks: {', '.join(all_data.columns[:5])}" + (f" and {len(all_data.columns) - 5} more" if len(all_data.columns) > 5 else ""))
    
    # Check for remaining NaN values
    nan_counts = all_data.isna().sum()
    if nan_counts.sum() > 0:
        print("\nWarning: There are still some NaN values in the data:")
        for ticker, count in nan_counts[nan_counts > 0].items():
            print(f"- {ticker}: {count} missing values")
        
        # Fill remaining NaNs with column mean
        all_data = all_data.fillna(all_data.mean())
        print("Filled remaining NaN values with column means")
    
    return all_data

### =====================================================================
### FEATURE ENGINEERING
### =====================================================================

def create_features(price_data):
    """
    Create features for machine learning models.
    
    Args:
        price_data (pd.DataFrame): Price data indexed by date
        
    Returns:
        pd.DataFrame: DataFrame with features and target variables
    """
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING".center(80))
    print("=" * 80)
    
    features_data = pd.DataFrame(index=price_data.index)
    
    # Process each stock
    for ticker in price_data.columns:
        print(f"Creating features for {ticker}...")
        
        # Get price series for this ticker
        prices = price_data[ticker]
        
        # Calculate returns
        returns = prices.pct_change()
        
        # Create lagged returns (1, 5, 10 days)
        features_data[f"{ticker}_return_1d"] = returns
        features_data[f"{ticker}_return_5d"] = prices.pct_change(5)
        features_data[f"{ticker}_return_10d"] = prices.pct_change(10)
        
        # Create rolling means (10, 20 days)
        features_data[f"{ticker}_mean_10d"] = prices.rolling(window=10).mean() / prices - 1
        features_data[f"{ticker}_mean_20d"] = prices.rolling(window=20).mean() / prices - 1
        
        # Create rolling standard deviation (10 days)
        features_data[f"{ticker}_std_10d"] = prices.rolling(window=10).std() / prices
        
        # Calculate RSI (14 days)
        try:
            rsi = ta.momentum.RSIIndicator(prices, window=14).rsi() / 100  # Normalize to 0-1
            features_data[f"{ticker}_rsi_14d"] = rsi
        except Exception as e:
            print(f"Warning: Could not calculate RSI for {ticker}: {str(e)}")
            features_data[f"{ticker}_rsi_14d"] = np.nan
        
        # Create target variables
        # Next day return (for regression)
        features_data[f"{ticker}_next_return"] = returns.shift(-1)
        
        # Next day direction (for classification)
        features_data[f"{ticker}_next_direction"] = (returns.shift(-1) > 0).astype(int)
    
    # Drop rows with NaN values (due to lagging/rolling calculations)
    features_data = features_data.dropna()
    
    print(f"\nFeatures created successfully:")
    print(f"- Number of features: {features_data.shape[1]}")
    print(f"- Time period: {features_data.index.min().strftime('%Y-%m-%d')} to {features_data.index.max().strftime('%Y-%m-%d')}")
    print(f"- Number of samples: {features_data.shape[0]}")
    
    return features_data

### =====================================================================
### MACHINE LEARNING
### =====================================================================

def train_predict_ml(features_data, price_data, initial_window_years=5, test_window_years=1):
    """
    Train ML models using walk-forward validation and make predictions.
    
    Args:
        features_data (pd.DataFrame): Features and target variables
        price_data (pd.DataFrame): Original price data
        initial_window_years (int): Number of years for initial training window
        test_window_years (int): Number of years for each test window
        
    Returns:
        tuple: (predictions_df, performance_metrics)
    """
    print("\n" + "=" * 80)
    print("MACHINE LEARNING PREDICTION".center(80))
    print("=" * 80)
    
    # Get all unique tickers from the price data
    tickers = price_data.columns.tolist()
    
    # Initialize DataFrames to store predictions and actual values
    all_predictions = pd.DataFrame()
    
    # Initialize dictionaries to store performance metrics
    direction_accuracy = {}
    return_mae = {}
    return_rmse = {}
    
    # Get the date range
    start_date = features_data.index.min()
    end_date = features_data.index.max()
    
    # Calculate initial window end date
    days_in_initial_window = int(initial_window_years * 365.25)  # Approximate days in years
    initial_window_end = start_date + pd.Timedelta(days=days_in_initial_window)
    
    # Check if we have enough data
    if initial_window_end >= end_date:
        print(f"Warning: Not enough data for {initial_window_years} years initial window.")
        print(f"Available data: {(end_date - start_date).days / 365.25:.2f} years")
        # Adjust initial window to use 2/3 of available data
        days_in_initial_window = int((end_date - start_date).days * 2/3)
        initial_window_end = start_date + pd.Timedelta(days=days_in_initial_window)
        print(f"Adjusted initial window to {days_in_initial_window / 365.25:.2f} years")
    
    # Initialize current window
    current_window_start = start_date
    current_window_end = initial_window_end
    
    # Calculate days in test window
    days_in_test_window = int(test_window_years * 365.25)  # Approximate days in years
    
    # Store weights for each test period (for backtesting)
    period_weights = {}
    period_dates = []
    
    # Run walk-forward validation
    window_count = 0
    while current_window_end < end_date:
        window_count += 1
        print(f"\nProcessing window {window_count}:")
        
        # Define test window
        test_window_start = current_window_end
        test_window_end = test_window_start + pd.Timedelta(days=days_in_test_window)
        
        # Ensure test window doesn't exceed available data
        if test_window_end > end_date:
            test_window_end = end_date
        
        print(f"Training: {current_window_start.strftime('%Y-%m-%d')} to {current_window_end.strftime('%Y-%m-%d')}")
        print(f"Testing:  {test_window_start.strftime('%Y-%m-%d')} to {test_window_end.strftime('%Y-%m-%d')}")
        
        # Store test period start date for backtesting
        period_dates.append(test_window_start)
        
        # Initialize predictions for this window
        window_predictions = pd.DataFrame(index=features_data[(features_data.index >= test_window_start) & 
                                                           (features_data.index < test_window_end)].index)
        
        # Process each ticker
        for ticker in tickers:
            # Get feature columns for this ticker
            feature_cols = [col for col in features_data.columns if col.startswith(f"{ticker}_") and 
                           not col.endswith("_next_return") and not col.endswith("_next_direction")]
            
            # Get target columns for this ticker
            return_target = f"{ticker}_next_return"
            direction_target = f"{ticker}_next_direction"
            
            # Split data into training and testing sets
            train_data = features_data[(features_data.index >= current_window_start) & 
                                      (features_data.index < current_window_end)]
            test_data = features_data[(features_data.index >= test_window_start) & 
                                     (features_data.index < test_window_end)]
            
            # Skip if not enough data
            if len(train_data) < 252:  # At least 1 year of training data
                print(f"  Skipping {ticker}: Not enough training data ({len(train_data)} samples)")
                continue
            
            if len(test_data) < 20:  # At least 1 month of testing data
                print(f"  Skipping {ticker}: Not enough testing data ({len(test_data)} samples)")
                continue
            
            # Prepare training data
            X_train = train_data[feature_cols]
            y_train_return = train_data[return_target]
            y_train_direction = train_data[direction_target]
            
            # Prepare testing data
            X_test = test_data[feature_cols]
            y_test_return = test_data[return_target]
            y_test_direction = test_data[direction_target]
            
            # Normalize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train return prediction model (regression)
            print(f"  Training models for {ticker}...")
            if XGBOOST_AVAILABLE:
                return_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            else:
                return_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            
            return_model.fit(X_train_scaled, y_train_return)
            
            # Train direction prediction model (classification)
            try:
                direction_model = LogisticRegression(random_state=42, max_iter=1000)
                # Check if we have at least two classes in the target
                if len(np.unique(y_train_direction)) < 2:
                    print(f"  Warning: Only one class in direction data for {ticker}. Using dummy classifier.")
                    # Create a dummy classifier that always predicts the majority class
                    majority_class = int(y_train_direction.iloc[0])
                    y_pred_direction = np.full(len(X_test_scaled), majority_class)
                else:
                    direction_model.fit(X_train_scaled, y_train_direction)
                    y_pred_direction = direction_model.predict(X_test_scaled)
            except Exception as e:
                print(f"  Error training direction model for {ticker}: {str(e)}")
                # Use a dummy classifier as fallback
                majority_class = int(y_train_direction.mode().iloc[0]) if not y_train_direction.empty else 1
                y_pred_direction = np.full(len(X_test_scaled), majority_class)
            
            # Make predictions
            y_pred_return = return_model.predict(X_test_scaled)
            
            # Direction predictions are already handled in the try-except block above
            
            # Store predictions
            window_predictions[f"{ticker}_pred_return"] = y_pred_return
            window_predictions[f"{ticker}_actual_return"] = y_test_return.values
            window_predictions[f"{ticker}_pred_direction"] = y_pred_direction
            window_predictions[f"{ticker}_actual_direction"] = y_test_direction.values
            
            # Calculate metrics
            acc = accuracy_score(y_test_direction, y_pred_direction)
            mae = mean_absolute_error(y_test_return, y_pred_return)
            rmse = np.sqrt(mean_squared_error(y_test_return, y_pred_return))
            
            # Update performance metrics
            if ticker not in direction_accuracy:
                direction_accuracy[ticker] = []
                return_mae[ticker] = []
                return_rmse[ticker] = []
            
            direction_accuracy[ticker].append(acc)
            return_mae[ticker].append(mae)
            return_rmse[ticker].append(rmse)
            
            print(f"  {ticker}: Direction Accuracy = {acc:.4f}, Return MAE = {mae:.6f}, RMSE = {rmse:.6f}")
        
        # Add window predictions to all predictions
        if not window_predictions.empty:
            if all_predictions.empty:
                all_predictions = window_predictions
            else:
                all_predictions = pd.concat([all_predictions, window_predictions])
        
        # Move to next window
        current_window_start = current_window_start + pd.Timedelta(days=days_in_test_window)
        current_window_end = current_window_end + pd.Timedelta(days=days_in_test_window)
    
    # Calculate overall performance metrics
    overall_metrics = {}
    print("\nOverall ML Performance Metrics:")
    print("-" * 40)
    
    for ticker in tickers:
        if ticker in direction_accuracy and direction_accuracy[ticker]:
            avg_acc = np.mean(direction_accuracy[ticker])
            avg_mae = np.mean(return_mae[ticker])
            avg_rmse = np.mean(return_rmse[ticker])
            
            overall_metrics[ticker] = {
                'direction_accuracy': avg_acc,
                'return_mae': avg_mae,
                'return_rmse': avg_rmse
            }
            
            print(f"{ticker}:")
            print(f"  Direction Accuracy: {avg_acc:.4f}")
            print(f"  Return MAE: {avg_mae:.6f}")
            print(f"  Return RMSE: {avg_rmse:.6f}")
    
    # Save predictions to CSV
    all_predictions.to_csv('ml_predictions.csv')
    print("\nPredictions saved to ml_predictions.csv")
    
    return all_predictions, overall_metrics, period_dates

### =====================================================================
### PORTFOLIO OPTIMIZATION
### =====================================================================

def optimize_portfolio(price_data, predictions_df, period_dates, use_ml_predictions=True):
    """
    Optimize portfolio using PyPortfolioOpt.
    
    Args:
        price_data (pd.DataFrame): Price data indexed by date
        predictions_df (pd.DataFrame): ML predictions
        period_dates (list): List of dates for each test period
        use_ml_predictions (bool): Whether to use ML predictions for expected returns
        
    Returns:
        dict: Dictionary with optimized weights for each period
    """
    print("\n" + "=" * 80)
    print("PORTFOLIO OPTIMIZATION".center(80))
    print("=" * 80)
    
    # Get all unique tickers
    tickers = price_data.columns.tolist()
    
    # Initialize dictionary to store weights for each period
    period_weights = {}
    
    # Process each period
    for i, period_start in enumerate(period_dates):
        print(f"\nOptimizing portfolio for period starting {period_start.strftime('%Y-%m-%d')}:")
        
        # Get data up to period start
        historical_data = price_data[price_data.index < period_start]
        
        # Skip if not enough data
        if len(historical_data) < 252:  # At least 1 year of data
            print(f"  Skipping: Not enough historical data ({len(historical_data)} days)")
            continue
        
        # Calculate expected returns
        if use_ml_predictions:
            print("  Using ML predictions for expected returns")
            
            # Get the most recent predictions for each ticker
            expected_returns_dict = {}
            for ticker in tickers:
                # Check if we have predictions for this ticker
                if f"{ticker}_pred_return" in predictions_df.columns:
                    # Get predictions for the next period
                    next_period_preds = predictions_df[predictions_df.index >= period_start]
                    
                    if not next_period_preds.empty:
                        # Use the average of predicted returns for the next period
                        expected_returns_dict[ticker] = next_period_preds[f"{ticker}_pred_return"].mean() * 252  # Annualize
                    else:
                        # Fallback to historical mean
                        expected_returns_dict[ticker] = historical_data[ticker].pct_change().mean() * 252
                else:
                    # Fallback to historical mean
                    expected_returns_dict[ticker] = historical_data[ticker].pct_change().mean() * 252
            
            # Convert to Series
            mu = pd.Series(expected_returns_dict)
        else:
            print("  Using historical mean returns")
            mu = expected_returns.mean_historical_return(historical_data, frequency=252)
        
        # Calculate sample covariance matrix
        S = risk_models.sample_cov(historical_data, frequency=252)
        
        # Create Efficient Frontier object
        try:
            ef = EfficientFrontier(mu, S)
            
            # Ask user for optimization method
            print("\nOptimization Methods:")
            print("1. Maximum Sharpe Ratio")
            print("2. Minimum Volatility")
            print("3. Target Return")
            print("4. Target Risk")
            
            # Restore stdout for user input
            original_stdout = sys.stdout
            sys.stdout = sys.__stdout__
            
            method = input("\nSelect optimization method (1-4): ")
            
            # Redirect back to original stdout
            sys.stdout = original_stdout
            
            # Optimize portfolio based on selected method
            if method == "1":
                print("  Optimizing for Maximum Sharpe Ratio")
                ef.max_sharpe()
                weights = ef.clean_weights()
                expected_return, expected_volatility, sharpe_ratio = ef.portfolio_performance()
                print(f"  Expected Annual Return: {expected_return:.2%}")
                print(f"  Annual Volatility: {expected_volatility:.2%}")
                print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
            
            elif method == "2":
                print("  Optimizing for Minimum Volatility")
                ef.min_volatility()
                weights = ef.clean_weights()
                expected_return, expected_volatility, sharpe_ratio = ef.portfolio_performance()
                print(f"  Expected Annual Return: {expected_return:.2%}")
                print(f"  Annual Volatility: {expected_volatility:.2%}")
                print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
            
            elif method == "3":
                # Restore stdout for user input
                sys.stdout = sys.__stdout__
                target_return = float(input("Enter target annual return (e.g., 0.15 for 15%): "))
                sys.stdout = original_stdout
                
                print(f"  Optimizing for Target Return: {target_return:.2%}")
                ef.efficient_return(target_return)
                weights = ef.clean_weights()
                expected_return, expected_volatility, sharpe_ratio = ef.portfolio_performance()
                print(f"  Expected Annual Return: {expected_return:.2%}")
                print(f"  Annual Volatility: {expected_volatility:.2%}")
                print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
            
            elif method == "4":
                # Restore stdout for user input
                sys.stdout = sys.__stdout__
                target_risk = float(input("Enter target annual volatility (e.g., 0.10 for 10%): "))
                sys.stdout = original_stdout
                
                print(f"  Optimizing for Target Risk: {target_risk:.2%}")
                ef.efficient_risk(target_risk)
                weights = ef.clean_weights()
                expected_return, expected_volatility, sharpe_ratio = ef.portfolio_performance()
                print(f"  Expected Annual Return: {expected_return:.2%}")
                print(f"  Annual Volatility: {expected_volatility:.2%}")
                print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
            
            else:
                print("  Invalid selection. Using Maximum Sharpe Ratio.")
                ef.max_sharpe()
                weights = ef.clean_weights()
                expected_return, expected_volatility, sharpe_ratio = ef.portfolio_performance()
                print(f"  Expected Annual Return: {expected_return:.2%}")
                print(f"  Annual Volatility: {expected_volatility:.2%}")
                print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
            
            # Store weights for this period
            period_weights[period_start] = {
                'weights': weights,
                'expected_return': expected_return,
                'expected_volatility': expected_volatility,
                'sharpe_ratio': sharpe_ratio
            }
            
            # Print weights
            print("\n  Optimized Weights:")
            for ticker, weight in weights.items():
                if weight > 0.01:  # Only show significant weights
                    print(f"    {ticker}: {weight:.4f}")
        
        except Exception as e:
            print(f"  Error optimizing portfolio: {str(e)}")
            continue
    
    return period_weights

### =====================================================================
### BACKTESTING
### =====================================================================

def backtest_portfolio(price_data, period_weights, period_dates, include_costs=True):
    """
    Backtest the optimized portfolio.
    
    Args:
        price_data (pd.DataFrame): Price data indexed by date
        period_weights (dict): Dictionary with optimized weights for each period
        period_dates (list): List of dates for each test period
        include_costs (bool): Whether to include transaction costs
        
    Returns:
        tuple: (backtest_results, backtest_metrics)
    """
    print("\n" + "=" * 80)
    print("PORTFOLIO BACKTESTING".center(80))
    print("=" * 80)
    
    # Parameters for transaction costs
    slippage = 0.001  # 0.1%
    commission = 0.0005  # 0.05%
    
    # Initialize DataFrame for backtest results
    backtest_results = pd.DataFrame(index=price_data.index)
    backtest_results['portfolio_value'] = np.nan
    
    # Initialize portfolio value
    initial_investment = 10000  # $10,000 initial investment
    current_value = initial_investment
    
    # Initialize holdings
    current_holdings = {ticker: 0 for ticker in price_data.columns}
    
    # Find the first date in the backtest (first period start date)
    if not period_dates:
        print("No period dates available for backtesting.")
        return None, None
    
    backtest_start = period_dates[0]
    
    # Find the last date in the backtest (last available price data)
    backtest_end = price_data.index[-1]
    
    print(f"Backtesting from {backtest_start.strftime('%Y-%m-%d')} to {backtest_end.strftime('%Y-%m-%d')}")
    
    # Get all dates in the backtest period
    backtest_dates = price_data[(price_data.index >= backtest_start) & 
                              (price_data.index <= backtest_end)].index
    
    # Initialize portfolio values for all dates
    portfolio_values = pd.Series(index=backtest_dates)
    
    # Track current period index
    current_period_idx = 0
    
    # Process each date in the backtest period
    for date in backtest_dates:
        # Check if we need to rebalance (start of a new period)
        if current_period_idx < len(period_dates) and date >= period_dates[current_period_idx]:
            # Get the period start date
            period_start = period_dates[current_period_idx]
            
            # Check if we have weights for this period
            if period_start in period_weights:
                print(f"\nRebalancing portfolio on {date.strftime('%Y-%m-%d')}")
                
                # Get the weights for this period
                weights = period_weights[period_start]['weights']
                
                # Get the prices for this date
                prices = price_data.loc[date]
                
                # Calculate the new target position values
                target_values = {ticker: current_value * weight for ticker, weight in weights.items()}
                
                # Calculate the new target positions (number of shares)
                target_positions = {ticker: int(value / prices[ticker]) for ticker, value in target_values.items() if ticker in prices}
                
                # Calculate transaction costs if enabled
                if include_costs:
                    # Calculate the cost of trading
                    trading_cost = 0
                    for ticker, target_pos in target_positions.items():
                        # Calculate the change in position
                        position_change = abs(target_pos - current_holdings[ticker])
                        
                        # Calculate the cost of trading
                        if position_change > 0:
                            # Cost = price * position_change * (slippage + commission)
                            cost = prices[ticker] * position_change * (slippage + commission)
                            trading_cost += cost
                    
                    # Subtract trading costs from current value
                    current_value -= trading_cost
                    print(f"  Trading costs: ${trading_cost:.2f}")
                
                # Update holdings
                current_holdings = target_positions
                
                # Move to the next period
                current_period_idx += 1
        
        # Calculate portfolio value for this date
        try:
            prices = price_data.loc[date]
            portfolio_value = sum(current_holdings[ticker] * prices[ticker] for ticker in current_holdings if ticker in prices)
            portfolio_values[date] = portfolio_value
        except Exception as e:
            print(f"Error calculating portfolio value for {date.strftime('%Y-%m-%d')}: {str(e)}")
            continue
    
    # Calculate daily returns
    daily_returns = portfolio_values.pct_change().dropna()
    
    # Calculate cumulative returns
    cumulative_returns = (1 + daily_returns).cumprod()
    
    # Calculate performance metrics
    total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0
    years = (backtest_end - backtest_start).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    annual_volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (cagr - 0.02) / annual_volatility if annual_volatility > 0 else 0  # Assuming 2% risk-free rate
    
    # Calculate drawdowns
    rolling_max = portfolio_values.cummax()
    drawdowns = (portfolio_values - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Calculate Sortino ratio (downside deviation)
    negative_returns = daily_returns[daily_returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(252)
    sortino_ratio = (cagr - 0.02) / downside_deviation if downside_deviation > 0 else 0
    
    # Calculate Value at Risk (95%)
    var_95 = np.percentile(daily_returns, 5)
    
    # Store results
    backtest_results = pd.DataFrame({
        'portfolio_value': portfolio_values,
        'daily_return': daily_returns,
        'cumulative_return': cumulative_returns
    })
    
    # Store metrics
    backtest_metrics = {
        'total_return': total_return,
        'cagr': cagr,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95
    }
    
    # Print performance metrics
    print("\nBacktest Performance Metrics:")
    print("-" * 40)
    print(f"Total Return: {total_return:.2%}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Annual Volatility: {annual_volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Value at Risk (95%): {var_95:.2%}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(backtest_results['cumulative_return'])
    plt.title('Cumulative Return')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(backtest_results['daily_return'].rolling(window=252).std() * np.sqrt(252))
    plt.title('Rolling Annual Volatility (252-day window)')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(drawdowns)
    plt.title('Drawdowns')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'backtest_results.png'))
    plt.close()
    
    # Save backtest results to CSV
    backtest_results.to_csv('backtest_report.csv')
    print("\nBacktest results saved to backtest_report.csv")
    print(f"Backtest plots saved to {os.path.join(PLOTS_DIR, 'backtest_results.png')}")
    
    return backtest_results, backtest_metrics

### =====================================================================
### INVESTMENT CALCULATOR
### =====================================================================

def investment_calculator(price_data, weights):
    """
    Calculate the number of shares to buy based on the investment amount.
    
    Args:
        price_data (pd.DataFrame): Price data indexed by date
        weights (dict): Dictionary with optimized weights
    """
    print("\n" + "=" * 80)
    print("INVESTMENT CALCULATOR".center(80))
    print("=" * 80)
    
    # Get the latest prices
    latest_date = price_data.index[-1]
    latest_prices = price_data.loc[latest_date]
    
    print(f"Using latest prices as of {latest_date.strftime('%Y-%m-%d')}")
    
    while True:
        # Restore stdout for user input
        original_stdout = sys.stdout
        sys.stdout = sys.__stdout__
        
        try:
            investment_amount = float(input("\nHow much capital do you want to invest? (e.g., 10000): "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        
        # Redirect back to original stdout
        sys.stdout = original_stdout
        
        print(f"\nCalculating allocation for ${investment_amount:.2f}...")
        
        # Create discrete allocation object
        try:
            da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=investment_amount)
            allocation, leftover = da.greedy_portfolio()
            
            # Print allocation
            print("\nDiscrete Allocation:")
            print("-" * 40)
            print(f"{'Ticker':<10} {'Shares':<10} {'Price':<10} {'Value':<10}")
            print("-" * 40)
            
            total_value = 0
            for ticker, shares in allocation.items():
                price = latest_prices[ticker]
                value = shares * price
                total_value += value
                print(f"{ticker:<10} {shares:<10} ${price:<9.2f} ${value:<9.2f}")
            
            print("-" * 40)
            print(f"{'Total':<10} {'':<10} {'':<10} ${total_value:<9.2f}")
            print(f"{'Leftover':<10} {'':<10} {'':<10} ${leftover:<9.2f}")
            print(f"{'Investment':<10} {'':<10} {'':<10} ${investment_amount:<9.2f}")
        
        except Exception as e:
            print(f"Error calculating allocation: {str(e)}")
        
        # Ask if the user wants to enter another amount
        sys.stdout = sys.__stdout__
        another = input("\nDo you want to enter another investment amount? (y/n): ")
        sys.stdout = original_stdout
        
        if another.lower() != 'y':
            break

### =====================================================================
### MAIN FUNCTION
### =====================================================================

def main():
    """
    Main function to run the portfolio assistant.
    """
    print("\n" + "=" * 80)
    print("PORTFOLIO ASSISTANT".center(80))
    print("=" * 80)
    print("A comprehensive tool for portfolio optimization and backtesting")
    print("=" * 80)
    
    try:
        # Step 1: Load data
        price_data = load_data()
        
        # Step 2: Create features
        features_data = create_features(price_data)
        
        # Step 3: Train ML models and make predictions
        predictions_df, ml_metrics, period_dates = train_predict_ml(features_data, price_data)
        
        # Step 4: Ask user whether to use ML predictions or historical returns
        original_stdout = sys.stdout
        sys.stdout = sys.__stdout__
        use_ml = input("\nDo you want to use ML predictions for expected returns? (y/n): ")
        sys.stdout = original_stdout
        
        use_ml_predictions = use_ml.lower() == 'y'
        
        # Step 5: Optimize portfolio
        period_weights = optimize_portfolio(price_data, predictions_df, period_dates, use_ml_predictions)
        
        # Step 6: Backtest portfolio
        backtest_results, backtest_metrics = backtest_portfolio(price_data, period_weights, period_dates)
        
        # Step 7: Investment calculator
        # Use the weights from the most recent period
        if period_dates and period_dates[-1] in period_weights:
            latest_weights = period_weights[period_dates[-1]]['weights']
            investment_calculator(price_data, latest_weights)
        else:
            print("\nNo weights available for investment calculator.")
        
        print("\n" + "=" * 80)
        print("PORTFOLIO ASSISTANT COMPLETED".center(80))
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError in main function: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()