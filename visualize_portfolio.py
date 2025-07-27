import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_portfolio_data():
    """Load the portfolio data from the CSV file.
    
    Returns:
        pd.DataFrame: Portfolio data with closing prices
    """
    try:
        portfolio_data = pd.read_csv('portfolio_data.csv', index_col='Date', parse_dates=True)
        return portfolio_data
    except FileNotFoundError:
        print("Error: portfolio_data.csv not found. Run stock_data_processor.py first.")
        return None

def plot_stock_prices(portfolio_data):
    """Plot the closing prices of all stocks in the portfolio.
    
    Args:
        portfolio_data (pd.DataFrame): Portfolio data with closing prices
    """
    plt.figure(figsize=(15, 8))
    
    # Plot each stock's closing price
    for column in portfolio_data.columns:
        plt.plot(portfolio_data.index, portfolio_data[column], alpha=0.7, linewidth=1, label=column)
    
    plt.title('Stock Closing Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (INR)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('stock_prices.png')
    plt.close()
    print("Stock prices plot saved to stock_prices.png")

def plot_normalized_prices(portfolio_data):
    """Plot the normalized prices of all stocks in the portfolio.
    
    Args:
        portfolio_data (pd.DataFrame): Portfolio data with closing prices
    """
    # Normalize the prices (divide by the first day's price)
    normalized_data = portfolio_data.div(portfolio_data.iloc[0])
    
    plt.figure(figsize=(15, 8))
    
    # Plot each stock's normalized price
    for column in normalized_data.columns:
        plt.plot(normalized_data.index, normalized_data[column], alpha=0.7, linewidth=1, label=column)
    
    plt.title('Normalized Stock Prices Over Time (Base = 1)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('normalized_prices.png')
    plt.close()
    print("Normalized prices plot saved to normalized_prices.png")

def plot_returns_distribution(portfolio_data):
    """Plot the distribution of daily returns for all stocks.
    
    Args:
        portfolio_data (pd.DataFrame): Portfolio data with closing prices
    """
    # Calculate daily returns
    daily_returns = portfolio_data.pct_change().dropna()
    
    plt.figure(figsize=(15, 8))
    
    # Plot the distribution of returns for each stock
    for column in daily_returns.columns:
        sns.kdeplot(daily_returns[column], label=column)
    
    plt.title('Distribution of Daily Returns')
    plt.xlabel('Daily Return')
    plt.ylabel('Density')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('returns_distribution.png')
    plt.close()
    print("Returns distribution plot saved to returns_distribution.png")

def plot_cumulative_returns(portfolio_data):
    """Plot the cumulative returns of all stocks in the portfolio.
    
    Args:
        portfolio_data (pd.DataFrame): Portfolio data with closing prices
    """
    # Calculate daily returns
    daily_returns = portfolio_data.pct_change().dropna()
    
    # Calculate cumulative returns
    cumulative_returns = (1 + daily_returns).cumprod()
    
    plt.figure(figsize=(15, 8))
    
    # Plot each stock's cumulative return
    for column in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[column], alpha=0.7, linewidth=1, label=column)
    
    plt.title('Cumulative Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('cumulative_returns.png')
    plt.close()
    print("Cumulative returns plot saved to cumulative_returns.png")

def plot_rolling_volatility(portfolio_data, window=30):
    """Plot the rolling volatility of all stocks in the portfolio.
    
    Args:
        portfolio_data (pd.DataFrame): Portfolio data with closing prices
        window (int): Rolling window size in days
    """
    # Calculate daily returns
    daily_returns = portfolio_data.pct_change().dropna()
    
    # Calculate rolling volatility (standard deviation)
    rolling_volatility = daily_returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    plt.figure(figsize=(15, 8))
    
    # Plot each stock's rolling volatility
    for column in rolling_volatility.columns:
        plt.plot(rolling_volatility.index, rolling_volatility[column], alpha=0.7, linewidth=1, label=column)
    
    plt.title(f'{window}-Day Rolling Volatility (Annualized)')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('rolling_volatility.png')
    plt.close()
    print(f"Rolling volatility plot saved to rolling_volatility.png")

def plot_optimal_portfolio_weights():
    """Plot the weights of the optimal portfolio."""
    try:
        # Load the optimal portfolio weights
        weights_df = pd.read_csv('optimal_portfolio_weights.csv')
        
        # Sort by weight in descending order
        weights_df = weights_df.sort_values('Weight', ascending=False)
        
        # Plot the weights
        plt.figure(figsize=(12, 8))
        
        # Create a bar plot
        plt.bar(weights_df['Stock'], weights_df['Weight'], color='skyblue')
        
        plt.title('Optimal Portfolio Weights')
        plt.xlabel('Stock')
        plt.ylabel('Weight')
        plt.xticks(rotation=90)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('optimal_portfolio_weights.png')
        plt.close()
        print("Optimal portfolio weights plot saved to optimal_portfolio_weights.png")
    except FileNotFoundError:
        print("Error: optimal_portfolio_weights.csv not found. Run mpt_optimizer.py first.")

def main():
    """Main function to visualize the portfolio data."""
    print("Visualizing portfolio data...")
    
    # Load the portfolio data
    portfolio_data = load_portfolio_data()
    
    if portfolio_data is not None:
        # Plot the stock prices
        plot_stock_prices(portfolio_data)
        
        # Plot the normalized prices
        plot_normalized_prices(portfolio_data)
        
        # Plot the returns distribution
        plot_returns_distribution(portfolio_data)
        
        # Plot the cumulative returns
        plot_cumulative_returns(portfolio_data)
        
        # Plot the rolling volatility
        plot_rolling_volatility(portfolio_data)
        
        # Plot the optimal portfolio weights
        plot_optimal_portfolio_weights()
        
        print("\nVisualization completed successfully!")

if __name__ == "__main__":
    main()