import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class PortfolioVisualizer:
    """Class for visualizing portfolio analysis results."""
    
    def __init__(self):
        """Initialize the portfolio visualizer."""
        self.output_dir = 'plots'
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Ensure the output directory exists."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def plot_correlation_matrix(self, prices_df):
        """Plot the correlation matrix of stock returns.
        
        Args:
            prices_df (pd.DataFrame): DataFrame of stock prices
            
        Returns:
            str: Path to the saved plot
        """
        try:
            # Calculate returns
            returns_df = prices_df.pct_change().dropna()
            
            # Calculate correlation matrix
            corr_matrix = returns_df.corr()
            
            # Create the plot
            plt.figure(figsize=(12, 10))
            plt.matshow(corr_matrix, fignum=1)
            plt.colorbar(label='Correlation')
            
            # Add ticks and labels
            tickers = corr_matrix.columns
            plt.xticks(range(len(tickers)), tickers, rotation=90)
            plt.yticks(range(len(tickers)), tickers)
            
            # Add title and labels
            plt.title('Correlation Matrix of Stock Returns', fontsize=16)
            
            # Save the plot
            plot_path = os.path.join(self.output_dir, 'correlation_matrix.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            
            print(f"Correlation matrix plot saved to {plot_path}")
            return plot_path
        
        except Exception as e:
            print(f"Error plotting correlation matrix: {e}")
            return None
    
    def plot_efficient_frontier(self, optimizer, show_assets=True):
        """Plot the efficient frontier with maximum Sharpe ratio and minimum volatility points.
        
        Args:
            optimizer (PortfolioOptimizer): The portfolio optimizer object
            show_assets (bool): Whether to show individual assets
            
        Returns:
            str: Path to the saved plot
        """
        try:
            # Get the efficient frontier
            returns, volatilities = optimizer.get_efficient_frontier(points=100)
            
            # Get random portfolios
            random_returns, random_volatilities, random_sharpe = optimizer.get_random_portfolios(n_portfolios=5000)
            
            # Get the max Sharpe and min vol portfolios
            max_sharpe_return, max_sharpe_vol, max_sharpe_ratio = optimizer.performance
            min_vol_weights, min_vol_performance = optimizer.get_min_volatility_portfolio()
            min_vol_return, min_vol_vol, min_vol_ratio = min_vol_performance
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Plot random portfolios
            plt.scatter(
                random_volatilities,
                random_returns,
                c=random_sharpe,
                cmap='viridis',
                marker='.',
                alpha=0.3,
                label='Random portfolios'
            )
            
            # Plot efficient frontier
            plt.plot(volatilities, returns, 'r-', linewidth=3, label='Efficient frontier')
            
            # Plot max Sharpe ratio point
            plt.scatter(
                max_sharpe_vol,
                max_sharpe_return,
                marker='*',
                color='red',
                s=300,
                label='Maximum Sharpe ratio'
            )
            
            # Plot min volatility point
            plt.scatter(
                min_vol_vol,
                min_vol_return,
                marker='o',
                color='green',
                s=200,
                label='Minimum volatility'
            )
            
            # Plot individual assets if requested
            if show_assets:
                for i, ticker in enumerate(optimizer.prices_df.columns):
                    asset_return = optimizer.mu[i]
                    asset_vol = np.sqrt(optimizer.S.iloc[i, i])
                    plt.scatter(
                        asset_vol,
                        asset_return,
                        marker='o',
                        s=100,
                        label=ticker
                    )
            
            # Add labels and title
            plt.title('Efficient Frontier', fontsize=16)
            plt.xlabel('Volatility (standard deviation)', fontsize=12)
            plt.ylabel('Expected Return', fontsize=12)
            plt.colorbar(label='Sharpe ratio')
            plt.legend(loc='best')
            plt.grid(True)
            
            # Save the plot
            plot_path = os.path.join(self.output_dir, 'efficient_frontier.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            
            print(f"Efficient frontier plot saved to {plot_path}")
            return plot_path
        
        except Exception as e:
            print(f"Error plotting efficient frontier: {e}")
            return None
    
    def plot_weights(self, weights, title="Portfolio Weights"):
        """Plot the portfolio weights.
        
        Args:
            weights (dict): Dictionary of weights
            title (str): Title for the plot
            
        Returns:
            str: Path to the saved plot
        """
        try:
            # Convert weights to Series
            weights_series = pd.Series(weights)
            
            # Filter out weights that are very small
            weights_series = weights_series[weights_series > 0.01]
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Create a pie chart
            plt.pie(
                weights_series,
                labels=weights_series.index,
                autopct='%1.1f%%',
                startangle=90,
                shadow=True,
                explode=[0.05] * len(weights_series)
            )
            
            # Add title
            plt.title(title, fontsize=16)
            
            # Save the plot
            filename = title.lower().replace(' ', '_')
            plot_path = os.path.join(self.output_dir, f'{filename}.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            
            print(f"{title} plot saved to {plot_path}")
            return plot_path
        
        except Exception as e:
            print(f"Error plotting weights: {e}")
            return None
    
    def plot_returns_histogram(self, prices_df):
        """Plot the histogram of returns for each stock.
        
        Args:
            prices_df (pd.DataFrame): DataFrame of stock prices
            
        Returns:
            str: Path to the saved plot
        """
        try:
            # Calculate returns
            returns_df = prices_df.pct_change().dropna()
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Plot histogram for each stock
            returns_df.plot(kind='hist', bins=50, alpha=0.5, ax=plt.gca())
            
            # Add title and labels
            plt.title('Distribution of Daily Returns', fontsize=16)
            plt.xlabel('Daily Return', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.legend()
            plt.grid(True)
            
            # Save the plot
            plot_path = os.path.join(self.output_dir, 'returns_histogram.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            
            print(f"Returns histogram plot saved to {plot_path}")
            return plot_path
        
        except Exception as e:
            print(f"Error plotting returns histogram: {e}")
            return None
    
    def plot_cumulative_returns(self, prices_df):
        """Plot the cumulative returns for each stock.
        
        Args:
            prices_df (pd.DataFrame): DataFrame of stock prices
            
        Returns:
            str: Path to the saved plot
        """
        try:
            # Calculate returns
            returns_df = prices_df.pct_change().dropna()
            
            # Calculate cumulative returns
            cumulative_returns = (1 + returns_df).cumprod()
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Plot cumulative returns for each stock
            cumulative_returns.plot(ax=plt.gca())
            
            # Add title and labels
            plt.title('Cumulative Returns', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Cumulative Return', fontsize=12)
            plt.legend()
            plt.grid(True)
            
            # Save the plot
            plot_path = os.path.join(self.output_dir, 'cumulative_returns.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            
            print(f"Cumulative returns plot saved to {plot_path}")
            return plot_path
        
        except Exception as e:
            print(f"Error plotting cumulative returns: {e}")
            return None
    
    def plot_allocation(self, allocation, latest_prices, investment_amount):
        """Plot the allocation of shares.
        
        Args:
            allocation (dict): Dictionary of allocation
            latest_prices (pd.Series): Series of latest prices
            investment_amount (float): Investment amount
            
        Returns:
            str: Path to the saved plot
        """
        try:
            # Calculate the cost of each allocation
            costs = {}
            for ticker, shares in allocation.items():
                costs[ticker] = shares * latest_prices[ticker]
            
            # Convert to Series
            costs_series = pd.Series(costs)
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Create a bar chart
            costs_series.plot(kind='bar', ax=plt.gca())
            
            # Add title and labels
            plt.title(f'Investment Allocation (Total: ₹{investment_amount:,.2f})', fontsize=16)
            plt.xlabel('Stock', fontsize=12)
            plt.ylabel('Investment (₹)', fontsize=12)
            plt.grid(True)
            
            # Add value labels on top of each bar
            for i, v in enumerate(costs_series):
                plt.text(i, v + 0.1, f'₹{v:,.0f}', ha='center')
            
            # Save the plot
            plot_path = os.path.join(self.output_dir, 'investment_allocation.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            
            print(f"Investment allocation plot saved to {plot_path}")
            return plot_path
        
        except Exception as e:
            print(f"Error plotting allocation: {e}")
            return None

# Example usage
def main():
    # This is just a placeholder to demonstrate how to use the visualizer
    # In practice, this would be called from main.py
    pass

if __name__ == "__main__":
    main()