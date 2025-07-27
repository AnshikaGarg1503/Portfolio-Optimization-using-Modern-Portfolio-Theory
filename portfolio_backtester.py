import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class PortfolioBacktester:
    """Class for backtesting portfolio performance."""
    
    def __init__(self, portfolio_data, weights_file=None, weights_dict=None):
        """Initialize the backtester with portfolio data and weights.
        
        Args:
            portfolio_data (pd.DataFrame): DataFrame with stock prices
            weights_file (str, optional): Path to CSV file with portfolio weights
            weights_dict (dict, optional): Dictionary with portfolio weights
        """
        self.portfolio_data = portfolio_data
        
        # Load weights from file or dictionary
        if weights_file is not None:
            weights_df = pd.read_csv(weights_file)
            self.weights = dict(zip(weights_df['Stock'], weights_df['Weight']))
        elif weights_dict is not None:
            self.weights = weights_dict
        else:
            # Equal weights if no weights provided
            self.weights = {ticker: 1.0 / len(portfolio_data.columns) for ticker in portfolio_data.columns}
        
        # Calculate daily returns
        self.daily_returns = portfolio_data.pct_change().dropna()
    
    def calculate_portfolio_returns(self):
        """Calculate the daily returns of the portfolio.
        
        Returns:
            pd.Series: Daily portfolio returns
        """
        # Initialize portfolio returns series
        portfolio_returns = pd.Series(0, index=self.daily_returns.index)
        
        # Calculate weighted returns for each stock
        for ticker, weight in self.weights.items():
            if ticker in self.daily_returns.columns:
                portfolio_returns += self.daily_returns[ticker] * weight
        
        return portfolio_returns
    
    def calculate_cumulative_returns(self, initial_investment=10000):
        """Calculate the cumulative returns of the portfolio.
        
        Args:
            initial_investment (float): Initial investment amount
            
        Returns:
            pd.Series: Cumulative portfolio value
        """
        # Calculate daily portfolio returns
        portfolio_returns = self.calculate_portfolio_returns()
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate portfolio value
        portfolio_value = initial_investment * cumulative_returns
        
        return portfolio_value
    
    def calculate_performance_metrics(self):
        """Calculate performance metrics for the portfolio.
        
        Returns:
            dict: Dictionary with performance metrics
        """
        # Calculate daily portfolio returns
        portfolio_returns = self.calculate_portfolio_returns()
        
        # Calculate annualized return
        annualized_return = portfolio_returns.mean() * 252
        
        # Calculate annualized volatility
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Calculate Sortino ratio (downside deviation)
        negative_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else np.nan
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
        
        # Calculate win rate
        win_rate = len(portfolio_returns[portfolio_returns > 0]) / len(portfolio_returns)
        
        # Calculate average win and loss
        avg_win = portfolio_returns[portfolio_returns > 0].mean()
        avg_loss = portfolio_returns[portfolio_returns < 0].mean()
        
        # Calculate profit factor
        profit_factor = (portfolio_returns[portfolio_returns > 0].sum() / 
                        abs(portfolio_returns[portfolio_returns < 0].sum())) if portfolio_returns[portfolio_returns < 0].sum() != 0 else np.nan
        
        # Create a dictionary with the results
        metrics = {
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
        
        return metrics
    
    def plot_portfolio_performance(self, initial_investment=10000, benchmark_ticker=None):
        """Plot the performance of the portfolio.
        
        Args:
            initial_investment (float): Initial investment amount
            benchmark_ticker (str, optional): Ticker of the benchmark to compare with
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Calculate portfolio value
        portfolio_value = self.calculate_cumulative_returns(initial_investment)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot portfolio value
        ax.plot(portfolio_value.index, portfolio_value, label='Portfolio', linewidth=2)
        
        # Plot benchmark if provided
        if benchmark_ticker is not None and benchmark_ticker in self.portfolio_data.columns:
            # Calculate benchmark returns
            benchmark_returns = self.portfolio_data[benchmark_ticker].pct_change().dropna()
            benchmark_value = initial_investment * (1 + benchmark_returns).cumprod()
            
            # Plot benchmark value
            ax.plot(benchmark_value.index, benchmark_value, label=f'Benchmark ({benchmark_ticker})', linewidth=2, alpha=0.7)
        
        # Add labels and legend
        ax.set_title('Portfolio Performance')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f'₹{int(x):,}'))
        
        plt.tight_layout()
        plt.savefig('portfolio_performance.png')
        plt.close()
        
        print("Portfolio performance plot saved to portfolio_performance.png")
        
        return fig
    
    def plot_drawdown(self):
        """Plot the drawdown of the portfolio.
        
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Calculate daily portfolio returns
        portfolio_returns = self.calculate_portfolio_returns()
        
        # Calculate drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot drawdown
        ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax.plot(drawdown.index, drawdown, color='red', alpha=0.5)
        
        # Add labels
        ax.set_title('Portfolio Drawdown')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f'{x:.1%}'))
        
        plt.tight_layout()
        plt.savefig('portfolio_drawdown.png')
        plt.close()
        
        print("Portfolio drawdown plot saved to portfolio_drawdown.png")
        
        return fig
    
    def print_performance_report(self, initial_investment=10000):
        """Print a performance report for the portfolio.
        
        Args:
            initial_investment (float): Initial investment amount
        """
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics()
        
        # Calculate portfolio value
        portfolio_value = self.calculate_cumulative_returns(initial_investment)
        final_value = portfolio_value.iloc[-1]
        total_return = (final_value / initial_investment) - 1
        
        # Calculate investment period
        start_date = portfolio_value.index[0]
        end_date = portfolio_value.index[-1]
        investment_period = (end_date - start_date).days / 365.25  # in years
        
        # Print report
        print("\n" + "=" * 50)
        print("PORTFOLIO PERFORMANCE REPORT".center(50))
        print("=" * 50)
        
        print(f"\nInvestment Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({investment_period:.2f} years)")
        print(f"Initial Investment: ₹{initial_investment:,.2f}")
        print(f"Final Value: ₹{final_value:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        
        print("\nRISK METRICS:")
        print(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        
        print("\nTRADING METRICS:")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Average Win: {metrics['avg_win']:.2%}")
        print(f"Average Loss: {metrics['avg_loss']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        
        print("\nPORTFOLIO ALLOCATION:")
        sorted_weights = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        for ticker, weight in sorted_weights:
            print(f"{ticker}: {weight:.2%}")
        
        print("\n" + "=" * 50)
        
        # Save report to file
        with open('portfolio_performance_report.txt', 'w') as f:
            f.write("PORTFOLIO PERFORMANCE REPORT\n\n")
            f.write(f"Investment Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({investment_period:.2f} years)\n")
            f.write(f"Initial Investment: ₹{initial_investment:,.2f}\n")
            f.write(f"Final Value: ₹{final_value:,.2f}\n")
            f.write(f"Total Return: {total_return:.2%}\n")
            f.write(f"Annualized Return: {metrics['annualized_return']:.2%}\n\n")
            
            f.write("RISK METRICS:\n")
            f.write(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}\n")
            f.write(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}\n")
            f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n")
            f.write(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}\n")
            f.write(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}\n\n")
            
            f.write("TRADING METRICS:\n")
            f.write(f"Win Rate: {metrics['win_rate']:.2%}\n")
            f.write(f"Average Win: {metrics['avg_win']:.2%}\n")
            f.write(f"Average Loss: {metrics['avg_loss']:.2%}\n")
            f.write(f"Profit Factor: {metrics['profit_factor']:.2f}\n\n")
            
            f.write("PORTFOLIO ALLOCATION:\n")
            for ticker, weight in sorted_weights:
                f.write(f"{ticker}: {weight:.2%}\n")
        
        print("Performance report saved to portfolio_performance_report.txt")

def main():
    """Main function to demonstrate the portfolio backtester."""
    try:
        # Load the portfolio data
        portfolio_data = pd.read_csv('portfolio_data.csv', index_col='Date', parse_dates=True)
        
        # Check if optimal weights file exists
        if os.path.exists('optimal_portfolio_weights.csv'):
            # Initialize the backtester with optimal weights
            backtester = PortfolioBacktester(portfolio_data, weights_file='optimal_portfolio_weights.csv')
            print("Using optimal portfolio weights from MPT optimization.")
        else:
            # Initialize the backtester with equal weights
            backtester = PortfolioBacktester(portfolio_data)
            print("Optimal weights file not found. Using equal weights for all stocks.")
        
        # Plot portfolio performance
        backtester.plot_portfolio_performance(initial_investment=100000, benchmark_ticker='NIFTY50.NS')
        
        # Plot drawdown
        backtester.plot_drawdown()
        
        # Print performance report
        backtester.print_performance_report(initial_investment=100000)
        
        print("\nPortfolio backtesting completed successfully!")
    except Exception as e:
        print(f"Error in portfolio backtesting: {e}")

if __name__ == "__main__":
    import os
    main()