import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

class PortfolioOptimizer:
    """Class for Modern Portfolio Theory optimization."""
    
    def __init__(self):
        """Initialize the portfolio optimizer."""
        self.prices_df = None
        self.returns_df = None
        self.mu = None
        self.S = None
        self.weights = None
        self.cleaned_weights = None
        self.latest_prices = None
        self.performance = {}
    
    def load_data(self):
        """Load stock data from the closing_prices_only folder.
        
        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        try:
            # Check if the directory exists
            if not os.path.exists('closing_prices_only'):
                print("Error: closing_prices_only directory not found.")
                return False
            
            # Get all CSV files in the directory
            csv_files = [f for f in os.listdir('closing_prices_only') if f.endswith('_closing.csv')]
            
            if not csv_files:
                print("Error: No CSV files found in closing_prices_only directory.")
                return False
            
            # Initialize an empty DataFrame
            combined_df = None
            
            # Process each CSV file
            for file in csv_files:
                # Extract ticker from filename
                ticker = file.split('_closing.csv')[0]
                
                # Read the CSV file
                file_path = os.path.join('closing_prices_only', file)
                df = pd.read_csv(file_path)
                
                # Ensure Date is in datetime format
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Rename Close column to ticker name
                df = df.rename(columns={'Close': ticker})
                
                if combined_df is None:
                    combined_df = df
                else:
                    # Merge with the combined DataFrame
                    combined_df = pd.merge(combined_df, df, on='Date', how='outer')
            
            # Drop rows with missing values
            combined_df.dropna(inplace=True)
            
            # Check if the DataFrame is empty after dropping NaN values
            if combined_df.empty:
                print("Error: DataFrame is empty after dropping missing values.")
                return False
            
            # Set Date as index
            combined_df.set_index('Date', inplace=True)
            
            # Sort by date
            combined_df.sort_index(inplace=True)
            
            # Store the prices DataFrame
            self.prices_df = combined_df
            
            # Store the latest prices for allocation
            self.latest_prices = self.prices_df.iloc[-1]
            
            print(f"Successfully loaded data for {len(combined_df.columns)} stocks.")
            print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
            
            return True
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def calculate_inputs(self):
        """Calculate the expected returns and covariance matrix.
        
        Returns:
            bool: True if inputs were calculated successfully, False otherwise
        """
        try:
            if self.prices_df is None:
                print("Error: No data loaded. Call load_data() first.")
                return False
            
            # Calculate daily returns
            self.returns_df = self.prices_df.pct_change().dropna()
            
            # Calculate expected returns (mu) - annualized
            self.mu = self.returns_df.mean() * 252
            
            # Calculate covariance matrix (S) - annualized
            self.S = self.returns_df.cov() * 252
            
            print("Calculated expected returns and covariance matrix.")
            return True
        
        except Exception as e:
            print(f"Error calculating inputs: {e}")
            return False
    
    def portfolio_performance(self, weights):
        """Calculate portfolio performance metrics.
        
        Args:
            weights (np.array): Portfolio weights
            
        Returns:
            tuple: (expected return, volatility, Sharpe ratio)
        """
        # Expected portfolio return
        returns = np.sum(self.mu * weights)
        
        # Expected portfolio volatility
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.S, weights)))
        
        # Sharpe ratio (assuming risk-free rate of 0.02)
        sharpe = (returns - 0.02) / volatility
        
        return returns, volatility, sharpe
    
    def negative_sharpe(self, weights):
        """Calculate the negative Sharpe ratio (for minimization).
        
        Args:
            weights (np.array): Portfolio weights
            
        Returns:
            float: Negative Sharpe ratio
        """
        returns, volatility, sharpe = self.portfolio_performance(weights)
        return -sharpe
    
    def portfolio_volatility(self, weights):
        """Calculate portfolio volatility.
        
        Args:
            weights (np.array): Portfolio weights
            
        Returns:
            float: Portfolio volatility
        """
        return np.sqrt(np.dot(weights.T, np.dot(self.S, weights)))
    
    def optimize_portfolio(self, risk_free_rate=0.02):
        """Optimize the portfolio for maximum Sharpe ratio.
        
        Args:
            risk_free_rate (float): Risk-free rate
            
        Returns:
            bool: True if portfolio was optimized successfully, False otherwise
        """
        try:
            if self.mu is None or self.S is None:
                print("Error: Inputs not calculated. Call calculate_inputs() first.")
                return False
            
            # Number of assets
            n_assets = len(self.mu)
            
            # Initial guess (equal weights)
            init_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Constraints
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # Bounds (no short selling)
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Optimize for maximum Sharpe ratio
            result = minimize(
                self.negative_sharpe,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            # Get the optimized weights
            self.weights = result['x']
            
            # Clean the weights (remove very small weights)
            self.cleaned_weights = {}
            for i, ticker in enumerate(self.prices_df.columns):
                if self.weights[i] > 0.01:  # Only include weights > 1%
                    self.cleaned_weights[ticker] = self.weights[i]
            
            # Normalize the cleaned weights to sum to 1
            total_weight = sum(self.cleaned_weights.values())
            for ticker in self.cleaned_weights:
                self.cleaned_weights[ticker] /= total_weight
            
            # Get performance metrics
            self.performance = self.portfolio_performance(self.weights)
            
            # Save weights to CSV
            pd.Series(self.cleaned_weights).to_csv('optimized_weights.csv', header=['Weight'])
            
            return True
        
        except Exception as e:
            print(f"Error optimizing portfolio: {e}")
            return False
    
    def get_min_volatility_portfolio(self, risk_free_rate=0.02):
        """Get the minimum volatility portfolio.
        
        Args:
            risk_free_rate (float): Risk-free rate
            
        Returns:
            tuple: (weights, performance)
        """
        try:
            if self.mu is None or self.S is None:
                print("Error: Inputs not calculated. Call calculate_inputs() first.")
                return None, None
            
            # Number of assets
            n_assets = len(self.mu)
            
            # Initial guess (equal weights)
            init_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Constraints
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # Bounds (no short selling)
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Optimize for minimum volatility
            result = minimize(
                self.portfolio_volatility,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            # Get the optimized weights
            min_vol_weights = result['x']
            
            # Clean the weights (remove very small weights)
            cleaned_weights = {}
            for i, ticker in enumerate(self.prices_df.columns):
                if min_vol_weights[i] > 0.01:  # Only include weights > 1%
                    cleaned_weights[ticker] = min_vol_weights[i]
            
            # Normalize the cleaned weights to sum to 1
            total_weight = sum(cleaned_weights.values())
            for ticker in cleaned_weights:
                cleaned_weights[ticker] /= total_weight
            
            # Get performance metrics
            performance = self.portfolio_performance(min_vol_weights)
            
            return cleaned_weights, performance
        
        except Exception as e:
            print(f"Error getting minimum volatility portfolio: {e}")
            return None, None
    
    def get_efficient_frontier(self, points=100):
        """Calculate the efficient frontier.
        
        Args:
            points (int): Number of points to calculate
            
        Returns:
            tuple: (returns, volatilities)
        """
        try:
            if self.mu is None or self.S is None:
                print("Error: Inputs not calculated. Call calculate_inputs() first.")
                return None, None
            
            # Number of assets
            n_assets = len(self.mu)
            
            # Target returns
            target_returns = np.linspace(self.mu.min(), self.mu.max(), points)
            
            # Efficient frontier volatilities
            volatilities = []
            
            # Initial guess (equal weights)
            init_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Bounds (no short selling)
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Calculate efficient frontier
            for target_return in target_returns:
                # Constraints
                constraints = (
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: np.sum(self.mu * x) - target_return}
                )
                
                # Optimize for minimum volatility at target return
                result = minimize(
                    self.portfolio_volatility,
                    init_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                # Get the volatility
                volatilities.append(self.portfolio_volatility(result['x']))
            
            return target_returns, np.array(volatilities)
        
        except Exception as e:
            print(f"Error calculating efficient frontier: {e}")
            return None, None
    
    def get_random_portfolios(self, n_portfolios=10000):
        """Generate random portfolios for plotting.
        
        Args:
            n_portfolios (int): Number of portfolios to generate
            
        Returns:
            tuple: (returns, volatilities, sharpe_ratios)
        """
        try:
            if self.mu is None or self.S is None:
                print("Error: Inputs not calculated. Call calculate_inputs() first.")
                return None, None, None
            
            # Number of assets
            n_assets = len(self.mu)
            
            # Arrays to store results
            returns = np.zeros(n_portfolios)
            volatilities = np.zeros(n_portfolios)
            sharpe_ratios = np.zeros(n_portfolios)
            
            # Generate random portfolios
            for i in range(n_portfolios):
                # Generate random weights
                weights = np.random.random(n_assets)
                weights /= np.sum(weights)
                
                # Calculate portfolio performance
                returns[i], volatilities[i], sharpe_ratios[i] = self.portfolio_performance(weights)
            
            return returns, volatilities, sharpe_ratios
        
        except Exception as e:
            print(f"Error generating random portfolios: {e}")
            return None, None, None
    
    def get_discrete_allocation(self, investment_amount, weights=None):
        """Get the discrete allocation of shares for a given investment amount.
        
        Args:
            investment_amount (float): The amount to invest
            weights (dict, optional): Custom weights to use. If None, uses self.cleaned_weights
            
        Returns:
            tuple: (allocation, leftover)
        """
        try:
            if weights is None:
                if self.cleaned_weights is None:
                    print("Error: Portfolio not optimized. Call optimize_portfolio() first.")
                    return None, None
                weights = self.cleaned_weights
            
            if self.latest_prices is None:
                print("Error: Latest prices not available.")
                return None, None
            
            # Calculate the allocation
            allocation = {}
            leftover = investment_amount
            
            # Sort weights in descending order
            sorted_weights = sorted(
                weights.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Allocate shares
            for ticker, weight in sorted_weights:
                price = self.latest_prices[ticker]
                target_value = weight * investment_amount
                shares = int(target_value / price)  # Whole shares only
                
                if shares > 0:
                    allocation[ticker] = shares
                    leftover -= shares * price
            
            return allocation, leftover
        
        except Exception as e:
            print(f"Error getting discrete allocation: {e}")
            return None, None

    def get_efficient_risk_portfolio(self, target_risk):
        """Get the portfolio with the highest return for a given risk level.
        
        Args:
            target_risk (float): The target risk (volatility)
            
        Returns:
            tuple: (weights, performance)
        """
        try:
            if self.mu is None or self.S is None:
                print("Error: Inputs not calculated. Call calculate_inputs() first.")
                return None, None
            
            # Number of assets
            n_assets = len(self.mu)
            
            # Initial guess (equal weights)
            init_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Define the objective function (negative expected return)
            def negative_return(weights):
                return -np.sum(self.mu * weights)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
                {'type': 'eq', 'fun': lambda x: self.portfolio_volatility(x) - target_risk}  # Target volatility
            ]
            
            # Bounds (no short selling)
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Check if the target risk is feasible
            min_vol_weights, min_vol_performance = self.get_min_volatility_portfolio()
            min_vol = min_vol_performance[1]  # Volatility of min vol portfolio
            
            # Get the volatility of the maximum return portfolio (100% in highest return asset)
            max_return_idx = np.argmax(self.mu)
            max_return_weights = np.zeros(n_assets)
            max_return_weights[max_return_idx] = 1
            max_vol = self.portfolio_volatility(max_return_weights)
            
            # Check if target risk is within feasible range
            if target_risk < min_vol or target_risk > max_vol:
                print(f"Target risk {target_risk:.2%} is outside the feasible range [{min_vol:.2%}, {max_vol:.2%}]")
                return None, None
            
            # Optimize for maximum return at target risk
            result = minimize(
                negative_return,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if not result['success']:
                print(f"Optimization failed: {result['message']}")
                return None, None
            
            # Get the optimized weights
            weights = result['x']
            
            # Clean the weights (remove very small weights)
            cleaned_weights = {}
            for i, ticker in enumerate(self.prices_df.columns):
                if weights[i] > 0.01:  # Only include weights > 1%
                    cleaned_weights[ticker] = weights[i]
            
            # Normalize the cleaned weights to sum to 1
            total_weight = sum(cleaned_weights.values())
            for ticker in cleaned_weights:
                cleaned_weights[ticker] /= total_weight
            
            # Get performance metrics
            performance = self.portfolio_performance(weights)
            
            return cleaned_weights, performance
        
        except Exception as e:
            print(f"Error getting efficient risk portfolio: {e}")
            return None, None
    
    def get_efficient_return_portfolio(self, target_return):
        """Get the portfolio with the lowest risk for a given return level.
        
        Args:
            target_return (float): The target return
            
        Returns:
            tuple: (weights, performance)
        """
        try:
            if self.mu is None or self.S is None:
                print("Error: Inputs not calculated. Call calculate_inputs() first.")
                return None, None
            
            # Number of assets
            n_assets = len(self.mu)
            
            # Initial guess (equal weights)
            init_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
                {'type': 'eq', 'fun': lambda x: np.sum(self.mu * x) - target_return}  # Target return
            ]
            
            # Bounds (no short selling)
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Check if the target return is feasible
            min_return = np.min(self.mu)
            max_return = np.max(self.mu)
            
            # Check if target return is within feasible range
            if target_return < min_return or target_return > max_return:
                print(f"Target return {target_return:.2%} is outside the feasible range [{min_return:.2%}, {max_return:.2%}]")
                return None, None
            
            # Optimize for minimum volatility at target return
            result = minimize(
                self.portfolio_volatility,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if not result['success']:
                print(f"Optimization failed: {result['message']}")
                return None, None
            
            # Get the optimized weights
            weights = result['x']
            
            # Clean the weights (remove very small weights)
            cleaned_weights = {}
            for i, ticker in enumerate(self.prices_df.columns):
                if weights[i] > 0.01:  # Only include weights > 1%
                    cleaned_weights[ticker] = weights[i]
            
            # Normalize the cleaned weights to sum to 1
            total_weight = sum(cleaned_weights.values())
            for ticker in cleaned_weights:
                cleaned_weights[ticker] /= total_weight
            
            # Get performance metrics
            performance = self.portfolio_performance(weights)
            
            return cleaned_weights, performance
        
        except Exception as e:
            print(f"Error getting efficient return portfolio: {e}")
            return None, None

# Example usage
def main():
    # Initialize the optimizer
    optimizer = PortfolioOptimizer()
    
    # Load the data
    if not optimizer.load_data():
        return
    
    # Calculate inputs
    if not optimizer.calculate_inputs():
        return
    
    # Optimize the portfolio
    if not optimizer.optimize_portfolio():
        return
    
    # Get the minimum volatility portfolio
    min_vol_weights, min_vol_performance = optimizer.get_min_volatility_portfolio()
    
    # Print the results
    print("\nOptimized portfolio weights:")
    for ticker, weight in optimizer.cleaned_weights.items():
        print(f"{ticker}: {weight:.4f}")
    
    print("\nMinimum volatility portfolio weights:")
    if min_vol_weights:
        for ticker, weight in min_vol_weights.items():
            print(f"{ticker}: {weight:.4f}")
    
    # Example investment allocation
    investment_amount = 100000  # ₹1,00,000
    allocation, leftover = optimizer.get_discrete_allocation(investment_amount)
    
    if allocation:
        print(f"\nDiscrete allocation for ₹{investment_amount:,.2f}:")
        for ticker, shares in allocation.items():
            price = optimizer.latest_prices[ticker]
            cost = shares * price
            print(f"{ticker}: {shares} shares (₹{cost:,.2f})")
        print(f"Leftover: ₹{leftover:,.2f}")

if __name__ == "__main__":
    main()