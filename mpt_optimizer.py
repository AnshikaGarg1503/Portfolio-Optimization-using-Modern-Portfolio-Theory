import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class MPTOptimizer:
    """Class for Modern Portfolio Theory optimization."""
    
    def __init__(self, returns_df):
        """Initialize the MPT optimizer with a DataFrame of returns.
        
        Args:
            returns_df (pd.DataFrame): DataFrame of daily returns for each stock
        """
        self.returns_df = returns_df
        self.mean_returns = returns_df.mean()
        self.cov_matrix = returns_df.cov()
        self.num_assets = len(returns_df.columns)
        self.asset_names = returns_df.columns
    
    def portfolio_annualized_performance(self, weights):
        """Calculate annualized return and volatility for a portfolio.
        
        Args:
            weights (np.array): Portfolio weights
            
        Returns:
            tuple: (returns, volatility)
        """
        returns = np.sum(self.mean_returns * weights) * 252
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        return returns, volatility
    
    def negative_sharpe_ratio(self, weights, risk_free_rate=0.02):
        """Calculate the negative Sharpe ratio for a portfolio.
        
        Args:
            weights (np.array): Portfolio weights
            risk_free_rate (float): Risk-free rate
            
        Returns:
            float: Negative Sharpe ratio
        """
        returns, volatility = self.portfolio_annualized_performance(weights)
        return -(returns - risk_free_rate) / volatility
    
    def portfolio_volatility(self, weights):
        """Calculate the volatility of a portfolio.
        
        Args:
            weights (np.array): Portfolio weights
            
        Returns:
            float: Portfolio volatility
        """
        return self.portfolio_annualized_performance(weights)[1]
    
    def optimize_portfolio(self, risk_free_rate=0.02, constraint_set=(0, 1)):
        """Optimize the portfolio for maximum Sharpe ratio.
        
        Args:
            risk_free_rate (float): Risk-free rate
            constraint_set (tuple): Constraints on weights (min, max)
            
        Returns:
            dict: Optimized portfolio information
        """
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple(constraint_set for asset in range(self.num_assets))
        
        # Initial guess (equal weights)
        init_guess = np.array([1.0 / self.num_assets] * self.num_assets)
        
        # Optimize for maximum Sharpe ratio
        optimal_sharpe = minimize(
            self.negative_sharpe_ratio,
            init_guess,
            args=(risk_free_rate,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Get optimal weights
        optimal_weights = optimal_sharpe['x']
        
        # Calculate performance metrics
        optimal_returns, optimal_volatility = self.portfolio_annualized_performance(optimal_weights)
        optimal_sharpe_ratio = (optimal_returns - risk_free_rate) / optimal_volatility
        
        # Create a dictionary with the results
        optimal_portfolio = {
            'weights': dict(zip(self.asset_names, optimal_weights)),
            'returns': optimal_returns,
            'volatility': optimal_volatility,
            'sharpe_ratio': optimal_sharpe_ratio
        }
        
        return optimal_portfolio
    
    def efficient_frontier(self, target_returns):
        """Calculate the efficient frontier for a range of target returns.
        
        Args:
            target_returns (np.array): Array of target returns
            
        Returns:
            tuple: (target_returns, efficient_volatilities)
        """
        efficient_volatilities = []
        
        for target in target_returns:
            # Constraints
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: self.portfolio_annualized_performance(x)[0] - target}
            )
            bounds = tuple((0, 1) for asset in range(self.num_assets))
            
            # Initial guess (equal weights)
            init_guess = np.array([1.0 / self.num_assets] * self.num_assets)
            
            # Optimize for minimum volatility
            efficient_portfolio = minimize(
                self.portfolio_volatility,
                init_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            efficient_volatilities.append(self.portfolio_volatility(efficient_portfolio['x']))
        
        return target_returns, efficient_volatilities
    
    def plot_efficient_frontier(self, risk_free_rate=0.02, num_portfolios=100):
        """Plot the efficient frontier with the optimal portfolio.
        
        Args:
            risk_free_rate (float): Risk-free rate
            num_portfolios (int): Number of random portfolios to generate
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Generate random portfolios
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            returns, volatility = self.portfolio_annualized_performance(weights)
            results[0, i] = volatility
            results[1, i] = returns
            results[2, i] = (returns - risk_free_rate) / volatility
        
        # Get optimal portfolio
        optimal_portfolio = self.optimize_portfolio(risk_free_rate)
        
        # Calculate efficient frontier
        min_return = min(results[1])
        max_return = max(results[1])
        target_returns = np.linspace(min_return, max_return, 50)
        efficient_returns, efficient_volatilities = self.efficient_frontier(target_returns)
        
        # Plot results
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Plot random portfolios
        scatter = ax.scatter(
            results[0, :],
            results[1, :],
            c=results[2, :],
            cmap='YlGnBu',
            marker='o',
            s=10,
            alpha=0.3
        )
        
        # Plot efficient frontier
        ax.plot(efficient_volatilities, efficient_returns, 'r--', linewidth=3)
        
        # Plot optimal portfolio
        ax.scatter(
            optimal_portfolio['volatility'],
            optimal_portfolio['returns'],
            marker='*',
            color='r',
            s=500,
            label='Optimal Portfolio'
        )
        
        # Add labels and legend
        ax.set_title('Portfolio Optimization with the Efficient Frontier')
        ax.set_xlabel('Annualized Volatility')
        ax.set_ylabel('Annualized Returns')
        ax.legend()
        
        # Add colorbar
        plt.colorbar(scatter, label='Sharpe Ratio')
        
        plt.savefig('efficient_frontier.png')
        plt.close()
        
        return fig
    
    def save_optimal_portfolio(self, risk_free_rate=0.02):
        """Save the optimal portfolio to a CSV file.
        
        Args:
            risk_free_rate (float): Risk-free rate
            
        Returns:
            pd.DataFrame: DataFrame with optimal weights
        """
        optimal_portfolio = self.optimize_portfolio(risk_free_rate)
        
        # Create a DataFrame with the weights
        weights_df = pd.DataFrame({
            'Stock': list(optimal_portfolio['weights'].keys()),
            'Weight': list(optimal_portfolio['weights'].values())
        })
        
        # Sort by weight in descending order
        weights_df = weights_df.sort_values('Weight', ascending=False)
        
        # Save to CSV
        weights_df.to_csv('optimal_portfolio_weights.csv', index=False)
        
        print(f"Optimal portfolio saved to optimal_portfolio_weights.csv")
        print(f"Expected annual return: {optimal_portfolio['returns']:.4f}")
        print(f"Expected annual volatility: {optimal_portfolio['volatility']:.4f}")
        print(f"Sharpe ratio: {optimal_portfolio['sharpe_ratio']:.4f}")
        
        return weights_df

def main():
    """Main function to demonstrate the MPT optimizer."""
    try:
        # Load the portfolio data
        portfolio_data = pd.read_csv('portfolio_data.csv', index_col='Date', parse_dates=True)
        
        # Calculate daily returns
        daily_returns = portfolio_data.pct_change().dropna()
        
        # Initialize the MPT optimizer
        mpt = MPTOptimizer(daily_returns)
        
        # Optimize the portfolio
        optimal_portfolio = mpt.optimize_portfolio()
        
        # Plot the efficient frontier
        mpt.plot_efficient_frontier()
        
        # Save the optimal portfolio
        mpt.save_optimal_portfolio()
        
        print("\nModern Portfolio Theory analysis completed successfully!")
        print("Check 'efficient_frontier.png' for the visualization.")
        print("Check 'optimal_portfolio_weights.csv' for the optimal weights.")
    except Exception as e:
        print(f"Error in MPT analysis: {e}")

if __name__ == "__main__":
    main()