import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
import sys
from optimizer import PortfolioOptimizer
from ml_backtester import MLBacktester

# Redirect output to a file
log_file = open('ml_backtesting_log.txt', 'w')
sys.stdout = log_file

# Modify the main function to directly run ML backtesting

def main():
    """Main function to run the portfolio optimizer with ML backtesting."""
    print("\n" + "=" * 60)
    print("MODERN PORTFOLIO THEORY (MPT) ANALYSIS".center(60))
    print("=" * 60 + "\n")
    
    # Check if portfolio_data.csv exists
    if not os.path.exists('portfolio_data.csv'):
        print("Error: portfolio_data.csv not found. Please run stock_data_processor.py first.")
        return
    
    # Run ML backtesting directly
    try:
        print("Phase 1: Loading and Processing Data")
        print("-" * 40)
        
        # Load the portfolio data to check its structure
        try:
            portfolio_data = pd.read_csv('portfolio_data.csv', index_col='Date', parse_dates=True)
            print(f"Portfolio data loaded successfully.")
            print(f"Shape: {portfolio_data.shape}")
            print(f"Columns: {portfolio_data.columns[:5]}... (and {len(portfolio_data.columns)-5} more)")
            print(f"Date range: {portfolio_data.index.min().strftime('%Y-%m-%d')} to {portfolio_data.index.max().strftime('%Y-%m-%d')}")
            print(f"First few rows:\n{portfolio_data.iloc[:3, :3]}")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            traceback.print_exc()
            return
        
        run_ml_backtesting()
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        traceback.print_exc()
    
    # Ask if user wants to run again
    # Restore stdout for user input
    sys.stdout = sys.__stdout__
    run_again = input("\nDo you want to run the program again? (y/n): ")
    # Redirect back to file
    sys.stdout = log_file
    
    if run_again.lower() == 'y':
        main()
    else:
        print("\nThank you for using the Portfolio Optimizer with ML Backtesting!")

def run_portfolio_optimization():
    """Run the traditional portfolio optimization using MPT."""
    try:
        print("\n" + "-" * 60)
        print("RUNNING TRADITIONAL PORTFOLIO OPTIMIZATION (MPT)".center(60))
        print("-" * 60 + "\n")
        
        # Initialize the portfolio optimizer
        optimizer = PortfolioOptimizer()
        
        # Load data
        optimizer.load_data()
        
        # Calculate inputs
        optimizer.calculate_inputs()
        
        # Optimize portfolio
        weights = optimizer.optimize_portfolio()
        
        # Get portfolio performance
        mu, sigma, sharpe = optimizer.portfolio_performance(weights)
        
        # Print results
        print("\nOptimized Portfolio (Maximum Sharpe Ratio):")
        print(f"Expected Annual Return: {mu:.2%}")
        print(f"Annual Volatility: {sigma:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        
        # Get discrete allocation
        # Restore stdout for user input
        sys.stdout = sys.__stdout__
        investment_amount = float(input("\nEnter investment amount (e.g., 10000): ") or "10000")
        # Redirect back to file
        sys.stdout = log_file
        
        allocation, leftover = optimizer.get_discrete_allocation(weights, investment_amount)
        
        print(f"\nDiscrete Allocation for ${investment_amount:.2f}:")
        for ticker, shares in allocation.items():
            print(f"{ticker}: {shares} shares (${shares * optimizer.latest_prices[ticker]:.2f})")
        print(f"Remaining cash: ${leftover:.2f}")
        
        print("\nOptimization completed successfully!")
        return optimizer
    
    except Exception as e:
        print(f"Error in portfolio optimization: {e}")
        return None

def run_ml_backtesting():
    """Run ML backtesting."""
    try:
        print("\n" + "-" * 60)
        print("RUNNING MACHINE LEARNING BACKTESTING".center(60))
        print("-" * 60 + "\n")
        
        # Use a specific ticker for testing
        ticker = "RELIANCE.NS"
        print(f"Using ticker: {ticker} for testing")
        
        # Initialize the ML backtester
        print("\nInitializing ML Backtester...")
        try:
            backtester = MLBacktester(ticker=ticker)
            print("ML Backtester initialized successfully.")
        except Exception as e:
            print(f"Error initializing ML Backtester: {str(e)}")
            traceback.print_exc()
            return None
        
        # Run the walk-forward backtest
        print("\nRunning walk-forward backtest...")
        try:
            results = backtester.run_walk_forward_backtest(initial_window_years=2, test_window_months=6)
            print("Walk-forward backtest completed.")
        except Exception as e:
            print(f"Error running walk-forward backtest: {str(e)}")
            traceback.print_exc()
            return None
        
        # Print information about the results
        print("\nBacktest results shape:", results.shape if hasattr(results, 'shape') else "No results")
        if hasattr(results, 'shape') and results.shape[0] > 0:
            print(f"Date range: {results.index.min().strftime('%Y-%m-%d')} to {results.index.max().strftime('%Y-%m-%d')}")
            print(f"Number of predictions: {results.shape[0]}")
            print(f"Columns: {results.columns.tolist()}")
            print(f"First few rows:\n{results.head()}")
        else:
            print("No valid results were generated from the backtest.")
            if hasattr(results, 'empty'):
                print(f"Results DataFrame is empty: {results.empty}")
        
        # Calculate strategy performance
        print("\nCalculating strategy performance...")
        try:
            performance = backtester.calculate_strategy_performance()
            print("Performance calculation completed successfully.")
            print(f"Performance metrics: {performance}")
        except Exception as e:
            print(f"Error calculating performance: {str(e)}")
            traceback.print_exc()
        
        # Plot results
        print("\nPlotting results...")
        try:
            backtester.plot_results()
            print("Plotting completed successfully.")
        except Exception as e:
            print(f"Error plotting results: {str(e)}")
            traceback.print_exc()
        
        # Print performance report
        print("\nGenerating performance report...")
        try:
            backtester.print_performance_report()
            print("Performance report generated successfully.")
        except Exception as e:
            print(f"Error generating performance report: {str(e)}")
            traceback.print_exc()
        
        # Ask if user wants to save results
        # Restore stdout for user input
        sys.stdout = sys.__stdout__
        save_choice = input("\nDo you want to save the prediction results to CSV? (y/n): ")
        # Redirect back to file
        sys.stdout = log_file
        
        if save_choice.lower() == 'y':
            if hasattr(backtester, 'combined_results') and not backtester.combined_results.empty:
                backtester.combined_results.to_csv('ml_prediction_results.csv')
                print("Prediction results saved to ml_prediction_results.csv")
            else:
                print("No results to save.")
        
        print("\nML backtesting completed successfully!")
        return backtester
    
    except Exception as e:
        print(f"\nError during ML backtesting: {str(e)}")
        traceback.print_exc()
        return None

# Fix the syntax error at the end of the file

def compare_strategies(optimizer, backtester):
    """Compare traditional MPT strategy with ML-based strategy."""
    if optimizer is None or backtester is None or not hasattr(backtester, 'combined_results'):
        print("Cannot compare strategies. Please run both optimization and ML backtesting first.")
        return
    
    try:
        print("\n" + "-" * 60)
        print("COMPARING MPT VS ML STRATEGIES".center(60))
        print("-" * 60 + "\n")
        
        # Get ML strategy performance
        ml_metrics = backtester.calculate_strategy_performance()
        ml_returns = ml_metrics['strategy_returns']
        
        # Calculate MPT strategy performance over the same period
        # This would require implementing a method to backtest the MPT strategy
        # For now, we'll just print the ML strategy performance
        
        print("ML Strategy Performance:")
        print(f"Total Return: {ml_metrics['total_return']:.2%}")
        print(f"Annualized Return: {ml_metrics['annualized_return']:.2%}")
        print(f"Volatility: {ml_metrics['volatility']:.2%}")
        print(f"Sharpe Ratio: {ml_metrics['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {ml_metrics['max_drawdown']:.2%}")
        
        print("\nComparison completed!")
    
    except Exception as e:
        print(f"Error comparing strategies: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    finally:
        # Close the log file and restore stdout
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"\nLog file created: ml_backtesting_log.txt")