import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optimizer import PortfolioOptimizer
from visualizer import PortfolioVisualizer

def get_portfolio_type():
    """Present a menu to the user to select the type of portfolio.
    
    Returns:
        str: The selected portfolio type
    """
    while True:
        print("\n" + "=" * 50)
        print("WHAT KIND OF PORTFOLIO DO YOU WANT?")
        print("=" * 50)
        print("i.   Minimum Volatility")
        print("ii.  Maximum Sharpe Ratio")
        print("iii. Target a Specific Risk (volatility)")
        print("iv.  Target a Specific Return")
        print("=" * 50)
        
        choice = input("Enter your choice (i, ii, iii, or iv): ").strip().lower()
        
        if choice in ['i', 'ii', 'iii', 'iv']:
            return choice
        else:
            print("Invalid choice. Please try again.")

def get_target_value(prompt, min_value=0.01, max_value=1.0):
    """Get a target value from the user within a specified range.
    
    Args:
        prompt (str): The prompt to display to the user
        min_value (float): The minimum allowed value
        max_value (float): The maximum allowed value
        
    Returns:
        float: The target value
    """
    while True:
        try:
            value_str = input(prompt)
            value = float(value_str)
            
            if value < min_value or value > max_value:
                print(f"Please enter a value between {min_value:.2f} and {max_value:.2f}.")
                continue
            
            return value
        
        except ValueError:
            print("Invalid value. Please enter a numeric value.")

def get_investment_amount():
    """Get the investment amount from the user.
    
    Returns:
        float: The investment amount
    """
    while True:
        try:
            amount_str = input("Enter total amount you want to invest (e.g. 100000): ")
            
            # Remove currency symbols and commas
            amount_str = amount_str.replace('₹', '').replace('$', '').replace(',', '')
            
            # Convert to float
            amount = float(amount_str)
            
            if amount <= 0:
                print("Please enter a positive amount.")
                continue
            
            return amount
        
        except ValueError:
            print("Invalid amount. Please enter a numeric value.")

def print_investment_advice(allocation, latest_prices, investment_amount, leftover):
    """Print investment advice based on the allocation.
    
    Args:
        allocation (dict): Dictionary of allocation
        latest_prices (pd.Series): Series of latest prices
        investment_amount (float): Investment amount
        leftover (float): Leftover amount
    """
    print("\n" + "=" * 50)
    print(f"INVESTMENT ADVICE FOR ₹{investment_amount:,.2f}")
    print("=" * 50)
    
    print("\nBased on the optimized portfolio, here's how you should allocate your investment:")
    print("\nStock\tShares\tPrice\t\tCost\t\tWeight")
    print("-" * 70)
    
    total_cost = 0
    
    for ticker, shares in allocation.items():
        price = latest_prices[ticker]
        cost = shares * price
        total_cost += cost
        weight = cost / (investment_amount - leftover)
        
        print(f"{ticker}\t{shares}\t₹{price:,.2f}\t₹{cost:,.2f}\t{weight:.2%}")
    
    print("-" * 70)
    print(f"Total\t\t\t\t₹{total_cost:,.2f}\t100.00%")
    print(f"Leftover\t\t\t₹{leftover:,.2f}")
    print(f"Investment\t\t\t₹{investment_amount:,.2f}")
    print("\nNote: The allocation is optimized based on your selected portfolio type while staying within your investment amount.")

def handle_investment(optimizer, visualizer, weights, portfolio_type):
    """Handle the investment amount input and allocation.
    
    Args:
        optimizer (PortfolioOptimizer): The portfolio optimizer
        visualizer (PortfolioVisualizer): The portfolio visualizer
        weights (dict): The portfolio weights
        portfolio_type (str): The type of portfolio
        
    Returns:
        bool: True if the user wants to create another portfolio, False otherwise
    """
    # Get investment amount from user
    investment_amount = get_investment_amount()
    
    # Get discrete allocation
    allocation, leftover = optimizer.get_discrete_allocation(investment_amount, weights)
    
    if allocation:
        # Print investment advice
        print_investment_advice(allocation, optimizer.latest_prices, investment_amount, leftover)
        
        # Plot the allocation
        visualizer.plot_allocation(allocation, optimizer.latest_prices, investment_amount)
    
    # Ask if the user wants to create another portfolio for a different amount
    while True:
        repeat = input("\nDo you want to create another portfolio for a different amount? (yes/no): ").strip().lower()
        if repeat in ['yes', 'no', 'y', 'n']:
            return repeat in ['yes', 'y']
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

def main():
    print("\n" + "=" * 50)
    print("MODERN PORTFOLIO THEORY (MPT) ANALYSIS")
    print("=" * 50)
    
    # Initialize the optimizer and visualizer
    optimizer = PortfolioOptimizer()
    visualizer = PortfolioVisualizer()
    
    print("\nPhase 1: Loading and Processing Data")
    print("-" * 40)
    
    # Load the data
    if not optimizer.load_data():
        print("Failed to load data. Exiting.")
        return
    
    # Calculate inputs (mu and S)
    if not optimizer.calculate_inputs():
        print("Failed to calculate inputs. Exiting.")
        return
    
    # Plot correlation matrix
    visualizer.plot_correlation_matrix(optimizer.prices_df)
    
    # Plot returns histogram
    visualizer.plot_returns_histogram(optimizer.prices_df)
    
    # Plot cumulative returns
    visualizer.plot_cumulative_returns(optimizer.prices_df)
    
    print("\nPhase 2: Portfolio Optimization")
    print("-" * 40)
    
    # Get the portfolio type from the user
    portfolio_type = get_portfolio_type()
    
    # Initialize variables for portfolio weights and performance
    weights = None
    performance = None
    portfolio_name = ""
    
    # Optimize the portfolio based on the user's choice
    if portfolio_type == 'i':  # Minimum Volatility
        weights, performance = optimizer.get_min_volatility_portfolio()
        portfolio_name = "Minimum Volatility Portfolio"
    
    elif portfolio_type == 'ii':  # Maximum Sharpe Ratio
        if not optimizer.optimize_portfolio():
            print("Failed to optimize portfolio. Exiting.")
            return
        weights = optimizer.cleaned_weights
        performance = optimizer.performance
        portfolio_name = "Maximum Sharpe Ratio Portfolio"
    
    elif portfolio_type == 'iii':  # Target Risk
        # Get the target risk from the user
        target_risk = get_target_value("Enter target volatility (e.g., 0.15 for 15%): ", 0.01, 0.5)
        
        # Get the efficient risk portfolio
        weights, performance = optimizer.get_efficient_risk_portfolio(target_risk)
        
        if weights is None:
            print("❌ Portfolio with given risk value not possible with selected stocks.")
            return main()  # Return to the main menu
        
        portfolio_name = f"Target Risk ({target_risk:.2%}) Portfolio"
    
    elif portfolio_type == 'iv':  # Target Return
        # Get the target return from the user
        target_return = get_target_value("Enter target annual return (e.g., 0.10 for 10%): ", 0.01, 0.5)
        
        # Get the efficient return portfolio
        weights, performance = optimizer.get_efficient_return_portfolio(target_return)
        
        if weights is None:
            print("❌ Portfolio with given return value not possible with selected stocks.")
            return main()  # Return to the main menu
        
        portfolio_name = f"Target Return ({target_return:.2%}) Portfolio"
    
    # Print the results
    print(f"\n{portfolio_name}:")
    if performance:
        expected_return, volatility, sharpe = performance
        print(f"Expected Annual Return: {expected_return:.2%}")
        print(f"Annual Volatility: {volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
    
    # Plot the weights
    if weights:
        visualizer.plot_weights(weights, portfolio_name)
    
    # Plot the efficient frontier
    visualizer.plot_efficient_frontier(optimizer)
    
    print("\nPhase 3: Investment Advice")
    print("-" * 40)
    
    # Handle investment amount and allocation
    repeat = True
    while repeat:
        repeat = handle_investment(optimizer, visualizer, weights, portfolio_type)
    
    print("\nThank you for using the MPT Analysis tool! All plots have been saved to the 'plots' directory.")

if __name__ == "__main__":
    main()