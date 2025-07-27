import os
import sys
import time

def print_header(message):
    """Print a formatted header message.
    
    Args:
        message (str): The message to print
    """
    print("\n" + "=" * 80)
    print(f" {message} ".center(80, "="))
    print("=" * 80 + "\n")

def run_script(script_name, description):
    """Run a Python script and handle any errors.
    
    Args:
        script_name (str): The name of the script to run
        description (str): A description of what the script does
    
    Returns:
        bool: True if the script ran successfully, False otherwise
    """
    print_header(f"RUNNING: {description}")
    
    try:
        # Use the Python executable that's running this script
        command = f"{sys.executable} {script_name}"
        exit_code = os.system(command)
        
        if exit_code == 0:
            print(f"\n✅ {script_name} completed successfully!")
            return True
        else:
            print(f"\n❌ {script_name} failed with exit code {exit_code}")
            return False
    except Exception as e:
        print(f"\n❌ Error running {script_name}: {e}")
        return False

def main():
    """Run the entire portfolio analysis workflow."""
    start_time = time.time()
    
    print_header("STOCK DATA PROCESSOR FOR MODERN PORTFOLIO THEORY")
    print("This script will run the entire workflow for stock data processing and MPT analysis.")
    print("The workflow consists of the following steps:")
    print("1. Download and process stock data")
    print("2. Apply Modern Portfolio Theory optimization")
    print("3. Visualize the portfolio data and results")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()
    
    # Step 1: Download and process stock data
    if not run_script("stock_data_processor.py", "Downloading and Processing Stock Data"):
        print("\nCannot continue with the workflow due to errors in stock data processing.")
        return
    
    # Step 2: Apply Modern Portfolio Theory optimization
    if not run_script("mpt_optimizer.py", "Applying Modern Portfolio Theory Optimization"):
        print("\nCannot continue with the workflow due to errors in MPT optimization.")
        return
    
    # Step 3: Visualize the portfolio data and results
    if not run_script("visualize_portfolio.py", "Visualizing Portfolio Data and Results"):
        print("\nVisualization failed, but the main analysis is complete.")
    
    # Calculate and display the total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    minutes, seconds = divmod(execution_time, 60)
    hours, minutes = divmod(minutes, 60)
    
    print_header("WORKFLOW COMPLETED")
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print("\nThe following files have been generated:")
    print("- raw_data/: Raw stock data for each ticker")
    print("- closing_prices_only/: Processed closing prices for each ticker")
    print("- portfolio_data.csv: Merged dataset with all stock closing prices")
    print("- correlation_matrix.png: Correlation matrix of stock returns")
    print("- efficient_frontier.png: Efficient frontier visualization")
    print("- optimal_portfolio_weights.csv: Optimal portfolio weights")
    print("- stock_prices.png: Stock closing prices over time")
    print("- normalized_prices.png: Normalized stock prices over time")
    print("- returns_distribution.png: Distribution of daily returns")
    print("- cumulative_returns.png: Cumulative returns over time")
    print("- rolling_volatility.png: Rolling volatility over time")
    print("- optimal_portfolio_weights.png: Visualization of optimal weights")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWorkflow cancelled by user.")
    except Exception as e:
        print(f"\n\nAn unexpected error occurred: {e}")