import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# Create necessary directories
def create_directories():
    """Create the necessary directories for the project."""
    directories = ['raw_data', 'closing_prices_only']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

# PHASE 1: Data Download
def download_stock_data(tickers, start_date, end_date):
    """Download historical stock data for the given tickers.
    
    Args:
        tickers (list): List of stock tickers
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    """
    print("\nPHASE 1: Downloading historical stock data...")
    
    for ticker in tqdm(tickers, desc="Downloading stock data"):
        try:
            # Download data
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            
            # Save to CSV
            file_path = os.path.join('raw_data', f'historical_data_of_{ticker}.csv')
            stock_data.to_csv(file_path)
            print(f"Downloaded data for {ticker} and saved to {file_path}")
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")

# PHASE 2: Data Processing
def process_stock_data(tickers):
    """Process the downloaded stock data to extract closing prices.
    
    Args:
        tickers (list): List of stock tickers
    """
    print("\nPHASE 2: Processing stock data...")
    
    for ticker in tqdm(tickers, desc="Processing stock data"):
        try:
            # Read the raw data
            file_path = os.path.join('raw_data', f'historical_data_of_{ticker}.csv')
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist. Skipping {ticker}.")
                continue
                
            # Read the CSV file with custom handling for the yfinance format
            # Skip the first 3 rows which contain header information
            df = pd.read_csv(file_path, skiprows=3)
            
            # Check if the dataframe is empty or has no rows
            if df.empty or len(df) == 0:
                print(f"Warning: No data found in {file_path}. Skipping {ticker}.")
                continue
            
            # The first column should be the date
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
            
            # Ensure Date is in datetime format
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Extract only Date and Close columns if Close exists
            if 'Close' in df.columns:
                close_col = 'Close'
            elif df.columns[1] == '0':
                # If columns are unnamed, assume second column is Close
                close_col = df.columns[1]
                df.rename(columns={close_col: 'Close'}, inplace=True)
                close_col = 'Close'
            else:
                # Try to find a column that might contain close prices
                potential_close_cols = [col for col in df.columns if 'clos' in col.lower()]
                if potential_close_cols:
                    close_col = potential_close_cols[0]
                    df.rename(columns={close_col: 'Close'}, inplace=True)
                    close_col = 'Close'
                else:
                    # If no close column found, use the second column
                    close_col = df.columns[1]
                    df.rename(columns={close_col: 'Close'}, inplace=True)
                    close_col = 'Close'
            
            # Create a clean dataframe with just Date and Close
            df_clean = df[['Date', 'Close']]
            
            # Handle missing values
            df_clean = df_clean.dropna()
            
            # Create the output directory if it doesn't exist
            if not os.path.exists('closing_prices_only'):
                os.makedirs('closing_prices_only')
            
            # Save to CSV
            output_path = os.path.join('closing_prices_only', f'{ticker}_closing.csv')
            df_clean.to_csv(output_path, index=False)
            print(f"Processed data for {ticker} and saved to {output_path}")
        except Exception as e:
            print(f"Error processing data for {ticker}: {e}")

# PHASE 3: Portfolio Dataset Construction
def construct_portfolio_dataset(tickers):
    """Merge all cleaned CSVs into a single DataFrame.
    
    Args:
        tickers (list): List of stock tickers
    
    Returns:
        pd.DataFrame: Merged DataFrame with all stock closing prices
    """
    print("\nPHASE 3: Constructing portfolio dataset...")
    
    # Initialize an empty DataFrame
    portfolio_df = None
    successful_tickers = []
    
    for ticker in tqdm(tickers, desc="Merging stock data"):
        try:
            # Read the cleaned data
            file_path = os.path.join('closing_prices_only', f'{ticker}_closing.csv')
            
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist. Skipping {ticker}.")
                continue
                
            df = pd.read_csv(file_path)
            
            # Ensure Date is in datetime format
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Rename Close column to ticker name
            df = df.rename(columns={'Close': ticker})
            
            if portfolio_df is None:
                portfolio_df = df
                successful_tickers.append(ticker)
            else:
                # Merge with the portfolio DataFrame
                portfolio_df = pd.merge(portfolio_df, df, on='Date', how='outer')
                successful_tickers.append(ticker)
        except Exception as e:
            print(f"Error merging data for {ticker}: {e}")
    
    if portfolio_df is None or len(successful_tickers) == 0:
        print("Error: No data could be merged. Check if any CSV files were created in Phase 2.")
        return None
    
    print(f"Successfully merged data for {len(successful_tickers)} stocks.")
    
    # Drop rows with missing values
    portfolio_df.dropna(inplace=True)
    
    # Set Date as index
    portfolio_df.set_index('Date', inplace=True)
    
    # Filter out future dates
    current_date = datetime.now().date()
    print(f"Current date: {current_date}")
    print(f"Min date in DataFrame: {portfolio_df.index.min().date()}")
    print(f"Max date in DataFrame: {portfolio_df.index.max().date()}")
    print(f"Number of rows before filtering: {len(portfolio_df)}")
    
    # Convert index to datetime if it's not already
    portfolio_df.index = pd.to_datetime(portfolio_df.index)
    
    # Filter out future dates
    portfolio_df = portfolio_df[portfolio_df.index.date <= current_date]
    
    print(f"Number of rows after filtering: {len(portfolio_df)}")
    print(f"Min date in filtered DataFrame: {portfolio_df.index.min().date() if not portfolio_df.empty else 'N/A'}")
    print(f"Max date in filtered DataFrame: {portfolio_df.index.max().date() if not portfolio_df.empty else 'N/A'}")
    
    # Save to CSV
    portfolio_df.to_csv('portfolio_data.csv')
    print(f"Portfolio dataset constructed and saved to portfolio_data.csv")
    
    return portfolio_df

# PHASE 4: Ready for MPT
def prepare_for_mpt(portfolio_df):
    """Prepare the portfolio data for Modern Portfolio Theory analysis.
    
    Args:
        portfolio_df (pd.DataFrame): Portfolio DataFrame with all stock closing prices
    """
    print("\nPHASE 4: Preparing data for Modern Portfolio Theory...")
    
    # Calculate daily returns
    daily_returns = portfolio_df.pct_change().dropna()
    
    # Calculate expected returns (mean of daily returns)
    expected_returns = daily_returns.mean()
    
    # Calculate covariance matrix
    cov_matrix = daily_returns.cov()
    
    print("Data is now ready for Modern Portfolio Theory analysis!")
    print(f"Number of stocks: {len(portfolio_df.columns)}")
    print(f"Date range: {portfolio_df.index.min()} to {portfolio_df.index.max()}")
    
    # Display expected returns
    print("\nExpected Returns:")
    print(expected_returns)
    
    # Plot correlation matrix heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = daily_returns.corr()
    plt.matshow(correlation_matrix, fignum=1)
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.colorbar()
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    print("\nCorrelation matrix heatmap saved to correlation_matrix.png")
    
    return daily_returns, expected_returns, cov_matrix

def main():
    # Define NIFTY FIFTY NSE stocks
    nifty_fifty_stocks = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'ITC.NS',
        'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'BAJFINANCE.NS',
        'HCLTECH.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS',
        'NESTLEIND.NS', 'TATAMOTORS.NS', 'ADANIPORTS.NS', 'POWERGRID.NS', 'NTPC.NS',
        'TECHM.NS', 'BAJAJFINSV.NS', 'HDFCLIFE.NS', 'DIVISLAB.NS', 'ONGC.NS',
        'JSWSTEEL.NS', 'TATASTEEL.NS', 'COALINDIA.NS', 'GRASIM.NS', 'SBILIFE.NS',
        'INDUSINDBK.NS', 'DRREDDY.NS', 'BPCL.NS', 'BRITANNIA.NS', 'M&M.NS',
        'CIPLA.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS', 'ADANIENT.NS', 'HINDALCO.NS',
        'UPL.NS', 'TATACONSUM.NS', 'APOLLOHOSP.NS', 'BAJAJ-AUTO.NS', 'SHREECEM.NS',
        'SUZLON.NS', 'JPPOWER.NS',
        'RVNL.NS', 'BEL.NS', 'RAILTEL.NS', 'NTPC.NS', 'PNB.NS', 'IDEA.NS'
    ]
    
    # Define date range
    start_date = '2015-01-01'
    #end_date = datetime.now().strftime('%Y-%m-%d')  # Use current date instead of future date
    end_date ='2024-12-31'
    # Create directories
    create_directories()
    
    # PHASE 1: Download stock data
    download_stock_data(nifty_fifty_stocks, start_date, end_date)
    
    # PHASE 2: Process stock data
    process_stock_data(nifty_fifty_stocks)
    
    # PHASE 3: Construct portfolio dataset
    portfolio_df = construct_portfolio_dataset(nifty_fifty_stocks)
    
    # PHASE 4: Prepare for MPT
    daily_returns, expected_returns, cov_matrix = prepare_for_mpt(portfolio_df)
    
    print("\nAll phases completed successfully!")

if __name__ == "__main__":
    main()