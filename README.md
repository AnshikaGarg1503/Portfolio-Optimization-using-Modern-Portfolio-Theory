# Stock Data Processor for Modern Portfolio Theory

This project downloads, processes, and prepares historical stock data for applying Modern Portfolio Theory (MPT) for portfolio optimization. It focuses on the NIFTY FIFTY NSE stocks from the Indian market.

## Project Structure

```
.
├── stock_data_processor.py  # Main script
├── requirements.txt         # Required packages
├── raw_data/                # Directory for raw stock data
├── closing_prices_only/     # Directory for processed closing prices
├── portfolio_data.csv       # Final merged dataset
└── correlation_matrix.png   # Visualization of stock correlations
```

## Features

### PHASE 1: Data Download
- Downloads historical daily stock data from 1st Jan 2020 to current date for NIFTY FIFTY NSE stocks using `yfinance`
- Saves the full data for each stock into the `raw_data` folder

### PHASE 2: Data Processing
- Extracts only the `Date` and `Close` columns from each stock's data
- Ensures the "Date" column is parsed as `datetime`
- Cleans any missing data
- Saves the cleaned data to the `closing_prices_only` folder

### PHASE 3: Portfolio Dataset Construction
- Merges all cleaned CSVs into a single DataFrame
- Aligns all stocks by date using an outer join
- Drops dates with missing values
- Saves the merged DataFrame to `portfolio_data.csv`

### PHASE 4: Ready for MPT
- Prepares the final dataset for Modern Portfolio Theory analysis
- Calculates daily returns, expected returns, and covariance matrix
- Generates a correlation matrix visualization

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python stock_data_processor.py
```

The script will:
1. Create necessary directories
2. Download historical stock data
3. Process the data to extract closing prices
4. Construct a portfolio dataset
5. Prepare the data for Modern Portfolio Theory analysis

## Output

- `raw_data/`: Contains raw historical data for each stock
- `closing_prices_only/`: Contains processed closing prices for each stock
- `portfolio_data.csv`: Final dataset with all stocks' closing prices
- `correlation_matrix.png`: Visualization of stock correlations

## Modern Portfolio Theory (MPT)

The final dataset (`portfolio_data.csv`) is ready for applying Modern Portfolio Theory, which can be used to:
- Calculate optimal portfolio weights
- Determine the efficient frontier
- Find the optimal portfolio based on risk-return preferences

## License

MIT