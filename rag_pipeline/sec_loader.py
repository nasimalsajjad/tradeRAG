import os
import argparse
from sec_edgar_downloader import Downloader
from datetime import datetime

# Define the root directory for storing raw SEC data
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'sec')

def download_filings(ticker: str, filing_type: str, start_date: str, end_date: str, email: str):
    """Downloads SEC filings for a given ticker and date range."""
    print(f"--- Starting download for {ticker}, {filing_type} ({start_date} to {end_date}) ---")

    # Define the base path where the library will create its structure
    download_path = os.path.join(RAW_DATA_DIR)
    print(f"Ensuring base download directory exists: {download_path}")
    os.makedirs(download_path, exist_ok=True) # Should already exist from previous runs if successful

    # Path where the library is expected to save files for this specific ticker/type
    expected_save_location = os.path.join(download_path, 'sec-edgar-filings', ticker, filing_type)
    print(f"Expected save location for library: {expected_save_location}")

    print(f"Initializing Downloader with email: {email}, download path: {download_path}")
    # The library requires a company name/email for the User-Agent string
    dl = Downloader("TradeRAG Project", email, download_path)

    try:
        print(f"Attempting to download using dl.get('{filing_type}', '{ticker}', after='{start_date}', before='{end_date}')...")
        # Download filings
        # Note: The library handles the date format check internally
        count = dl.get(filing_type, ticker, after=start_date, before=end_date)
        print(f"Download attempt finished. Reported count: {count}")
        if count > 0:
            print(f"Successfully downloaded {count} {filing_type} filings for {ticker}.")
            print(f"Check for filings in: {expected_save_location}")
        else:
            print(f"No {filing_type} filings found or downloaded for {ticker} in the specified date range.")

    except Exception as e:
        print(f"An error occurred during download for {ticker}: {e}")
    print(f"--- Finished download attempt for {ticker}, {filing_type} ---")


def main():
    parser = argparse.ArgumentParser(description="Download SEC filings (10-K, 10-Q) for specified tickers.")
    parser.add_argument("-t", "--tickers", required=True, nargs='+', help="List of stock tickers (e.g., AAPL MSFT GOOG).")
    parser.add_argument("-f", "--filing_types", nargs='+', default=["10-K", "10-Q"], help="List of filing types (e.g., 10-K 10-Q). Defaults to 10-K and 10-Q.")
    parser.add_argument("-s", "--start_date", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("-e", "--end_date", default=datetime.now().strftime('%Y-%m-%d'), help="End date in YYYY-MM-DD format (defaults to today).")
    parser.add_argument("--email", required=True, help="Your email address (required by SEC EDGAR).")

    args = parser.parse_args()

    # Validate dates (basic check)
    try:
        datetime.strptime(args.start_date, '%Y-%m-%d')
        datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError:
        print("Error: Dates must be in YYYY-MM-DD format.")
        return

    print(f"Processing tickers: {args.tickers}")
    print(f"Processing filing types: {args.filing_types}")

    for ticker in args.tickers:
        for filing_type in args.filing_types:
            download_filings(ticker.upper(), filing_type.upper(), args.start_date, args.end_date, args.email)

if __name__ == "__main__":
    main() 