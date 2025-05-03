import os
import argparse
import finnhub
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Define the root directory for storing raw Finnhub earnings-related data
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'finnhub_earnings_data')

# --- Fetch Functions ---

def fetch_earnings_calendar(client, ticker: str, start_date: str, end_date: str, output_dir: str):
    """Fetches earnings calendar data from Finnhub API."""
    print(f"Fetching earnings calendar for {ticker} from {start_date} to {end_date}...")
    try:
        calendar_data = client.earnings_calendar(_from=start_date, to=end_date, symbol=ticker, international=False)
        if not calendar_data or not calendar_data.get('earningsCalendar'):
            print(f"  -> No earnings calendar events found.")
            return

        earnings_list = calendar_data.get('earningsCalendar', [])
        print(f"  -> Found {len(earnings_list)} events.")

        filename = f"calendar_{start_date}_{end_date}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(calendar_data, f, ensure_ascii=False, indent=4)
        print(f"  -> Saved calendar data to {filepath}")

    except finnhub.FinnhubAPIException as api_e:
        print(f"  -> Finnhub API error (Calendar): {api_e}")
    except Exception as e:
        print(f"  -> Unexpected error (Calendar): {e}")

def fetch_earnings_estimates(client, ticker: str, output_dir: str):
    """Fetches earnings estimates (quarterly & annual) from Finnhub API."""
    print(f"Fetching earnings estimates for {ticker}...")
    try:
        estimates = client.company_eps_estimates(ticker, freq='quarterly') # Can also fetch 'annual'
        if not estimates:
            print(f"  -> No earnings estimates found.")
            return

        print(f"  -> Found estimates data.") # Adjust count/message later if structure is clear

        filename = f"estimates_quarterly.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            # Save the raw response dictionary/list
            json.dump(estimates, f, ensure_ascii=False, indent=4)
        print(f"  -> Saved quarterly estimates data to {filepath}")

    except finnhub.FinnhubAPIException as api_e:
        print(f"  -> Finnhub API error (Estimates): {api_e}")
    except Exception as e:
        print(f"  -> Unexpected error (Estimates): {e}")

def fetch_eps_surprises(client, ticker: str, output_dir: str, limit: int = 4):
    """Fetches recent EPS surprises from Finnhub API."""
    print(f"Fetching EPS surprises for {ticker} (limit: {limit})...")
    try:
        surprises = client.company_earnings(ticker, limit=limit)
        if not surprises:
            print(f"  -> No EPS surprises found.")
            return

        print(f"  -> Found {len(surprises)} EPS surprise records.")

        filename = f"eps_surprises_last_{limit}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            # Finnhub returns a list directly for this endpoint
            json.dump(surprises, f, ensure_ascii=False, indent=4)
        print(f"  -> Saved EPS surprises data to {filepath}")

    except finnhub.FinnhubAPIException as api_e:
        print(f"  -> Finnhub API error (EPS Surprises): {api_e}")
    except Exception as e:
        print(f"  -> Unexpected error (EPS Surprises): {e}")

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Fetch various earnings-related data (Calendar, Estimates, Surprises) from Finnhub.")
    parser.add_argument("-t", "--ticker", type=str, required=True, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("-s", "--start_date", type=str, required=True, help="Start date for Earnings Calendar (YYYY-MM-DD)")
    parser.add_argument("-e", "--end_date", type=str, required=True, help="End date for Earnings Calendar (YYYY-MM-DD)")
    parser.add_argument("-l", "--limit", type=int, default=4, help="Limit for EPS Surprises query (default: 4)")

    args = parser.parse_args()

    # Basic date validation
    try:
        start_dt = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(args.end_date, '%Y-%m-%d')
        if start_dt > end_dt:
             print("Error: Start date cannot be after end date for calendar.")
             return
    except ValueError:
        print("Error: Dates must be in YYYY-MM-DD format.")
        return

    # Load API Key
    # Load environment variables from config.env file in project root
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'config.env')
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        print("Error: FINNHUB_API_KEY not found in environment variables (config.env file).")
        return

    finnhub_client = finnhub.Client(api_key=api_key)
    ticker = args.ticker.upper()

    # Prepare output directory for the ticker
    ticker_output_dir = os.path.join(RAW_DATA_DIR, ticker)
    os.makedirs(ticker_output_dir, exist_ok=True)
    print(f"--- Processing Finnhub earnings data for {ticker} ---")
    print(f"Output directory: {ticker_output_dir}")

    # Fetch all types of data
    fetch_earnings_calendar(finnhub_client, ticker, args.start_date, args.end_date, ticker_output_dir)
    fetch_earnings_estimates(finnhub_client, ticker, ticker_output_dir)
    fetch_eps_surprises(finnhub_client, ticker, ticker_output_dir, limit=args.limit)

    print(f"--- Finished processing Finnhub earnings data for {ticker} ---")

if __name__ == "__main__":
    main() 