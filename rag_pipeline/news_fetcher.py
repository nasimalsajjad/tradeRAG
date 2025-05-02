import os
import argparse
import finnhub
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Define the root directory for storing raw news data
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'news')

def fetch_news(ticker: str, start_date: str, end_date: str):
    """Fetches company news from Finnhub API for a given ticker and date range."""
    load_dotenv() # Load environment variables from .env file
    api_key = os.getenv("FINNHUB_API_KEY")

    if not api_key:
        print("Error: FINNHUB_API_KEY not found in environment variables.")
        print("Please ensure it is set in your .env file.")
        return

    print(f"Fetching news for {ticker} from {start_date} to {end_date}...")
    finnhub_client = finnhub.Client(api_key=api_key)

    try:
        # Finnhub API call for company news
        # API requires dates in YYYY-MM-DD format
        news_list = finnhub_client.company_news(ticker, _from=start_date, to=end_date)

        if not news_list:
            print(f"No news found for {ticker} in the specified date range.")
            return

        print(f"Found {len(news_list)} news articles.")

        # Ensure the target directory exists
        output_dir = os.path.join(RAW_DATA_DIR, ticker)
        os.makedirs(output_dir, exist_ok=True)

        saved_count = 0
        for news_item in news_list:
            # Create a unique filename, e.g., using news ID and timestamp
            # Convert timestamp to readable date for filename
            dt_object = datetime.fromtimestamp(news_item['datetime'])
            date_str = dt_object.strftime('%Y%m%d_%H%M%S')
            filename = f"{date_str}_{news_item['id']}.json"
            filepath = os.path.join(output_dir, filename)

            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(news_item, f, ensure_ascii=False, indent=4)
                saved_count += 1
            except Exception as e:
                print(f"  -> Error saving news item {news_item['id']} to {filepath}: {e}")

        print(f"Successfully saved {saved_count} news articles to {output_dir}")

    except finnhub.FinnhubAPIException as api_e:
        print(f"Finnhub API error: {api_e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Fetch company news from Finnhub for a given ticker and date range.")
    parser.add_argument("-t", "--ticker", type=str, required=True, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("-s", "--start_date", type=str, required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("-e", "--end_date", type=str, help="End date in YYYY-MM-DD format (defaults to today)")

    args = parser.parse_args()

    end_date = args.end_date if args.end_date else datetime.now().strftime('%Y-%m-%d')

    # Basic date validation
    try:
        start_dt = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        if start_dt > end_dt:
             print("Error: Start date cannot be after end date.")
             return
    except ValueError:
        print("Error: Dates must be in YYYY-MM-DD format.")
        return

    fetch_news(args.ticker.upper(), args.start_date, end_date)

if __name__ == "__main__":
    main() 