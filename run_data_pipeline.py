import argparse
import os
import logging
from dotenv import load_dotenv
import finnhub # Import finnhub here to initialize client once

# Import specific functions/classes from the pipeline modules
from rag_pipeline.sec_loader import download_filings as sec_download
from rag_pipeline.news_fetcher import fetch_news
from rag_pipeline.yfinance_data_loader import fetch_yfinance_data
from rag_pipeline.finnhub_earnings_data import (
    fetch_earnings_calendar,
    fetch_earnings_estimates,
    fetch_eps_surprises
)
from rag_pipeline.preprocessor import preprocess_ticker_data
from rag_pipeline.chunk_index import load_processed_data, build_and_persist_index

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(ticker: str, start_date: str, end_date: str, sec_email: str, finnhub_client: finnhub.Client):
    """Runs the full data pipeline for a given ticker and date range."""
    logger.info(f"=== Starting Data Pipeline for Ticker: {ticker} [{start_date} to {end_date}] ===")

    # --- 1. Fetch SEC Filings ---
    logger.info("--- Stage 1: Fetching SEC Filings (10-K, 10-Q) ---")
    for filing_type in ["10-K", "10-Q"]:
        try:
            sec_download(ticker, filing_type, start_date, end_date, sec_email)
        except Exception as e:
            logger.error(f"Error during SEC download for {filing_type}: {e}", exc_info=True)

    # --- 2. Fetch Finnhub News ---
    logger.info("--- Stage 2: Fetching Finnhub News ---")
    try:
        fetch_news(ticker, start_date, end_date)
    except Exception as e:
        logger.error(f"Error during Finnhub news fetching: {e}", exc_info=True)

    # --- 3. Fetch yfinance Data ---
    # Note: yfinance component doesn't use start/end dates in its current form
    logger.info("--- Stage 3: Fetching yfinance Data ---")
    try:
        fetch_yfinance_data(ticker)
    except Exception as e:
        logger.error(f"Error during yfinance data fetching: {e}", exc_info=True)

    # --- 4. Fetch Finnhub Earnings Data ---
    logger.info("--- Stage 4: Fetching Finnhub Earnings Data ---")
    # Prepare output directory (consistent with finnhub_earnings_data.py)
    finnhub_earnings_raw_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'finnhub_earnings_data')
    ticker_output_dir = os.path.join(finnhub_earnings_raw_dir, ticker)
    os.makedirs(ticker_output_dir, exist_ok=True)
    try:
        fetch_earnings_calendar(finnhub_client, ticker, start_date, end_date, ticker_output_dir)
    except Exception as e:
        logger.error(f"Error during Finnhub calendar fetching: {e}", exc_info=True)
    try:
        fetch_earnings_estimates(finnhub_client, ticker, ticker_output_dir)
    except Exception as e:
        logger.error(f"Error during Finnhub estimates fetching: {e}", exc_info=True)
    try:
        fetch_eps_surprises(finnhub_client, ticker, ticker_output_dir) # Default limit is 4
    except Exception as e:
        logger.error(f"Error during Finnhub surprises fetching: {e}", exc_info=True)


    # --- 5. Preprocess All Fetched Data ---
    logger.info("--- Stage 5: Preprocessing All Raw Data ---")
    try:
        preprocess_ticker_data(ticker)
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)

    # --- 6. Chunk and Index Processed Data ---
    logger.info("--- Stage 6: Chunking and Indexing Data ---")
    try:
        logger.info(f"Loading processed documents for {ticker}...")
        documents = load_processed_data(ticker)
        if documents:
            logger.info(f"Building/persisting vector index for {ticker}...")
            build_and_persist_index(ticker, documents)
        else:
            logger.warning(f"No processed documents found for {ticker}. Skipping indexing.")
    except Exception as e:
        logger.error(f"Error during chunking/indexing: {e}", exc_info=True)

    logger.info(f"=== Data Pipeline Finished for Ticker: {ticker} ===")


def main():
    # Load environment variables from config.env file in project root
    # Ensure this runs before accessing env vars
    dotenv_path = os.path.join(os.path.dirname(__file__), 'config.env')
    if not os.path.exists(dotenv_path):
         logger.error(f"Environment file not found at: {dotenv_path}")
         logger.error("Please ensure 'config.env' exists in the project root directory.")
         return
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(f"Loaded environment variables from: {dotenv_path}")

    sec_email = os.getenv("SEC_EMAIL")
    finnhub_api_key = os.getenv("FINNHUB_API_KEY")

    if not sec_email:
        logger.error("SEC_EMAIL not found in environment variables (config.env). Required for SEC downloads.")
        return
    if not finnhub_api_key:
         logger.error("FINNHUB_API_KEY not found in environment variables (config.env). Required for Finnhub API.")
         return
    
    # Initialize Finnhub client once
    finnhub_client = finnhub.Client(api_key=finnhub_api_key)

    parser = argparse.ArgumentParser(description="Run the full data processing and indexing pipeline for a stock ticker.")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--start_date", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", required=True, help="End date in YYYY-MM-DD format")

    args = parser.parse_args()

    # Basic date validation (optional, as sub-scripts might handle it)
    # Consider adding date validation here if needed

    run_pipeline(args.ticker.upper(), args.start_date, args.end_date, sec_email, finnhub_client)

if __name__ == "__main__":
    main() 