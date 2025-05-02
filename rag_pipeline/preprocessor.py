import os
import json
import re
import glob
import argparse
from bs4 import BeautifulSoup
from datetime import datetime

# Define base directories
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

# --- Helper Functions ---

def clean_text(text: str) -> str:
    """Basic text cleaning: remove excessive whitespace, potentially other boilerplate."""
    if not isinstance(text, str):
        return "" # Return empty string if input is not text
    text = re.sub(r'\s+', ' ', text) # Replace multiple whitespace with single space
    text = text.strip()
    # Add more specific cleaning rules here as needed (e.g., remove boilerplate lines)
    return text

# --- Processing Functions for Each Data Source ---

def process_sec_filing(filepath: str) -> list[dict]:
    """Processes a single SEC filing text file (expecting HTML/text format)."""
    print(f"  Processing SEC filing: {os.path.basename(filepath)}")
    processed_docs = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Basic HTML parsing with BeautifulSoup
        # This is a simplified approach; SEC filings can be complex
        soup = BeautifulSoup(content, 'html.parser')

        # Attempt to extract text content
        # Prioritize main document body if possible, otherwise take all text
        body = soup.find('body')
        text_content = body.get_text(separator='\n', strip=True) if body else soup.get_text(separator='\n', strip=True)

        cleaned_text = clean_text(text_content)

        if cleaned_text:
            # Try to extract filing date from path or metadata if possible (placeholder)
            # Example: sec-edgar-filings/TSLA/10-Q/0001628280-23-034847/full-submission.txt
            # More robust date extraction would be needed
            date_str = "unknown"
            processed_docs.append({
                "text": cleaned_text,
                "source_type": "sec_filing",
                "source_file": os.path.basename(filepath),
                "date": date_str,
                # Add more metadata like filing type (10-K/10-Q) if derivable from path
            })
        else:
            print(f"    -> No text content extracted.")

    except Exception as e:
        print(f"    -> Error processing SEC filing {filepath}: {e}")
    return processed_docs

def process_finnhub_news(filepath: str) -> list[dict]:
    """Processes a single Finnhub news JSON file."""
    print(f"  Processing Finnhub news: {os.path.basename(filepath)}")
    processed_docs = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            news_item = json.load(f)

        # Extract relevant fields
        headline = news_item.get('headline', '')
        summary = news_item.get('summary', '')
        content = f"{headline}\n\n{summary}" # Combine headline and summary
        cleaned_text = clean_text(content)
        source = news_item.get('source')
        news_id = news_item.get('id')
        url = news_item.get('url')
        timestamp = news_item.get('datetime')
        date_str = datetime.fromtimestamp(timestamp).isoformat() if timestamp else "unknown"

        if cleaned_text:
            processed_docs.append({
                "text": cleaned_text,
                "source_type": "finnhub_news",
                "source_file": os.path.basename(filepath),
                "news_id": news_id,
                "headline": headline,
                "url": url,
                "news_source": source,
                "date": date_str,
            })
        else:
            print(f"    -> No text content extracted.")

    except json.JSONDecodeError:
        print(f"    -> Error decoding JSON from {filepath}")
    except Exception as e:
        print(f"    -> Error processing Finnhub news {filepath}: {e}")
    return processed_docs

def process_finnhub_earnings_data(filepath: str) -> list[dict]:
    """Processes Finnhub earnings metadata JSON files (like EPS surprises)."""
    filename = os.path.basename(filepath)
    print(f"  Processing Finnhub earnings data: {filename}")
    processed_docs = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert the structured data into a textual representation
        # Example for EPS surprises (list of dicts)
        text_representation = f"Finnhub Data: {filename}\n\n"
        if isinstance(data, list):
            for item in data[:10]: # Limit items displayed in text for brevity
                text_representation += json.dumps(item) + "\n"
            if len(data) > 10:
                text_representation += "... (and more)\n"
        elif isinstance(data, dict):
             text_representation += json.dumps(data, indent=2)
        else:
             text_representation += str(data)

        cleaned_text = clean_text(text_representation)

        if cleaned_text:
            # Use file modification time as a proxy if no date in data
            try:
                mtime = os.path.getmtime(filepath)
                date_str = datetime.fromtimestamp(mtime).isoformat()
            except Exception:
                date_str = "unknown"

            processed_docs.append({
                "text": cleaned_text,
                "source_type": "finnhub_earnings_metadata",
                "source_file": filename,
                "date": date_str,
            })
        else:
             print(f"    -> No text content generated.")

    except json.JSONDecodeError:
        print(f"    -> Error decoding JSON from {filepath}")
    except Exception as e:
        print(f"    -> Error processing Finnhub earnings data {filepath}: {e}")
    return processed_docs

def process_yfinance_data(filepath: str) -> list[dict]:
    """Processes yfinance JSON data files."""
    filename = os.path.basename(filepath)
    print(f"  Processing yfinance data: {filename}")
    processed_docs = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        text_representation = f"yfinance Data: {filename}\n\n"
        
        # Special handling for info.json to extract business summary
        if filename == "info.json" and isinstance(data, dict) and 'longBusinessSummary' in data:
             summary = data.get('longBusinessSummary', '')
             if summary:
                  text_representation += "Business Summary:\n" + summary + "\n\n---\n"
             text_representation += "Full Info Dict:\n" + json.dumps(data, indent=2)
        elif isinstance(data, (dict, list)):
             # General case: just dump the JSON structure as text
             text_representation += json.dumps(data, indent=2)
        else:
            text_representation += str(data)

        cleaned_text = clean_text(text_representation)

        if cleaned_text:
             # Use file modification time as a proxy
            try:
                mtime = os.path.getmtime(filepath)
                date_str = datetime.fromtimestamp(mtime).isoformat()
            except Exception:
                date_str = "unknown"

            processed_docs.append({
                "text": cleaned_text,
                "source_type": "yfinance_data",
                "source_file": filename,
                "date": date_str,
            })
        else:
             print(f"    -> No text content generated.")

    except json.JSONDecodeError:
        print(f"    -> Error decoding JSON from {filepath}")
    except Exception as e:
        print(f"    -> Error processing yfinance data {filepath}: {e}")
    return processed_docs


# --- Main Preprocessing Logic ---

def preprocess_ticker_data(ticker: str):
    """Finds all raw data for a ticker and processes it."""
    print(f"--- Starting preprocessing for ticker: {ticker} ---")
    all_processed_docs = []
    ticker = ticker.upper()

    # Define potential source directories
    source_dirs = {
        "sec_filing": os.path.join(RAW_DIR, 'sec', 'sec-edgar-filings', ticker),
        "finnhub_news": os.path.join(RAW_DIR, 'news', ticker),
        "finnhub_earnings_metadata": os.path.join(RAW_DIR, 'finnhub_earnings_data', ticker),
        "yfinance_data": os.path.join(RAW_DIR, 'yfinance_data', ticker),
        # Add manual earnings dir here if re-enabled
        # "earnings_manual": os.path.join(RAW_DIR, 'earnings', ticker),
    }

    processing_map = {
        "sec_filing": process_sec_filing,
        "finnhub_news": process_finnhub_news,
        "finnhub_earnings_metadata": process_finnhub_earnings_data,
        "yfinance_data": process_yfinance_data,
        # "earnings_manual": process_manual_earnings, # Need function if added
    }

    for source_type, source_dir in source_dirs.items():
        print(f"Checking source directory: {source_dir}")
        if not os.path.exists(source_dir):
            print(f"  -> Directory not found. Skipping.")
            continue

        process_func = processing_map.get(source_type)
        if not process_func:
            print(f"  -> No processing function defined for {source_type}. Skipping.")
            continue

        # Find relevant files based on source type
        if source_type == "sec_filing":
            # Look for the main filing text file recursively
            glob_pattern = os.path.join(source_dir, '**', 'full-submission.txt')
            files_to_process = glob.glob(glob_pattern, recursive=True)
        else:
            # Assume JSON files for other sources
            glob_pattern = os.path.join(source_dir, '*.json')
            files_to_process = glob.glob(glob_pattern)

        if not files_to_process:
            print(f"  -> No files found matching pattern for {source_type}.")
            continue
        
        print(f"Found {len(files_to_process)} files to process for {source_type}.")
        for filepath in files_to_process:
            processed_docs = process_func(filepath)
            # Add ticker info to each processed doc
            for doc in processed_docs:
                doc['ticker'] = ticker
            all_processed_docs.extend(processed_docs)

    # Save all processed documents for the ticker
    if not all_processed_docs:
        print(f"No data was successfully processed for ticker {ticker}.")
    else:
        processed_ticker_dir = os.path.join(PROCESSED_DIR, ticker)
        os.makedirs(processed_ticker_dir, exist_ok=True)
        output_filepath = os.path.join(processed_ticker_dir, 'processed_data.jsonl') # Use JSON Lines

        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                for doc in all_processed_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            print(f"Successfully saved {len(all_processed_docs)} processed documents to {output_filepath}")
        except Exception as e:
            print(f"Error saving processed data to {output_filepath}: {e}")

    print(f"--- Finished preprocessing for ticker: {ticker} ---")

def main():
    parser = argparse.ArgumentParser(description="Preprocess raw data for a given stock ticker.")
    parser.add_argument("-t", "--ticker", type=str, required=True, help="Stock ticker symbol to preprocess (e.g., AAPL)")
    # Optional: Add --force flag to re-process even if output exists

    args = parser.parse_args()
    preprocess_ticker_data(args.ticker.upper())

if __name__ == "__main__":
    main() 