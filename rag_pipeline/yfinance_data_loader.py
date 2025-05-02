import os
import argparse
import yfinance as yf
import json
import pandas as pd
from datetime import datetime, date

# Define the root directory for storing raw yfinance data
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'yfinance_data')

# Custom JSON encoder to handle date/datetime objects
def date_converter(o):
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    # Add handling for Pandas Timestamp if it appears
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    # Handle numpy int64 if it causes issues
    if isinstance(o, pd.Int64Dtype().type):
         return int(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

# Helper function to save data (handles dicts and DataFrames)
def save_data_to_json(data, filename: str, output_dir: str):
    filepath = os.path.join(output_dir, filename)
    try:
        if data is None:
            print(f"  -> No data to save for {filename}")
            return

        if isinstance(data, dict):
             # If data contains DataFrames (like .info might), convert them first
            serializable_data = {} 
            all_serializable = True
            for k, v in data.items():
                 if isinstance(v, pd.DataFrame):
                      try:
                           # Attempt conversion, handle Timestamps
                           v.columns = v.columns.astype(str) # Ensure column names are strings
                           v.index = v.index.astype(str) # Ensure index is string
                           serializable_data[k] = json.loads(v.to_json(orient='index', date_format='iso'))
                      except Exception as df_conv_err:
                           print(f"    -> Warning: Could not serialize DataFrame in dict key '{k}': {df_conv_err}")
                           serializable_data[k] = "<DataFrame Conversion Error>"
                           all_serializable = False
                 elif isinstance(v, pd.Timestamp): 
                       serializable_data[k] = v.isoformat()
                 else:
                       serializable_data[k] = v
            data_to_save = serializable_data
        elif isinstance(data, pd.DataFrame):
            if data.empty:
                print(f"  -> DataFrame for {filename} is empty, not saving.")
                return
            # Convert DataFrame to JSON serializable dict format
            data.columns = data.columns.astype(str) # Ensure column names are strings
            data.index = data.index.astype(str) # Ensure index is string
            data_to_save = json.loads(data.to_json(orient='index', date_format='iso'))
        elif isinstance(data, pd.Series):
             if data.empty:
                  print(f"  -> Series for {filename} is empty, not saving.")
                  return
             data.index = data.index.astype(str) # Ensure index is string
             data_to_save = json.loads(data.to_json(orient='index', date_format='iso'))
        else:
             # Assume it's already JSON serializable (e.g., list, basic types)
            data_to_save = data 

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4, default=date_converter)
        print(f"  -> Saved {filename}")

    except TypeError as te:
         print(f"  -> Error saving {filename}: JSON Serialization Error - {te}. Data type: {type(data)}")
    except Exception as e:
        print(f"  -> Error saving {filename}: {e}")

def fetch_yfinance_data(ticker: str):
    """Fetches various data points from yfinance for a given ticker."""
    print(f"--- Processing yfinance data for {ticker} ---")
    try:
        yf_ticker = yf.Ticker(ticker)

        # Prepare output directory
        output_dir = os.path.join(RAW_DATA_DIR, ticker)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

        data_fetch_map = {
            "info": lambda: yf_ticker.info,
            "calendar": lambda: yf_ticker.calendar,
            "earnings_dates_history": lambda: yf_ticker.earnings_dates,
            "financials_annual": lambda: yf_ticker.financials,
            "financials_quarterly": lambda: yf_ticker.quarterly_financials,
            "balance_sheet_annual": lambda: yf_ticker.balance_sheet,
            "balance_sheet_quarterly": lambda: yf_ticker.quarterly_balance_sheet,
            "cashflow_annual": lambda: yf_ticker.cashflow,
            "cashflow_quarterly": lambda: yf_ticker.quarterly_cashflow,
            "major_holders": lambda: yf_ticker.major_holders,
            "institutional_holders": lambda: yf_ticker.institutional_holders,
            "recommendations": lambda: yf_ticker.recommendations,
            # Add others like .splits, .dividends if needed
        }

        for name, fetch_func in data_fetch_map.items():
            print(f"Fetching {name}...")
            try:
                data = fetch_func()
                save_data_to_json(data, f"{name}.json", output_dir)
            except AttributeError:
                 print(f"  -> Attribute for '{name}' not found for ticker {ticker}.")
            except Exception as e:
                print(f"  -> Error fetching/processing {name}: {e}")

    except Exception as e:
        print(f"An error occurred initializing yfinance Ticker for {ticker}: {e}")

    print(f"--- Finished processing yfinance data for {ticker} ---")

def main():
    parser = argparse.ArgumentParser(description="Fetch comprehensive data from yfinance for a given ticker.")
    parser.add_argument("-t", "--ticker", type=str, required=True, help="Stock ticker symbol (e.g., AAPL)")

    args = parser.parse_args()
    fetch_yfinance_data(args.ticker.upper())

if __name__ == "__main__":
    main() 