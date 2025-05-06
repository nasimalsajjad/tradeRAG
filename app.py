# app.py
import streamlit as st
import requests
import os

# --- Configuration ---
# Get the API endpoint from environment variable or use default
# This allows flexibility if you deploy the API elsewhere later.
API_ENDPOINT = os.getenv("TRADERAG_API_URL", "http://localhost:8000/query")

# --- Streamlit App Layout ---
st.set_page_config(page_title="TradeRAG", layout="wide")
st.title("📈 TradeRAG Query Interface")
st.caption("Ask questions about financial data for specific stock tickers.")

# --- Input Fields ---
col1, col2 = st.columns([3, 1]) # Make query input wider

with col1:
    user_query = st.text_input("Enter your query:", placeholder="e.g., What were the recent earnings? What are the risks?")

with col2:
    ticker = st.text_input("Stock Ticker:", placeholder="e.g., AAPL, MSFT, BAC", max_chars=5)

# --- Submit Button and Response Area ---
submit_button = st.button("Get Answer")

if submit_button:
    if not user_query:
        st.warning("Please enter a query.")
    elif not ticker:
        st.warning("Please enter a stock ticker.")
    else:
        ticker_upper = ticker.strip().upper()
        with st.spinner(f"Thinking about {ticker_upper}..."): # Show loading indicator
            try:
                # Prepare the request payload
                payload = {"query": user_query, "ticker": ticker_upper}

                # Send POST request to the FastAPI backend
                response = requests.post(API_ENDPOINT, json=payload, timeout=180) # Increased timeout

                # Check the response status code
                if response.status_code == 200:
                    # Display the answer from the API
                    api_response = response.json()
                    st.success("Answer:")
                    st.markdown(api_response.get("answer", "No answer found.")) # Use markdown for better formatting
                else:
                    # Display error from the API
                    try:
                        error_detail = response.json().get("detail", response.text)
                    except requests.exceptions.JSONDecodeError:
                        error_detail = response.text # Fallback if response is not JSON
                    st.error(f"API Error (Status {response.status_code}): {error_detail}")

            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to the backend API at {API_ENDPOINT}: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}") 