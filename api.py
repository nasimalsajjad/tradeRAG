# api.py
import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure the rag_pipeline directory is in the Python path
# Adjust based on your project structure if needed
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, 'rag_pipeline'))
sys.path.append(current_dir) # Add root directory as well

# Import the core RAG function
try:
    # Need to ensure llm_utils and ner_er can also be found by rag_chain
    from rag_pipeline.rag_chain import run_rag_query
except ImportError as e:
    logging.error(f"Failed to import run_rag_query: {e}")
    # Define a dummy function if import fails, so the API can still start
    # (though queries will fail)
    def run_rag_query(query: str, ticker: str):
        return f"ERROR: Backend RAG chain failed to load ({e}). Check imports and paths."

# Configure logging (optional, but good practice for API)
# Ensure this doesn't conflict with logging in imported modules
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
# logger = logging.getLogger(__name__)

# --- Pydantic Model for Request Body ---
class QueryRequest(BaseModel):
    query: str
    ticker: str

# --- FastAPI App Initialization ---
app = FastAPI(
    title="TradeRAG API",
    description="API for querying financial information using RAG pipeline.",
    version="0.1.0",
)

# --- API Endpoints ---
@app.post("/query",
          summary="Run RAG query",
          description="Takes a user query and a stock ticker, runs the RAG pipeline (Vector + KG + LLM), and returns the answer.")
async def handle_query(request: QueryRequest):
    """Handles incoming queries to the RAG pipeline."""
    try:
        # logger.info(f"Received query for ticker '{request.ticker}': '{request.query}'")
        print(f"Received query for ticker '{request.ticker}': '{request.query}'") # Use print for now if logging is tricky

        # Run the RAG pipeline function
        # Note: run_rag_query might take time, FastAPI runs this synchronously by default.
        # For production, consider running long tasks in the background (e.g., using BackgroundTasks or Celery).
        result = run_rag_query(query=request.query, ticker=request.ticker.upper())

        # Check if the result indicates an error from the backend
        if result.startswith("Error:"):
            # logger.error(f"RAG query failed: {result}")
            print(f"RAG query failed: {result}")
            # Return a 500 Internal Server Error for backend failures
            raise HTTPException(status_code=500, detail=result)

        # logger.info(f"Returning successful response.")
        print(f"Returning successful response.")
        return {"answer": result}

    except HTTPException as http_exc: # Re-raise known HTTP exceptions
        raise http_exc
    except Exception as e:
        # logger.error(f"Unexpected error handling query: {e}", exc_info=True)
        print(f"Unexpected error handling query: {e}")
        # Return a generic 500 error for other unexpected issues
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.get("/", summary="Root/Health Check", description="Basic endpoint to check if the API is running.")
async def read_root():
    return {"message": "TradeRAG API is running."}

# --- Optional: Add main block for running with uvicorn (for simple testing) ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 