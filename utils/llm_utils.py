# utils/llm_utils.py
import os
import requests
import json
import logging
import time
from dotenv import load_dotenv

# Configure logging - REMOVING THIS LINE (should be configured in main script)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
DEFAULT_MAX_NEW_TOKENS = 500 # Using this as max_tokens for OpenAI format
DEFAULT_TEMPERATURE = 0.6
RUNPOD_TIMEOUT = 180 # Timeout for RunPod API calls in seconds (adjust as needed)

# --- Global Variables (Loaded from .env) ---
_runpod_api_key = None
_runpod_pod_id = "55j0rcttb3ssv3" # Hardcoding the Pod ID provided
_runpod_model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" # Hardcoding model name
_runpod_pod_url = None
_is_configured = False

def _load_runpod_config():
    """Loads RunPod configuration from environment variables."""
    global _runpod_api_key, _runpod_pod_url, _is_configured
    if _is_configured:
        return True

    # Load .env file from the project root
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'config.env')
    load_dotenv(dotenv_path=dotenv_path)
    logging.info(f"Loaded .env from: {dotenv_path}")

    _runpod_api_key = os.getenv("RUNPOD_API_KEY")
    # We no longer need RUNPOD_ENDPOINT_ID for Pods
    # _runpod_endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")

    if not _runpod_api_key:
        logging.error("RUNPOD_API_KEY not found in environment variables.")
        return False
    # if not _runpod_endpoint_id: # No longer needed
    #     logging.error("RUNPOD_ENDPOINT_ID not found in environment variables.")
    #     return False

    # Construct the RunPod Pod API URL (OpenAI compatible)
    _runpod_pod_url = f"https://{_runpod_pod_id}-8000.proxy.runpod.net/v1/chat/completions"
    logging.info(f"RunPod Pod endpoint configured: {_runpod_pod_url}")
    _is_configured = True
    return True

def query_llm(prompt: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, temperature: float = DEFAULT_TEMPERATURE) -> str | None:
    """Queries the RunPod Pod OpenAI-compatible endpoint."""
    if not _load_runpod_config():
        logging.error("RunPod configuration failed. Cannot query LLM.")
        return None

    if not isinstance(prompt, str) or not prompt.strip():
        logging.warning("Received empty or invalid prompt.")
        return None

    headers = {
        "Authorization": f"Bearer {_runpod_api_key}",
        "Content-Type": "application/json"
    }

    # Construct the payload according to OpenAI Chat Completions format
    payload = {
        "model": _runpod_model_name, # Specify the model
        "messages": [
            {"role": "user", "content": prompt} # Simple user message structure
        ],
        "max_tokens": max_new_tokens, # OpenAI uses max_tokens
        "temperature": temperature,
        # "stop": ["stop_sequence"], # Optional stop sequences
    }

    logging.info(f"Querying RunPod Pod LLM (URL: {_runpod_pod_url}, Model: {_runpod_model_name}, max_tokens={max_new_tokens}, temp={temperature})...")
    start_time = time.time()

    try:
        response = requests.post(_runpod_pod_url, headers=headers, json=payload, timeout=RUNPOD_TIMEOUT)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        end_time = time.time()
        logging.info(f"RunPod Pod LLM query completed in {end_time - start_time:.2f} seconds.")

        result = response.json()

        # Log the full response for debugging
        logging.debug(f"Full RunPod response JSON: {json.dumps(result, indent=2)}")

        # Process the response - check for errors from the API
        if "error" in result:
            logging.error(f"RunPod Pod API returned an error: {result['error']}")
            return None
        
        # Extract the response based on OpenAI Chat Completions structure
        if 'choices' in result and isinstance(result['choices'], list) and len(result['choices']) > 0:
            first_choice = result['choices'][0]
            if 'message' in first_choice and 'content' in first_choice['message']:
                generated_text = first_choice['message']['content']
                return generated_text.strip() if isinstance(generated_text, str) else None
            else:
                logging.warning(f"Could not find 'content' in message of first choice: {first_choice}")
                return None
        else:
             logging.warning(f"Could not parse expected 'choices' structure from RunPod Pod output: {result}")
             return None

    except requests.exceptions.Timeout:
        logging.error(f"RunPod Pod API request timed out after {RUNPOD_TIMEOUT} seconds.")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error during RunPod Pod API request: {e}", exc_info=True)
        # Log response content if available for debugging non-200 status codes
        if hasattr(e, 'response') and e.response is not None:
             try:
                  logging.error(f"RunPod Response Content: {e.response.text}")
             except Exception:
                  logging.error("Could not decode RunPod error response.")
        return None
    except Exception as e:
         logging.error(f"An unexpected error occurred: {e}", exc_info=True)
         return None

# --- Example Usage (for direct script execution testing) ---
# (Keep existing example usage, though it will now hit the Pod API)
if __name__ == "__main__":
    # Ensure logging is configured if run directly
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("Running example RunPod Pod LLM query...")
    example_prompt = "Explain the concept of Retrieval-Augmented Generation in simple terms."
    print(f"Prompt: {example_prompt}")
    response = query_llm(example_prompt)
    if response:
        print(f"\nResponse:\n{response}")
    else:
        print("\nFailed to get response from LLM via RunPod Pod.") 