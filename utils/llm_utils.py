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
_runpod_api_key = None # Keep loading this, might be needed for other RunPod API calls later
_vllm_api_key = "sk-IrR7Bwxtin0haWagUnPrBgq5PurnUz86" # API key for vLLM endpoint itself
_runpod_pod_id = "bhpsvsi3j3q0fm" # Hardcoding the new Pod ID provided
_runpod_model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" # Model name for this pod
_runpod_pod_url = None
_is_configured = False

def _load_runpod_config():
    """Loads RunPod configuration from environment variables."""
    global _runpod_api_key, _runpod_pod_url, _is_configured
    # Only configure the URL part once, API key loading remains dynamic if needed
    if _runpod_pod_url is None:
        # Load .env file from the project root just for RUNPOD_API_KEY if other RunPod calls are needed
        dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'config.env')
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f"Loaded .env from: {dotenv_path}")
        _runpod_api_key = os.getenv("RUNPOD_API_KEY") # Load the general RunPod key
        # We no longer need RUNPOD_ENDPOINT_ID for Pods

        # Construct the RunPod Pod API URL (OpenAI compatible)
        _runpod_pod_url = f"https://{_runpod_pod_id}-8000.proxy.runpod.net/v1/chat/completions"
        logging.info(f"RunPod Pod endpoint configured: {_runpod_pod_url}")
        _is_configured = True # Mark basic config as done
        
    return True # Always return true once URL is set

def query_llm(prompt: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, temperature: float = DEFAULT_TEMPERATURE) -> str | None:
    """Queries the RunPod Pod OpenAI-compatible endpoint using the vLLM-specific API key."""
    if not _load_runpod_config(): # Ensure URL is configured
        logging.error("RunPod configuration failed. Cannot query LLM.")
        return None

    if not isinstance(prompt, str) or not prompt.strip():
        logging.warning("Received empty or invalid prompt.")
        return None

    if not _vllm_api_key: # Check if the vLLM key is set
        logging.error("vLLM API Key is not set in llm_utils.py. Cannot authenticate with the Pod.")
        return None
        
    headers = {
        "Authorization": f"Bearer {_vllm_api_key}", # Use the vLLM specific key
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