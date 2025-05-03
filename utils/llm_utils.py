# utils/llm_utils.py
import os
import requests
import json
import logging
import time
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
DEFAULT_MAX_NEW_TOKENS = 500 # Increased from 300
DEFAULT_TEMPERATURE = 0.6
RUNPOD_TIMEOUT = 180 # Timeout for RunPod API calls in seconds (adjust as needed)

# --- Global Variables (Loaded from .env) ---
_runpod_api_key = None
_runpod_endpoint_id = None
_runpod_url = None
_is_configured = False

def _load_runpod_config():
    """Loads RunPod configuration from environment variables."""
    global _runpod_api_key, _runpod_endpoint_id, _runpod_url, _is_configured
    if _is_configured:
        return True

    # Load .env file from the project root
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'config.env')
    load_dotenv(dotenv_path=dotenv_path)
    logging.info(f"Loaded .env from: {dotenv_path}")

    _runpod_api_key = os.getenv("RUNPOD_API_KEY")
    _runpod_endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")

    if not _runpod_api_key:
        logging.error("RUNPOD_API_KEY not found in environment variables.")
        return False
    if not _runpod_endpoint_id:
        logging.error("RUNPOD_ENDPOINT_ID not found in environment variables.")
        return False

    # Construct the RunPod API URL (using runsync for simplicity)
    _runpod_url = f"https://api.runpod.ai/v2/{_runpod_endpoint_id}/runsync"
    logging.info(f"RunPod endpoint configured: {_runpod_url}")
    _is_configured = True
    return True

def query_llm(prompt: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, temperature: float = DEFAULT_TEMPERATURE) -> str | None:
    """Queries the RunPod Serverless vLLM endpoint."""
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

    # Construct the payload according to RunPod vLLM template expectations
    # See: https://docs.runpod.io/serverless/endpoints/vllm
    payload = {
        "input": {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            # Add other vLLM supported parameters if needed:
            # "top_p": 0.9,
            # "stop": ["stop_sequence"],
        }
    }

    logging.info(f"Querying RunPod LLM (Endpoint: {_runpod_endpoint_id}, max_new_tokens={max_new_tokens}, temp={temperature})...")
    start_time = time.time()

    try:
        response = requests.post(_runpod_url, headers=headers, json=payload, timeout=RUNPOD_TIMEOUT)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        end_time = time.time()
        logging.info(f"RunPod LLM query completed in {end_time - start_time:.2f} seconds.")

        result = response.json()

        # Process the response - check for errors reported by RunPod/vLLM
        if "error" in result:
            logging.error(f"RunPod endpoint returned an error: {result['error']}")
            return None
        
        # Check the status and output (assuming OpenAI compatible structure)
        if result.get("status") == "COMPLETED" and "output" in result:
             output_data = result["output"]
             # Handle potential list wrapping and OpenAI-like structure
             if isinstance(output_data, list) and len(output_data) > 0:
                  # Take the first item if it's a list
                  output_data = output_data[0]

             if isinstance(output_data, dict):
                 if 'choices' in output_data and isinstance(output_data['choices'], list) and len(output_data['choices']) > 0:
                     first_choice = output_data['choices'][0]
                     if 'text' in first_choice: # Common structure
                         generated_text = first_choice['text']
                         return generated_text.strip() if isinstance(generated_text, str) else None
                     elif 'message' in first_choice and 'content' in first_choice['message']:
                         generated_text = first_choice['message']['content'] # Chat completion structure
                         return generated_text.strip() if isinstance(generated_text, str) else None
                     elif 'tokens' in first_choice and isinstance(first_choice['tokens'], list): # Sometimes raw tokens are returned
                         generated_text = "".join(first_choice['tokens'])
                         return generated_text.strip()

                 # Fallback if structure is simpler (e.g., just {'text': '...'} )
                 elif 'text' in output_data:
                      generated_text = output_data['text']
                      return generated_text.strip() if isinstance(generated_text, str) else None

             # Handle case where output might be just the string directly
             elif isinstance(output_data, str):
                 return output_data.strip()

             # If parsing failed
             logging.warning(f"Could not parse expected text from RunPod output: {result['output']}")
             return None
        else:
            logging.warning(f"RunPod job did not complete successfully or output missing. Status: {result.get('status')}")
            logging.debug(f"Full RunPod response: {result}")
            return None

    except requests.exceptions.Timeout:
        logging.error(f"RunPod API request timed out after {RUNPOD_TIMEOUT} seconds.")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error during RunPod API request: {e}", exc_info=True)
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
if __name__ == "__main__":
    logging.info("Running example RunPod LLM query...")
    example_prompt = "Explain the concept of Retrieval-Augmented Generation in simple terms."
    print(f"Prompt: {example_prompt}")
    response = query_llm(example_prompt)
    if response:
        print(f"\nResponse:\n{response}")
    else:
        print("\nFailed to get response from LLM via RunPod.") 