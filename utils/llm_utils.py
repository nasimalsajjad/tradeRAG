# utils/llm_utils.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging
import time 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
DEFAULT_MAX_NEW_TOKENS = 300 # Keep relatively low for CPU performance
DEFAULT_TEMPERATURE = 0.6

# --- Global Variables (Lazy Loaded) ---
_tokenizer = None
_model = None
_generator_pipeline = None

def _initialize_llm():
    """Initializes the tokenizer, model, and pipeline on first use."""
    global _tokenizer, _model, _generator_pipeline
    if _generator_pipeline is None:
        logging.info(f"Initializing LLM: {MODEL_NAME} for CPU inference...")
        start_time = time.time()
        try:
            logging.info("Loading tokenizer...")
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
            logging.info("Loading model (using float32 for CPU compatibility)...")
            # Load model explicitly asking for float32, device_map='cpu' can be added for more safety
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, 
                torch_dtype=torch.float32, # Use float32 for CPU
                # device_map='cpu' # Can explicitly force CPU here too
            )
            
            logging.info("Creating text-generation pipeline for CPU (device=-1)...")
            # device=-1 forces CPU usage in the pipeline
            _generator_pipeline = pipeline(
                "text-generation", 
                model=_model, 
                tokenizer=_tokenizer, 
                device=-1 
            )
            end_time = time.time()
            logging.info(f"LLM initialization complete. Took {end_time - start_time:.2f} seconds.")
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {e}", exc_info=True)
            # Prevent further attempts if initialization fails
            _generator_pipeline = False # Use False to indicate failure

def query_llm(prompt: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, temperature: float = DEFAULT_TEMPERATURE) -> str | None:
    """Queries the loaded Mistral-7B model on CPU."""
    _initialize_llm() # Ensure LLM is loaded

    if _generator_pipeline is None or _generator_pipeline is False:
        logging.error("LLM Pipeline not available. Cannot query.")
        return None
    
    if not isinstance(prompt, str) or not prompt.strip():
        logging.warning("Received empty or invalid prompt.")
        return None
        
    logging.info(f"Querying LLM (max_new_tokens={max_new_tokens}, temp={temperature})...")
    start_time = time.time()
    
    try:
        # Ensure prompt formatting for instruct models (if needed - Mistral Instruct usually uses [INST]...[/INST])
        # Basic check, adjust if model requires specific format not added by pipeline
        # if not prompt.strip().startswith("[INST]"):
        #    formatted_prompt = f"[INST] {prompt.strip()} [/INST]"
        # else:
        #    formatted_prompt = prompt
        
        # The pipeline should handle basic prompt formatting for most instruct models
        formatted_prompt = prompt 
        
        outputs = _generator_pipeline(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=_tokenizer.eos_token_id # Prevent warnings
        )
        
        end_time = time.time()
        logging.info(f"LLM query completed in {end_time - start_time:.2f} seconds.")
        
        if outputs and len(outputs) > 0 and "generated_text" in outputs[0]:
            # Depending on the pipeline, the output might include the prompt.
            # We usually want only the newly generated text.
            full_text = outputs[0]["generated_text"]
            # Simple way to remove prompt if present:
            if full_text.startswith(formatted_prompt):
                 response = full_text[len(formatted_prompt):].strip()
            else:
                 response = full_text # Or handle differently if prompt isn't included
            return response
        else:
            logging.warning("LLM generated no valid output.")
            return None
            
    except Exception as e:
        logging.error(f"Error during LLM query: {e}", exc_info=True)
        return None

# --- Example Usage (for direct script execution testing) ---
if __name__ == "__main__":
    logging.info("Running example LLM query...")
    example_prompt = "Explain the concept of Retrieval-Augmented Generation in simple terms."
    print(f"Prompt: {example_prompt}")
    response = query_llm(example_prompt)
    if response:
        print(f"\nResponse:\n{response}")
    else:
        print("\nFailed to get response from LLM.") 