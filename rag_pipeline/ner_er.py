# Placeholder for NER and Entity Resolution logic
import os
import json
import argparse
import re
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Define directories
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
KG_INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'kg_input')

# --- Helper Functions (from chunk_index.py, modified) ---

def load_processed_data_for_ner(ticker: str) -> list[dict]:
    """Loads processed data from the JSONL file for a ticker, returning list of dicts."""
    processed_file = os.path.join(PROCESSED_DIR, ticker, 'processed_data.jsonl')
    if not os.path.exists(processed_file):
        print(f"Error: Processed data file not found at {processed_file}")
        return []

    documents_data = []
    line_count = 0
    try:
        with open(processed_file, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                try:
                    data = json.loads(line)
                    # Keep essential fields for NER input and context
                    documents_data.append({
                        "text": data.get('text', ''),
                        "metadata": {
                            'ticker': data.get('ticker', ticker),
                            'source_type': data.get('source_type', 'unknown'),
                            'source_file': data.get('source_file', 'unknown'),
                            'date': data.get('date', 'unknown'),
                        }
                    })
                except json.JSONDecodeError as json_err:
                    print(f"Skipping line {line_count} due to JSON decode error: {json_err} - Line: {line[:100]}...") # Show start of line
                except Exception as e:
                     print(f"Skipping line {line_count} due to error: {e} - Line: {line[:100]}...")

        print(f"Loaded {len(documents_data)} data items from {processed_file}")
    except Exception as e:
        print(f"Error reading or processing {processed_file}: {e}")
    return documents_data


# --- NER Logic ---

def run_ner(documents_data: list[dict], model_name="Jean-Baptiste/roberta-large-ner-english"):
    """Runs NER on the text field of each document dictionary."""
    if not documents_data:
        print("No documents provided for NER.")
        return []

    print(f"Initializing NER pipeline with model: {model_name}...")
    # Explicitly load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        # Use device=0 if CUDA is available and desired, otherwise defaults to CPU
        # ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=0)
        ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        print(f"NER pipeline initialized on device: {ner_pipeline.device}")
    except Exception as e:
        print(f"Error initializing NER pipeline: {e}")
        print("Check model name and internet connection.")
        # Consider installing required packages if missing: pip install torch torchvision torchaudio (or tensorflow/flax)
        return []

    results = []
    print(f"Running NER on {len(documents_data)} documents...")
    for i, doc_data in enumerate(documents_data):
        metadata = doc_data.get("metadata", {})
        source_file_info = f"(source: {metadata.get('source_file', 'unknown')})"
        if i % 50 == 0 and i > 0: # Print progress
            print(f"  Processed {i}/{len(documents_data)} documents...")

        text = doc_data.get("text")
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            print(f"Skipping document {i} {source_file_info} due to missing or empty text.")
            results.append({"metadata": metadata, "entities": []})
            continue

        try:
             # Truncate long texts to avoid potential issues with model limits
             # RoBERTa's max sequence length is typically 512 tokens.
             # We truncate characters generously here; pipeline might handle tokenization limits better.
            max_chars = 3000 # Adjust as needed
            truncated_text = text[:max_chars]
            if len(text) > max_chars:
                # Don't print truncation warning every time unless debugging
                pass # print(f"  Warning: Truncating text for document {i} {source_file_info}")

            # Apply NER - Use try-except per document
            entities = ner_pipeline(truncated_text)

            # Convert scores to standard floats for JSON serialization
            serializable_entities = []
            if entities:
                for entity in entities:
                    # Ensure score exists and convert
                    entity['score'] = float(entity.get('score', 0.0))
                    serializable_entities.append(entity)

            results.append({"metadata": metadata, "entities": serializable_entities})

        except Exception as e:
             print(f"Error processing document {i} {source_file_info} with NER: {e}")
             results.append({"metadata": metadata, "entities": []}) # Add empty entities on error


    print(f"Finished NER processing. {len(results)} results generated.")
    return results # List of dicts, each containing metadata + entities


# --- Entity Resolution Logic ---

def normalize_date(date_str: str):
    """Attempts to parse a date string into YYYY-MM-DD format."""
    if not date_str or not isinstance(date_str, str):
        return None
    # Add more formats as needed
    formats_to_try = [
        "%Y-%m-%d", # ISO
        "%B %d, %Y", # January 1, 2023
        "%b %d, %Y", # Jan 1, 2023
        "%m/%d/%Y",   # 01/01/2023
        "%m/%d/%y",   # 01/01/23
    ]
    for fmt in formats_to_try:
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    # Add more complex parsing logic (e.g., using dateutil) if required
    # print(f"Could not parse date: {date_str}")
    return None

def resolve_entities(ner_results: list[dict], target_ticker: str):
    """Basic entity resolution: filter types, normalize ORG to ticker."""
    print("Starting Entity Resolution step...")
    resolved_data = []
    target_ticker = target_ticker.upper()
    # Define known variations for the target ticker (case-insensitive)
    # This should ideally come from a config or external source
    org_variations = {
        "AMD": ["amd", "advanced micro devices", "advanced micro devices, inc."],
        "TSLA": ["tesla", "tesla, inc.", "tesla inc"],
        "AAPL": ["apple", "apple inc", "apple computer"],
        "MSFT": ["microsoft", "microsoft corporation"],
        "NVDA": ["nvidia", "nvidia corporation"],
        # Add more known tickers and variations
    }
    target_variations = [v.lower() for v in org_variations.get(target_ticker, [target_ticker.lower()])]

    allowed_entity_types = {"ORG", "PER", "DATE", "MONEY"}

    processed_count = 0
    for item in ner_results:
        resolved_entities = []
        original_entities = item.get("entities", [])

        for entity in original_entities:
            entity_group = entity.get("entity_group")
            entity_word = entity.get("word", "").strip()

            if entity_group in allowed_entity_types and entity_word:
                resolved_entity = entity.copy()

                # 1. ORG Normalization
                if entity_group == "ORG":
                    if entity_word.lower() in target_variations:
                        resolved_entity["normalized_id"] = target_ticker
                        resolved_entity["normalized_label"] = "Company" # Assign a graph label
                    # Potential: Add logic here for other ORGs if needed (e.g., competitors, partners)
                    # For now, only normalize the target ticker
                    else:
                         resolved_entity["normalized_label"] = "Organization"

                # 2. DATE Normalization (Basic Example)
                elif entity_group == "DATE":
                    normalized = normalize_date(entity_word)
                    if normalized:
                        resolved_entity["normalized_date"] = normalized
                    resolved_entity["normalized_label"] = "Date"

                # 3. PERSON Normalization (Placeholder)
                elif entity_group == "PER":
                    # Simple pass-through, could add clustering/canonical names later
                     resolved_entity["normalized_label"] = "Person"

                # 4. MONEY Normalization (Placeholder)
                elif entity_group == "MONEY":
                     # Simple pass-through, could add value/currency extraction later
                     resolved_entity["normalized_label"] = "MonetaryAmount"

                resolved_entities.append(resolved_entity)

        resolved_data.append({
            "metadata": item.get("metadata", {}),
            "resolved_entities": resolved_entities
        })
        processed_count += 1
        if processed_count % 100 == 0:
             print(f"  Resolved entities for {processed_count}/{len(ner_results)} documents...")

    print(f"Finished Entity Resolution. Processed {len(resolved_data)} documents.")
    return resolved_data

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Run NER and Entity Resolution on processed data.")
    parser.add_argument("-t", "--ticker", type=str, required=True, help="Stock ticker symbol to process (e.g., AMD)")
    parser.add_argument("--ner_model", type=str, default="Jean-Baptiste/roberta-large-ner-english", help="NER model name from Hugging Face.")
    # Add ER model name argument later

    args = parser.parse_args()
    ticker = args.ticker.upper()

    print(f"--- Starting NER/ER for ticker: {ticker} ---")

    # 1. Load processed data
    documents_data = load_processed_data_for_ner(ticker)
    if not documents_data:
        print("Exiting: No documents loaded.")
        return

    # 2. Run NER
    ner_results = run_ner(documents_data, model_name=args.ner_model)
    if not ner_results:
         print("Exiting: NER step failed or produced no results.")
         return

    # 3. Run Entity Resolution
    final_results = resolve_entities(ner_results, ticker)

    # 4. Save results
    output_dir = os.path.join(KG_INPUT_DIR, ticker)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "entities_resolved.jsonl") # Changed filename
    print(f"Saving resolved entities to: {output_filename}")
    saved_count = 0
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            for item in final_results:
                # Ensure item is serializable (floats handled in NER step)
                try:
                    f.write(json.dumps(item) + '\n')
                    saved_count += 1
                except TypeError as json_err:
                     print(f"Error saving item to JSON: {json_err}. Skipping item.")
                     # Optionally print the problematic item for debugging:
                     # print(f"Problematic item: {item}")
        print(f"Successfully saved {saved_count} resolved items to {output_filename}")
    except Exception as e:
         print(f"Error writing results to {output_filename}: {e}")


    print(f"--- Finished NER/ER for ticker: {ticker} ---")


if __name__ == "__main__":
    main() 