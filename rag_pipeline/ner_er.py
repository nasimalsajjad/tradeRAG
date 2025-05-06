# Placeholder for NER and Entity Resolution logic
import os
import json
import argparse
import re
from datetime import datetime
# Remove transformers imports
# from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import spacy # Add spacy import
from sentence_transformers import SentenceTransformer
import numpy as np

# Define directories
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
KG_INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'kg_input')

# --- Canonical Knowledge Base (Simple Example) ---
# In production, load this from a file/database
canonical_kb = {
    "Company": {
        "MU": {
            "name": "Micron Technology, Inc.",
            "aliases": ["Micron", "Micron Technology"]
        },
        "TSLA": {
            "name": "Tesla, Inc.",
            "aliases": ["Tesla"]
        },
        "AMD": {
            "name": "Advanced Micro Devices, Inc.",
            "aliases": ["AMD", "Advanced Micro Devices"]
        },
        # Add more known companies
    },
    # Can add "Person", etc. if doing embedding-based resolution for them
}

# --- Global Variables for Embeddings (Load once) ---
_embedding_model = None
_kb_embeddings = {} # Structure: {"Company": {"MU": [emb1, emb2...], "TSLA": [...]}}

# --- Helper Functions ---

def get_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Loads the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {model_name}...")
        try:
            _embedding_model = SentenceTransformer(model_name)
            print("Embedding model loaded.")
        except Exception as e:
            print(f"Error loading embedding model {model_name}: {e}")
            raise # Re-raise the exception to stop if model can't load
    return _embedding_model

def compute_kb_embeddings():
    """Computes and stores embeddings for the canonical KB."""
    global _kb_embeddings
    if _kb_embeddings: # Only compute once
        return

    print("Computing embeddings for Canonical Knowledge Base...")
    model = get_embedding_model()
    if not model: return

    embeddings_computed = 0
    for entity_type, entities in canonical_kb.items():
        if entity_type not in _kb_embeddings:
            _kb_embeddings[entity_type] = {}
        for entity_id, data in entities.items():
            entity_embeddings = []
            texts_to_embed = [data["name"]] + data.get("aliases", [])
            try:
                 # Compute embeddings in batch if possible
                 computed = model.encode(texts_to_embed, convert_to_numpy=True)
                 entity_embeddings.extend(computed)
                 embeddings_computed += len(texts_to_embed)
            except Exception as e:
                 print(f"Error computing embedding for {entity_type} {entity_id}: {e}")
            _kb_embeddings[entity_type][entity_id] = entity_embeddings

    print(f"Computed {embeddings_computed} embeddings for KB.")

def cosine_similarity(v1, v2):
    """Computes cosine similarity between two numpy vectors."""
    # Ensure inputs are numpy arrays
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    # Normalize vectors
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0 # Avoid division by zero
    v1_normalized = v1 / norm_v1
    v2_normalized = v2 / norm_v2
    # Compute dot product
    return np.dot(v1_normalized, v2_normalized)

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

def run_ner(documents_data: list[dict], model_name="en_core_web_trf"): # Changed default model name
    """Runs NER on the text field of each document dictionary using spaCy."""
    if not documents_data:
        print("No documents provided for NER.")
        return []

    print(f"Initializing spaCy NER model: {model_name}...")
    try:
        # Load the spaCy model
        # Consider adding download instructions if model not found:
        # spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
        print(f"spaCy NER model '{model_name}' loaded successfully.")
        # Increase the max length limit after loading
        nlp.max_length = 1500000
        print(f"Increased nlp.max_length to {nlp.max_length}")
        # Check if GPU is available and enabled for spaCy if desired
        # Example: spacy.require_gpu() -> True if available
        # Check loaded pipeline components
        print(f"Loaded spaCy pipeline components: {nlp.pipe_names}")
        if 'ner' not in nlp.pipe_names:
            print(f"Warning: 'ner' component not found in the loaded spaCy model '{model_name}'.")
            # You might want to handle this case, e.g., by returning [] or raising an error
            # return []

    except OSError as e:
        print(f"Error loading spaCy model '{model_name}': {e}")
        print(f"Please ensure the model is downloaded, e.g., run: python -m spacy download {model_name}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during spaCy model loading: {e}")
        return []


    results = []
    print(f"Running NER on {len(documents_data)} documents...")
    for i, doc_data in enumerate(documents_data):
        metadata = doc_data.get("metadata", {})
        source_file_info = f"(source: {metadata.get('source_file', 'unknown')})"
        # Log progress more frequently initially, then less often
        if i < 25 and i % 5 == 0: # Log every 5 docs for the first 25
            print(f"  Processed {i}/{len(documents_data)} documents...")
        elif i >= 25 and i % 50 == 0: # Log every 50 docs after the first 25
            print(f"  Processed {i}/{len(documents_data)} documents...")

        text = doc_data.get("text")
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            print(f"Skipping document {i} {source_file_info} due to missing or empty text.")
            results.append({"metadata": metadata, "entities": []})
            continue

        # spaCy handles text length internally, but consider limits for very large texts
        # max_length = nlp.max_length
        # if len(text) > max_length:
        #     print(f"Warning: Text for document {i} exceeds model max length ({max_length}), truncating.")
        #     text = text[:max_length]

        try:
            # Apply spaCy NER
            doc = nlp(text)

            # Extract entities in the format expected by resolve_entities
            extracted_entities = []
            if doc.ents:
                for ent in doc.ents:
                    extracted_entities.append({
                        'word': ent.text,
                        'entity_group': ent.label_, # spaCy uses label_
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'score': 1.0 # spaCy entities don't have built-in scores like HF pipeline
                    })

            results.append({"metadata": metadata, "entities": extracted_entities})

        except Exception as e:
             # Consider more specific error handling if needed
             print(f"Error processing document {i} {source_file_info} with spaCy NER: {e}")
             # Check for potential memory issues with large documents
             if "MemoryError" in str(e):
                 print("  Potential MemoryError encountered. Consider processing smaller text chunks.")
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

def resolve_entities(ner_results: list[dict], target_ticker: str, similarity_threshold=0.85):
    """Resolves entities using embedding similarity for ORGs and rules for others."""
    print("Starting Entity Resolution step...")
    resolved_data = []
    target_ticker = target_ticker.upper()

    # Ensure KB embeddings are computed
    try:
        compute_kb_embeddings()
        model = get_embedding_model() # Ensure model is loaded
        if not model:
             print("ERROR: Embedding model not available. Cannot perform embedding-based ER.")
             return [] # Or handle differently
    except Exception as load_err:
         print(f"ERROR during embedding setup: {load_err}")
         return []

    allowed_entity_types = {"ORG", "PER", "DATE", "MONEY"}
    processed_count = 0
    
    # Get precomputed Company embeddings from the global structure
    company_kb_embeddings = _kb_embeddings.get("Company", {})

    for item in ner_results:
        resolved_entities = []
        original_entities = item.get("entities", [])

        for entity in original_entities:
            entity_group = entity.get("entity_group")
            entity_word = entity.get("word", "").strip()

            if entity_group in allowed_entity_types and entity_word:
                resolved_entity = entity.copy()

                # 1. ORG Normalization (Embedding-based)
                if entity_group == "ORG":
                    try:
                        entity_embedding = model.encode(entity_word, convert_to_numpy=True)
                        
                        best_match_id = None
                        max_similarity = -1.0 # Initialize with value lower than possible similarity

                        # Compare against canonical Company embeddings
                        for canonical_id, kb_embed_list in company_kb_embeddings.items():
                            for kb_embedding in kb_embed_list:
                                sim = cosine_similarity(entity_embedding, kb_embedding)
                                if sim > max_similarity:
                                    max_similarity = sim
                                    best_match_id = canonical_id
                        
                        # Apply threshold
                        if max_similarity >= similarity_threshold:
                            # Found a confident match to a known company
                            resolved_entity["normalized_id"] = best_match_id
                            resolved_entity["normalized_label"] = "Company"
                            resolved_entity["match_score"] = float(max_similarity) # Store score
                        else:
                            # Not similar enough to a known company, treat as generic org
                            resolved_entity["normalized_label"] = "Organization"
                    except Exception as e:
                         print(f"Error during ORG entity embedding/matching for '{entity_word}': {e}")
                         resolved_entity["normalized_label"] = "Organization" # Fallback

                # 2. DATE Normalization (Rule-based - Keep existing)
                elif entity_group == "DATE":
                    normalized = normalize_date(entity_word)
                    if normalized:
                        resolved_entity["normalized_date"] = normalized
                    resolved_entity["normalized_label"] = "Date"

                # 3. PERSON Normalization (Placeholder - Keep existing)
                elif entity_group == "PER":
                    resolved_entity["normalized_label"] = "Person"

                # 4. MONEY Normalization (Placeholder - Keep existing)
                elif entity_group == "MONEY":
                    resolved_entity["normalized_label"] = "MonetaryAmount"

                # Only append if a label was assigned
                if "normalized_label" in resolved_entity:
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
    # Updated default and help text for the NER model argument
    parser.add_argument("--ner_model", type=str, default="en_core_web_trf", help="SpaCy NER model name (e.g., en_core_web_sm, en_core_web_lg, en_core_web_trf). Ensure it's downloaded.")
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
    # Optional: Add a check/download for the default spacy model if needed
    # try:
    #     spacy.load("en_core_web_trf")
    # except OSError:
    #     print("Default spaCy model 'en_core_web_trf' not found.")
    #     print("Downloading model...")
    #     try:
    #         spacy.cli.download("en_core_web_trf")
    #         print("Model downloaded successfully.")
    #     except Exception as download_err:
    #         print(f"Failed to download model: {download_err}")
    #         # Decide whether to exit or continue
    #         # exit(1)
    main() 