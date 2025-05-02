import os
import json
import argparse
from dotenv import load_dotenv, find_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore

# Define directories
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector_store')

def load_processed_data(ticker: str) -> list[Document]:
    """Loads processed data from the JSONL file for a ticker."""
    processed_file = os.path.join(PROCESSED_DIR, ticker, 'processed_data.jsonl')
    if not os.path.exists(processed_file):
        print(f"Error: Processed data file not found at {processed_file}")
        return []

    llama_documents = []
    try:
        with open(processed_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # Create a LlamaIndex Document object
                # Metadata is crucial for filtering/context later
                doc = Document(
                    text=data.get('text', ''),
                    metadata={
                        'ticker': data.get('ticker', ticker),
                        'source_type': data.get('source_type', 'unknown'),
                        'source_file': data.get('source_file', 'unknown'),
                        'date': data.get('date', 'unknown'),
                        # Add other relevant metadata fields from preprocessor output
                        'news_id': data.get('news_id'),
                        'headline': data.get('headline'),
                        'url': data.get('url'),
                        'news_source': data.get('news_source'),
                    },
                    # Exclude metadata fields that might be too large or complex if needed
                    excluded_llm_metadata_keys=['news_id', 'url', 'news_source', 'headline'],
                    excluded_embed_metadata_keys=['news_id', 'url', 'news_source', 'headline']
                )
                # Remove None values from metadata
                doc.metadata = {k: v for k, v in doc.metadata.items() if v is not None}
                llama_documents.append(doc)
        print(f"Loaded {len(llama_documents)} documents from {processed_file}")
    except Exception as e:
        print(f"Error loading or processing {processed_file}: {e}")
    return llama_documents

def build_and_persist_index(ticker: str, documents: list[Document]):
    """Builds or updates the vector index for a ticker and persists it."""
    if not documents:
        print("No documents provided to build index. Skipping.")
        return

    # --- Configure LlamaIndex Components ---
    # No longer need to load OpenAI key for embeddings
    # dotenv_path = find_dotenv()
    # print(f"Attempting to load .env file from: {dotenv_path}")
    # load_dotenv(dotenv_path=dotenv_path, verbose=True)
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # if not openai_api_key:
    #     print("Error: OPENAI_API_KEY not found in environment variables after loading .env.")
    #     print("Cannot proceed with OpenAI embeddings.")
    #     return

    # Set up global settings (Newer LlamaIndex way)
    # Use HuggingFace embeddings
    print("Setting up HuggingFace embedding model (all-MiniLM-L6-v2)... This may download the model.")
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # You can adjust chunk_size and chunk_overlap
    Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    Settings.llm = None # Not needed for indexing stage

    # Define storage path for this ticker
    persist_dir = os.path.join(VECTOR_STORE_DIR, ticker)
    os.makedirs(persist_dir, exist_ok=True)
    print(f"Vector store persistence directory: {persist_dir}")

    # --- Check if index exists --- 
    try:
        # Try loading first to avoid rebuilding if possible (though current logic rebuilds)
        print(f"Checking for existing index at {persist_dir}...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        # Simple check if index exists - load_index_from_storage raises error if not fully present
        index = load_index_from_storage(storage_context, embed_model=Settings.embed_model)
        print(f"Loaded existing index from {persist_dir}")
        # TODO: Implement update logic if needed (e.g., insert new docs, delete old)
        # For now, we'll just rebuild if it exists, which is simpler but less efficient
        print("Rebuilding index from scratch using HuggingFace embeddings...")
        index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True
        )
        index.storage_context.persist(persist_dir=persist_dir)

    except FileNotFoundError:
        print(f"No existing index found at {persist_dir}. Building new index with HuggingFace embeddings...")
        # Build index from scratch
        index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True
        )
        # Persist index to disk
        index.storage_context.persist(persist_dir=persist_dir)
    
    except Exception as e:
         # Catch other potential errors during index loading/building
         print(f"An error occurred during index loading/building: {e}")
         # Print traceback for more details if needed
         # import traceback
         # traceback.print_exc()
         return

    print(f"Index building/persistence for {ticker} complete.")


def main():
    parser = argparse.ArgumentParser(description="Chunk and index processed data for a ticker using LlamaIndex and HuggingFace embeddings.")
    parser.add_argument("-t", "--ticker", type=str, required=True, help="Stock ticker symbol to index (e.g., AAPL)")
    # Add flags like --rebuild later if needed

    args = parser.parse_args()
    ticker = args.ticker.upper()

    print(f"--- Starting Chunking and Indexing for {ticker} ---")
    # 1. Load processed data into LlamaIndex Documents
    documents = load_processed_data(ticker)

    # 2. Build and persist the vector index
    if documents:
        build_and_persist_index(ticker, documents)
    else:
        print("No documents loaded, skipping index building.")
    
    print(f"--- Finished Chunking and Indexing for {ticker} ---")

if __name__ == "__main__":
    main() 