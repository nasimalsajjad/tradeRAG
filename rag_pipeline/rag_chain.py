# rag_pipeline/rag_chain.py
import os
import argparse
import logging
from dotenv import load_dotenv

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
    QueryBundle
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Neo4j imports
from neo4j import GraphDatabase, exceptions as neo4j_exceptions
import traceback

# Local LLM Query (now via RunPod)
# Need to adjust path if rag_chain is run from project root
# Assuming run from root: from utils.llm_utils import query_llm
# If run from rag_pipeline: from ../utils.llm_utils import query_llm
# For now, let's assume run from root for simplicity
import sys
sys.path.append(os.path.dirname(__file__)) # Add current dir to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add project root
from utils.llm_utils import query_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directories
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector_store')

# --- Neo4j Connection (Reusing logic from graph_ingestor) ---
def get_neo4j_driver():
    """Establishes connection to Neo4j using environment variables."""
    # Load config.env file from the project root
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'config.env')
    load_dotenv(dotenv_path=dotenv_path)

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    # --- TEMPORARY HARDCODED TEST --- REMOVED
    # uri = "bolt://localhost:7687"
    # user = "neo4j"
    # password = "changeme123"
    # logging.info("DEBUG: Using hardcoded credentials for connection test in rag_chain.")
    # --------------------------------

    if not uri or not user or not password:
        logging.error("NEO4J_URI, NEO4J_USERNAME, or NEO4J_PASSWORD not found in environment variables.")
        return None

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        logging.info("Successfully connected to Neo4j for querying.")
        return driver
    except neo4j_exceptions.AuthError as auth_err:
         logging.error(f"Neo4j Authentication Error: {auth_err}.")
         return None
    except neo4j_exceptions.ServiceUnavailable as conn_err:
         logging.error(f"Neo4j Connection Error: {conn_err}. Is Neo4j running?")
         return None
    except Exception as e:
        logging.error(f"Error connecting to Neo4j: {e}", exc_info=True)
        return None

# --- LlamaIndex Vector Store Loading ---
def load_vector_index(ticker: str):
    """Loads the persisted LlamaIndex vector store for a given ticker."""
    persist_dir = os.path.join(VECTOR_STORE_DIR, ticker.upper())
    if not os.path.exists(persist_dir):
        logging.error(f"Vector store directory not found for ticker {ticker} at: {persist_dir}")
        return None

    try:
        logging.info(f"Loading vector index from: {persist_dir}")
        # Ensure embedding model is configured (same as used for indexing)
        # Settings.embed_model should ideally be set globally once, but setting here is safe
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        logging.info(f"Successfully loaded vector index for {ticker}.")
        return index
    except Exception as e:
        logging.error(f"Error loading vector index for {ticker}: {e}", exc_info=True)
        return None

# --- RAG Query Logic ---
def run_rag_query(query: str, ticker: str):
    """Executes the RAG query using Vector Store and Knowledge Graph."""
    logging.info(f"Running RAG query for ticker '{ticker}': '{query}'")

    # 1. Load Vector Index
    vector_index = load_vector_index(ticker)
    if not vector_index:
        return "Error: Could not load vector index."

    # 2. Connect to Neo4j
    neo4j_driver = get_neo4j_driver()
    # Proceed even if Neo4j fails for now, relying only on vector search initially
    if not neo4j_driver:
        logging.warning("Could not connect to Neo4j. Proceeding with vector search only.")

    # 3. Retrieve from Vector Store
    logging.info("Step 1: Retrieving relevant chunks from Vector DB...")
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5) # Retrieve top 5 chunks (increased from 3)
    retrieved_nodes = vector_retriever.retrieve(query)
    
    if not retrieved_nodes:
        logging.warning("No relevant chunks found in vector store.")
        vector_context = "No relevant information found in text documents."
    else:
        logging.info(f"Retrieved {len(retrieved_nodes)} node(s) from vector store.")
        # Format vector context
        vector_context = "\n\n-- Relevant Text Context --\n"
        for i, node in enumerate(retrieved_nodes):
             # Accessing text content via node.get_content()
            vector_context += f"Chunk {i+1} (Source: {node.metadata.get('source_file', 'unknown')}, Score: {node.score:.4f}):\n"
            vector_context += node.get_content(metadata_mode="none").strip()
            vector_context += "\n---\n"
        # print(vector_context) # Debug print

    # 4. Retrieve from Knowledge Graph (Placeholder)
    kg_context = "" # Initialize KG context as empty
    if neo4j_driver:
        logging.info("Step 2: Expanding context with linked facts from Neo4j KG (Placeholder)...")
        # TODO: Implement KG retrieval based on query or entities from vector context
        # Example: Extract entities (Company, Person) from query or retrieved_nodes
        # Run Cypher queries like:
        # MATCH (c:Company {ticker: $ticker})-[:MENTIONED_IN]->(d:Document)<-[:MENTIONED_IN]-(p:Person)
        # WHERE p.name CONTAINS $person_name
        # RETURN c.ticker, p.name, d.source_file, d.document_date
        # kg_context = "\n\n-- Related Knowledge Graph Facts --\nKnowledge Graph retrieval not yet implemented.\n---\n" # <-- Commenting this out
        # Remember to close the driver session if opened here
        neo4j_driver.close() # Close connection if we opened it

    # 5. Construct Prompt for LLM
    logging.info("Step 3: Constructing prompt for LLM...")
    final_prompt = f"""
User Query: {query}
Ticker Focus: {ticker}

Based on the following context from financial documents and knowledge graph facts, please answer the user query.

{vector_context}
{kg_context}
Answer:
"""
    # print(f"\n--- Final Prompt ---\n{final_prompt}") # Debug print

    # 6. Query LLM (via RunPod)
    logging.info("Step 4: Sending context and query to LLM...")
    final_answer = query_llm(final_prompt)

    if not final_answer:
        return "Error: Failed to get response from LLM."

    return final_answer

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a RAG query using Vector Store and Knowledge Graph.")
    parser.add_argument("-q", "--query", type=str, required=True, help="The question to ask.")
    parser.add_argument("-t", "--ticker", type=str, required=True, help="The target stock ticker (e.g., AMD).")

    args = parser.parse_args()

    # Run the RAG query
    result = run_rag_query(args.query, args.ticker.upper())

    print("\n======== Query Result ========")
    print(result)
    print("==============================") 