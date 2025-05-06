# rag_pipeline/rag_chain.py
import os
import argparse
import logging
from dotenv import load_dotenv
import spacy
import numpy as np

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
from neo4j.time import Date as Neo4jDate

# Local LLM Query (now via RunPod)
# Need to adjust path if rag_chain is run from project root
# Assuming run from root: from utils.llm_utils import query_llm
# If run from rag_pipeline: from ../utils.llm_utils import query_llm
# For now, let's assume run from root for simplicity
import sys
sys.path.append(os.path.dirname(__file__)) # Add current dir to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add project root
from utils.llm_utils import query_llm

# Import from ner_er for entity resolution logic
# Assuming ner_er.py is in the same directory (rag_pipeline)
try:
    from ner_er import (
        get_embedding_model,
        compute_kb_embeddings,
        cosine_similarity,
        normalize_date,
        canonical_kb,
        _kb_embeddings # Import the global dict holding precomputed KB embeddings
    )
except ImportError as e:
     logging.error(f"Could not import from ner_er.py: {e}. Ensure it's in the same directory.")
     # Define dummy functions/variables if import fails to allow script to load
     def get_embedding_model(model_name=None): return None
     def compute_kb_embeddings(): pass
     def cosine_similarity(v1, v2): return 0.0
     def normalize_date(d): return None
     canonical_kb = {"Company": {}}
     _kb_embeddings = {}

# Configure logging
# Avoid duplicate handlers if already configured by llm_utils
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define directories
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector_store')

# --- Global Variables for Models (Load Once) ---
_spacy_nlp = None
_query_embedding_model = None # Separate variable for query embedding model if needed

# --- spaCy Model Loading ---
def _load_spacy_model(model_name="en_core_web_trf"):
    """Loads the spaCy model, handling potential errors."""
    global _spacy_nlp
    if _spacy_nlp is None:
        logger.info(f"Initializing spaCy NER model: {model_name}...")
        try:
            # Check if model is available, download if necessary (optional)
            # try:
            #     spacy.load(model_name)
            # except OSError:
            #     logger.warning(f"SpaCy model '{model_name}' not found. Attempting download...")
            #     spacy.cli.download(model_name)
                
            _spacy_nlp = spacy.load(model_name)
            # Increase max length for potentially long query contexts if needed, though less likely for queries
            _spacy_nlp.max_length = 1500000 # Keep consistent with ner_er for now
            logger.info(f"SpaCy NER model '{model_name}' loaded successfully.")
            logger.info(f"Loaded spaCy pipeline components: {_spacy_nlp.pipe_names}")
            if 'ner' not in _spacy_nlp.pipe_names:
                 logger.error(f"'ner' component not found in the loaded spaCy model '{model_name}'. Cannot extract query entities.")
                 _spacy_nlp = None # Mark as failed
        except OSError as e:
            logger.error(f"Error loading spaCy model '{model_name}': {e}. Please ensure it's downloaded.")
            _spacy_nlp = None
        except Exception as e:
            logger.error(f"An unexpected error occurred during spaCy model loading: {e}", exc_info=True)
            _spacy_nlp = None
    return _spacy_nlp

# --- Entity Extraction and Resolution for Query ---
def extract_resolve_query_entities(query: str, target_ticker: str, similarity_threshold=0.85):
    """Extracts and resolves entities (ORG, PER, DATE) from the user query."""
    logger.info(f"Extracting/resolving entities from query: '{query}'")
    resolved_entities = []
    target_ticker = target_ticker.upper()
    
    # Ensure spaCy model is loaded
    nlp = _load_spacy_model()
    if not nlp:
        logger.warning("SpaCy model not loaded, cannot extract entities from query.")
        return resolved_entities
        
    # Ensure KB embeddings and embedding model are loaded for ORG resolution
    try:
        compute_kb_embeddings()
        embedding_model = get_embedding_model()
        if not embedding_model:
             logger.warning("Embedding model not available. Cannot resolve ORG entities in query.")
             # Continue without embedding resolution for ORGs
        company_kb_embeddings = _kb_embeddings.get("Company", {})
    except Exception as load_err:
        logger.error(f"Error during embedding setup for query resolution: {load_err}", exc_info=True)
        embedding_model = None # Mark as failed
        company_kb_embeddings = {}

    allowed_entity_types = {"ORG", "PERSON", "DATE", "MONEY"} # SpaCy labels might differ slightly (e.g., PER)
    
    try:
        doc = nlp(query)
        logger.info(f"Query NER raw results: {[(ent.text, ent.label_) for ent in doc.ents]}")

        for ent in doc.ents:
            entity_group = ent.label_
            entity_word = ent.text.strip()
            
            # Map SpaCy labels if necessary (e.g., PER to PERSON if KB uses PERSON)
            normalized_group = "Person" if entity_group == "PER" else entity_group

            if normalized_group in allowed_entity_types and entity_word:
                resolved_entity = {
                    "word": entity_word,
                    "label": normalized_group, # Use consistent label
                    "start": ent.start_char,
                    "end": ent.end_char
                }

                # 1. ORG Resolution (Embedding-based)
                if normalized_group == "ORG" and embedding_model and company_kb_embeddings:
                    try:
                        entity_embedding = embedding_model.encode(entity_word, convert_to_numpy=True)
                        best_match_id = None
                        max_similarity = -1.0

                        for canonical_id, kb_embed_list in company_kb_embeddings.items():
                            for kb_embedding in kb_embed_list:
                                sim = cosine_similarity(entity_embedding, kb_embedding)
                                if sim > max_similarity:
                                    max_similarity = sim
                                    best_match_id = canonical_id
                        
                        if max_similarity >= similarity_threshold:
                            # If the resolved company is the same as the target ticker, ignore it for graph context generation
                            if best_match_id == target_ticker:
                                logger.info(f"Ignoring resolved ORG entity '{entity_word}' matching target ticker '{target_ticker}'.")
                                continue # Skip adding this entity
                            resolved_entity["label"] = "Company" # Promote to Company
                            resolved_entity["ticker"] = best_match_id # Add ticker ID
                            resolved_entity["score"] = float(max_similarity)
                        else:
                            resolved_entity["label"] = "Organization" # Keep as generic Org
                    except Exception as e:
                         logger.warning(f"Error during ORG entity embedding/matching for query word '{entity_word}': {e}", exc_info=True)
                         resolved_entity["label"] = "Organization" # Fallback

                # 2. DATE Resolution (Rule-based)
                elif normalized_group == "DATE":
                    normalized = normalize_date(entity_word)
                    if normalized:
                        resolved_entity["iso_date"] = normalized
                    # Keep the entity even if normalization fails, maybe query based on text? (future enhancement)

                # 3. PERSON/MONEY (Keep as is for now)
                elif normalized_group in ["Person", "MONEY"]:
                    pass # No specific resolution yet, label already set

                resolved_entities.append(resolved_entity)

    except Exception as e:
        logger.error(f"Error processing query with spaCy NER: {e}", exc_info=True)

    logger.info(f"Resolved query entities: {resolved_entities}")
    return resolved_entities


# --- Cypher Query Generation (Enhanced) ---
def generate_cypher_queries(target_ticker: str, resolved_entities: list[dict]) -> list[tuple[str, str, dict]]:
    """Generates Cypher queries based on the target ticker and resolved entities from the query."""
    queries = []
    base_params = {"target_ticker": target_ticker}

    # Separate entities by type for easier logic
    company_entities = [e for e in resolved_entities if e["label"] == "Company"] # Already resolved to tickers
    person_entities = [e for e in resolved_entities if e["label"] == "Person"]
    date_entities = [e for e in resolved_entities if e["label"] == "Date" and "normalized_date" in e]
    # TODO: Add MonetaryAmount handling if needed

    # --- Query Generation Logic ---

    # Helper to add a query if it's not a duplicate description
    existing_descriptions = set()
    def add_query(desc, query, params):
        if desc not in existing_descriptions:
            queries.append((desc, query, params))
            existing_descriptions.add(desc)

    # === Scenario 1: Specific Date(s) mentioned ===
    if date_entities:
        params = base_params.copy()
        date_params = {}
        date_match_clauses = []
        for i, date_entity in enumerate(date_entities):
            try:
                param_name = f"date_{i}"
                year, month, day = map(int, date_entity["normalized_date"].split('-'))
                date_params[param_name] = Neo4jDate(year, month, day)
                date_match_clauses.append(f"(date_node:Date {{date: ${param_name}}})-[:MENTIONED_IN]->(d)")
            except (ValueError, TypeError):
                 print(f"Warning: Skipping invalid date for Cypher: {date_entity.get('normalized_date')}")
                 continue # Skip if date is invalid

        if date_match_clauses: # Only proceed if we have valid dates
            params.update(date_params)
            # Query: Find documents for the target company mentioned on specific dates
            query = (
                f"""
                MATCH (target_co:Company {{ticker: $target_ticker}})<-[:MENTIONED_IN]-(d:Document),
                      {' MATCH '.join(date_match_clauses)}
                RETURN d.source_file AS document, d.date_string AS date
                ORDER BY d.date DESC NULLS LAST
                LIMIT 5
                """
            )
            add_query(f"Find docs for {target_ticker} on specific date(s)", query, params)

            # Query: Find *other* companies mentioned alongside target on specific dates
            query_other_co = (
                f"""
                MATCH (target_co:Company {{ticker: $target_ticker}})<-[:MENTIONED_IN]-(d:Document),
                      {' MATCH '.join(date_match_clauses)},
                      (other_co:Company)<-[:MENTIONED_IN]-(d)
                WHERE other_co <> target_co
                RETURN DISTINCT other_co.ticker AS other_company, d.source_file AS document, d.date_string AS date
                ORDER BY d.date DESC NULLS LAST, other_co.ticker
                LIMIT 10
                """
            )
            add_query(f"Find other companies mentioned with {target_ticker} on specific date(s)", query_other_co, params)

    # === Scenario 2: Other Company mentioned ===
    if company_entities:
        # Take the first other company mentioned for simplicity in this example
        other_company = company_entities[0]
        other_ticker = other_company.get("ticker")
        if other_ticker and other_ticker != target_ticker: # Ensure it's valid and not the target itself
            params = base_params.copy()
            params["other_ticker"] = other_ticker

            # Query 2a: Find documents where BOTH companies are mentioned (via direct MENTIONED_IN)
            query_both_mentioned = (
                """
                MATCH (target_co:Company {ticker: $target_ticker})<-[:MENTIONED_IN]-(d:Document),
                      (other_co:Company {ticker: $other_ticker})<-[:MENTIONED_IN]-(d)
                RETURN d.source_file AS document, d.date_string AS date
                ORDER BY d.date DESC NULLS LAST
                LIMIT 5
                """
            )
            add_query(f"Find docs mentioning both {target_ticker} and {other_ticker}", query_both_mentioned, params)

            # Query 2b: Find documents where target mentions the other company (using MENTIONS relationship)
            query_target_mentions_other = (
                 """
                 MATCH (target_co:Company {ticker: $target_ticker})-[m:MENTIONS]-(other_co:Company {ticker: $other_ticker})
                 // Find documents associated with this mention relationship
                 MATCH (d:Document) WHERE d.doc_id IN m.source_doc_ids
                 RETURN d.source_file AS document, d.date_string AS date
                 ORDER BY d.date DESC NULLS LAST
                 LIMIT 5
                 """
            )
            add_query(f"Find docs where {target_ticker} mentions {other_ticker}", query_target_mentions_other, params)


    # === Scenario 3: Person mentioned ===
    if person_entities:
        person = person_entities[0] # Take first person
        params = base_params.copy()
        params["person_name"] = person["word"]

        # Query 3a: Find documents mentioning both the target company AND the person
        query_person_and_company = (
            """
            MATCH (target_co:Company {ticker: $target_ticker})<-[:MENTIONED_IN]-(d:Document),
                  (p:Person {name: $person_name})<-[:MENTIONED_IN]-(d)
            RETURN d.source_file AS document, d.date_string AS date
            ORDER BY d.date DESC NULLS LAST
            LIMIT 5
            """
        )
        add_query(f"Find docs mentioning {target_ticker} and {params['person_name']}", query_person_and_company, params)

        # Query 3b: Find *other* companies mentioned in documents involving the target company and the person
        query_other_co_with_person = (
            """
            MATCH (target_co:Company {ticker: $target_ticker})<-[:MENTIONED_IN]-(d:Document),
                  (p:Person {name: $person_name})<-[:MENTIONED_IN]-(d),
                  (other_co:Company)<-[:MENTIONED_IN]-(d)
            WHERE other_co <> target_co
            RETURN DISTINCT other_co.ticker AS other_company, d.source_file AS document, d.date_string AS date
            ORDER BY d.date DESC NULLS LAST, other_co.ticker
            LIMIT 10
            """
        )
        add_query(f"Find other companies mentioned with {target_ticker} and {params['person_name']}", query_other_co_with_person, params)


    # === Default Case: Only target ticker or unhandled combination ===
    if not queries:
        params = base_params.copy()
        # Default Query: Find recent documents mentioning the target company
        query = (
            """
            MATCH (d:Document {ticker: $target_ticker})
            // Optional: Also match via Company node relationship if direct ticker fails
            // OPTIONAL MATCH (c:Company {ticker: $target_ticker})<-[:MENTIONED_IN]-(d_rel:Document)
            // WITH d, d_rel
            // RETURN coalesce(d, d_rel).source_file AS document, coalesce(d, d_rel).date_string AS date
            RETURN d.source_file AS document, d.date_string AS date
            ORDER BY d.date DESC NULLS LAST
            LIMIT 5
            """
        )
        add_query(f"Find recent docs for {target_ticker}", query, params)

    # Log the generated queries
    logger.info(f"Generated {len(queries)} KG queries based on resolved entities:")
    for i, (desc, q, p) in enumerate(queries):
        logger.info(f"  Query {i+1} ({desc}): Params={p}")
        # logger.debug(f"    Cypher: {q}") # Log query only in debug mode

    return queries


# --- Neo4j Query Execution ---
def execute_cypher_query(driver, query: str, params: dict = None):
    """Executes a given Cypher query with parameters and returns results."""
    results = []
    try:
        with driver.session(database="neo4j") as session:
            response = session.run(query, params or {})
            # Convert Neo4j Records to dictionaries for easier handling
            results = [record.data() for record in response]
            return results
    except neo4j_exceptions.Neo4jError as db_error:
        logger.error(f"Neo4j Query Error executing: {query} with params {params}")
        logger.error(f"Database error: {db_error}")
        # Optionally re-raise or return specific error info
        return [{"error": f"Neo4j Query Error: {db_error.message}"}]
    except Exception as e:
        logger.error(f"Unexpected Error executing Cypher: {query} with params {params}")
        logger.error(f"Error details: {e}")
        traceback.print_exc()
        return [{"error": f"Unexpected error during query execution: {e}"}]


# --- Format KG Results ---
def format_kg_results(results_list: list[tuple[str, list[dict]]]) -> str:
    """Formats the results from multiple KG queries into a readable string."""
    if not results_list:
        return "No information found in the knowledge graph for the query specifics."

    formatted_string = "Knowledge Graph Information:\n"
    has_content = False

    for description, results in results_list:
        if results:
            # Check for error marker
            if len(results) == 1 and "error" in results[0]:
                 formatted_string += f"- Query '{description}' failed: {results[0]['error']}\n"
                 continue # Skip formatting if query failed

            has_content = True
            formatted_string += f"- {description}:\n"
            # Limit results display if too many
            display_limit = 10
            for i, record in enumerate(results[:display_limit]):
                record_str = ", ".join([f"{key}: {value}" for key, value in record.items()])
                formatted_string += f"  - {record_str}\n"
            if len(results) > display_limit:
                formatted_string += f"  - ... (and {len(results) - display_limit} more)\n"

    if not has_content:
        return "No specific information found in the knowledge graph based on the entities in your query."

    return formatted_string.strip()


# --- Main RAG Function ---
def run_rag_query(query: str, ticker: str) -> str:
    """
    Runs the full RAG query process:
    1. Load Vector Index
    2. Retrieve relevant chunks
    3. Resolve Entities in Query
    4. Generate & Execute KG Queries (Cypher)
    5. Combine Context (Vector + KG)
    6. Query LLM
    """
    logger.info(f"--- RAG Query Start ---")
    logger.info(f"Ticker: {ticker}")
    logger.info(f"Query: {query}")

    # 0. Setup (Ensure models/drivers are ready)
    # Load environment variables (needed for Neo4j connection and LLM)
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'config.env')
    load_dotenv(dotenv_path=dotenv_path)

    # Ensure spaCy model is loaded
    global nlp_ner
    if nlp_ner is None:
        load_spacy_model()
        if nlp_ner is None:
            return "Error: Failed to load spaCy NER model."

    # Ensure embedding model is ready (needed for entity resolution)
    global embedding_model_er
    if embedding_model_er is None:
        embedding_model_er = get_embedding_model() # From ner_er

    # Ensure Neo4j driver is ready
    neo4j_driver = get_neo4j_driver()
    if not neo4j_driver:
        # Attempt to return KG error without failing entire query?
        kg_context = "Error: Could not connect to Knowledge Graph."
    else:
        kg_context = "" # Initialize KG context

    # 1. Load Vector Index
    try:
        logger.info(f"Loading vector index for {ticker}...")
        index_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector_store', ticker)
        if not os.path.exists(index_dir):
            return f"Error: Vector index for ticker '{ticker}' not found at {index_dir}. Please run the data pipeline first."

        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.embed_model = embed_model
        index = load_index_from_storage(storage_context)
        logger.info(f"Vector index for {ticker} loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load vector index for {ticker}: {e}")
        traceback.print_exc()
        return f"Error: Failed to load vector store for {ticker}."

    # 2. Retrieve relevant chunks
    logger.info("Retrieving relevant documents from vector store...")
    retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
    retrieved_nodes = retriever.retrieve(QueryBundle(query))
    vector_context = "\n\n".join([node.get_content() for node in retrieved_nodes])
    logger.info(f"Retrieved {len(retrieved_nodes)} document chunks.")
    # logger.debug(f"Vector Context:\n{vector_context[:500]}...") # Log snippet

    # === Knowledge Graph Integration ===
    if neo4j_driver:
        try:
            # 3. Resolve Entities in Query
            logger.info("Resolving entities in the user query...")
            resolved_entities_query = extract_resolve_query_entities(query, ticker)
            logger.info(f"Resolved entities from query: {resolved_entities_query}")

            # 4. Generate & Execute KG Queries
            if resolved_entities_query:
                logger.info("Generating Cypher queries based on resolved entities...")
                cypher_queries_info = generate_cypher_queries(ticker, resolved_entities_query)

                kg_query_results = []
                if cypher_queries_info:
                    logger.info(f"Executing {len(cypher_queries_info)} KG queries...")
                    for i, (desc, cypher_q, params) in enumerate(cypher_queries_info):
                        logger.info(f"  Executing KG Query {i+1} ({desc})...")
                        results = execute_cypher_query(neo4j_driver, cypher_q, params)
                        logger.info(f"  Query {i+1} returned {len(results)} results.")
                        kg_query_results.append((desc, results))

                    # 5. Format KG Results
                    kg_context = format_kg_results(kg_query_results)
                    logger.info("Knowledge Graph context generated.")
                    # logger.debug(f"KG Context:\n{kg_context}")
                else:
                    logger.info("No specific Cypher queries generated based on resolved entities.")
                    kg_context = "No specific information found in knowledge graph based on query entities."
            else:
                logger.info("No entities resolved from the query for KG lookup.")
                kg_context = "No specific entities identified in the query for knowledge graph lookup."

        except Exception as kg_err:
            logger.error(f"Error during Knowledge Graph query phase: {kg_err}")
            traceback.print_exc()
            kg_context = f"Error querying Knowledge Graph: {kg_err}"
        finally:
            # Ensure driver is closed if it was successfully created
            if neo4j_driver:
                neo4j_driver.close()
                logger.info("Neo4j driver closed.")
    # === End Knowledge Graph Integration ===

    # 6. Combine Context
    logger.info("Combining vector context and KG context...")
    combined_context = f"Vector Store Context:\n{vector_context}\n\n{kg_context}"

    # 7. Query LLM
    logger.info("Querying LLM with combined context...")
    prompt = f"""
User Query: {query}
Ticker: {ticker}

Based ONLY on the following context, answer the user query. Do not use any prior knowledge. If the context does not contain the answer, say "The provided context does not contain specific information to answer this query."

Context:
---
{combined_context}
---

Answer:
"""
    try:
        response = query_llm(prompt)
        logger.info("LLM response received.")
        # logger.debug(f"LLM Raw Response: {response}")
        final_answer = response # Assuming query_llm returns the answer string directly
    except Exception as llm_err:
        logger.error(f"Error querying LLM: {llm_err}")
        traceback.print_exc()
        final_answer = f"Error: Failed to get response from LLM ({llm_err})"

    logger.info(f"--- RAG Query End ---")
    return final_answer

# --- Main Execution (for testing) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG query for a given ticker and query.")
    parser.add_argument("-t", "--ticker", type=str, required=True, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("-q", "--query", type=str, required=True, help="Query string")

    args = parser.parse_args()

    # Load environment variables (especially for API keys needed by LLM utils or Neo4j)
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'config.env'))

    final_answer = run_rag_query(query=args.query, ticker=args.ticker)

    print("\n--- Final RAG Answer ---")
    print(final_answer) 