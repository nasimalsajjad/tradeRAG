# Placeholder for Graph Ingestion logic
import os
import json
import argparse
from neo4j import GraphDatabase, exceptions as neo4j_exceptions
from dotenv import load_dotenv
import traceback

print("graph_ingestor.py loaded")

# Define directories
KG_INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'kg_input')

# --- Neo4j Connection ---

def get_neo4j_driver():
    """Establishes connection to Neo4j using environment variables."""
    # Load config.env file from the project root
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'config.env')
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded config.env from: {dotenv_path}")

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    # --- HARDCODING NEO4J CREDS TO BYPASS DOTENV ISSUES --- REMOVED
    # uri = "bolt://localhost:7687"
    # user = "neo4j"
    # password = "changeme123"
    # print("DEBUG: Using hardcoded credentials for Neo4j connection.")
    # ------------------------------------------------------

    print(f"Read from env: URI={uri}, User={user}, Password={'*' * len(password) if password else None}") # Debug print
    # TEMPORARY DEBUG: Print the actual password and its length - REMOVING
    # if password:
    #   print(f"DEBUG: Attempting connection with password: '{password}' (Length: {len(password)})")
    # else:
    #   print("DEBUG: Password read from env is None or empty.")

    if not uri or not user or not password:
        print("Error: NEO4J_URI, NEO4J_USERNAME, or NEO4J_PASSWORD not found in environment variables.")
        print("Please ensure they are set in your .env file.")
        return None

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("Successfully connected to Neo4j.")
        return driver
    except neo4j_exceptions.AuthError as auth_err:
         print(f"Neo4j Authentication Error: {auth_err}. Check username/password in .env")
         return None
    except neo4j_exceptions.ServiceUnavailable as conn_err:
         print(f"Neo4j Connection Error: {conn_err}. Is the Neo4j server running at {uri}?")
         return None
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        traceback.print_exc() # Print full traceback for unexpected errors
        return None

# --- Graph Creation Logic ---

def create_document_node(tx, metadata):
    """Creates or merges a Document node based on a unique doc_id."""
    source_type = metadata.get('source_type', 'unknown')
    source_file = metadata.get('source_file', 'unknown')
    ticker = metadata.get('ticker', 'unknown')
    doc_date = metadata.get('date', 'unknown') # Original date string from metadata

    # Create a unique ID for the document node
    doc_id = f"{source_type}::{source_file}"

    # Properties to set on the document node
    props = {
        "source_type": source_type,
        "source_file": source_file,
        "ticker": ticker,
        "document_date": doc_date # Store original date info
    }

    # Cypher query to MERGE the document node
    query = (
        "MERGE (d:Document {doc_id: $doc_id}) "
        "SET d += $props " # Use SET to update properties on match or create
        "RETURN d.doc_id as doc_id" # Return the property instead of internal id()
    )
    result = tx.run(query, doc_id=doc_id, props=props)
    record = result.single()
    return record["doc_id"] if record else None # Return the doc_id string

def create_entity_node_and_relationship(tx, document_id, entity):
    """Creates/merges an Entity node and links it via MENTIONED_IN to the Document."""
    # Match the document node using its doc_id property
    match_doc_query = "MATCH (doc:Document {doc_id: $document_id}) "

    label = entity.get('normalized_label')
    word = entity.get('word')
    if not label or not word:
        print(f"Skipping entity due to missing label or word: {entity}")
        return

    # Prepare parameters for entity node merging
    params = {"word": word}
    merge_clause = ""
    properties_clause = "ON CREATE SET e.created_at = timestamp() " # Common property

    if label == "Company":
        ticker = entity.get("normalized_id")
        if not ticker:
            print(f"Skipping Company entity due to missing normalized_id: {entity}")
            return
        params["ticker"] = ticker
        merge_clause = f"MERGE (e:{label} {{ticker: $ticker}}) "
        properties_clause += "ON CREATE SET e.name = $word " # Set name only on creation
    elif label == "Person":
        params["name"] = word
        merge_clause = f"MERGE (e:{label} {{name: $name}}) "
    elif label == "Date":
        iso_date = entity.get("normalized_date")
        if not iso_date:
            print(f"Skipping Date entity due to missing normalized_date: {entity}")
            return
        params["iso_date"] = iso_date
        merge_clause = f"MERGE (e:{label} {{iso_date: $iso_date}}) "
        properties_clause += "ON CREATE SET e.display_text = $word "
    elif label == "Organization":
        params["name"] = word
        merge_clause = f"MERGE (e:{label} {{name: $name}}) "
    elif label == "MonetaryAmount":
        # Simple approach: use the word as the primary key for now
        # TODO: Enhance to extract numeric value and currency
        params["value_text"] = word
        merge_clause = f"MERGE (e:{label} {{value_text: $value_text}}) "
    else:
        print(f"Skipping entity with unhandled label: {label}")
        return

    # Merge the relationship
    # Add entity properties like score, start, end to the relationship
    rel_props = {
        "score": entity.get("score", 0.0),
        "start_char": entity.get("start"),
        "end_char": entity.get("end")
    }
    # Filter out None values from relationship properties
    rel_props = {k: v for k, v in rel_props.items() if v is not None}

    merge_rel_clause = "MERGE (e)-[r:MENTIONED_IN]->(doc) SET r += $rel_props"

    # Combine the query parts
    full_query = f"{match_doc_query} {merge_clause} {properties_clause} {merge_rel_clause}"

    # Add document_id and relationship properties to parameters
    params["document_id"] = document_id
    params["rel_props"] = rel_props

    # Execute the query
    tx.run(full_query, **params)

# --- Main Ingestion Function ---

def ingest_data_to_neo4j(driver, ticker: str):
    """Loads resolved entities and ingests them into Neo4j."""
    resolved_file = os.path.join(KG_INPUT_DIR, ticker, 'entities_resolved.jsonl')
    if not os.path.exists(resolved_file):
        print(f"Error: Resolved entities file not found at {resolved_file}")
        return

    print(f"Starting ingestion from {resolved_file}...")
    processed_lines = 0
    processed_docs_success = 0
    error_count = 0
    skipped_entities = 0
    with open(resolved_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            processed_lines += 1
            try:
                data = json.loads(line)
                metadata = data.get("metadata", {})
                resolved_entities = data.get("resolved_entities", [])

                if not metadata or not metadata.get('source_file'):
                    print(f"Skipping line {line_num} due to missing metadata or source_file: {line.strip()}")
                    error_count += 1
                    continue

                # Use a managed transaction for each document
                # This ensures atomicity for the document and its entities
                with driver.session(database="neo4j") as session: # Specify database if needed
                    try:
                        # Create the Document node first
                        # We pass the function and its arguments to execute_write
                        document_id = session.execute_write(create_document_node, metadata)

                        if not document_id:
                             print(f"Skipping entities for document {metadata.get('source_file')} on line {line_num} due to node creation failure.")
                             error_count += 1
                             continue

                        # Create entity nodes and relationships
                        entities_in_doc = 0
                        for entity in resolved_entities:
                            # Pass the necessary function and its arguments
                            session.execute_write(create_entity_node_and_relationship, document_id, entity)
                            entities_in_doc += 1

                        processed_docs_success += 1
                        if processed_docs_success % 50 == 0:
                             print(f"  Successfully processed {processed_docs_success} documents (current line: {line_num})...")

                    except Exception as tx_error:
                        print(f"Error during transaction for document {metadata.get('source_file')} on line {line_num}: {tx_error}")
                        traceback.print_exc() # Print stack trace for transaction errors
                        error_count += 1

            except json.JSONDecodeError as json_err:
                print(f"Skipping line {line_num} due to JSON decode error: {json_err} - Line: {line.strip()}")
                error_count += 1
            except Exception as e:
                print(f"Skipping line {line_num} due to unexpected error: {e} - Line: {line.strip()}")
                traceback.print_exc() # Print stack trace for other errors
                error_count += 1

    print(f"Finished ingestion. Processed {processed_lines} lines.")
    print(f"Successfully ingested data for {processed_docs_success} documents.")
    print(f"Encountered {error_count} errors/skipped lines.")


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Ingest resolved entities into Neo4j.")
    parser.add_argument("-t", "--ticker", type=str, required=True, help="Stock ticker symbol to ingest data for (e.g., AMD)")

    args = parser.parse_args()
    ticker = args.ticker.upper()

    print(f"--- Starting Neo4j Ingestion for ticker: {ticker} ---")

    driver = get_neo4j_driver()

    if driver:
        try:
            # Optional: Add constraints for uniqueness beforehand for better performance and data integrity
            print("Applying Neo4j constraints (if they don't exist)...")
            with driver.session(database="neo4j") as session:
                try:
                    session.run("CREATE CONSTRAINT company_ticker IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE")
                    session.run("CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE")
                    session.run("CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE")
                    session.run("CREATE CONSTRAINT date_iso IF NOT EXISTS FOR (dt:Date) REQUIRE dt.iso_date IS UNIQUE")
                    session.run("CREATE CONSTRAINT org_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE")
                    session.run("CREATE CONSTRAINT money_value IF NOT EXISTS FOR (m:MonetaryAmount) REQUIRE m.value_text IS UNIQUE") # Simple constraint for now
                    print("Constraints applied successfully or already exist.")
                except neo4j_exceptions.ClientError as constraint_err:
                     # Ignore errors if constraints already exist, handle others
                     if "already exists" in str(constraint_err):
                          print("Constraints already exist.")
                     else:
                          print(f"Error applying constraints: {constraint_err}")
                          # Decide if you want to continue without constraints or stop

            # Start the main ingestion process
            ingest_data_to_neo4j(driver, ticker)

        finally:
            driver.close()
            print("Neo4j connection closed.")
    else:
        print("Could not connect to Neo4j. Ingestion aborted.")

    print(f"--- Finished Neo4j Ingestion for ticker: {ticker} ---")


if __name__ == "__main__":
    main() 