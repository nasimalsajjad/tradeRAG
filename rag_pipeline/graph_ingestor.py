# Placeholder for Graph Ingestion logic
import os
import json
import argparse
from neo4j import GraphDatabase, exceptions as neo4j_exceptions, basic_auth
from neo4j.time import Date as Neo4jDate # Import Neo4j Date type
from dotenv import load_dotenv
import traceback
import re # Import regex for parsing money

print("graph_ingestor.py loaded")

# Define directories
KG_INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'kg_input')

# --- Money Parsing Helper ---
def parse_monetary_value(text):
    """Attempts to parse currency symbol and numeric value from text."""
    # Remove commas
    text = text.replace(',', '')
    # Regex to find currency symbols ($, €, £) and numbers (int/float, k, m, b)
    # More robust parsing might require a dedicated library
    match = re.search(r'([$€£]?)\s*([\d\\.]+)\s*([kKmMbB]?)', text)
    if match:
        currency = match.group(1) if match.group(1) else None # Default currency if needed?
        value_str = match.group(2)
        multiplier_char = match.group(3).lower()
        
        try:
            value = float(value_str)
            multiplier = 1
            if multiplier_char == 'k':
                multiplier = 1_000
            elif multiplier_char == 'm':
                multiplier = 1_000_000
            elif multiplier_char == 'b':
                multiplier = 1_000_000_000
            
            final_value = value * multiplier
            return {"value": final_value, "currency": currency, "text": text}
        except ValueError:
            pass # Failed to convert value string to float
    return {"value": None, "currency": None, "text": text} # Return original text if parsing fails

# --- Neo4j Connection ---

def get_neo4j_driver():
    """Establishes connection to Neo4j using environment variables."""
    # Load config.env file from the project root
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'config.env')
    load_dotenv(dotenv_path=dotenv_path)
    # print(f"Loaded config.env from: {dotenv_path}")

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687") # Default URI
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    # print(f"Read from env: URI={uri}, User={user}, Password={'*' * len(password) if password else None}")

    if not user or not password:
        print("Error: NEO4J_USERNAME, or NEO4J_PASSWORD not found in environment variables.")
        print("Please ensure they are set in your config.env file.")
        return None

    try:
        # Use basic_auth for clarity
        auth = basic_auth(user, password)
        driver = GraphDatabase.driver(uri, auth=auth)
        driver.verify_connectivity()
        print("Successfully connected to Neo4j.")
        return driver
    except neo4j_exceptions.AuthError as auth_err:
         print(f"Neo4j Authentication Error: {auth_err}. Check username/password in config.env")
         return None
    except neo4j_exceptions.ServiceUnavailable as conn_err:
         print(f"Neo4j Connection Error: {conn_err}. Is the Neo4j server running at {uri}?")
         return None
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        traceback.print_exc() # Print full traceback for unexpected errors
        return None

# --- Graph Creation Logic ---

def execute_write_wrapper(tx, func, *args):
    """Wrapper to execute a function within a transaction."""
    return func(tx, *args)

def create_document_node(tx, metadata):
    """Creates/merges Document node, setting props carefully in two steps."""
    source_type = metadata.get('source_type', 'unknown')
    source_file = metadata.get('source_file', 'unknown')
    ticker = metadata.get('ticker', 'unknown')
    doc_date_str = metadata.get('date', None)
    doc_id = f"{source_type}::{ticker}::{source_file}" 

    # Step 1: Ensure the node exists with the core identifier
    merge_query = (
        "MERGE (d:Document {doc_id: $doc_id}) "
        "ON CREATE SET d.ticker = $ticker, d.created_at = timestamp() "
        "RETURN d.doc_id AS returned_doc_id" 
    )
    result = tx.run(merge_query, doc_id=doc_id, ticker=ticker)
    record = result.single()
    
    # Explicitly check if the merge returned the ID
    if not record or not record["returned_doc_id"]:
        print(f"!!! CRITICAL: MERGE failed for doc_id {doc_id}. Cannot set properties.")
        return None # Cannot proceed if merge failed
    
    returned_doc_id = record["returned_doc_id"]

    # Step 2: Set additional properties using SET, avoiding overwrites/conflicts
    props_to_set = {
        "p_source_type": source_type,
        "p_source_file": source_file,
        "p_ticker": ticker # Can set ticker again safely
    }
    set_clauses = [
        "d.source_type = coalesce(d.source_type, $p_source_type)",
        "d.source_file = coalesce(d.source_file, $p_source_file)",
        "d.ticker = coalesce(d.ticker, $p_ticker)" # Use coalesce for safety
    ]

    # Attempt to parse date and add set clauses
    if doc_date_str:
        props_to_set["p_date_string"] = doc_date_str
        set_clauses.append("d.date_string = coalesce(d.date_string, $p_date_string)")
        parsed_date = None
        try:
            year, month, day = map(int, doc_date_str.split('-'))
            parsed_date = Neo4jDate(year, month, day)
        except (ValueError, TypeError):
            try:
                if 'T' in doc_date_str:
                    date_part = doc_date_str.split('T')[0]
                    year, month, day = map(int, date_part.split('-'))
                    parsed_date = Neo4jDate(year, month, day)
                # else: # No need to print warning here if parse fails, just won't set date prop
            except (ValueError, TypeError):
                pass # Silently ignore date parse errors for SET
        
        if parsed_date:
            props_to_set["p_date"] = parsed_date
            set_clauses.append("d.date = coalesce(d.date, $p_date)")

    # Combine SET clauses
    set_query = (
        f"MATCH (d:Document {{doc_id: $doc_id}}) "
        f"SET { ', '.join(set_clauses) } "
    )
    
    # Execute the SET query
    tx.run(set_query, doc_id=returned_doc_id, **props_to_set)
    
    return returned_doc_id # Return the confirmed doc_id

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

    try:
        if label == "Company":
            ticker = entity.get("normalized_id")
            if not ticker:
                print(f"Skipping Company entity due to missing normalized_id: {entity}")
                return
            params["ticker"] = ticker
            merge_clause = f"MERGE (e:{label} {{ticker: $ticker}}) "
            properties_clause += "ON CREATE SET e.name = $word " # Set name only on creation
            properties_clause += "ON MATCH SET e.name = CASE WHEN e.name IS NULL THEN $word ELSE e.name END" # Update name if it was null
        elif label == "Person":
            params["name"] = word
            merge_clause = f"MERGE (e:{label} {{name: $name}}) "
        elif label == "Date":
            iso_date = entity.get("normalized_date")
            if not iso_date:
                print(f"Skipping Date entity due to missing normalized_date: {entity}")
                return
            try:
                year, month, day = map(int, iso_date.split('-'))
                neo4j_date = Neo4jDate(year, month, day)
                params["date"] = neo4j_date
                merge_clause = f"MERGE (e:{label} {{date: $date}}) "
                properties_clause += "ON CREATE SET e.display_text = $word "
            except (ValueError, TypeError) as date_err:
                 print(f"Warning: Could not parse date '{iso_date}' for Date entity node '{word}'. Skipping node creation. Error: {date_err}")
                 return # Skip creating this node/rel if date fails
        elif label == "Organization":
            params["name"] = word
            merge_clause = f"MERGE (e:{label} {{name: $name}}) "
        elif label == "MonetaryAmount":
            parsed_money = parse_monetary_value(word)
            params["value_text"] = parsed_money["text"] # Always store original text
            # Merge based on text for simplicity, avoids issues with parsing errors
            merge_clause = f"MERGE (e:{label} {{value_text: $value_text}}) "
            if parsed_money["value"] is not None:
                 params["value"] = parsed_money["value"]
                 properties_clause += "ON CREATE SET e.value = $value "
                 properties_clause += "ON MATCH SET e.value = coalesce(e.value, $value) " # Set if null
            if parsed_money["currency"]:
                 params["currency"] = parsed_money["currency"]
                 properties_clause += "ON CREATE SET e.currency = $currency "
                 properties_clause += "ON MATCH SET e.currency = coalesce(e.currency, $currency) "
        else:
            # Log other labels but maybe don't create nodes? Or create generic 'Entity'?
            # print(f"Skipping entity with unhandled label: {label} ('{word}')")
            label = "Entity" # Create a generic Entity node
            params["name"] = word
            params["original_label"] = entity.get('label', 'unknown') # Store original label
            merge_clause = f"MERGE (e:{label} {{name: $name, original_label: $original_label}}) "


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

    except Exception as e:
        print(f"ERROR processing entity: {entity} for doc {document_id}. Error: {e}")
        traceback.print_exc()

def create_mention_relationships(tx, document_id, entities_in_doc):
    """Creates MENTIONS relationships between Company entities in the same document."""
    # Filter for Company entities from the list for this document
    company_tickers = [e.get("normalized_id") for e in entities_in_doc if e.get("normalized_label") == "Company" and e.get("normalized_id")]
    unique_tickers = sorted(list(set(company_tickers))) # Get unique tickers, sort for consistent pairing

    # If less than 2 companies, no pairs to create
    if len(unique_tickers) < 2:
        return

    # Create pairs of distinct tickers
    pairs = []
    for i in range(len(unique_tickers)):
        for j in range(i + 1, len(unique_tickers)):
            pairs.append((unique_tickers[i], unique_tickers[j]))

    if not pairs:
        return

    # Cypher query to merge relationships between pairs
    # Using an undirected relationship for simplicity
    query = (
        "MATCH (c1:Company {ticker: $ticker1}) "
        "MATCH (c2:Company {ticker: $ticker2}) "
        "MERGE (c1)-[r:MENTIONS]-(c2) " # Undirected relationship
        "ON CREATE SET r.created_at = timestamp(), r.source_doc_ids = [$doc_id] "
        "ON MATCH SET r.source_doc_ids = CASE WHEN NOT $doc_id IN r.source_doc_ids THEN r.source_doc_ids + $doc_id ELSE r.source_doc_ids END"
    )
    # Add count? SET r.mention_count = coalesce(r.mention_count, 0) + 1 ? Might be complex.

    # Run the query for each pair
    for ticker1, ticker2 in pairs:
        params = {
            "ticker1": ticker1,
            "ticker2": ticker2,
            "doc_id": document_id
        }
        tx.run(query, **params)
    # print(f"  Created/updated MENTIONS relationships for {len(pairs)} pairs in doc {document_id}")


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
    # skipped_entities = 0 # Less relevant now?

    with open(resolved_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            processed_lines += 1
            if line_num % 100 == 0:
                print(f"  Processing line {line_num}...")
            try:
                data = json.loads(line)
                metadata = data.get("metadata", {})
                resolved_entities = data.get("resolved_entities", []) 

                if not metadata or not metadata.get('source_file'):
                    print(f"Skipping line {line_num} due to missing metadata or source_file: {line.strip()}")
                    error_count += 1
                    continue

                # Re-enable entity creation AND mentions relationship
                document_id = None
                with driver.session(database="neo4j") as session:
                    try:
                        # Step 1: Create/Merge Document Node (Simplified Version)
                        document_id = session.execute_write(create_document_node, metadata)
                        
                        if not document_id:
                            print(f"Skipping line {line_num}: Failed to create/merge document node.")
                            error_count += 1
                            continue 
                        
                        # Step 2: Create Entity Nodes and Relationships
                        entity_creation_failed = False
                        for entity in resolved_entities:
                            try:
                                session.execute_write(create_entity_node_and_relationship, document_id, entity)
                            except Exception as entity_err:
                                print(f"ERROR processing entity (line {line_num}, doc {document_id}): {entity}. Error: {entity_err}")
                                traceback.print_exc()
                                entity_creation_failed = True
                        
                        # === RE-ENABLED MENTIONS Creation ===
                        # Step 3: Create Mention Relationships (Only if entities were OK)
                        mention_creation_failed = False
                        if not entity_creation_failed:
                            try:
                                session.execute_write(create_mention_relationships, document_id, resolved_entities)
                            except Exception as mention_err:
                                 print(f"ERROR creating mention relationships (line {line_num}, doc {document_id}). Error: {mention_err}")
                                 traceback.print_exc()
                                 mention_creation_failed = True 
                                 error_count += 1 # Count as error if mentions fail
                        elif entity_creation_failed:
                             # Log that mentions are skipped due to prior errors
                             print(f"Skipping mention relationships for doc {document_id} due to entity errors.")
                             # error_count was already incremented during entity processing
                        # === END RE-ENABLED SECTION ===

                        # Count success only if doc created AND no entity/mention errors occurred
                        if document_id and not entity_creation_failed and not mention_creation_failed:
                            processed_docs_success += 1
                        # Ensure error_count reflects any failure in entity/mention steps
                        elif document_id and (entity_creation_failed or mention_creation_failed):
                             pass # error_count was already incremented in the respective failure block

                    except Exception as tx_error:
                        print(f"Unexpected error during session for line {line_num}: {tx_error}")
                        traceback.print_exc()
                        error_count += 1 

            except json.JSONDecodeError as json_err:
                print(f"Skipping line {line_num} due to JSON decode error: {json_err} - Line: {line.strip()}")
                error_count += 1
            except Exception as e:
                print(f"Skipping line {line_num} due to unexpected error: {e} - Line: {line.strip()}")
                traceback.print_exc() # Print stack trace for other errors
                error_count += 1

    print(f"--- Finished ingestion for {ticker} ---")
    print(f"Processed {processed_lines} lines from {resolved_file}.")
    print(f"Successfully processed transactions for {processed_docs_success} documents.")
    print(f"Encountered {error_count} errors/skipped lines.")


# --- Utility: Apply Constraints ---
def apply_constraints(driver):
    """Applies unique constraints to the Neo4j database."""
    constraints = [
        "CREATE CONSTRAINT company_ticker IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
        "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT date_val IF NOT EXISTS FOR (dt:Date) REQUIRE dt.date IS UNIQUE",
        "CREATE CONSTRAINT org_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE",
        # Add others as needed, e.g., for MonetaryAmount.value_text if desired
        # "CREATE CONSTRAINT money_text IF NOT EXISTS FOR (m:MonetaryAmount) REQUIRE m.value_text IS UNIQUE"
    ]
    print("Applying Neo4j constraints (if they don't exist)...")
    with driver.session(database="neo4j") as session:
        for i, constraint in enumerate(constraints):
            try:
                session.run(constraint)
                # print(f" Applied constraint {i+1}/{len(constraints)}")
            except Exception as e:
                print(f" Error applying constraint: {constraint} - Error: {e}")
    print("Constraint application finished.")


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Ingest resolved entities into Neo4j.")
    parser.add_argument("-t", "--ticker", type=str, required=True, help="Stock ticker symbol to ingest data for (e.g., AMD)")
    parser.add_argument("--apply-constraints-only", action="store_true", help="Only apply constraints and exit.")

    args = parser.parse_args()
    ticker = args.ticker.upper()

    print(f"--- Starting Neo4j Ingestion Process ---")

    driver = get_neo4j_driver()

    if driver:
        try:
            apply_constraints(driver) # Apply constraints first

            if args.apply_constraints_only:
                print("Exiting after applying constraints.")
                return

            print(f"--- Starting Data Ingestion for ticker: {ticker} ---")
            ingest_data_to_neo4j(driver, ticker)

        finally:
            if driver:
                driver.close()
                print("Neo4j driver closed.")
    else:
        print("Failed to get Neo4j driver. Exiting.")


if __name__ == "__main__":
    main() 