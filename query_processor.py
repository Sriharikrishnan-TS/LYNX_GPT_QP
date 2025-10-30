import requests
import json
import psycopg2
import os
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Load database credentials from .env file
load_dotenv()
DB_PARAMS = {
    'dbname': os.getenv("DB_NAME", "qp_ingestion2"), # Added default
    'user': os.getenv("DB_USER", "postgres"),       # Added default
    'password': os.getenv("DB_PASSWORD"),
    'host': os.getenv("DB_HOST", "localhost"),     # Added default
    'port': os.getenv("DB_PORT", "5432"),          # Added default
    'sslmode': 'require'
}

def extract_metadata_from_query(query_text: str) -> dict:
    """Uses Gemma to extract metadata (dept, subject, year) from a user query."""
    print(f"INFO: Sending query to gpt-oss:20b for metadata extraction: '{query_text}'")
    
    # --- UPDATED PROMPT ---
    prompt = f"""
You are an expert at analyzing user queries about question papers and extracting key metadata into a JSON object. Your goal is to identify the department, subject, and year.

Follow these rules exactly:
1.  **Output Format:** Respond ONLY with a single, valid JSON object. Do not include any explanations, reasoning, or markdown.
2.  **Fields:** You must extract "department", "subject", and "year".
3.  **Full Subject Extraction:** When a subject is mentioned, extract the **fullest possible name** of the subject, not just one keyword. (e.g., "design and analysis of algorithms", not "algorithms").

4.  **Subject Normalization:** You **must** convert the extracted subject name to **all lowercase**.

5.  **Department Standardization (CRITICAL):** You must normalize all department names (full names, common typos, abbreviations) to their standard abbreviation.
    * `computer science and engineering`, `computer science & engineering`, `computr science`, `cs` -> `cse`
    * `electronics and communication engineering` -> `ece`
    * `electrical and electronics engineering` -> `eee`
    * `instrumentation and control engineering` -> `ice`
    * `mechanical engineering` -> `mech`
    * `chemical engineering` -> `chem`
    * `production engineering` -> `prod`
    * `metallurgical and material science engineering` -> `mme`
    * `civil engineering` -> `civil`
    * If the department is not in this list, output it as-is.

6.  **Disambiguation (NEW RULE):** A phrase identified as the department (e.g., "computer science and engineering") **CANNOT** also be extracted as the subject. The subject must be a separate topic.

7.  **Robust Subject Typo Correction:** You **must** correct common misspellings in *subject* names.
    * `algoritms` -> `algorithms`
    * `desing and analyis` -> `design and analysis`

8.  **Year:** Extract the four-digit year.

9.  **Completeness:** If a value is genuinely not present or ambiguous, use `null`. (This rule is for you, the model, even though the examples below are all complete).

[EXAMPLE 1 - UPDATED]
Query: "find papers for cse department from 2023 on data structures"
JSON Output: {{"department": "cse", "subject": "data structures", "year": "2023"}}

[EXAMPLE 2 - UPDATED]
Query: "show me the 2021 algoritms papers from computr science"
JSON Output: {{"department": "cse", "subject": "algorithms", "year": "2021"}}

[EXAMPLE 3 - UPDATED]
Query: "any question paper for mech 2022 on thermodynamics?"
JSON Output: {{"department": "mech", "subject": "thermodynamics", "year": "2022"}}

[EXAMPLE 4 - UPDATED]
Query: "get me computer science and engineering desing and analyis of algorithms of 2023 paper"
JSON Output: {{"department": "cse", "subject": "design and analysis of algorithms", "year": "2023"}}

[EXAMPLE 5 - NEW]
Query: "get me Computer science and engineering 2023 automata and formal languages paper"
JSON Output: {{"department": "cse", "subject": "automata and formal languages", "year": "2023"}}

Now, analyze the following query:

Query: "{query_text}"
JSON Output:
"""
    # --- END OF UPDATED PROMPT ---

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=45)
        response.raise_for_status()
        response_data = json.loads(response.text)
        metadata = json.loads(response_data.get("response", "{}"))
        # Ensure all expected keys are present, defaulting to None
        for key in ['department', 'subject', 'year']:
            if key not in metadata:
                metadata[key] = None
        print(f"INFO: Metadata extracted by LLM: {metadata}")
        return metadata
    except Exception as e:
        print(f"[ERROR] LLM request failed: {e}")
        return {'department': None, 'subject': None, 'year': None, 'error': str(e)}

def build_sql_query(metadata: dict) -> tuple[str, list]:
    """Builds a SQL query based on the extracted metadata."""
    # Selects filename, dept, subject, year from the metadata table
    base_query = "SELECT department, subject, year, file_url FROM metadata.metadata WHERE 1=1"
    conditions = []
    params = []

    if metadata.get("department"):
        conditions.append("department ILIKE %s")
        params.append(f"%{metadata['department']}%")

    if metadata.get("subject"):
        conditions.append("subject ILIKE %s")
        params.append(f"%{metadata['subject']}%")

    if metadata.get("year"):
        try:
            year_val = int(metadata['year'])
            if 1900 < year_val < 2100:
                 conditions.append("year = %s")
                 params.append(year_val)
            else:
                 print(f"[WARN] Invalid year '{metadata['year']}' extracted by LLM, ignoring.")
        except (ValueError, TypeError):
             print(f"[WARN] Non-integer year '{metadata['year']}' extracted by LLM, ignoring.")

    if conditions:
        base_query += " AND " + " AND ".join(conditions)

    base_query += " ORDER BY year DESC, department, subject;"

    print(f"INFO: Built SQL Query: {base_query}")
    print(f"INFO: Query Params: {params}")
    return base_query, params

def run_query(sql_query: str, params: list) -> list:
    """Connects to the DB, runs the query, and returns results."""
    results = []
    conn = None
    try:
        # Check if DB password is provided
        if not DB_PARAMS.get('password'):
             print("[ERROR] Database password not found in .env file.")
             return [{"error": "Database configuration error (password missing)"}]

        conn = psycopg2.connect(**DB_PARAMS)
        with conn.cursor() as cur:
            cur.execute("SET search_path TO metadata, public;")
            cur.execute(sql_query, params)
            colnames = [desc[0] for desc in cur.description]
            results = [dict(zip(colnames, row)) for row in cur.fetchall()]
        print(f"INFO: Query executed successfully, found {len(results)} results.")
    except psycopg2.OperationalError as e:
        print(f"[ERROR] Database connection failed: {e}")
        return [{"error": "Database connection failed"}]
    except (Exception, psycopg2.DatabaseError) as e:
        print(f"[ERROR] Database query failed: {e}")
        return [{"error": f"Database query failed: {e}"}]
    finally:
        if conn:
            conn.close()
    return results

def process_user_query(user_query: str) -> dict:
    """Orchestrates the query processing pipeline."""
    metadata = extract_metadata_from_query(user_query)
    if 'error' in metadata:
        return {"error": f"LLM failed: {metadata['error']}", "results": []}

    sql_query, params = build_sql_query(metadata)
    db_results = run_query(sql_query, params)

    if db_results and isinstance(db_results[0], dict) and 'error' in db_results[0]:
         db_error = db_results[0]['error']
         return {"metadata": metadata, "sql": sql_query, "error": db_error, "results": []}

    return {
        "metadata": metadata,
        "sql": sql_query,
        "results": db_results
    }

# --- Example Usage (for testing this script directly) ---
if __name__ == '__main__':
    test_query = "show me CSE papers from 2023 about algoritms"
    output = process_user_query(test_query)
    print("\n--- Final Output ---")
    print(json.dumps(output, indent=2, default=str))