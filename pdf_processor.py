# pdf_processor.py
import os
import requests
import json
import pytesseract
from pdf2image import convert_from_bytes # Import convert_from_bytes
import cv2
import numpy as np
import re
from dotenv import load_dotenv
import psycopg2

# --- CONFIGURATION ---
TOP_N_CHARACTERS = 600

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

DB_PARAMS = {
    'dbname': os.getenv("DB_NAME", "qp_ingestion2"),
    'user': os.getenv("DB_USER", "postgres"),
    'password': os.getenv("DB_PASSWORD","Imtherealg@at18"), # Make sure this is set in .env
    'host': os.getenv("DB_HOST", "localhost"),
    'port': os.getenv("DB_PORT", "5432")
}

if not GROQ_API_KEY:
    print("[ERROR] GROQ_API_KEY not found in .env file.")
# Basic check for DB password
if not DB_PARAMS.get('password'):
     print("[WARN] DB_PASSWORD not found in .env file. DB operations might fail.")

# --- OCR & Preprocessing ---
def preprocess_image(image):
    open_cv_image = np.array(image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text_from_bytes(pdf_bytes: bytes, filename: str) -> str:
    print(f"\nINFO: Starting Tesseract OCR for '{filename}'...")
    try:
        # Use convert_from_bytes for in-memory processing
        images = convert_from_bytes(pdf_bytes, last_page=1, dpi=300)
        if not images:
            print("[ERROR] PDF seems empty or unreadable.")
            return ""
        preprocessed_img = preprocess_image(images[0])
        text = pytesseract.image_to_string(preprocessed_img)
        print("INFO: OCR completed.")
        return text
    except Exception as e:
        print(f"[ERROR] Failed during Tesseract extraction for {filename}: {e}")
        return ""

# --- LLM Metadata Extraction ---
def extract_metadata_with_groq_llama3(text: str, filename: str) -> dict:
    print(f"INFO: Sending text from '{filename}' to Groq Llama 3 API...")
    prompt = f"""
You are a precision data extraction engine. Your sole task is to analyze text from a university question paper and extract specific metadata fields into a clean JSON object.

Follow these rules exactly:
1.  **Output Format:** Respond ONLY with a single, valid JSON object. Do not include any other text or explanations.
2.  **department:** Extract the full official department name, often found after "DEPARTMENT OF".
3.  **subject:** Extract only the subject name. The name often follows keywords like "SUBJECT:" or "Sub. Code & Title :". Explicitly exclude any subject codes (e.g., "(ENIR 11)").
4.  **year:** Extract the four-digit year of the examination, if present.
5.  **Missing Values:** If any field cannot be found, its value must be null.

---
[EXAMPLE 1]
Input Text:
\"\"\"
NATIONAL INSTITUTE OF TECHONOLOGY, TIRUCHIRAPALLI
DEPARTMENT OF ENERGY & ENVIRONMENT
SUBJECT: ENERGY AND ENVIRONMENT (ENIR 11)
\"\"\"
Correct JSON Output:
{{
  "department": "ENERGY & ENVIRONMENT",
  "subject": "ENERGY AND ENVIRONMENT",
  "year": null
}}
---
[EXAMPLE 2]
Input Text:
\"\"\"
B.Tech. DEGREE EXAMINATION, NOVEMBER/DECEMBER 2023.
Computer Science and Engineering
CS 8351 – Digital Principles and System Design
\"\"\"
Correct JSON Output:
{{
  "department": "Computer Science and Engineering",
  "subject": "Digital Principles and System Design",
  "year": 2023
}}
---

Now, perform the same task on the following text:

[TEXT TO ANALYZE]
\"\"\"
{text}
\"\"\"
"""
    headers = { "Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json" }
    data = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        response_data = response.json()
        message_content = response_data['choices'][0]['message']['content']
        metadata = json.loads(message_content)
        metadata['filename'] = filename
        for key in ['department', 'subject', 'year']: # Ensure keys exist
            if key not in metadata: metadata[key] = None
        return metadata
    except Exception as e:
        print(f"[ERROR] Groq API request failed for {filename}: {e}")
        # Add error details if available
        error_details = str(e)
        if hasattr(e, 'response') and e.response is not None:
             try: error_details = json.dumps(e.response.json(), indent=2)
             except: error_details = e.response.text
        return {'filename': filename, 'error': f"LLM Error: {error_details}"}


# --- Database Insertion ---
def insert_metadata_into_db(metadata: dict, conn):
    print(f"INFO: Inserting metadata for '{metadata.get('filename', 'Unknown file')}'...")
    sql = "INSERT INTO metadata.metadata (department, subject, year) VALUES (%(department)s, %(subject)s, %(year)s);"
    try:
        with conn.cursor() as cur:
            cur.execute("SET search_path TO metadata, public;") # Set schema context
            cur.execute(sql, metadata)
        conn.commit()
        print("[SUCCESS] Data inserted into the database.")
        return True # Indicate success
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"[ERROR] Database insertion failed: {error}")
        conn.rollback()
        return False # Indicate failure

# --- Main Processing Function ---
def process_single_pdf(pdf_bytes: bytes, filename: str) -> dict:
    """Processes a single PDF (bytes) and inserts metadata into the DB."""
    
    # 1. Extract Text
    raw_text = extract_text_from_bytes(pdf_bytes, filename)
    if not raw_text or not raw_text.strip():
        return {"filename": filename, "status": "Error", "message": "No text extracted from PDF."}

    # 2. Filter Text
    top_text = raw_text[:TOP_N_CHARACTERS]

    # 3. Extract Metadata via LLM
    metadata = extract_metadata_with_groq_llama3(top_text, filename)
    if 'error' in metadata:
        return {"filename": filename, "status": "Error", "message": metadata['error']}

    # 4. Compulsory Regex for Year
    print(f"INFO: Applying compulsory regex for year extraction on '{filename}'...")
    year_match = re.search(r"\b(20\d{2})\b", top_text)
    if year_match:
        found_year = int(year_match.group(1))
        metadata["year"] = found_year
        print(f"      -> Set year to '{found_year}' using regex.")
    else:
        metadata["year"] = None
        print(f"      -> No year found using regex.")

    # 5. Insert into Database
    db_connection = None
    insertion_success = False
    db_error_message = "DB connection could not be established."
    try:
        db_connection = psycopg2.connect(**DB_PARAMS)
        insertion_success = insert_metadata_into_db(metadata, db_connection)
        if not insertion_success:
             db_error_message = "Failed to insert data into DB." # More specific error captured by insert function
    except psycopg2.OperationalError as e:
        print(f"[ERROR] Could not connect to the database: {e}")
        db_error_message = f"Database connection error: {e}"
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during DB operation: {e}")
        db_error_message = f"Unexpected DB error: {e}"
    finally:
        if db_connection:
            db_connection.close()

    if insertion_success:
        return {"filename": filename, "status": "Success", "metadata": metadata,"raw_text":top_text}
    else:
        return {"filename": filename, "status": "Error", "message": db_error_message, "metadata": metadata,"raw_text":top_text}

# Note: The batch processing loop (`if __name__ == "__main__":`) from testing.py is removed
# as this script will now be imported and used function by function.