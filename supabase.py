#the entire pipeline which is supposed to work when we connect with supabase(just skeleton structure)

import os
import requests
import json
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import psycopg2
from dotenv import load_dotenv

# --- ⚙️ CONFIGURATION ---

# 1. Load environment variables from a .env file
load_dotenv()
# USER = os.getenv("USER")
# PASSWORD = os.getenv("PASSWORD")
# HOST = os.getenv("HOST")
# PORT = os.getenv("PORT")
# DBNAME = os.getenv("DBNAME")


# 2. File Paths and API endpoints
PDF_PATH = "lynx_gpt_qp-2.pdf"
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"

# 3. Set to True to save the image that Tesseract sees (for debugging OCR)
SAVE_DEBUG_IMAGE = True

# 4. (Optional) Tesseract executable path for Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# --- 🖼️ 1. IMAGE PREPROCESSING & OCR ---
# (This entire section is well-written, no changes needed)

def preprocess_image(image):
    """
    A simpler, more robust preprocessing function.
    - Converts to grayscale
    - Applies a simple global threshold (Otsu's method)
    """
    open_cv_image = np.array(image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text_with_tesseract(pdf_path: str) -> str:
    # ... (Your function is perfect as is) ...
    print(f"INFO: Starting Tesseract OCR for '{pdf_path}'...")
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF file not found at: {pdf_path}")
        return ""
        
    try:
        images = convert_from_path(pdf_path)
        full_text = []

        for i, image in enumerate(images):
            page_num = i + 1
            print(f"INFO: Processing page {page_num}/{len(images)}...")
            preprocessed_img = preprocess_image(image)

            if SAVE_DEBUG_IMAGE:
                debug_filename = f"debug_page_{page_num}.png"
                cv2.imwrite(debug_filename, preprocessed_img)
                print(f"      -> Saved debug image to '{debug_filename}'")

            text = pytesseract.image_to_string(preprocessed_img)
            
            if text:
                full_text.append(text)
        
        print("INFO: OCR completed.")
        return "\n\n--- Page Break ---\n\n".join(full_text)

    except Exception as e:
        print(f"[ERROR] Failed during Tesseract extraction: {e}")
        print("[INFO] Make sure Tesseract and Poppler are installed and in your system's PATH.")
        return ""


# --- 🧠 2. LLM METADATA EXTRACTION ---
# (This section is also excellent, no changes needed)

def extract_metadata_with_llama3(text: str) -> dict:
    # ... (Your function is perfect as is) ...
    print("\nINFO: Sending text to Llama 3 with improvised prompt...")
    
    prompt = f"""
You are a precision data extraction engine. Your sole task is to analyze text from a university question paper and extract specific metadata fields into a clean JSON object.

Follow these rules exactly:
1.  **Output Format:** Respond ONLY with a single, valid JSON object. Do not include any other text, explanations, or markdown formatting like ```json.
2.  **Department:** Extract the full official department name (e.g., "Mechanical Engineering").
3.  **Semester:** Extract only the number or Roman numeral (e.g., "3" or "III"). Do not include the word "Semester".
4.  **subject**: Extract only the subject name (e.g., "Physics", "Digital Principles and System Design"). Explicitly exclude any preceding subject codes (e.g., "PH8151", "CS 8351").
5.  **Year:** Extract the four-digit year the examination took place (e.g., 2023).
6.  **Missing Values:** If any field cannot be found, its value must be null.
7.  **No Guessing:** Do not invent or infer information not explicitly present in the text.

---
[EXAMPLE 1]

Input Text:
\"\"\"
B.Tech. DEGREE EXAMINATION, NOVEMBER/DECEMBER 2023.
Third Semester
Computer Science and Engineering
CS 8351 – Digital Principles and System Design
\"\"\"

Correct JSON Output:
{{
  "department": "Computer Science and Engineering",
  "semester": "Third",
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
    payload = {
        "model": "gemma:2b",
        "prompt": prompt,
        "format": "json",
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        response_data = json.loads(response.text)
        return json.loads(response_data.get("response", "{}"))
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Could not connect to the Ollama server. Is it running? Error: {e}")
        return {}
    except json.JSONDecodeError:
        print(f"[ERROR] Failed to decode JSON from Llama 3's response.")
        return {}
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        return {}


# --- 🐘 3. DATABASE INSERTION ---

def insert_metadata_into_db(metadata: dict, connection):
    """Inserts the extracted metadata into the PostgreSQL database."""
    print("\nINFO: Inserting metadata into the database...")
    
    # <<< FIX 2: Define the SQL query as a constant string.
    # Assumes a table like: CREATE TABLE question_papers (department TEXT, semester TEXT, subject TEXT, year INTEGER);
    sql = """
        INSERT INTO question_papers (department, semester, subject, year)
        VALUES (%s, %s, %s, %s);
    """
    
    try:
        with connection.cursor() as cur:
            cur.execute(sql, (
                metadata.get("department"),
                metadata.get("semester"),
                metadata.get("subject"),
                metadata.get("year")
            ))
        connection.commit()
        print("[SUCCESS] Data inserted into the database.")
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"[ERROR] Database insertion failed: {error}")
        connection.rollback()
    


# --- ▶️ 4. MAIN EXECUTION ---

if __name__ == "__main__":
    raw_text = extract_text_with_tesseract(PDF_PATH)
    
    if raw_text and raw_text.strip():
        extracted_metadata = extract_metadata_with_llama3(raw_text)
        
        if extracted_metadata:
            print("\n--- Extracted Metadata (from Llama 3) ---")
            print(json.dumps(extracted_metadata, indent=2))
            
            db_connection = None
            try:
                # <<< FIX 1: Connect using environment variables for security and flexibility.
                db_connection = psycopg2.connect(
                    dbname=os.getenv("NAME"),
                    user=os.getenv("USER"),
                    password=os.getenv("PASSWORD"),
                    host=os.getenv("HOST"),
                    port=os.getenv("PORT")
                )
                insert_metadata_into_db(extracted_metadata, db_connection)
            except psycopg2.OperationalError as e:
                print(f"[ERROR] Could not connect to the database. Check your .env file and that the database is running: {e}")
            finally:
                if db_connection:
                    db_connection.close()
                    print("\nINFO: Database connection closed.")
    else:
        print("\n[CONCLUSION] No text was extracted from the PDF. The process has stopped.")