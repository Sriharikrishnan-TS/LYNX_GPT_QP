import os
import requests
import json
import pytesseract
from pdf2image import convert_from_bytes 
import cv2
import numpy as np
import re
from dotenv import load_dotenv
from supabase import create_client, Client
import uuid

TOP_N_CHARACTERS = 600

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

supabase_URL = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY") 

if not GROQ_API_KEY:
    print("[ERROR] GROQ_API_KEY not found in .env file.")
if not supabase_URL or not supabase_key:
    print("[ERROR] SUPABASE_URL or SUPABASE_SERVICE_KEY not found in .env file.")

bucketname = 'questionpapers' 

# --- OCR & Preprocessing ---
def preprocess_image(image):
    open_cv_image = np.array(image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text_from_bytes(pdf_bytes: bytes, filename: str) -> str:
    print(f"\nINFO: Starting Tesseract OCR for '{filename}'...")
    try:
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
    print(f"INFO: Sending text from '{filename}' to Groq GPT OSS")
    
    prompt = f"""
You are a precision data extraction engine. Your sole task is to analyze text from a university question paper and extract specific metadata fields into a clean JSON object.

Follow these rules exactly:
1.  **Output Format:** Respond ONLY with a single, valid JSON object. Do not include any other text or explanations.

2.  **Department Standardization (CRITICAL):** This is your most important rule. You must normalize the extracted department name to its standard abbreviation. If you find the full name, a common typo, or the abbreviation itself, you must output the standard abbreviation.
    * `computer science and engineering`, `computer science & engineering`, `computer science and engg`, `cs` -> `cse`
    * `electronics and communication engineering` -> `ece`
    * `electrical and electronics engineering` -> `eee`
    * `instrumentation and control engineering` -> `ice`
    * `mechanical engineering` -> `mech`
    * `chemical engineering` -> `chem`
    * `production engineering` -> `prod`
    * `metallurgical and material science engineering` -> `mme`
    * `civil engineering` -> `civil`

3.  **subject:** Extract only the subject name. The name often follows keywords like "SUBJECT:" or "Sub. Code & Title :". Explicitly exclude any subject codes (e.g., "(ENIR 11)").

4.  **Subject Normalization:** You **must** convert the extracted subject name to **all lowercase**.

5.  **year:** Extract the four-digit year of the examination, if present.

6.  **Missing Values:** If any field cannot be found, its value must be null.

7.  **Year Inference (Advice):** Pay close attention to dates.
    * A string like "2210-24" or "22-10-24" on an exam paper likely means the date "22-10-2024", so the year is 2024.
    * An academic year like "2023-24" means the exam is for the 2023-2024 session. 2024 is preferred.
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
[EXAMPLE 2 - UPDATED]
Input Text:
\"\"\"
B.Tech. DEGREE EXAMINATION, NOVEMBER/DECEMBER 2023.
Computer Science and Engineering
CS 8351 - Digital Principles and System Design
\"\"\"
Correct JSON Output:
{{
  "department": "cse",
  "subject": "Digital Principles and System Design",
  "year": 2023
}}
---
[EXAMPLE 3 - UPDATED]
Input Text:
\"\"\"
DEPARTMENT OF COMPUTER SCIENCE AND ENGG.
NATIONAL INSTITUTE OF TECHNOLOGY, TIRUCHIRAPPALLE 620 O15. CYCLE TEST II CSHOI7 BIG DATA MINING 2210-24 TIME: 60 Mins
\"\"\"
Correct JSON Output:
{{
  "department": "cse",
  "subject": "BIG DATA MINING",
  "year": 2024
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

        # Standardize year to integer or None
        for key in ['department', 'subject']:
            if key not in metadata: metadata[key] = None
        
        llm_year = metadata.get('year')
        if isinstance(llm_year, str):
            try:
                metadata['year'] = int(llm_year)
            except (ValueError, TypeError):
                metadata['year'] = None
        elif not isinstance(llm_year, int):
             metadata['year'] = None

        return metadata
    except Exception as e:
        print(f"[ERROR] Groq API request failed for {filename}: {e}")
        error_details = str(e)
        if hasattr(e, 'response') and e.response is not None:
             try: error_details = json.dumps(e.response.json(), indent=2)
             except: error_details = e.response.text
        return {'filename': filename, 'error': f"LLM Error: {error_details}"}

def supabase_bucket(client: Client, file_bytes: bytes, storage_path: str) -> str:
    if not client:
        print("[ERROR] Supabase client is invalid. Cannot upload file.")
        return None
    try: 
        print(f"INFO: Uploading to Supabase bucket '{bucketname}' at path '{storage_path}'...")
        bucket_response = client.storage.from_(bucketname).upload(
                                                                    file = file_bytes,
                                                                    path = storage_path,
                                                                    file_options = {"content-type": "application/pdf"}
                                                                    )

        url_response = client.storage.from_(bucketname).get_public_url(storage_path)
        print(f"INFO: File upload successful. Public URL: {url_response}")
        return url_response
    except Exception as e:
        print(f"[ERROR] Supabase file upload failed: {e}")
        return None

def insert_metadata_into_db(client: Client, metadata: dict) -> bool:
    if not client:
        print("[ERROR] Supabase client is invalid. Cannot insert metadata.")
        return False
        
    print(f"INFO: Inserting metadata for 'file_url: {metadata.get('file_url', 'Unknown file')}' into Supabase table...")
    
    db_data = {
        "department": metadata.get("department"),
        "subject": metadata.get("subject"),
        "year": metadata.get("year"),
        "file_url": metadata.get("file_url")
    }

    try:
        data, count = client.schema('metadata').table('metadata').insert(db_data).execute()
        print("[SUCCESS] Data inserted into the Supabase database.")
        return True 
    except Exception as e:
        print(f"[ERROR] Supabase database insertion failed: {e}")
        return False 

# --- Main Processing Function  ---
def process_single_pdf(pdf_bytes: bytes, filename: str) -> dict:
    """Processes a single PDF (bytes), uploads to storage, and inserts metadata."""
    
    try:
        if not supabase_URL or not supabase_key:
            print("[ERROR] Cannot create client, .env variables missing.")
            return {"filename": filename, "status": "Error", "message": "Server-side error: .env variables not loaded."}
            
        supabase_client: Client = create_client(supabase_URL, supabase_key)
        print("INFO: Supabase client initialized for this job.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Supabase client: {e}")
        return {"filename": filename, "status": "Error", "message": f"Failed to initialize Supabase client: {e}"}
         

    org_filename = filename
    unique_id = uuid.uuid4()
    file_extension = os.path.splitext(org_filename)[1].lower()
    if not file_extension: file_extension = ".pdf" 
    strg_path = f"{unique_id}{file_extension}"

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
    
    # 4. Compulsory Regex for Year (FALLBACK LOGIC)
    print(f"INFO: Checking year extraction for '{filename}'...")
    if metadata.get("year") is None:
        print("INFO: LLM found no year. Applying regex fallback...")
        # Simpler regex for "20XX"
        year_match = re.search(r"\b(20\d{2})\b", top_text) 
        if year_match:
            found_year = int(year_match.group(1))
            metadata["year"] = found_year
            print(f"      -> Set year to '{found_year}' using regex fallback.")
        else:
            print(f"      -> No year found using regex fallback.")
    else:
        print(f"INFO: LLM successfully found year: {metadata.get('year')}. Skipping regex.")


    # 5. Upload to Supabase Storage
    file_url = supabase_bucket(supabase_client, pdf_bytes, strg_path)
    if not file_url:
        return {"filename": filename, "status": "Error", "message": "Failed to upload file to Supabase storage.", "metadata": metadata,"raw_text":top_text}
    
    metadata["file_url"] = file_url

    # 6. Insert into Supabase Database
    insertion_success = insert_metadata_into_db(supabase_client, metadata)

    if insertion_success:
        return {"filename": filename, "status": "Success", "metadata": metadata,"raw_text":top_text}
    else:
        return {"filename": filename, "status": "Error", "message": "Failed to insert metadata into Supabase database.", "metadata": metadata,"raw_text":top_text}

