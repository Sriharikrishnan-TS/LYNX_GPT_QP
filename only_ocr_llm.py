import os
import requests
import json
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import re # Import the regex library

# --- ⚙️ CONFIGURATION ---
PDF_PATH = "lynx_gpt_qp-1.pdf"
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
SAVE_DEBUG_IMAGE = True
TOP_N_CHARACTERS = 300 # Set the character limit

# --- 🖼️ 1. IMAGE PREPROCESSING & OCR ---

def preprocess_image(image):
    """A simple preprocessing function to convert the image to black and white."""
    open_cv_image = np.array(image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text_with_tesseract(pdf_path: str) -> str:
    """Extracts text from the FIRST PAGE of a PDF using Tesseract."""
    print(f"INFO: Starting Tesseract OCR for '{pdf_path}'...")
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF file not found at: {pdf_path}")
        return ""
    try:
        # MODIFIED: Only process the first page
        images = convert_from_path(pdf_path, last_page=1, dpi=300) 
        
        if not images:
            print("[ERROR] PDF seems to be empty or could not be read.")
            return ""

        print(f"INFO: Processing page 1...")
        preprocessed_img = preprocess_image(images[0])
        if SAVE_DEBUG_IMAGE:
            debug_filename = "debug_page_1.png"
            cv2.imwrite(debug_filename, preprocessed_img)
            print(f"      -> Saved debug image to '{debug_filename}'")
        
        text = pytesseract.image_to_string(preprocessed_img)
        print("INFO: OCR completed.")
        return text # Return text from the first page only

    except Exception as e:
        print(f"[ERROR] Failed during Tesseract extraction: {e}")
        print("[INFO] Make sure Tesseract and Poppler are installed and in your system's PATH.")
        return ""

# --- 🧠 2. LLM METADATA EXTRACTION ---

def extract_metadata_with_llm(text: str) -> dict:
    """Extracts metadata from text using a locally running model via Ollama."""
    print("\nINFO: Sending text to Gemma for metadata extraction...")
    
    prompt = f"""
You are a precision data extraction engine. Your sole task is to analyze text from a university question paper and extract specific metadata fields into a clean JSON object.

Follow these rules exactly:
1.  **Output Format:** Respond ONLY with a single, valid JSON object. Do not include any other text or explanations.
2.  **department:** Extract the full official department name, often found after "DEPARTMENT OF".
3.  **subject:** Extract only the subject name. The name often follows a keyword like "SUBJECT:" or "Sub. Code & Title :". Explicitly exclude any subject codes (e.g., "(ENIR 11)").
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
    payload = { "model": "gemma:2b", "prompt": prompt, "format": "json", "stream": False }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        response_data = json.loads(response.text)
        return json.loads(response_data.get("response", "{}"))
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        return {}

# --- ▶️ 3. MAIN EXECUTION ---

if __name__ == "__main__":
    raw_text = extract_text_with_tesseract(PDF_PATH)
    
    if raw_text and raw_text.strip():
        # --- MODIFIED: Filter text to top N characters ---
        print(f"\nINFO: Filtering extracted text to top {TOP_N_CHARACTERS} characters.")
        top_text = raw_text[:TOP_N_CHARACTERS]
        
        print("\n--- Filtered Top Text (Preview) ---")
        print(top_text)
        print("-----------------------------------")
        
        # Pass the filtered text to the LLM
        extracted_metadata = extract_metadata_with_llm(top_text)
        
        # --- MODIFIED: Run regex fallback on the top_text ---
        if extracted_metadata and extracted_metadata.get("year") is None:
            print("INFO: LLM did not find a year. Trying regex fallback...")
            year_match = re.search(r"\b(20\d{2})\b", top_text) # Use top_text
            if year_match:
                extracted_metadata["year"] = int(year_match.group(1))
                print(f"      -> Found year '{extracted_metadata['year']}' using regex.")
        
        print("\n--- Extracted Metadata (from Gemma with Regex Fallback) ---")
        if extracted_metadata:
            print(json.dumps(extracted_metadata, indent=2))
        else:
            print("Could not extract metadata.")
    else:
        print("\n[CONCLUSION] No text was extracted from the PDF. The process has stopped.")