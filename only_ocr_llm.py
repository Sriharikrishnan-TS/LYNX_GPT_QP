import os
import requests
import json
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import re

# --- ⚙️ CONFIGURATION ---
PDF_PATH = "lynx_gpt_qp-25.pdf"
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
SAVE_DEBUG_IMAGE = True

# --- 🖼️ 1. IMAGE PREPROCESSING & OCR ---

def preprocess_image(image):
    open_cv_image = np.array(image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text_with_tesseract(pdf_path: str) -> str:
    print(f"INFO: Starting Tesseract OCR for '{pdf_path}'...")
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF file not found at: {pdf_path}")
        return ""
    try:
        # custom_config = r'--oem 3 --psm 6'
        images = convert_from_path(pdf_path,last_page=1,dpi=300)
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
        return ""

# --- 🧠 2. LLM METADATA EXTRACTION ---

def extract_metadata_with_llm(text: str) -> dict:
    print("\nINFO: Sending text to Gemma for metadata extraction...")
    
    prompt = f"""
You are a precision data extraction engine. Your sole task is to analyze text from a university question paper and extract specific metadata fields into a clean JSON object.

Follow these rules exactly:
1.  **Output Format:** Respond ONLY with a single, valid JSON object. Do not include any other text or explanations.
2.  **department:** Extract the students' department. If two departments are mentioned (one offering the course and one taking it), prioritize the department listed next to the "Degree" or "Branch".
3.  **subject:** Extract only the subject name. The name often follows a subject code (e.g., "MEIR11") and keywords like "Title" or a dash (–). Exclude the code.This is typically a short phrase (2-5 words) found in the header of the document, often in ALL CAPS or next to a subject code. **Crucially, the subject is never a long sentence or a question copied from the exam body.
4.  **year:** Extract the four-digit year. If a date range is given (e.g., "Dec. 2022/Jan. 2023"), use the later year.
5.  **Missing Values:** If any field cannot be found, its value must be null.
6.  **No Guessing:** Do not invent or infer information not explicitly present in the text.

---
[EXAMPLE 1]
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
[EXAMPLE 2]
Input Text:
\"\"\"
B.E./B.Tech. Examinations, Dec. 2022/Jan. 2023
Common to all branches
22PHYS12 – Engineering Physics
\"\"\"
Correct JSON Output:
{{
  "department": null,
  "subject": "Engineering Physics",
  "year": 2023
}}
---
[EXAMPLE 3]
Input Text:
\"\"\"
DEPARTMENT OF MECHANICAL ENGINEERING
Degree : B.Tech. — Electronics and Communication Engineering
Sub. Code & Title : MEIRI1, Basics of Mechanical Engineering
\"\"\"
Correct JSON Output:
{{
  "department": "Electronics and Communication Engineering",
  "subject": "Basics of Mechanical Engineering",
  "year": null
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
        print("\n--- Full Extracted Text (from Tesseract) ---")
        print(raw_text)
        print("--------------------------------------------")
        
        extracted_metadata = extract_metadata_with_llm(raw_text)

        # --- NEW: REGEX FALLBACK FOR YEAR ---
        if extracted_metadata and extracted_metadata.get("year") is None:
            print("INFO: LLM did not find a year. Trying regex fallback...")
            # This pattern looks for a 4-digit number starting with 20xx
            year_match = re.search(r"\b(20\d{2})\b", raw_text)
            if year_match:
                extracted_metadata["year"] = int(year_match.group(1))
                print(f"      -> Found year '{extracted_metadata['year']}' using regex.")
        # --- END OF NEW CODE ---
        
        print("\n--- Extracted Metadata (from Gemma) ---")
        if extracted_metadata:
            print(json.dumps(extracted_metadata, indent=2))
        else:
            print("Could not extract metadata.")
    else:
        print("\n[CONCLUSION] No text was extracted from the PDF.")