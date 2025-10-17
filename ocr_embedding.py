#extracting using embeddings

import os
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
from sentence_transformers import SentenceTransformer, util

# --- ⚙️ CONFIGURATION ---
PDF_PATH = "lynx_gpt_qp-2.pdf"
# This is a popular, high-performance model that's small and fast.
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- 🖼️ 1. OCR (Text Extraction with Tesseract) ---

def simple_preprocess_image(image):
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
        images = convert_from_path(pdf_path)
        full_text = []
        for i, image in enumerate(images):
            print(f"INFO: Processing page {i + 1}/{len(images)}...")
            preprocessed_img = simple_preprocess_image(image)
            text = pytesseract.image_to_string(preprocessed_img)
            if text:
                full_text.append(text)
        print("INFO: OCR completed.")
        return "\n".join(full_text) # Use a single newline for easier chunking
    except Exception as e:
        print(f"[ERROR] Failed during Tesseract extraction: {e}")
        return ""

# --- 🧠 2. METADATA EXTRACTION (Sentence Transformers) ---

def extract_metadata_with_embeddings(text: str) -> dict:
    """
    Extracts metadata by performing semantic search on text chunks.
    """
    print(f"\nINFO: Initializing Sentence Transformer model '{MODEL_NAME}'...")
    # Load the pre-trained model. This will download it on the first run.
    model = SentenceTransformer(MODEL_NAME)

    # 1. Chunk the text. Splitting by lines is a good start for question papers.
    chunks = [chunk.strip() for chunk in text.split('\n') if chunk.strip()]
    if not chunks:
        print("[ERROR] Text is empty or could not be chunked.")
        return {}

    # 2. Define the questions (queries) we want to ask.
    queries = {
        "department": "What is the name of the department or branch?",
        "subject": "What is the name of the subject or course?",
        "semester": "What is the semester number or name?",
        "year": "What is the year of the examination?"
    }

    print("INFO: Encoding text chunks and queries...")
    # 3. Create embeddings for both the text chunks and our queries.
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
    query_embeddings = model.encode(list(queries.values()), convert_to_tensor=True)

    print("INFO: Performing semantic search...")
    # 4. Perform semantic search to find the best chunk for each query.
    # We use cosine similarity to find the closest matches.
    search_results = util.semantic_search(query_embeddings, chunk_embeddings, top_k=1)

    # 5. Extract the answers.
    metadata = {}
    query_keys = list(queries.keys()) # department, subject, etc.

    for i, query_key in enumerate(query_keys):
        top_hit = search_results[i][0] # Get the top hit for the i-th query
        best_chunk_index = top_hit['corpus_id']
        best_chunk_text = chunks[best_chunk_index]
        
        # The result is the entire line that matched best.
        metadata[query_key] = best_chunk_text

    return metadata

# --- ▶️ 3. MAIN EXECUTION ---

if __name__ == "__main__":
    raw_text = extract_text_with_tesseract(PDF_PATH)
    
    if raw_text and raw_text.strip():
        print("\n--- Raw Extracted Text (Preview) ---")
        print(raw_text[:1000].strip() + "...")
        
        extracted_metadata = extract_metadata_with_embeddings(raw_text)
        
        print("\n--- Extracted Metadata (from Embeddings) ---")
        if extracted_metadata:
            # Print the final dictionary nicely formatted
            for key, value in extracted_metadata.items():
                print(f"  - {key.capitalize()}: {value}")
        else:
            print("Could not extract metadata.")
    else:
        print("\n[CONCLUSION] No text was extracted from the PDF. The process has stopped.")