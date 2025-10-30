import streamlit as st
import pandas as pd
import json
import os
import requests  # <-- Added for download button
from pdf_processor import process_single_pdf  # Import the PDF processing function
from query_processor import process_user_query # Import the query processing function

st.set_page_config(layout="wide")
st.title("Question Paper Hub")

# --- Helper function to fetch and cache PDF bytes ---
@st.cache_data
def get_pdf_bytes(url):
    """Fetches and caches the PDF bytes from a public URL."""
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        return response.content
    except requests.RequestException as e:
        st.error(f"Error fetching PDF for download: {e}")
        return None

# --- Use tabs for different sections ---
tab1, tab2 = st.tabs(["Upload Question Papers", "Query Question Papers"])

# --- Tab 1: Uploading (No changes here) ---
with tab1:
    st.header("Upload New Question Papers")
    st.markdown("Select one or more PDF question papers to extract metadata and save to the database.")

    # File Uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader" # Unique key for this uploader
    )

    # Processing Logic for Uploads
    if uploaded_files:
        st.markdown("---")
        st.subheader("Processing Status:")

        # Process each uploaded file
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            st.write(f"Processing **{file_name}**...")

            with st.spinner(f"Extracting metadata for {file_name}..."):
                # Read the file content as bytes
                pdf_bytes = uploaded_file.getvalue()

                # Call the processing function from pdf_processor.py
                result = process_single_pdf(pdf_bytes, file_name)

            # Display the result for each file
            if result.get("status") == "Success":
                st.success(f"Successfully processed and saved **{file_name}**.")
                
            else:
                st.error(f"Failed to process **{file_name}**: {result.get('message', 'Unknown error')}")
                # Optionally show partial metadata if available on error
                if "metadata" in result:
                    with st.expander("Show Partially Extracted Metadata (Before DB Error)"):
                        st.json(result.get("metadata", {}))
                    with st.expander("show raw extracted text from ocr"):
                        st.write(result.get("raw_text"))
            st.markdown("---") # Separator between file results
    else:
        st.info("Upload PDF files using the button above.")


# Tab 2: Querying 
with tab2:
    st.header("Query Existing Question Papers")
    st.markdown("Enter your query to search the database for question papers.")

    # --- 1. Create a form ---
    with st.form(key="query_form"):
        user_query = st.text_input(
            "Enter query (e.g., 'CSE papers 2023', 'algorithms endsem'):",
            key="query_input" # Unique key
        )
        
        submit_button = st.form_submit_button(label="Send Query") # Changed label for clarity

    # --- 3. Check if the button was pressed ---
    if submit_button:
        if user_query:
            st.markdown("---") # Separator
            with st.spinner("Searching the database..."):
                query_result = process_user_query(user_query)

            # Display Debug Info (Optional)
            with st.expander("Show Query Processing Details"):
                st.write("**Extracted Search Criteria (from LLM):**")
                st.json(query_result.get("metadata", {}))
                st.write("**Generated SQL Query:**")
                st.code(query_result.get("sql", "N/A"), language="sql")
            

            # Display Results or Error
            st.markdown("---")
            st.subheader("Search Results:")

            if query_result.get("error"):
                st.error(f"An error occurred: {query_result['error']}")
            elif query_result.get("results"):
                
                st.success(f"Found {len(query_result['results'])} matching question papers.")
                
                for index, paper in enumerate(query_result['results']):
                    st.markdown(f"**Department:** {paper.get('department', 'N/A')}")
                    st.markdown(f"**Subject:** {paper.get('subject','N/A')}")
                    st.markdown(f"**Year:** {paper.get('year', 'N/A')}")

                    file_url = paper.get('file_url')
                    if file_url:
                        file_extension = os.path.splitext(file_url)[1].lower()
                        
                        if file_extension == '.pdf':
                            # --- FIX 1: Replaced st.pdf() with a reliable iframe ---
                            with st.expander("View PDF", expanded=False):
                                st.markdown(
                                    f'<iframe src="{file_url}" width="100%" height="700" style="border: none; border-radius: 8px;"></iframe>', 
                                    unsafe_allow_html=True
                                )

                        elif file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                            with st.expander("View Image", expanded=False):
                                st.image(file_url)
                        else:
                            st.warning(f"Cannot display file type '{file_extension}' directly.")

                        # --- FIX 2 & 3: Fixed link_button and added download_button ---
                        col1, col2 = st.columns(2)
                        with col1:
                            # 1. Open in new tab (key removed)
                            st.link_button("Open in New Tab", file_url, use_container_width=True)

                        with col2:
                            # 2. Download button (newly added)
                            pdf_bytes = get_pdf_bytes(file_url)
                            
                            # Create a clean filename
                            file_name = f"{paper.get('subject', 'paper')}_{paper.get('year', 'NA')}.pdf".replace(" ", "_").replace("/", "_")
                            
                            if pdf_bytes:
                                st.download_button(
                                    label="Download PDF",
                                    data=pdf_bytes,
                                    file_name=file_name,
                                    mime="application/pdf",
                                    use_container_width=True
                                )

                    else:
                        st.warning("File URL missing for this database entry.")

                    st.divider() # Use st.divider for a modern separator

            else:
                st.warning("No matching question papers found for your query.")
        else:
            st.warning("Please enter a query before sending.")

    else:
        st.info("Enter a query above and press 'Send Query' to search.")
