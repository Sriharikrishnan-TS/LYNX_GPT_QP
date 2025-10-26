import streamlit as st
import pandas as pd
import json
import os
from pdf_processor import process_single_pdf  # Import the PDF processing function
from query_processor import process_user_query # Import the query processing function

st.set_page_config(layout="wide")
st.title("Question Paper Hub")

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
    # This groups the text input and submit button
    with st.form(key="query_form"):
        # User Input for Query (now inside the form)
        user_query = st.text_input(
            "Enter query (e.g., 'CSE papers 2023', 'algorithms endsem'):",
            key="query_input" # Unique key
        )
        
        submit_button = st.form_submit_button(label="Send")

    # --- 3. Check if the button was pressed ---
    if submit_button:
        # We also check if the user actually typed something
        if user_query:
            st.markdown("---") # Separator
            with st.spinner("Searching the database..."):
                # Call the backend processing function from query_processor.py
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
                
                st.success(f"found {len(query_result['results'])} matching question papers")
                for index, paper in enumerate(query_result['results']):
                    key_prefix = f"paper_{index}_"
                    st.markdown(f"**Department:** {paper.get('department', 'N/A')}")
                    st.markdown(f"**Subject:**{paper.get('subject','N/A')}")
                    st.markdown(f"**Year:** {paper.get('year', 'N/A')}")

                    file_url = paper.get('file_url')
                    if file_url:
                        file_extension = os.path.splitext(file_url)[1].lower()
                        if file_extension == '.pdf':
                            with st.expander("View PDF", expanded=False):
                                try:
                                    st.pdf(file_url, height=700)
                                except Exception as pdf_error:
                                    st.error(f"Could not load PDF viewer. Error: {pdf_error}")
                                    st.markdown(f"[Direct Link to PDF]({file_url})")

                        elif file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                            with st.expander("View Image", expanded=False):
                                try:
                                    st.image(file_url)
                                except Exception as img_error:
                                    st.error(f"Could not load image. Error: {img_error}")
                                    st.markdown(f"[Direct Link to Image]({file_url})")
                        else:
                            st.warning(f"Cannot display file type '{file_extension}' directly.")
                            st.markdown(f"[Download File]({file_url})")

                        st.link_button("Open File in New Tab", file_url, key=f"{key_prefix}_link")

                    else:
                        st.warning("File URL missing for this database entry.")

                    st.divider()

            else:
                st.warning("No matching question papers found for your query.")
        else:
            # Show a warning if they click "Send" with no text
            st.warning("Please enter a query before sending.")

    else:
        # This is the default message before the button is clicked
        st.info("Enter a query above and press 'Send Query' to search.")