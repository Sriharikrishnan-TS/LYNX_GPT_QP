import streamlit as st
import pandas as pd
import json
from pdf_processor import process_single_pdf  # Import the PDF processing function
from query_processor import process_user_query # Import the query processing function

st.set_page_config(layout="wide")
st.title("📄 Question Paper Hub")

# --- Use tabs for different sections ---
tab1, tab2 = st.tabs(["Upload Question Papers", "Query Question Papers"])

# --- Tab 1: Uploading ---
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
            st.markdown("---") # Separator between file results
    else:
        st.info("Upload PDF files using the button above.")


# --- Tab 2: Querying ---
with tab2:
    st.header("Query Existing Question Papers")
    st.markdown("Enter your query to search the database for question papers.")

    # User Input for Query
    user_query = st.text_input(
        "Enter query (e.g., 'CSE papers 2023', 'algorithms endsem'):",
        key="query_input" # Unique key
    )

    # Processing Logic for Queries
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
            # Display results in a table using Pandas DataFrame
            try:
                df = pd.DataFrame(query_result["results"])
                # Display without the index column
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.success(f"Found {len(query_result['results'])} matching question papers.")
            except Exception as e:
                st.error(f"Error displaying results: {e}")
                st.write(query_result["results"]) # Show raw results if DataFrame fails
        else:
            st.warning("No matching question papers found for your query.")

    else:
        st.info("Enter a query above to search the database.")