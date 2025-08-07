import streamlit as st
from PIL import Image
import fitz  # PyMuPDF


# Upload PDF
uploaded_pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_pdf_file is not None:
    # Load PDF file into PyMuPDF
    with fitz.open(stream=uploaded_pdf_file.read(), filetype="pdf") as doc:
        st.write(f"Total pages: {len(doc)}")

        # Option to select a page
        page_number = st.number_input("Page number", min_value=1, max_value=len(doc), value=1)
        page = doc[page_number - 1]

        # Extract and display text
        text = page.get_text()
        st.text_area("Page text:", text, height=400)

