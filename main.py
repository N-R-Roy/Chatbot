import streamlit as st
import re
import fitz
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.documents import Document


# API_URL = "https://router.huggingface.co/featherless-ai/v1/completions"
# here use haggingface api key
# headers = {"Authorization": "Bearer use_your_key"}

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)


PROMPT_TEMPLATE = """
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Use the following context to answer the question.

Context:
{context}

Question:
{question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def clean_text(text):
    # Remove common header/footer patterns (example: "Page 1", "Confidential")
    text = re.sub(r"Page\s+\d+", "", text)
    text = re.sub(r"Confidential", "", text, flags=re.IGNORECASE)

    # Remove extra whitespace
    text = re.sub(r'\n{2,}', '\n', text)  # Collapse multiple newlines
    text = text.strip()

    return text


def clean_pdf_text(doc):
    docs = []
    for page in doc:
        text = page.get_text()

        text = clean_text(text)

        doc = Document(page_content=text, metadata={"page": page.number+1})
        docs.append(doc)

    return docs


def convert_document_into_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=250,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


def create_vector_store(chunks):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local("faiss_index/")


def generate_prompt(query_text):
    vectorstore = FAISS.load_local("faiss_index/", embedding_model, allow_dangerous_deserialization=True)

    results = vectorstore.similarity_search(query_text, k=3)

    context_text = "\n\n".join(
        [f"[{i + 1}] {result.page_content}" for i, result in enumerate(results)]
    )

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    prompt = prompt_template.format(context=context_text, question=query_text)

    return prompt


def generate_answer(query_text):
    prompt = generate_prompt(query_text)

    payload = {
        "model": "meta-llama/Llama-3.1-8B",
        "prompt": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    result = response.json()

    # Extract the generated text
    if "choices" in result and len(result["choices"]) > 0:
        return result["choices"][0]["text"]
    else:
        return "No response text generated."


# Upload PDF
uploaded_pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_pdf_file is not None:
    # Load PDF file into PyMuPDF
    with fitz.open(stream=uploaded_pdf_file.read(), filetype="pdf") as doc:
        st.write(f"Total pages: {len(doc)}")

        doc = clean_pdf_text(doc)

        chunks = convert_document_into_chunks(doc)

        create_vector_store(chunks)


user_query = st.text_input("Ask a question:")

if st.button("Submit"):
    st.write("You asked:", user_query)
    answer = generate_answer(user_query)
    st.text_area(answer, height=200)


