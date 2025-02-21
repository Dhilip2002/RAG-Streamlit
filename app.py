import os
import tempfile
import warnings
import requests
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredExcelLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma  # Correct import for ChromaDB
from pathlib import Path
# import chromadb  # Ensure chromadb is installed

warnings.filterwarnings("ignore", message="Warning: Empty content on page")

# Constants
COLLECTION_NAME = "cv_collection"
PERSIST_DIRECTORY = "./chroma_db"  # Use relative path for Streamlit Cloud
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

# Initialize Streamlit
st.title("ChromaDB & LM Studio AI Query System")
st.subheader("Upload and Store Documents for AI-Powered Retrieval")

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary directory and return its path."""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    return file_path

def load_and_store_documents(uploaded_files):
    """Processes uploaded files, extracts embeddings, and stores them in ChromaDB."""
    if not uploaded_files:
        st.warning("No files uploaded.")
        return
    
    st.write("Processing uploaded documents...")
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-mpnet-base-v2")
    documents = []
    
    for uploaded_file in uploaded_files:
        file_path = save_uploaded_file(uploaded_file)
        
        if uploaded_file.name.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
        elif uploaded_file.name.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(file_path)
        else:
            st.warning(f"Unsupported file format: {uploaded_file.name}")
            continue
        
        documents.extend(loader.load())
    
    documents = [doc for doc in documents if doc.page_content.strip()]
    
    if not documents:
        st.error("No valid content extracted from uploaded documents.")
        return
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    if not texts:
        st.error("Text splitting failed. No content available for processing.")
        return
    
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model, collection_name=COLLECTION_NAME)
    
    try:
        if vectordb._collection.count() == 0:
            vectordb = Chroma.from_documents(documents=texts, embedding=embedding_model, persist_directory=PERSIST_DIRECTORY, collection_name=COLLECTION_NAME)
        
        st.success(f"Final number of stored documents: {vectordb._collection.count()}")
        st.write("Chroma DB successfully updated.")
    except Exception as e:
        st.error(f"Error updating ChromaDB: {e}")

def retrieve_and_answer_query(query):
    """Retrieves relevant documents and queries the AI model."""
    if not os.path.exists(PERSIST_DIRECTORY):
        st.error("Chroma DB directory does not exist. Please upload documents first.")
        return
    
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-mpnet-base-v2")
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, collection_name=COLLECTION_NAME, embedding_function=embedding_function)
    retriever = vectordb.as_retriever()
    
    try:
        search_results = retriever.invoke(query)
        if not search_results:
            st.warning("No relevant context found in the database.")
            return
        
        context = "\n\n".join([result.page_content for result in search_results])
        payload = {
            "model": "phi-3.1-mini-128k-instruct",
            "messages": [
                {"role": "system", "content": "Answer queries using the provided context."},
                {"role": "user", "content": f"Context: {context}\n\nUser Query: {query}"}
            ],
            "temperature": 0.7,
            "max_new_tokens": 1024,
            "stream": False
        }
        
        response = requests.post(LM_STUDIO_URL, headers={"Content-Type": "application/json"}, json=payload)
        response.raise_for_status()
        ai_response = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        st.subheader("AI Response:")
        st.write(ai_response)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Streamlit UI Components
uploaded_files = st.file_uploader("Upload PDF and Excel Files", accept_multiple_files=True, type=["pdf", "xlsx"])

if st.button("Load & Store Documents"):
    load_and_store_documents(uploaded_files)

query = st.text_input("Enter your query:")
if st.button("Retrieve & Answer") and query:
    retrieve_and_answer_query(query)
