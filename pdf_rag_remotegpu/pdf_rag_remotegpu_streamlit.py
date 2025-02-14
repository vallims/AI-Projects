# app.py

import streamlit as st
import os
import logging
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
import nltk
from nltk.tokenize import sent_tokenize
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants

MODEL_NAME = "llama3.2"
# Change ip and port to your GPU where the model is running.
# In case of local model,url = "http://127.0.0.1:11434"
url = "http://10.21.34.152:8010"
EMBEDDING_MODEL = "mxbai-embed-large"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"

def sentence_based_splitter(documents, chunk_size=5):
    """Split documents using sentence-based tokenizer."""
    chunks = []
    for doc in documents:
        sentences = sent_tokenize(doc.page_content)
        for i in range(0, len(sentences), chunk_size):
            chunk_text = " ".join(sentences[i:i+chunk_size])
            chunks.append(doc.__class__(page_content=chunk_text))
    print("Documents split into sentence-based chunks.")
    return chunks

def is_model_available(model_name):
    """Check if the model is available on a remote Ollama server."""
    client = ollama.Client(host=url)  # Connect to remote Ollama server
    available_models = [m.model.split(":")[0] for m in client.list().models]
    return model_name in available_models

def ingest_pdf(uploaded_file):

    """Load PDF document from uploaded file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())  # Save in temp file
            tmp_path = tmp_file.name  # Get temp file path

        loader = PyMuPDFLoader(file_path=tmp_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading PDF: {str(e)}")
        st.error("Failed to load the PDF.")
        return None

def split_documents(documents):
    """Split documents into smaller chunks."""

    chunks = sentence_based_splitter(documents)
    logging.info("Documents split into chunks.")
    return chunks


@st.cache_resource
def load_vector_db(uploaded_file):
    """Load or create the vector database."""
    # Pull the embedding model if not already available
    if uploaded_file is None:
        return None
    
    if not is_model_available(EMBEDDING_MODEL):
        logging.error(f"Embedding model '{EMBEDDING_MODEL}' not found.")
        st.error(f"Embedding model '{EMBEDDING_MODEL}' not found. Pulling the model...")
        client = ollama.Client(host=url)  # Connect to remote Ollama server
        client.pull(EMBEDDING_MODEL)

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL,base_url=url)
    
    data = ingest_pdf(uploaded_file)
    if data is None:
        return None

    # Split the documents into chunks
    chunks = split_documents(data)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name=VECTOR_STORE_NAME,
        persist_directory=PERSIST_DIRECTORY,
    )
    #vector_db.persist()
    logging.info("Vector database created and persisted.")
    return vector_db


def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",

    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    """Create the chain with preserved syntax."""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created with preserved syntax.")
    return chain


def main():
    st.title("Document Assistant")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing uploaded PDF..."):
            vector_db = load_vector_db(uploaded_file)
            if vector_db is None:
                st.error("Failed to load or create the vector database.")
                return

            st.success("PDF processed successfully. You can now ask questions!")

    # User input
    user_input = st.text_input("Enter your question:", "")

    if user_input and uploaded_file is not None:
        with st.spinner("Generating response..."):
            try:
                if not is_model_available(MODEL_NAME):
                    logging.error(f"Ollama model '{MODEL_NAME}' not found.")
                    st.error(f"Embedding model '{MODEL_NAME}' not found. Pulling the model...")
                    client = ollama.Client(host=url)  # Connect to remote Ollama server
                    client.pull(MODEL_NAME)
                llm = ChatOllama(model=MODEL_NAME,base_url=url)
                retriever = create_retriever(vector_db, llm)
                chain = create_chain(retriever, llm)
                response = chain.invoke(input=user_input)

                st.markdown("**Assistant:**")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload a PDF and enter a question to get started.")

    
if __name__ == "__main__":
    main()
