import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("RAG Enterprise Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embeddings)

    query = st.text_input("Ask a question from the document")

    if query:
        results = db.similarity_search(query)
        answer = "\n".join([doc.page_content for doc in results])
        st.write(answer)
