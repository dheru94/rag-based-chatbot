# build_index.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

# Load PDF
loader = PyPDFLoader("dataset/12th_chemistry.pdf")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS vector store
vectorstore = FAISS.from_documents(docs, embedding_model)

# Save to disk
vectorstore.save_local("vectorstore")
print("✅ Vector store built and saved to 'vectorstore/'")
