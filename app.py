import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import os

from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


# Local development fallback
grok_key = st.secrets.get("grok_key", os.environ["grok_key"])


# ---- Set page config ----
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📚 RAG Chatbot with Groq")

# ---- Initialize session state ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- Load vector store and retriever ----



@st.cache_resource
def load_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vectorstore", embedding_model, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":10})

# @st.cache_resource
# def load_retriever():
#     # Load PDF
#     loader = PyPDFLoader("dataset/12th_chemistry.pdf")
#     documents = loader.load()

#     # Split text
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     docs = splitter.split_documents(documents)

#     # Embeddings
#     embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#     # Vector Store
#     vectorstore = FAISS.from_documents(docs, embedding_model)
#     # retriever = vectorstore.as_retriever()

#     return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":10})

retriever = load_retriever()

def format_docs(retriever_docs):
    context_text = "\n\n".join(doc.page_content for doc in retriever_docs)
    return context_text


parser = StrOutputParser()


# prompt for the llm

prompt = PromptTemplate(
    
    template="""
    You're a helpful chemistry tutor.
    Answer ONLY from the provided context,
    If the context is insufficient, just say out of syllabus.

    Context:
    {context}

    Question:
    {question}

    Answer:""",
    input_variables=["context", "question"],
)


# ---- Load LLM (Groq) ----
llm = ChatGroq(
    model_name="llama3-70b-8192",  # or llama3-70b-8192, etc.
    api_key= grok_key
).bind()


# ---- Build RAG chain ----

parallel_chain = RunnableParallel({
    'context': RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
    'question': RunnableLambda(lambda x: x["question"])
})

# parallel_chain = RunnableParallel({
#     'context': retriever | RunnableLambda(format_docs),
#     'question': RunnablePassthrough()
# })

main_chain = parallel_chain | prompt | llm | parser


# ---- Chat UI ----
user_input = st.chat_input("Ask me something...")

if user_input:
    response = main_chain.invoke({"question": str(user_input)})
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", response))

# ---- Display chat history ----
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)
