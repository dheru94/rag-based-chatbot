## 📚 RAG Chatbot with Groq & Streamlit
A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, and Groq's LLaMA 3 models. Upload your textbook (PDF), and ask natural language questions — the bot will answer using only the provided content.

<!-- You can replace this with a real screenshot -->

## 🚀 Features
💬 Ask questions about Chemistry textbook class 12th

🔎 Retrieval-based: answers only from the book using FAISS vector store

🤖 Powered by Groq's LLaMA 3 (llama3-70b-8192)

🧠 Embeddings with HuggingFace (all-MiniLM-L6-v2)

⚡ Fast local vector store using FAISS

🧾 Streamlit interface with chat history

## 📂 Project Structure
bash
Copy
Edit
.
├── app.py                  # Main Streamlit app
├── generate_vectorstore.py # (Optional) Script to build FAISS index
├── dataset/
│   └── 12th_chemistry.pdf  # Example PDF
├── vectorstore/            # FAISS index (created automatically or manually)
├── .env                    # API keys and secrets
├── requirements.txt        # All dependencies
└── README.md               # This file
## ⚙️ Setup Instructions
# 1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/rag-streamlit-chatbot.git
cd rag-streamlit-chatbot
# 2. Create and Activate a Virtual Environment
bash
Copy
Edit
python -m venv myenv
source myenv/bin/activate  # or `myenv\Scripts\activate` on Windows
# 3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
# 4. Add Environment Variables
Create a .env file:

bash
Copy
Edit
touch .env
Paste your Groq API Key into it:

env
Copy
Edit
grok_key=your_groq_api_key_here
## 🛠️ Usage
Option A: Generate the vectorstore first (recommended for faster app start)
bash
Copy
Edit
python generate_vectorstore.py
Option B: Let the app build it (slower on first run)
Just run the app:

bash
Copy
Edit
streamlit run app.py
Then open http://localhost:8501 in your browser.

## 📌 Example Question
"What are d-block elements and their properties?"

🧠 The chatbot retrieves the relevant chunks from your PDF and responds using Groq’s LLM, strictly limited to your textbook's content.

## 📦 Requirements
Python 3.10+

Streamlit

LangChain (core, community)

FAISS

HuggingFace Transformers

sentence-transformers

python-dotenv

# You can install them all via:

bash
Copy
Edit
pip install -r requirements.txt
# ✅ TODO / Improvements
 Upload new PDFs dynamically

 Add support for multiple documents

 UI improvements with expandable context view

 Add history export/download

## 🤝 Credits
LangChain

Groq API

Sentence Transformers

Streamlit

## 📜 License
MIT — feel free to use, fork, and modify.

Let me know if you'd like a requirements.txt, a generate_vectorstore.py script, or a version of this README with dynamic PDF uploading features included.