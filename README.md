 🩺 MediBot – AI-Powered Medical Chatbot

**MediBot** is an interactive medical chatbot powered by **Large Language Models (LLMs)**, **LangChain**, and **FAISS**.
It allows users to ask medical questions and get context-aware, accurate responses with traceable sources extracted from a trusted medical encyclopedia.

The application is built with a user-friendly **Streamlit interface**, uses **Groq-hosted Mistral or HuggingFace models**, and retrieves relevant document chunks using semantic vector search.

---
# ✨ Features

- 🔍 **Semantic Search** using FAISS and SentenceTransformers
- 🤖 **LLM-Powered Answers** (Groq-hosted LLaMA or HuggingFace Mistral)
- 📄 **Contextual QA** from embedded medical PDFs
- 📚 **Source Document Tracing** in responses
- 💬 **Chat Interface** built with Streamlit
- ⚡ **Cached Vector Store** for fast response time

---

# 🧠 How It Works

## 🔧 `create_memory_for_llm.py`
- Loads medical PDFs from the `data/` directory
- Splits text into chunks using `RecursiveCharacterTextSplitter`
- Converts chunks into embeddings via `sentence-transformers/all-MiniLM-L6-v2`
- Stores embeddings locally in a **FAISS** vector database

## 🔗 `connect_memory_with_llm.py`
- Loads the FAISS vector store
- Defines a custom prompt template for controlled responses
- Connects to **Mistral-7B** hosted on HuggingFace
- Executes a single QA inference from the command line

## 🖥️ `medibot.py`
- Runs a **Streamlit** web app
- Supports conversational chat with persistent session state
- Queries Groq-hosted LLaMA or any LLM via HuggingFace
- Displays answers and their source documents cleanly


## 🛠 Tools and Technologies Used

| Tool/Library      | Description |
|------------------|-------------|
| **LangChain**     | Framework for building LLM applications |
| **FAISS**         | Facebook AI Similarity Search – Vector database |
| **Mistral 7B**    | Lightweight LLM used via HuggingFace or Groq |
| **HuggingFace**   | ML/LLM model hub |
| **Groq API**      | Ultra-fast inference API for open-source models |
| **Python**        | Primary programming language |
| **Streamlit**     | UI framework for interactive web apps |
| **VS Code**       | Development environment |
| **dotenv**        | For environment variable management |

---

## 🚀 Getting Started
pip install -r requirements.txt
3. Add API Keys
Create a .env file in the root directory:

GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token  # Optional if using HuggingFace

4. Prepare the Vectorstore

python create_memory_for_llm.py

5. Run the Chatbot UI
streamlit run medibot.py
💬 Example Output
User Prompt:

How to cure cancer?
Response:
The best chance for a surgical cure is usually with the first operation...
Source Docs:

[Document(metadata={'source': 'data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf', 'page': 26}, ...)]
📌 To-Do List
 File upload for custom documents

 Source highlighting in document text

 Result formatting improvements

 Deployment on Streamlit Cloud or HuggingFace Spaces

🧑‍💻 Author
Abdu Raqib Hidayathulla
LinkedIn - https://www.linkedin.com/in/abdu-raqib-hidayathulla-6b8664244/ 
Email- abduraqibfarhan@gmail.com

