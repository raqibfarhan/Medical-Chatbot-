# 🩺 MediBot – AI-Powered Medical Chatbot

**MediBot** is a Retrieval-Augmented Generation (RAG) medical chatbot powered by **LangChain**, **FAISS**, and **Large Language Models**.
It answers medical questions with context-aware, accurate responses and traceable sources extracted from *The Gale Encyclopedia of Medicine (2nd Edition)*.

The project includes:
- A **Streamlit web app** using Groq-hosted LLaMA 4 Maverick for fast inference
- A **CLI tool** using HuggingFace-hosted Mistral-7B for terminal-based Q&A

---

## ✨ Features

- 🔍 **Semantic Search** – FAISS vector database with SentenceTransformer embeddings
- 🤖 **Dual LLM Support** – Groq-hosted LLaMA 4 Maverick (web) and HuggingFace Mistral-7B (CLI)
- 📄 **RAG Pipeline** – Retrieval-Augmented Generation from embedded medical PDFs
- 📚 **Source Tracing** – Every answer includes the source documents and page numbers
- 💬 **Chat Interface** – Conversational UI built with Streamlit and persistent chat history
- ⚡ **Cached Vector Store** – `@st.cache_resource` for fast repeated queries

---

## 📁 Project Structure

```
Medical-Chatbot/
└── Medibot/
    └── medical-chatbot/
        ├── data/
        │   └── The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf
        ├── vectorstore/
        │   └── db_faiss/
        │       ├── index.faiss
        │       └── index.pkl
        ├── create_memory_for_llm.py   # Step 1: PDF → Chunks → Embeddings → FAISS
        ├── connect_memory_with_llm.py # Step 2: CLI-based QA with Mistral-7B
        ├── medibot.py                 # Step 3: Streamlit web app with Groq LLaMA
        ├── Pipfile                    # Pipenv dependency config
        ├── Pipfile.lock
        ├── requirements.txt           # Pip dependency list
        └── README.md                  # Pipenv setup guide
```

---

## 🧠 How It Works

### Step 1 – Data Ingestion & Embedding (`create_memory_for_llm.py`)

1. Loads all PDFs from the `data/` directory using `PyPDFLoader`
2. Splits the extracted text into **500-character chunks** with 50-character overlap using `RecursiveCharacterTextSplitter`
3. Generates vector embeddings using the **`sentence-transformers/all-MiniLM-L6-v2`** model
4. Stores the embeddings locally in a **FAISS** vector database at `vectorstore/db_faiss/`

### Step 2 – CLI Q&A with HuggingFace (`connect_memory_with_llm.py`)

1. Loads the pre-built FAISS vector store
2. Connects to **Mistral-7B-Instruct-v0.3** via the HuggingFace Inference API
3. Retrieves the **top 3** most relevant document chunks for the user's query
4. Feeds the chunks as context into a custom prompt template that constrains the LLM to only answer from the provided context
5. Prints the result and source documents to the terminal

### Step 3 – Streamlit Web App with Groq (`medibot.py`)

1. Loads and **caches** the FAISS vector store using `@st.cache_resource`
2. Accepts user questions through a Streamlit chat input
3. Retrieves the top 3 relevant chunks and sends them with the query to **LLaMA 4 Maverick** via the **Groq API**
4. Displays the LLM's answer along with source documents in a conversational chat interface
5. Maintains full **chat history** across the session using `st.session_state`

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **LLM Framework** | LangChain |
| **Vector Database** | FAISS (Facebook AI Similarity Search) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **LLM (Web App)** | LLaMA 4 Maverick via Groq API |
| **LLM (CLI)** | Mistral-7B-Instruct-v0.3 via HuggingFace |
| **Web UI** | Streamlit |
| **Language** | Python 3.12 |
| **Env Management** | Pipenv / pip + python-dotenv |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Medical-Chatbot.git
cd Medical-Chatbot/Medibot/medical-chatbot
```

### 2. Install Dependencies

**Using Pipenv (recommended):**

```bash
pipenv install
pipenv shell
```

**Using pip:**

```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys

Create a `.env` file in the `medical-chatbot/` directory:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

- **GROQ_API_KEY** – Required for the Streamlit web app. Get one at [console.groq.com](https://console.groq.com)
- **HF_TOKEN** – Required for the CLI tool. Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 4. Build the Vector Store (first time only)

```bash
python create_memory_for_llm.py
```

> This processes the PDF, generates embeddings, and saves them to `vectorstore/db_faiss/`. The pre-built vector store is already included in the repo, so you can skip this step if you don't need to regenerate it.

### 5. Run the Chatbot

**Streamlit Web App (Groq):**

```bash
streamlit run medibot.py
```

**CLI Mode (HuggingFace):**

```bash
python connect_memory_with_llm.py
```

---

## 💬 Example

**User:** How to cure cancer?

**MediBot:** The best chance for a surgical cure is usually with the first operation...

**Source Docs:**

```
Document(metadata={'source': 'data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf', 'page': 26})
```

---

## 📌 To-Do

- [ ] File upload for custom documents
- [ ] Source highlighting in document text
- [ ] Result formatting improvements
- [ ] Deployment on Streamlit Cloud or HuggingFace Spaces

---

## 🧑‍💻 Author

**Abdu Raqib Hidayathulla**

- LinkedIn: [linkedin.com/in/abdu-raqib-hidayathulla-6b8664244](https://www.linkedin.com/in/abdu-raqib-hidayathulla-6b8664244/)
- Email: abduraqibfarhan@gmail.com
