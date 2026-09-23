import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Same thing as from connect file only difference is
# when the model is loaded Database will be saved in CACHE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")

# A few starter questions so users know what MediBot can do
EXAMPLE_QUESTIONS = [
    "What are the symptoms of anemia?",
    "What causes asthma?",
    "How is diabetes diagnosed?",
    "What is the treatment for migraine?",
]


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def render_sources(source_documents):
    """Show the retrieved encyclopedia passages in a tidy, collapsible section."""
    if not source_documents:
        return
    with st.expander(f"📚 Sources ({len(source_documents)} references)"):
        for i, doc in enumerate(source_documents, start=1):
            page = doc.metadata.get("page", "N/A")
            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:300] + "…"
            st.markdown(f"**Reference {i}** — page {page}")
            st.caption(snippet)


def build_qa_chain():
    custom_prompt_template = """
You are MediBot, a warm and friendly medical information assistant.
Use ONLY the information in the context below to answer the user's question.

Follow these rules carefully:
- If the user's message is a greeting, small talk, or is NOT a medical or health-related
  question, do NOT use the context. Reply exactly with:
  "Hi there! 👋 I'm MediBot 🩺 — your friendly medical information assistant. I can only help
  with medical and health-related questions, like symptoms, diseases, treatments, and medicines.
  Ask me anything health-related and I'll do my best to help! 😊"
- If it IS a medical question but the answer is not found in the context, reply with:
  "I'm sorry, I couldn't find that in my medical encyclopedia. 🩺 Please try rephrasing, or ask
  me about another medical topic!"
- Otherwise, answer clearly and directly using the context. No small talk, just the answer.

Context: {context}
Question: {question}

Answer:
"""
    vectorstore = get_vectorstore()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGroq(
            model_name="openai/gpt-oss-20b",  # Groq-hosted model available on this account
            temperature=0.0,
            groq_api_key=os.environ["GROQ_API_KEY"],
        ),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)},
    )
    return qa_chain


def answer_question(prompt):
    """Run one question through the RAG chain and render the assistant reply."""
    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("MediBot is looking that up… 🔎"):
            try:
                qa_chain = build_qa_chain()
                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                source_documents = response.get("source_documents", [])

                st.markdown(result)
                render_sources(source_documents)

                st.session_state.messages.append({'role': 'assistant', 'content': result})
            except Exception as e:
                error_msg = f"⚠️ Something went wrong: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({'role': 'assistant', 'content': error_msg})


def main():
    st.set_page_config(page_title="MediBot 🩺", page_icon="🩺", layout="centered")

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("🩺 MediBot")
        st.write("Your friendly AI medical information assistant.")

        st.subheader("💡 Try asking")
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, use_container_width=True, key=f"ex_{q}"):
                st.session_state.pending_prompt = q

        st.divider()
        st.subheader("ℹ️ How it works")
        st.write(
            "MediBot searches a trusted medical encyclopedia and uses AI to answer "
            "your health questions — with the source pages it used."
        )

        st.divider()
        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.caption("👨‍💻 Built by Abdu Raqib Hidayathulla")

    # ---------------- Header ----------------
    st.title("🩺 MediBot")
    st.caption("Ask me anything about symptoms, diseases, treatments and medicines.")

    st.warning(
        "⚠️ **Disclaimer:** MediBot is for informational and educational purposes only. "
        "It is **not** a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare provider for any medical concerns."
    )

    # ---------------- Session state ----------------
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Friendly welcome when the chat is empty
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🩺"):
            st.markdown(
                "Hi there! 👋 I'm **MediBot** — your friendly medical information assistant.\n\n"
                "Ask me a health question, or tap one of the examples in the sidebar to get started! 😊"
            )

    # Replay chat history
    for message in st.session_state.messages:
        avatar = "🧑" if message['role'] == 'user' else "🩺"
        st.chat_message(message['role'], avatar=avatar).markdown(message['content'])

    # ---------------- Handle input ----------------
    prompt = st.chat_input("Type your medical question here…")

    # An example-question button may have queued a prompt
    if not prompt and st.session_state.get("pending_prompt"):
        prompt = st.session_state.pop("pending_prompt")

    if prompt:
        st.chat_message('user', avatar="🧑").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        answer_question(prompt)


if __name__ == "__main__":
    main()
