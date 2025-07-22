import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA                 # # Conversational retrieval where previous QA is Saved
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Step 1: Setup LLM (Mistral with HuggingFace)

#only this part of code that is making easy to use LLM
# this is the repoid from Huggingface
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,                     # this number 0.5 will give more precise answer
        model_kwargs={"token":HF_TOKEN,      # HF token is the repo that we took from huggungface
                      "max_length":"512"}    # 512 is the lenth of the answer in chat how long 
    )
    return llm

# Step 2: Connect LLM with FAISS and Create chain
# context- it will retreive from the datset or pdf
# question- question asked by the user- Eg: what is diabtes?

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

# "promptTemplate" library used from Langchain 

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Now we need to load the database by which model it has embedded we need to load 
# Modle we created datbase need to load that is  emebedded

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"    # this is the path for database
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")   #minilm-l6-v2 is the model 
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)   #we are going to load the database with LLM
# allow_dangerous_deserialization=True - This is for the [Source of information is trusted and verified then TRUE"vectorstore"]


# Create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),  # thats the MISTRAL REPOID yo took from huggingface
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),   # DB shouls be itself as retrievr/ ARGUMENT PASSED- KWARGS - "3" - Ranked best TOP three results as document should retreive from similar words from document
    return_source_documents=True,      # TRUE beacuse we need documents to show 
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}    # CUSTOM prompt that we have created you recall the function 
)

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
