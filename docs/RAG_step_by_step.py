import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

import streamlit as st
from groq import Groq

df = pd.read_csv("./data/amazon_books_data.csv")
# Convert each row of a DataFrame (df) into a LangChain Document object.
# `row.to_string()` converts the entire row into a plain text string.
# This is useful when preparing data for embedding and retrieval.
texts = [Document(page_content=row.to_string()) for _, row in df.iterrows()]
# Initialize a text splitter.
# This will break large text documents into smaller overlapping chunks.
# chunk_size=500 means each chunk will be at most 500 characters long.
# chunk_overlap=50 ensures each chunk overlaps with the previous one by 50 characters,
# which helps preserve context continuity across chunks.
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# Split the list of full-row documents into smaller chunks using the text splitter.
# This step ensures each chunk stays within LLM token limits and improves retrieval accuracy.
docs = splitter.split_documents(texts)

# Embed the Chunks
# Use a sentence-transformers model to generate vector embeddings.
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store in a Vector DB
# Use ChromaDB to store and retrieve relevant chunks.
vectordb = Chroma.from_documents(
    documents=docs, embedding=embedding, persist_directory="./rag_store"
)
vectordb.persist()

# Retrieve Context for Query
# Use similarity search to fetch top-k relevant pieces of the CSV.
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
query = input("Enter your question: ")
relevant_docs = retriever.get_relevant_documents(query)
context = "\n\n".join([doc.page_content for doc in relevant_docs])

# Modify LLM Prompt to Include Retrieved Context
# Update your call to Groq like this:
system_prompt = (
    f"""You are a data analyst who answers based on the following data:\n\n{context}"""
    "\n\nUse the context to answer the user's question. "
)

chat_history = [{"role": "system", "content": system_prompt}]
chat_history += st.session_state.messages  # append chat history

client = Groq(api_key=st.secrets["GROQ_API_KEY"])
stream = client.chat.completions.create(
    model=st.session_state["groq_model"],
    messages=chat_history,
    stream=True,
)
