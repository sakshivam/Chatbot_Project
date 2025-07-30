import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import tempfile
import os

# Title
st.title("📄 RAG-based Chatbot for PDF using LangChain")

# Load API key securely from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 1. Upload PDF File
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if pdf_file:
    # Save PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    # 2. Load PDF and Split into Chunks
    @st.cache_resource
    def get_documents(pdf_path):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(docs)

    documents = get_documents(tmp_path)

    # 3. Create Embeddings and Vector Store
    @st.cache_resource
    def get_vectorstore(_docs):
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectordb = Chroma.from_documents(
            _docs, embedding, persist_directory="./chroma_db_pdf"
        )
        vectordb.persist()
        return vectordb

    vectordb = get_vectorstore(documents)

    # 4. Setup LangChain RAG pipeline
    retriever = vectordb.as_retriever()

    groq_llm = ChatOpenAI(
        model="llama3-70b-8192",
        openai_api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        temperature=0,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=groq_llm,
        retriever=retriever,
        return_source_documents=True,
    )

    # 5. Display previous chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 6. Chat UI
    query = st.chat_input("Ask a question about your PDF 👇")
    if query:
        # Save user message
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain({"query": query})
                answer = response["result"]
                sources = response["source_documents"]

                # Format assistant reply
                assistant_reply = answer
                st.markdown(assistant_reply)

                # Show sources
                source_str = ""
                for i, doc in enumerate(sources):
                    source_str += f"**Chunk {i+1}:**\n```\n{doc.page_content}\n```\n"

                with st.expander("📄 Sources"):
                    st.markdown(source_str)

                # Save assistant message with sources
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": assistant_reply}
                )

    # Optional: Clear history button
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()

    # Clean up temp file
    os.remove(tmp_path)

else:
    st.info("📥 Upload a PDF file to begin chatting with it.")
