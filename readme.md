# 📄 RAG-based PDF Chatbot with Groq and Streamlit

An intelligent PDF chatbot built using LangChain, Streamlit, and Groq's LLMs (e.g. LLaMA3). This app allows users to:

- 📤 Upload a PDF
- 🔍 Ask semantic questions based on its content
- 🧠 Uses Retrieval-Augmented Generation (RAG) pipeline
- 💬 Maintains chat history per session
- ⚡ Powered by Groq API for blazing-fast inference

---

## 📦 Requirements

- Python 3.8+
- `streamlit`
- `langchain`
- `chromadb`
- `sentence-transformers`
- `groq` (via `ChatOpenAI` API compatibility)

Install all dependencies with:

```bash
pip install -r requirements.txt

## Setup
1. Add your Groq API key
Create a .streamlit/secrets.toml file in your project root:

```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "your-groq-api-key-here"
```

## ▶️ Run the App
```python
streamlit run Semantic_Search_based_RAG_chatbot.py
```
The app will launch in your default browser.

## 📁 Project Structure
RAG-Chatbot/
├── RAG_main.py              # Main Streamlit app
├── requirements.txt         # Python dependencies
├── readme.md                # Project documentation
├── chroma_db_pdf/           # Chroma vector store (auto-created)
└── .streamlit/
    └── secrets.toml         # Groq API key (secure)

## Example Questions (for this PDF)
Assuming you're using a PDF like:
14 Examples of How LLMs Can Transform Materials Science and Chemistry

You could ask:
“What are the top use cases of LLMs in materials science?”
"Summarize the key findings or reflections discussed."
"What are the challenges of using LLMs in materials science workflows?"
"Explain the workflow used in project #3"
"Which tools or libraries were integrated with LLMs in these experiments?"

## 🧠 How it Works
* PDF is loaded and chunked using LangChain.
* Text chunks are embedded via sentence-transformers and stored in a Chroma vector DB.
* User's query is semantically matched with relevant chunks.
* Groq’s LLaMA3 responds with context-aware answers via LangChain's RetrievalQA.






