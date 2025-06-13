# 💬 Groq Chatbot with Streamlit

A simple conversational chatbot powered by Groq's LLMs (e.g. LLaMA3) and built with [Streamlit](https://streamlit.io/). This app supports:

- 🧠 LLM-powered responses via Groq API
- 💬 Real-time streaming responses
- 🔄 Persistent chat history (per session)
- ⚙️ Model selection capability

---

## 📦 Requirements

- Python 3.8+
- `streamlit`
- `groq`

Install with:

```bash
pip install streamlit groq
```
## Setup
1. Add your Groq API key
Create a .streamlit/secrets.toml file in your project root:

```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "your-groq-api-key-here"
```

## ▶️ Run the App
```python
streamlit run main.py
```
The app will launch in your default browser.

## 📁 Project Structure
Streamlit-chatbot/
├── main.py
├── readme.md
├── requirements.txt
└── .streamlit/
    └── secrets.toml





