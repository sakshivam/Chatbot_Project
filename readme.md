# 🤖 CrewAI Chatbot with Groq + Streamlit

A conversational chatbot powered by [Groq's LLaMA3](https://console.groq.com), orchestrated via [CrewAI](https://docs.crewai.com), and presented through a [Streamlit](https://streamlit.io/) UI.

This project demonstrates:

- 🧠 CrewAI agents powered by Groq LLMs (LLaMA3, Mixtral, etc.)
- 🔁 Real-time conversational interaction
- 🔒 Secure API usage via `.env` or `secrets.toml`
- 🗂️ Clean modular project structure using `agent`, `crew`, `client`, and UI layers

---

## ⚙️ Features

- 🤖 CrewAI agent using Groq-hosted LLMs via LiteLLM
- 💬 Chat memory with Streamlit session state
- 🔄 Interactive Streamlit UI
- 🔐 Environment-configurable API access
- 🧩 Easily extendable to support tools, multi-agent reasoning, or file-based input

---

## 📦 Requirements

- Python 3.10+
- [Groq API key](https://console.groq.com)
- `streamlit`, `crewai`, `litellm`, `python-dotenv`

Install all dependencies:

```bash
pip install -r requirements.txt
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
streamlit run streamlit_app.py
```
The app will launch in your default browser.

## 📁 Project Structure
streamlit-chatbot/
├── streamlit_app.py           # 🚀 Streamlit frontend app
├── groq_client.py             # 🔌 Groq LLM wrapper (LiteLLM + provider)
├── crew/
│   ├── agent.py               # 🧠 CrewAI agent definition
│   └── crew_config.py         # ⚙️ Crew setup with task + agent
├── requirements.txt
├── .env                       # 🔐 API key (optional if using secrets.toml)
├── .streamlit/
│   └── secrets.toml           # 🔐 Streamlit secrets (optional)
└── readme.md






