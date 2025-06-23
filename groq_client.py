from litellm import completion
import streamlit as st


def get_groq_llm(model_name="groq/llama3-8b-8192"):
    def chat_llm(prompt):
        messages = [{"role": "user", "content": prompt}]
        response = completion(
            model=model_name,
            messages=messages,
            api_key=st.secrets["GROQ_API_KEY"],
            llm_provider="groq",  # ✅ Explicitly set provider
        )
        return response["choices"][0]["message"]["content"]

    # ✅ Add metadata so CrewAI can interpret it
    chat_llm.model = model_name
    chat_llm.llm_provider = "groq"

    return chat_llm
