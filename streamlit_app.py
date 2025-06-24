import streamlit as st
import time
from groq_client import get_groq_llm
from crew.crew_config import get_crew

st.title("🤖 Data Science Assistant")

# Init model
if "groq_model" not in st.session_state:
    st.session_state["groq_model"] = "groq/llama3-8b-8192"

# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        # Get LLM from Groq
        llm_callable = get_groq_llm(st.session_state["groq_model"])

        # Create Crew and run
        crew = get_crew(prompt, llm_callable=llm_callable)
        # Kickoff the crew to get the response
        # This will run the task and return the response
        response = crew.kickoff()
        full_response = str(response)  # ✅ Extract the final string

        try:
            full_response = str(response.final_output)
        except AttributeError:
            # fallback: manually parse from text
            full_text = str(response)
            if "Final Answer:" in full_text:
                full_response = full_text.split("Final Answer:")[-1].strip()
            else:
                full_response = full_text.strip()

        # Simulate typing effect
        typed_text = ""
        for char in full_response:
            typed_text += char
            response_container.markdown(typed_text + "▌")
            time.sleep(0.01)  # ⏱️ Adjust speed here (0.02 = fast, 0.05 = slower)
        response_container.markdown(full_response)

    # Add to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
