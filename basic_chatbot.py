import streamlit as st
from openai import OpenAI
from groq import Groq


st.title("Groq Chatbot with Streamlit")

# # Set OpenAI API key from Streamlit secrets
# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Set a default model
# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"

# Set default Groq model in session state
if "groq_model" not in st.session_state:
    st.session_state["groq_model"] = "llama3-8b-8192"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # # Display assistant response in chat message container
    # with st.chat_message("assistant"):
    #     stream = client.chat.completions.create(
    #         model=st.session_state["groq_model"],
    #         messages=[
    #             {"role": m["role"], "content": m["content"]}
    #             for m in st.session_state.messages
    #         ],
    #         stream=True,
    #     )
    #     response = st.write_stream(stream)
    # st.session_state.messages.append({"role": "assistant", "content": response})

    # Prepare assistant response container
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        # Stream response from Groq
        for chunk in client.chat.completions.create(
            model=st.session_state["groq_model"],
            messages=st.session_state.messages,
            stream=True,
        ):
            delta = chunk.choices[0].delta.content or ""
            full_response += delta
            response_container.markdown(full_response + "▌")  # Live typing effect

        response_container.markdown(full_response)  # Final display

    # Save assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
