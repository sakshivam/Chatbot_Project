import streamlit as st
import pandas as pd
import traceback
import io
import contextlib
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI

# Load API key
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


# --- Load CSV (you can replace this with file uploader too) ---
@st.cache_data
def load_csv():
    return pd.read_csv("./data/Employee.csv")


df = load_csv()

# --- Initialize LLM (use your actual key here or from .env) ---
llm = ChatOpenAI(
    model="llama3-70b-8192",  # or any model Groq supports
    openai_api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",  # Groq-compatible OpenAI endpoint
    temperature=0,
)

# --- Define system prompt ---
system_msg = SystemMessage(
    content=(
        "You are a data analyst. A Pandas DataFrame named `df` is already loaded. "
        "Use column names as Education,JoiningYear,City,PaymentTier,Age,Gender,EverBenched,ExperienceInCurrentDomain,LeaveOrNot"
        "Generate Python code to answer the user's question using this dataframe. "
        "Always assign the final answer to a variable named `result`. "
        "Do not include print statements or explanations."
    )
)

# --- Store message history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [system_msg]

# --- Chat UI Header ---
st.title("🧠 Structured Query based CSV Data Analysis Assistant")
st.caption("Ask anything from the dataset (e.g. 'How many rows?)")

# --- Display chat messages ---
for msg in st.session_state.chat_history[1:]:  # skip system_msg
    with st.chat_message("assistant" if isinstance(msg, AIMessage) else "user"):
        st.markdown(msg.content)

# --- Chat input ---
user_query = st.chat_input("Ask a data question from your CSV...")

if user_query:
    # Display user message
    st.chat_message("user").markdown(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    # Query LLM for Python code
    messages = st.session_state.chat_history
    try:
        code = llm.invoke(messages).content
        st.session_state.chat_history.append(AIMessage(content=code))

        # Display AI message (code)
        with st.chat_message("assistant"):
            st.markdown("```python\n" + code + "\n```")

            # Run the code safely
            local_vars = {"df": df.copy()}
            output_buffer = io.StringIO()

            try:
                with contextlib.redirect_stdout(output_buffer):
                    exec(code, {}, local_vars)
                result = local_vars.get("result", None)

                if result is None:
                    result = output_buffer.getvalue().strip()

                # Display result
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                else:
                    st.markdown(f"**Result:** {result}")
            except Exception as e:
                st.error("Code execution failed:")
                st.code(traceback.format_exc())

    except Exception as e:
        st.error("LLM invocation failed:")
        st.code(traceback.format_exc())
