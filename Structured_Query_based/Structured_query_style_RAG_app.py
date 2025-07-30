import streamlit as st
import pandas as pd
from groq import Groq
import os

# 1. Set your Groq API key
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

# 2. App title
st.title("📊 CSV Query Assistant (Structured Approach)")
st.markdown(
    "Upload a CSV and ask questions in natural language. LLM will convert it to pandas code."
)

# 3. File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head(5), use_container_width=True)

    # 4. User query input
    user_query = st.text_input("🧠 Ask a question about your CSV:")

    if st.button("🔍 Run Query"):
        if not user_query.strip():
            st.warning("Please enter a query.")
        else:
            # 5. Generate pandas code using Groq
            def generate_code(query, df_sample):
                prompt = f"""
                        You are an AI assistant that writes pandas code to analyze tabular data.
                        Use the DataFrame called `df`.

                        DataFrame sample:
                        {df_sample.head(3).to_string(index=False)}

                        Columns: {', '.join(df_sample.columns)}

                        Question: "{query}"

                        Write only the Python code that uses `df` to answer the question. Return just the pandas expression like df.shape[0] or df['column'].mean(). Do not explain anything.
                                        """

                try:
                    response = client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                    )
                    return response.choices[0].message.content.strip().strip("`")
                except Exception as e:
                    return f"LLM Error: {e}"

            # 6. Execute code
            def execute_code(code, df):
                local_vars = {"df": df.copy()}
                try:
                    exec(f"result = {code}", {}, local_vars)
                    result = local_vars["result"]
                    if isinstance(result, pd.DataFrame):
                        return result
                    elif hasattr(result, "to_string"):
                        return result.to_string(index=False)
                    else:
                        return str(result)
                except Exception as e:
                    return f"Execution Error: {e}"

            # Run generation + execution
            with st.spinner("Thinking..."):
                generated_code = generate_code(user_query, df)
                st.code(generated_code, language="python")

                output = execute_code(generated_code, df)
                if isinstance(output, pd.DataFrame):
                    st.write("✅ Result:")
                    st.dataframe(output, use_container_width=True)
                else:
                    st.success("✅ Answer:")
                    st.write(output)
