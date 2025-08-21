import streamlit as st
from openai import OpenAI

st.title("Concept RAG Chatbot")

# ✅ get the key from Streamlit Secrets (no text box needed)
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("No OPENAI_API_KEY found. In Streamlit Cloud: App → ⋮ → Settings → Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

question = st.text_area("Ask me anything:")
if st.button("Ask") and question.strip():
    with st.spinner("Thinking..."):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful concept-based chatbot."},
                    {"role": "user", "content": question},
                ],
            )
            st.write(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"Error: {e}")
