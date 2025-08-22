# app.py — Streamlit + Groq (uses key from Streamlit Secrets)
import streamlit as st
from groq import Groq

st.title("Concept RAG Chatbot (Groq)")

# 1) Get Groq API key from Streamlit Secrets
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("No GROQ_API_KEY found. In Streamlit Cloud: App → ⋮ → Settings → Secrets.")
    st.stop()

# 2) Initialize Groq client
client = Groq(api_key=api_key)

# 3) Simple chat UI
question = st.text_area("Ask me anything:")

if st.button("Ask") and question.strip():
    with st.spinner("Thinking..."):
        try:
            chat = client.chat.completions.create(
                model="llama-3.1-8b-instant",  # fast/free Groq model
                messages=[
                    {"role": "system", "content": "You are a helpful concept-based chatbot."},
                    {"role": "user", "content": question},
                ],
                temperature=0.2,
            )
            st.write(chat.choices[0].message.content)
        except Exception as e:
            st.error(f"Error: {e}")

st.caption("Now running on Groq. Next step later: add file upload + retrieval to answer from your documents (RAG).")
