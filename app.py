import streamlit as st
from openai import OpenAI

st.title("Concept RAG Chatbot")

# API key input
api_key = st.text_input("Enter your OpenAI API key:", type="password")

if api_key:
    client = OpenAI(api_key=api_key)

    # Question box
    user_question = st.text_area("Ask me anything:")
    if st.button("Ask"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful concept-based chatbot."},
                    {"role": "user", "content": user_question},
                ],
            )
            st.success(response.choices[0].message.content)

