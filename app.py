# app.py — Groq RAG (with file upload)
import streamlit as st
from groq import Groq

st.title("Concept RAG Chatbot (Groq + File Upload)")

# 1) Get Groq API key from Streamlit Secrets
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("No GROQ_API_KEY found. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

# 2) Initialize Groq client
client = Groq(api_key=api_key)

# 3) Let user upload text or PDF files
uploaded_files = st.file_uploader("Upload files (TXT, PDF, DOCX)", accept_multiple_files=True)

# Store uploaded text
file_texts = []
if uploaded_files:
    for file in uploaded_files:
        if file.type == "text/plain":
            file_texts.append(file.read().decode("utf-8"))
        elif file.type == "application/pdf":
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                file_texts.append(text)
            except Exception as e:
                st.warning(f"Could not read {file.name}: {e}")
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                import docx
                doc = docx.Document(file)
                text = "\n".join([p.text for p in doc.paragraphs])
                file_texts.append(text)
            except Exception as e:
                st.warning(f"Could not read {file.name}: {e}")

context = "\n\n".join(file_texts)

# 4) Ask questions
question = st.text_area("Ask a question:")

if st.button("Ask") and question.strip():
    with st.spinner("Searching your files..."):
        try:
            # Include uploaded text as context
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers based only on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
            chat = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.2,
            )
            st.write(chat.choices[0].message.content)
        except Exception as e:
            st.error(f"Error: {e}")

st.caption("Now the bot answers from your uploaded files. 🚀")
