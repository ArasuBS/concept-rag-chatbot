# app.py — Groq RAG with Built-in PDFs or Uploads
import os, io
from pathlib import Path
import streamlit as st
from groq import Groq

# file readers
import PyPDF2
import docx  # python-docx

# simple retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------ Config ------------
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
CONTEXT_CHAR_BUDGET = 3500
KNOWLEDGE_DIR = Path("knowledge")  # <-- your baked-in PDFs live here

st.title("Cozza–Arasu RAG Assistant 🚀")
st.caption("Built in collaboration spirit – inspired by Lisa Cozza & Arasu")

# ------------ API ------------
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("No GROQ_API_KEY found. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()
client = Groq(api_key=api_key)

# ------------ Helpers ------------
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.replace("\r\n", "\n")
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        out.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return [c.strip() for c in out if c.strip()]

def read_pdf(fileobj):
    reader = PyPDF2.PdfReader(fileobj)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def read_docx(fileobj):
    d = docx.Document(fileobj)
    return "\n".join(p.text for p in d.paragraphs)

def read_any(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return read_pdf(uploaded_file)
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8", errors="ignore")
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return read_docx(uploaded_file)
    return ""

def select_top_chunks(question, chunks, top_k=TOP_K, budget=CONTEXT_CHAR_BUDGET):
    if not chunks:
        return []
    docs = chunks + [question]
    vec = TfidfVectorizer(stop_words="english").fit_transform(docs)
    q_vec, d_vecs = vec[-1], vec[:-1]
    sims = cosine_similarity(q_vec, d_vecs)[0]
    ranked = sorted(zip(sims, range(len(chunks))), reverse=True)
    selected, used = [], 0
    for _score, idx in ranked[: top_k * 3]:
        c = chunks[idx]
        if used + len(c) <= budget:
            selected.append(c)
            used += len(c)
        if len(selected) >= top_k or used >= budget:
            break
    return selected

# ------------ Load built-in knowledge (cached) ------------
@st.cache_resource(show_spinner=False)
def load_built_in_chunks():
    chunks = []
    if KNOWLEDGE_DIR.exists():
        for p in sorted(KNOWLEDGE_DIR.glob("*.pdf")):
            try:
                with p.open("rb") as f:
                    text = read_pdf(f)
                chunks.extend(chunk_text(text))
            except Exception as e:
                st.warning(f"Could not read {p.name}: {e}")
    return chunks

# ------------ UI: source selector ------------
with st.sidebar:
    source = st.radio("Knowledge source", ["Built-in (Cozza pack)", "Upload"], index=0)
    st.caption("Switch to *Upload* if you want to add extra files just for this session.")

all_chunks = []

if source == "Built-in (Cozza pack)":
    all_chunks = load_built_in_chunks()
    if not all_chunks:
        st.info("No PDFs found in /knowledge. Add files in your repo to preload.")
else:
    uploaded_files = st.file_uploader("Upload files (TXT, PDF, DOCX)", accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            try:
                text = read_any(f)
                if text:
                    all_chunks.extend(chunk_text(text))
                else:
                    st.warning(f"Unsupported or empty file: {f.name}")
            except Exception as e:
                st.warning(f"Failed reading {f.name}: {e}")

st.divider()
q = st.text_area("Ask a question:")

if st.button("Ask") and q.strip():
    if not all_chunks:
        st.info("No knowledge loaded. Add PDFs to /knowledge or upload files.")
    else:
        with st.spinner("Retrieving relevant passages..."):
            support = select_top_chunks(q, all_chunks)
            context = "\n\n".join(f"[CHUNK]\n{c}" for c in support)

        with st.spinner("Thinking…"):
            try:
                msg = [
                    {"role": "system",
                     "content": "Answer ONLY using the provided context. If not enough info, say you don't know."},
                    {"role": "user",
                     "content": f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer clearly with practical steps."}
                ]
                chat = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=msg,
                    temperature=0.2,
                )
                st.write(chat.choices[0].message.content)
                with st.expander("Show retrieved context"):
                    st.write(context)
            except Exception as e:
                st.error(f"Error: {e}")
