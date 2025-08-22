# app.py — Groq RAG with chunking + retrieval (token-safe)
import io
import streamlit as st
from groq import Groq

# For reading files
import PyPDF2
import docx  # python-docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Cozza–Arasu RAG Assistant")
st.caption("Built in collaboration spirit – inspired by Lisa Cozza & Arasu")

# --- Config (tweak if needed) ---
CHUNK_SIZE = 900        # characters per chunk
CHUNK_OVERLAP = 150     # characters overlap between chunks
TOP_K = 5               # max number of chunks to pass
CONTEXT_CHAR_BUDGET = 3500  # total chars sent to the model as context

# --- Groq key from Secrets ---
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("No GROQ_API_KEY found. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()
client = Groq(api_key=api_key)

# ---------- Helpers ----------
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return [c.strip() for c in chunks if c.strip()]

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    out = []
    for page in reader.pages:
        out.append(page.extract_text() or "")
    return "\n".join(out)

def read_docx(file):
    d = docx.Document(file)
    return "\n".join(p.text for p in d.paragraphs)

def read_any(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return read_pdf(uploaded_file)
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8", errors="ignore")
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return read_docx(uploaded_file)
    return ""  # unsupported

def select_top_chunks(question, chunks, top_k=TOP_K, budget=CONTEXT_CHAR_BUDGET):
    if not chunks:
        return []
    # TF-IDF retrieval
    docs = chunks + [question]
    vec = TfidfVectorizer(stop_words="english").fit_transform(docs)
    q_vec = vec[-1]
    d_vecs = vec[:-1]
    sims = cosine_similarity(q_vec, d_vecs)[0]
    ranked = sorted(zip(sims, range(len(chunks))), reverse=True)
    selected, used = [], 0
    for score, idx in ranked[:top_k*3]:  # check a bit more than top_k to fit budget
        c = chunks[idx]
        if used + len(c) <= budget:
            selected.append((score, c))
            used += len(c)
        if len(selected) >= top_k or used >= budget:
            break
    return [c for _, c in selected]

# ---------- UI ----------
uploaded_files = st.file_uploader("Upload files (TXT, PDF, DOCX)", accept_multiple_files=True)
all_chunks = []

if uploaded_files:
    for f in uploaded_files:
        try:
            text = read_any(f)
            if not text:
                st.warning(f"Could not read {f.name} (unsupported or empty).")
                continue
            all_chunks.extend(chunk_text(text))
        except Exception as e:
            st.warning(f"Failed to read {f.name}: {e}")

question = st.text_area("Ask a question from the uploaded files:")

if st.button("Ask") and question.strip():
    if not all_chunks:
        st.info("Please upload at least one TXT/PDF/DOCX with text.")
    else:
        with st.spinner("Retrieving relevant passages..."):
            support = select_top_chunks(question, all_chunks, top_k=TOP_K, budget=CONTEXT_CHAR_BUDGET)
            context = "\n\n".join(f"[CHUNK]\n{c}" for c in support)

        with st.spinner("Thinking with Groq..."):
            try:
                msg = [
                    {"role": "system",
                     "content": "Answer ONLY using the provided context. If the answer is not in context, say you don't know."},
                    {"role": "user",
                     "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer clearly in bullets where helpful."}
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

st.caption("We limit context size and send only the most relevant chunks to avoid token-limit errors.")
