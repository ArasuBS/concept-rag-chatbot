# app.py — Cozza–Arasu RAG Assistant (Groq) ✅
# - Preloads PDFs from /knowledge (so Lisa can ask with no uploads)
# - Optional "Upload" mode for ad-hoc files
# - Chunking + TF-IDF retrieval with a strict context budget (token-safe)
# - Safety rails: question length limit, graceful error handling

from pathlib import Path
import streamlit as st
from groq import Groq

# File reading
import PyPDF2
import docx  # python-docx

# Retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ===================== Config =====================
# Chunking / retrieval
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
CONTEXT_CHAR_BUDGET = 3500

# Built-in knowledge folder (PDFs live here in your repo)
KNOWLEDGE_DIR = Path("knowledge")

# Safety rails
MAX_QUESTION_LENGTH = 1000  # characters
GUIDANCE_TEXT = "Tip: keep questions under 3–4 sentences for best results."

# Title
st.title("Cozza–Arasu RAG Assistant 🚀")
st.caption("Built in collaboration spirit – inspired by Lisa Cozza & Arasu")

# Groq key
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("No GROQ_API_KEY found. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()
client = Groq(api_key=api_key)


# ===================== Helpers =====================
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Simple character-based splitter with overlap."""
    text = text.replace("\r\n", "\n")
    chunks, start = [], 0
    N = len(text)
    while start < N:
        end = min(start + size, N)
        chunks.append(text[start:end])
        # move start forward keeping overlap (avoid infinite loop)
        next_start = end - overlap
        start = next_start if next_start > start else end
    return [c.strip() for c in chunks if c.strip()]


def read_pdf_filelike(filelike) -> str:
    """Extract text from a PDF given a file-like object (opened 'rb' or UploadedFile)."""
    reader = PyPDF2.PdfReader(filelike)
    out = []
    for p in reader.pages:
        out.append(p.extract_text() or "")
    return "\n".join(out)


def read_docx_filelike(filelike) -> str:
    d = docx.Document(filelike)
    return "\n".join(p.text for p in d.paragraphs)


def read_uploaded_any(uploaded_file) -> str:
    if uploaded_file.type == "application/pdf":
        return read_pdf_filelike(uploaded_file)
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8", errors="ignore")
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return read_docx_filelike(uploaded_file)
    return ""


def select_top_chunks(question: str, chunks: list[str],
                      top_k: int = TOP_K, budget: int = CONTEXT_CHAR_BUDGET) -> list[str]:
    """TF-IDF retrieve the most relevant chunks under a total char budget."""
    if not chunks:
        return []
    docs = chunks + [question]
    vec = TfidfVectorizer(stop_words="english").fit_transform(docs)
    q_vec, d_vecs = vec[-1], vec[:-1]
    sims = cosine_similarity(q_vec, d_vecs)[0]
    ranked = sorted(zip(sims, range(len(chunks))), reverse=True)

    selected, used = [], 0
    # Check more than top_k to fit the budget nicely
    for _score, idx in ranked[: top_k * 3]:
        c = chunks[idx]
        if used + len(c) <= budget:
            selected.append(c)
            used += len(c)
        if len(selected) >= top_k or used >= budget:
            break
    return selected


@st.cache_resource(show_spinner=False)
def load_built_in_chunks() -> list[str]:
    """Read & chunk all PDFs inside /knowledge once (cached)."""
    out = []
    if KNOWLEDGE_DIR.exists():
        for p in sorted(KNOWLEDGE_DIR.glob("*.pdf")):
            try:
                with p.open("rb") as f:
                    text = read_pdf_filelike(f)
                out.extend(chunk_text(text))
            except Exception as e:
                st.warning(f"Could not read {p.name}: {e}")
    return out


# ===================== UI: source selector =====================
with st.sidebar:
    source = st.radio("Knowledge source", ["Built-in (Cozza pack)", "Upload"], index=0)
    st.caption("Switch to *Upload* if you want to add extra files just for this session.")

all_chunks: list[str] = []

if source == "Built-in (Cozza pack)":
    all_chunks = load_built_in_chunks()
    if not all_chunks:
        st.info("No PDFs found in /knowledge. Add files in your repo to preload.")
else:
    uploaded = st.file_uploader("Upload files (TXT, PDF, DOCX)", accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            try:
                txt = read_uploaded_any(f)
                if txt:
                    all_chunks.extend(chunk_text(txt))
                else:
                    st.warning(f"Unsupported or empty file: {f.name}")
            except Exception as e:
                st.warning(f"Failed reading {f.name}: {e}")

st.divider()


# ===================== Q/A UI with safety rails =====================
q = st.text_area("Ask a question:", help=GUIDANCE_TEXT)
chars = len(q or "")
st.caption(f"Characters: {chars} / {MAX_QUESTION_LENGTH}")

ask_disabled = (chars == 0) or (chars > MAX_QUESTION_LENGTH)
if chars > MAX_QUESTION_LENGTH:
    st.error(f"Your question is too long. Please shorten it (max {MAX_QUESTION_LENGTH} characters).")

if st.button("Ask", disabled=ask_disabled) and q.strip():
    if not all_chunks:
        st.info("No knowledge loaded. Add PDFs to /knowledge or upload files.")
    else:
        try:
            with st.spinner("Retrieving relevant passages..."):
                support = select_top_chunks(q, all_chunks)
                context = "\n\n".join(f"[CHUNK]\n{c}" for c in support)

            with st.spinner("Thinking…"):
                messages = [
                    {"role": "system",
                     "content": "Answer ONLY using the provided context. If the answer is not present, say you don't know."},
                    {"role": "user",
                     "content": f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer clearly with practical steps or bullets."}
                ]
                resp = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    temperature=0.2,
                )
                st.write(resp.choices[0].message.content)

            with st.expander("Show retrieved context"):
                st.write(context)

        except Exception as e:
            st.error(
                "Oops — the request was too large or something unexpected happened. "
                "Please shorten the question or try again. If the issue persists, try fewer/smaller PDFs."
            )
            # Show technical detail privately for you
            st.caption(f"(Technical detail: {e})")
