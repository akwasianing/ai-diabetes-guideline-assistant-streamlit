from __future__ import annotations

import hashlib
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
INDEX_DIR = APP_DIR / ".index"
INDEX_FILE = INDEX_DIR / "semantic_index.pkl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_LLM = "openai/gpt-oss-20b"


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: int


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"Downloaded from .*?(?=\n|$)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    source: str,
    page: int,
    chunk_size: int = 1100,
    overlap: int = 180,
) -> List[Chunk]:
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[Chunk] = []
    current = ""
    idx = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
            continue

        if current:
            chunks.append(Chunk(current, source, page, idx))
            idx += 1

        tail = current[-overlap:] if current else ""
        current = f"{tail} {sentence}".strip()

        while len(current) > chunk_size:
            piece = current[:chunk_size]
            chunks.append(Chunk(piece, source, page, idx))
            idx += 1
            current = current[chunk_size - overlap :]

    if current:
        chunks.append(Chunk(current, source, page, idx))

    return chunks


def corpus_signature(pdf_paths: List[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(pdf_paths):
        stat = path.stat()
        digest.update(path.name.encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def load_reranker():
    return CrossEncoder(RERANKER_MODEL)


@st.cache_resource(show_spinner=False)
def build_or_load_index(signature: str):
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    INDEX_DIR.mkdir(exist_ok=True)

    if INDEX_FILE.exists():
        try:
            with open(INDEX_FILE, "rb") as f:
                payload = pickle.load(f)
            if payload.get("signature") == signature:
                saved_chunks = payload["chunks"]
                chunks = [
                    Chunk(**item) if isinstance(item, dict) else item
                    for item in saved_chunks
                ]
                embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
                return chunks, embeddings
        except Exception:
            pass

    chunks: List[Chunk] = []
    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            chunks.extend(chunk_text(text, pdf_path.name, page_number))

    if not chunks:
        raise RuntimeError("No readable text was extracted from the PDFs.")

    model = load_embedding_model()
    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump(
            {
                "signature": signature,
                "chunks": [
                    {
                        "text": chunk.text,
                        "source": chunk.source,
                        "page": chunk.page,
                        "chunk_id": chunk.chunk_id,
                    }
                    for chunk in chunks
                ],
                "embeddings": embeddings,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return chunks, embeddings


def retrieve(
    question: str,
    chunks: List[Chunk],
    embeddings: np.ndarray,
    initial_k: int = 18,
    final_k: int = 6,
    use_reranker: bool = True,
) -> List[Tuple[Chunk, float]]:
    model = load_embedding_model()
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    scores = cosine_similarity(query_embedding, embeddings).ravel()
    candidate_indices = np.argsort(scores)[::-1][:initial_k]

    candidates = [(chunks[i], float(scores[i])) for i in candidate_indices]

    if use_reranker and candidates:
        reranker = load_reranker()
        pairs = [[question, chunk.text] for chunk, _ in candidates]
        rerank_scores = reranker.predict(pairs)
        reranked = sorted(
            [(chunk, float(score)) for (chunk, _), score in zip(candidates, rerank_scores)],
            key=lambda item: item[1],
            reverse=True,
        )
        return reranked[:final_k]

    return candidates[:final_k]


def get_api_key() -> str | None:
    try:
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"])
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")


def build_context(results: List[Tuple[Chunk, float]]) -> str:
    sections = []
    for idx, (chunk, _) in enumerate(results, start=1):
        citation = f"[S{idx}: {chunk.source}, page {chunk.page}]"
        sections.append(f"{citation}\n{chunk.text}")
    return "\n\n".join(sections)


def generate_ai_answer(
    question: str,
    results: List[Tuple[Chunk, float]],
    history: List[dict],
    model_name: str,
) -> str:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "No Groq API key was found. Add GROQ_API_KEY to .streamlit/secrets.toml "
            "or the Streamlit Community Cloud Secrets page."
        )

    context = build_context(results)
    recent_history = history[-6:]
    history_text = "\n".join(
        f"{item['role'].upper()}: {item['content']}" for item in recent_history
    )

    system_prompt = """
You are an evidence-grounded educational assistant for the ADA Standards of Care
in Diabetes—2026 corpus supplied by the application.

Rules:
1. Answer only from the retrieved source passages.
2. Do not use outside medical knowledge.
3. Do not diagnose, prescribe, or provide patient-specific medical advice.
4. If the passages do not adequately support an answer, say that the available
   guideline sections do not provide enough evidence.
5. Cite every substantive clinical claim using the exact source labels supplied,
   such as [S1] or [S2].
6. Do not invent citations, page numbers, recommendations, thresholds, or evidence grades.
7. Distinguish general guideline information from individualized clinical decisions.
8. Use clear professional language and concise paragraphs.
9. End with: "Educational use only; this does not replace clinical judgment."
""".strip()

    user_prompt = f"""
Conversation context:
{history_text or "No previous conversation."}

Question:
{question}

Retrieved guideline passages:
{context}

Write a direct, evidence-grounded answer. Use only the source labels above.
""".strip()

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=900,
        seed=42,
    )
    return completion.choices[0].message.content.strip()


def fallback_answer(results: List[Tuple[Chunk, float]]) -> str:
    if not results:
        return (
            "I could not find sufficiently relevant evidence in the current PDF corpus. "
            "Try a more specific question."
        )

    statements = []
    for idx, (chunk, _) in enumerate(results[:3], start=1):
        text = re.split(r"(?<=[.!?])\s+", chunk.text)
        excerpt = " ".join(text[:2]).strip()
        statements.append(f"{excerpt} [S{idx}]")
    return "\n\n".join(statements)


st.set_page_config(
    page_title="AI Diabetes Guideline Assistant",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 AI Diabetes Guideline Assistant")
st.caption(
    "Semantic retrieval + reranking + a Groq-hosted open model, grounded in the "
    "included ADA Standards of Care in Diabetes—2026 sections."
)

with st.sidebar:
    st.header("AI and retrieval settings")
    model_name = st.selectbox(
        "Groq model",
        [DEFAULT_LLM],
        help="The model generates answers only from retrieved guideline passages.",
    )
    final_k = st.slider("Passages sent to AI", 3, 8, 6)
    use_reranker = st.checkbox("Use Cross-Encoder reranking", value=True)
    show_sources = st.checkbox("Show retrieved passages", value=True)

    st.divider()
    api_key_present = bool(get_api_key())
    if api_key_present:
        st.success("Groq API key detected")
    else:
        st.error("Groq API key not configured")

    st.divider()
    st.warning(
        "Educational prototype only. Do not enter identifiable patient information. "
        "This app does not diagnose, prescribe, or replace clinical judgment."
    )

pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
if not pdf_paths:
    st.error("No PDF files were found in the data folder.")
    st.stop()

try:
    signature = corpus_signature(pdf_paths)
    with st.spinner("Loading semantic models and guideline index..."):
        chunks, embeddings = build_or_load_index(signature)
except Exception as exc:
    st.error(f"Could not initialize the semantic index: {exc}")
    st.stop()

st.success(
    f"Ready: {len(pdf_paths)} guideline files indexed into {len(chunks):,} semantic passages."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask about diagnosis, glycemic goals, lifestyle, medications, or cardiovascular risk..."
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating a grounded answer..."):
            try:
                results = retrieve(
                    question,
                    chunks,
                    embeddings,
                    initial_k=max(final_k * 3, 12),
                    final_k=final_k,
                    use_reranker=use_reranker,
                )

                if results and get_api_key():
                    answer = generate_ai_answer(
                        question,
                        results,
                        st.session_state.messages[:-1],
                        model_name,
                    )
                else:
                    answer = fallback_answer(results)
                    if not get_api_key():
                        answer = (
                            "AI generation is unavailable because the Groq API key has not "
                            "been configured. Showing a retrieval-only fallback:\n\n" + answer
                        )

                st.markdown(answer)

                if show_sources and results:
                    st.subheader("Retrieved evidence")
                    for idx, (chunk, score) in enumerate(results, start=1):
                        with st.expander(
                            f"S{idx} · {chunk.source}, page {chunk.page} · score {score:.3f}"
                        ):
                            st.write(chunk.text)

            except Exception as exc:
                answer = f"The application could not generate an answer: {exc}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.expander("How the AI pipeline works"):
    st.markdown(
        """
1. Extract text from the guideline PDFs.
2. Divide each page into overlapping passages.
3. Create semantic embeddings with `all-MiniLM-L6-v2`.
4. Retrieve the most relevant passages using cosine similarity.
5. Rerank them using `ms-marco-MiniLM-L-6-v2`.
6. Send only those passages to the Groq-hosted `openai/gpt-oss-20b` model.
7. Require the model to answer from the supplied evidence and cite source labels.

The first launch is slower because the embedding and reranking models must download.
        """
    )
