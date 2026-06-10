"""
RAG Engine for FAQ answering.

Pipeline:
  1. Load FAQ data (question + answer pairs)
  2. Embed each FAQ chunk using sentence-transformers (local, free, fast)
  3. On query: embed the question, cosine-similarity search, retrieve top-k chunks
  4. Pass retrieved context + user question to Claude for a grounded answer
"""

import json
import math
import os
import re
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Embedding model (local, no API key needed)
# ---------------------------------------------------------------------------

_embedder = None

def _get_embedder():
    """Lazy-load sentence-transformers model (downloads ~90 MB on first run)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ---------------------------------------------------------------------------
# Vector store (in-memory, pure Python — no Chroma / Pinecone needed)
# ---------------------------------------------------------------------------

class VectorStore:
    """Simple cosine-similarity vector store backed by a list of float arrays."""

    def __init__(self):
        self.chunks: list[dict] = []        # {question, answer, text}
        self.embeddings: list[list[float]] = []

    # --- Indexing -----------------------------------------------------------

    def add(self, question: str, answer: str) -> None:
        """Embed a question+answer pair and store it."""
        # We embed the question text so similarity is between user query and FAQ question.
        # The answer travels along as metadata.
        text = question  # embed only the question for retrieval
        embedding = _get_embedder().encode(text, convert_to_numpy=True).tolist()
        self.chunks.append({"question": question, "answer": answer})
        self.embeddings.append(embedding)

    def build_from_file(self, path: str | Path) -> None:
        """Load a JSON array of {question, answer} objects and index them."""
        with open(path, "r", encoding="utf-8") as f:
            faqs: list[dict] = json.load(f)
        for faq in faqs:
            self.add(faq["question"], faq["answer"])
        print(f"[VectorStore] Indexed {len(self.chunks)} FAQ entries.")

    # --- Retrieval ----------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Return the top_k most relevant FAQ chunks for a query.
        Each result is {question, answer, score}.
        """
        if not self.chunks:
            raise RuntimeError("Vector store is empty — call build_from_file first.")

        query_embedding = _get_embedder().encode(query, convert_to_numpy=True).tolist()
        scored = [
            {**chunk, "score": self._cosine_similarity(query_embedding, emb)}
            for chunk, emb in zip(self.chunks, self.embeddings)
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Answer generation via Claude
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful assistant for a fertility clinic FAQ system.
Answer the user's question using ONLY the FAQ context provided below.
If the context contains a direct or closely related answer, give it clearly and concisely.
If the context does not contain enough information to answer, say so honestly — do not invent medical details.
Keep your answer friendly and plain-language. Do not repeat the question back."""


from groq import Groq

def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    client,
    model: str = "llama-3.3-70b-versatile",
) -> str:

    context_lines = []

    for i, chunk in enumerate(retrieved_chunks, 1):
        context_lines.append(
            f"FAQ {i}:\nQ: {chunk['question']}\nA: {chunk['answer']}"
        )

    context_block = "\n\n".join(context_lines)

    user_message = f"""
FAQ Context:
{context_block}

User question:
{question}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Public façade
# ---------------------------------------------------------------------------

class FAQBot:
    """
    One-stop interface:
        bot = FAQBot("data/faqs.json")
        answer = bot.ask("Can I work during IVF?")
    """

    def __init__(self, faqs_path: str | Path, top_k: int = 3):
        self.top_k = top_k
        self.store = VectorStore()
        self.store.build_from_file(faqs_path)
        self.client = Groq(
        api_key=os.environ["GROQ_API_KEY"]
        )       

    def ask(self, question: str) -> dict:
        """
        Returns {answer, sources} where sources is the list of retrieved FAQs.
        """
        retrieved = self.store.search(question, top_k=self.top_k)
        answer = generate_answer(question, retrieved, self.client)
        return {
            "answer": answer,
            "sources": [
                {"question": r["question"], "score": round(r["score"], 3)}
                for r in retrieved
            ],
        }
