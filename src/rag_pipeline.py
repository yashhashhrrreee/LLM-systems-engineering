"""
rag_pipeline.py
===============
Advanced RAG pipeline implementing:
  1. Hierarchical (Small-to-Big) Retrieval — search on 200-char child chunks,
     return 1000-char parent chunks to the LLM.
  2. Hypothetical Document Embeddings (HyDE) — generate a fake academic answer
     before embedding, to bridge query-document vocabulary asymmetry.

Both components are built manually (no LangChain wrappers).

Usage:
    python src/rag_pipeline.py --query "What is multi-head attention?"
"""

import uuid
import argparse
import pathlib
import requests
import numpy as np
import faiss
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PDF_URL = "https://arxiv.org/pdf/1706.03762.pdf"
PDF_PATH = pathlib.Path("attention_is_all_you_need.pdf")
PARENT_CHUNK_SIZE = 1000
CHILD_CHUNK_SIZE = 200
EMBED_MODEL = "all-MiniLM-L6-v2"
HYDE_MODEL = "gpt2-medium"


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def download_pdf(url: str = PDF_URL, path: pathlib.Path = PDF_PATH) -> pathlib.Path:
    if not path.exists():
        print(f"Downloading {url} ...")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
        print(f"Saved to {path}")
    return path


def extract_text(pdf_path: pathlib.Path) -> str:
    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    print(f"Extracted {len(text):,} chars from {len(reader.pages)} pages.")
    return text


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def build_parent_store(full_text: str, chunk_size: int = PARENT_CHUNK_SIZE) -> dict[str, str]:
    """Returns {parent_id: parent_text} for non-overlapping parent chunks."""
    store = {}
    for i in range(0, len(full_text), chunk_size):
        pid = str(uuid.uuid4())
        store[pid] = full_text[i: i + chunk_size]
    print(f"Created {len(store)} parent chunks (size={chunk_size})")
    return store


def build_child_chunks(
    parent_store: dict[str, str],
    chunk_size: int = CHILD_CHUNK_SIZE,
) -> list[dict]:
    """
    Returns a list of dicts: {"text": str, "parent_id": str}
    Each child carries its parent's UUID so we can trace back after FAISS search.
    """
    children = []
    for pid, text in parent_store.items():
        for i in range(0, len(text), chunk_size):
            children.append({"text": text[i: i + chunk_size], "parent_id": pid})
    print(f"Created {len(children)} child chunks (size={chunk_size})")
    return children


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(
    child_chunks: list[dict],
    embedder: SentenceTransformer,
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Embed child chunk texts and store in a FAISS flat inner-product index.
    Returns (index, child_chunks) — the child_chunks list preserves metadata.
    """
    texts = [c["text"] for c in child_chunks]
    print(f"Embedding {len(texts)} child chunks...")
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index, child_chunks


# ---------------------------------------------------------------------------
# Small-to-Big retriever
# ---------------------------------------------------------------------------

class SmallToBigRetriever:
    """
    Custom hierarchical retriever built without LangChain.

    Search flow:
      query → bi-encoder → FAISS (child chunks) → parent_id lookup → parent text
    """

    def __init__(
        self,
        parent_store: dict[str, str],
        child_chunks: list[dict],
        faiss_index: faiss.IndexFlatIP,
        embedder: SentenceTransformer,
    ):
        self.parent_store = parent_store
        self.child_chunks = child_chunks
        self.index = faiss_index
        self.embedder = embedder

    def search(self, query: str, k: int = 3) -> list[str]:
        """
        Returns the top-k unique parent chunk texts for the given query.
        Deduplicates by parent_id so we don't return the same parent twice.
        """
        query_vec = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(query_vec, k * 3)  # over-fetch for dedup

        seen_parent_ids = set()
        results = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.child_chunks):
                continue
            child = self.child_chunks[idx]
            pid = child["parent_id"]
            if pid not in seen_parent_ids:
                seen_parent_ids.add(pid)
                results.append(self.parent_store[pid])
            if len(results) >= k:
                break

        return results


# ---------------------------------------------------------------------------
# HyDE generator
# ---------------------------------------------------------------------------

class HyDERetriever:
    """
    Wraps SmallToBigRetriever with a HyDE (Hypothetical Document Embeddings) step.

    Instead of embedding the raw user query, we ask an LLM to write a fake
    academic answer, then embed THAT. This bridges the vocabulary gap between
    informal queries and formal academic text.

    Reference: Gao et al., 2022 — https://arxiv.org/abs/2212.10496
    """

    HYDE_PROMPT_TEMPLATE = (
        "Write a short, academic paragraph answering the following question. "
        "Do not worry about exact factual accuracy — just use relevant academic "
        "vocabulary and technical terminology.\n\nQuestion: {query}\n\nAnswer:"
    )

    def __init__(self, retriever: SmallToBigRetriever, hyde_model: str = HYDE_MODEL):
        has_gpu = torch.cuda.is_available()
        print(f"Loading HyDE generator ({hyde_model})...")
        self.generator = pipeline(
            "text-generation",
            model=hyde_model,
            device=0 if has_gpu else -1,
            torch_dtype=torch.float16 if has_gpu else torch.float32,
        )
        self.retriever = retriever

    def generate_hypothesis(self, query: str, max_new_tokens: int = 150) -> str:
        prompt = self.HYDE_PROMPT_TEMPLATE.format(query=query)
        output = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.generator.tokenizer.eos_token_id,
        )
        full_text = output[0]["generated_text"]
        # Strip the prompt prefix to get only the generated hypothesis
        return full_text[len(prompt):].strip()

    def search(self, query: str, k: int = 3) -> tuple[str, list[str]]:
        """
        Returns (hypothetical_doc, retrieved_parent_chunks).
        Use hypothetical_doc for display/logging; parent_chunks go to the LLM.
        """
        hypothesis = self.generate_hypothesis(query)
        results = self.retriever.search(hypothesis, k=k)
        return hypothesis, results


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def build_pipeline(pdf_path: pathlib.Path = PDF_PATH) -> tuple[SmallToBigRetriever, HyDERetriever]:
    text = extract_text(pdf_path)
    parent_store = build_parent_store(text)
    child_chunks = build_child_chunks(parent_store)

    embedder = SentenceTransformer(EMBED_MODEL)
    index, child_chunks = build_faiss_index(child_chunks, embedder)

    retriever = SmallToBigRetriever(parent_store, child_chunks, index, embedder)
    hyde = HyDERetriever(retriever)
    return retriever, hyde


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Advanced RAG Pipeline")
    parser.add_argument(
        "--query",
        default="What is the purpose of multi-head attention?",
        help="Query to run against the pipeline",
    )
    parser.add_argument("--k", type=int, default=3, help="Number of parent chunks to retrieve")
    parser.add_argument("--mode", choices=["standard", "hyde"], default="hyde")
    args = parser.parse_args()

    download_pdf()
    retriever, hyde = build_pipeline()

    print(f"\n{'='*60}")
    print(f"Query: {args.query}")
    print(f"Mode:  {args.mode.upper()}")
    print(f"{'='*60}\n")

    if args.mode == "hyde":
        hypothesis, results = hyde.search(args.query, k=args.k)
        print(f"[HyDE] Hypothetical document:\n{hypothesis}\n")
    else:
        results = retriever.search(args.query, k=args.k)

    for i, chunk in enumerate(results, 1):
        print(f"--- Retrieved Parent Chunk {i} ---")
        print(chunk[:500])
        print()


if __name__ == "__main__":
    main()
