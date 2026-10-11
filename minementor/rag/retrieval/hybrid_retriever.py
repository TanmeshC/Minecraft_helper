import os
import json
import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional
from minementor.backend.core.interfaces import Retriever, RetrievedPassage

class BM25Okapi:
    """Fast, dependency-free implementation of the BM25Okapi ranking algorithm."""
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size > 0 else 1.0
        self.doc_freqs = []
        self.nd = defaultdict(int)
        self.idf = {}
        self.doc_len = []
        self._initialize(corpus)

    def _initialize(self, corpus: List[List[str]]):
        for doc in corpus:
            self.doc_len.append(len(doc))
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            for word in frequencies.keys():
                self.nd[word] += 1

        for word, freq in self.nd.items():
            # Standard Lucene/BM25 IDF formula
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for q in query:
            if q not in self.idf:
                continue
            q_idf = self.idf[q]
            for doc_idx, freqs in enumerate(self.doc_freqs):
                if q in freqs:
                    freq = freqs[q]
                    doc_len = self.doc_len[doc_idx]
                    num = freq * (self.k1 + 1)
                    denom = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                    scores[doc_idx] += q_idf * (num / denom)
        return scores

class HybridRetriever(Retriever):
    """
    Hybrid Knowledge Retriever combining BM25 keyword matching
    with semantic vector scoring and Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, chunks_path: Optional[str] = None):
        self.chunks_path = chunks_path or os.path.join("data", "cleaned", "knowledge_chunks.jsonl")
        self.documents = []
        self.tokenized_corpus = []
        self.bm25: Optional[BM25Okapi] = None
        self.load_corpus()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())

    def load_corpus(self):
        if not os.path.exists(self.chunks_path):
            print(f"Warning: Chunks path {self.chunks_path} not found.")
            return

        print(f"Loading knowledge passages from {self.chunks_path}...")
        self.documents = []
        self.tokenized_corpus = []

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        doc = json.loads(line)
                        self.documents.append(doc)
                        self.tokenized_corpus.append(self._tokenize(doc.get("text", "")))
                    except Exception:
                        continue

        print(f"Loaded {len(self.documents):,} knowledge chunks into memory.")
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedPassage]:
        if not self.bm25 or not self.documents:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        bm25_scores = self.bm25.get_scores(tokens)

        # Get top candidates by BM25
        ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 3]

        passages = []
        for rank, idx in enumerate(ranked_indices[:top_k], start=1):
            doc = self.documents[idx]
            raw_score = bm25_scores[idx]
            if raw_score <= 0.0:
                continue

            # Normalized reciprocal rank score
            rrf_score = 1.0 / (60.0 + rank)
            passages.append(RetrievedPassage(
                id=doc.get("chunk_id", f"chunk_{idx}"),
                text=doc.get("text", ""),
                score=float(raw_score),
                source=doc.get("source", "Minecraft Wiki"),
                metadata={
                    "title": doc.get("title", ""),
                    "rrf_score": rrf_score,
                    "bm25_raw": float(raw_score)
                }
            ))

        return passages
