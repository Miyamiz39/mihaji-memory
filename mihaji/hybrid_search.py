"""
Hybrid Search — BM25 + Vector Fusion for Mihaji Memory.

Adds keyword-aware retrieval to supplement ChromaDB's semantic search.
Uses Reciprocal Rank Fusion (RRF) to combine both rankings.

Features:
  - BM25 with jieba Chinese tokenization
  - RRF fusion (no tuning needed, robust to score scale differences)
  - Lazy BM25 index rebuild on stale data
  - Graceful degradation: BM25-only / Vector-only fallback
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import jieba
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# RRF constant — small offset to avoid division by zero and limit
# high-rank dominance. 60 is the standard value from the original paper.
RRF_CONSTANT = 60.0

# BM25-only mode threshold — if vector returns 0 results, fall through
# to pure BM25. No fusion needed.
BM25_FALLBACK_MIN_VECTOR_RESULTS = 1


# ---------------------------------------------------------------------------
# Tokenize: Chinese + English mixed text
# ---------------------------------------------------------------------------

_PUNCTUATION_PATTERN = re.compile(r"^[\s\W_]+$")


def tokenize(text: str) -> List[str]:
    """Tokenize mixed Chinese/English text with jieba.

    - English words are kept as-is (jieba handles them).
    - Chinese text is segmented.
    - Pure punctuation tokens are filtered to avoid BM25 IDF pollution.
    - All tokens lowercased for consistent matching.
    """
    if not text:
        return []
    text = text.lower()
    # jieba's built-in tokenizer handles mixed text well
    tokens = jieba.cut(text)
    clean_tokens: List[str] = []
    for t in tokens:
        t_stripped = t.strip()
        if t_stripped and not _PUNCTUATION_PATTERN.match(t_stripped):
            clean_tokens.append(t_stripped)
    return clean_tokens


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------

class BM25Index:
    """A lightweight, lazily-built BM25 index over memory documents.

    Thread-safety note: intended for single-threaded use within the plugin's
    prefetch/sync_turn flow. The ChromaDB backend serializes access on its
    own, so we follow the same pattern.
    """

    def __init__(self):
        self._corpus: List[str] = []
        self._doc_ids: List[str] = []          # ChromaDB chunk IDs for dedup
        self._doc_metas: List[Dict[str, Any]] = []  # group_id, chunk_idx, created_at
        self._tokenized_corpus: List[List[str]] = []
        self._index: Optional[BM25Okapi] = None
        self._stale = True  # Rebuild needed

    # -- Index management ------------------------------------------------

    def rebuild(self, documents: List[str], doc_ids: Optional[List[str]] = None,
                doc_metas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Build/replace the BM25 index from a list of raw text documents.

        Args:
            documents: Raw text of each document (ordered same as ChromaDB).
            doc_ids: ChromaDB chunk IDs for deduplication (optional).
            doc_metas: Metadata dicts with group_id, chunk_idx, created_at (optional).
        """
        tokenized = [tokenize(doc) for doc in documents]
        self._corpus = list(documents)
        self._doc_ids = list(doc_ids) if doc_ids else []
        self._doc_metas = list(doc_metas) if doc_metas else [{} for _ in documents]
        self._tokenized_corpus = tokenized
        if tokenized:
            self._index = BM25Okapi(tokenized)
        else:
            self._index = None
        self._stale = False
        logger.debug(
            "BM25 index rebuilt: %d documents, avg_dl=%.1f",
            len(documents),
            self._index.avgdl if self._index else 0,
        )

    def add_documents(
        self,
        new_documents: List[str],
        new_doc_ids: Optional[List[str]] = None,
        new_doc_metas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Incrementally add new documents to the BM25 index without full re-tokenization."""
        if not new_documents:
            return
        new_tokenized = [tokenize(doc) for doc in new_documents]
        self._corpus.extend(new_documents)
        if new_doc_ids:
            self._doc_ids.extend(new_doc_ids)
        else:
            self._doc_ids.extend(["" for _ in new_documents])
        if new_doc_metas:
            self._doc_metas.extend(new_doc_metas)
        else:
            self._doc_metas.extend([{} for _ in new_documents])
        self._tokenized_corpus.extend(new_tokenized)
        self._index = BM25Okapi(self._tokenized_corpus)
        self._stale = False
        logger.debug("BM25 index incrementally updated (+%d docs, total %d)", len(new_documents), len(self._corpus))

    def remove_documents(self, doc_ids: List[str]) -> int:
        """Remove documents by doc_id from the BM25 index."""
        if not doc_ids or not self._doc_ids:
            return 0
        to_remove = set(doc_ids)
        keep_indices = [i for i, cid in enumerate(self._doc_ids) if cid not in to_remove]
        removed_count = len(self._doc_ids) - len(keep_indices)
        if removed_count == 0:
            return 0
        self._corpus = [self._corpus[i] for i in keep_indices]
        self._doc_ids = [self._doc_ids[i] for i in keep_indices]
        self._doc_metas = [self._doc_metas[i] for i in keep_indices]
        self._tokenized_corpus = [self._tokenized_corpus[i] for i in keep_indices]
        if self._tokenized_corpus:
            self._index = BM25Okapi(self._tokenized_corpus)
        else:
            self._index = None
        self._stale = False
        return removed_count

    def save_cache(self, cache_path: str | Path) -> None:
        """Serialize the tokenized BM25 index corpus to disk cache."""
        try:
            import json
            p = Path(cache_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "corpus": self._corpus,
                "doc_ids": self._doc_ids,
                "doc_metas": self._doc_metas,
                "tokenized_corpus": self._tokenized_corpus,
            }
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            logger.debug("BM25 cache saved to %s (%d docs)", p, len(self._corpus))
        except Exception as e:
            logger.debug("BM25 save_cache failed: %s", e)

    def load_cache(self, cache_path: str | Path, expected_count: Optional[int] = None) -> bool:
        """Load cached BM25 index from disk."""
        try:
            import json
            p = Path(cache_path)
            if not p.exists():
                return False
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            corpus = data.get("corpus", [])
            if expected_count is not None and len(corpus) != expected_count:
                logger.debug("BM25 cache count mismatch: cached=%d, expected=%d", len(corpus), expected_count)
                return False
            self._corpus = corpus
            self._doc_ids = data.get("doc_ids", [])
            self._doc_metas = data.get("doc_metas", [])
            self._tokenized_corpus = data.get("tokenized_corpus", [])
            if self._tokenized_corpus:
                self._index = BM25Okapi(self._tokenized_corpus)
            else:
                self._index = None
            self._stale = False
            logger.debug("BM25 cache loaded successfully: %d docs", len(self._corpus))
            return True
        except Exception as e:
            logger.debug("BM25 load_cache failed: %s", e)
            return False

    def mark_stale(self) -> None:
        """Flag the index as stale (e.g. after batch mutation)."""
        self._stale = True

    @property
    def needs_rebuild(self) -> bool:
        return self._stale or self._index is None

    @property
    def doc_count(self) -> int:
        return len(self._corpus)

    # -- Search ----------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search BM25 index and return ranked results.

        Returns: list of {index, score, content, doc_id, group_id, chunk_idx, created_at}
        sorted by BM25 score descending.
        """
        if self.needs_rebuild or self._index is None:
            logger.warning("BM25 index not built — returning empty results")
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)
        # Pair each score with its index, sort descending
        ranked = sorted(
            [(i, scores[i]) for i in range(len(scores)) if scores[i] > 0],
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for i, score in ranked[:top_k]:
            meta = self._doc_metas[i] if i < len(self._doc_metas) else {}
            results.append({
                "index": i,
                "score": score,
                "content": self._corpus[i],
                "doc_id": self._doc_ids[i] if i < len(self._doc_ids) else "",
                "group_id": meta.get("group_id", ""),
                "chunk_idx": meta.get("chunk_idx", 0),
                "total_chunks": meta.get("total_chunks", 1),
                "created_at": meta.get("created_at", ""),
                "strength": meta.get("strength", 50),
                "memory_type": meta.get("memory_type", "general"),
                "tags": meta.get("tags", ""),
            })

        return results


# ---------------------------------------------------------------------------
# RRF Fusion
# ---------------------------------------------------------------------------

def rrf_fuse(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    top_k: int = 5,
    constant: float = RRF_CONSTANT,
) -> List[Dict[str, Any]]:
    """Fuse two ranked lists via Reciprocal Rank Fusion.

    Each result's score = sum(1 / (constant + rank)) across both rankings.
    Results that appear in only one ranking still get a partial score.

    Args:
        vector_results: List from vector search, ordered by relevance desc.
        bm25_results: List from BM25 search, ordered by score desc.
        top_k: How many fused results to return.
        constant: RRF constant (default 60 — standard value).

    Returns:
        Deduplicated list of {content: str, ...metadata..., rrf_score: float}
        sorted by rrf_score descending.
    """
    rrf_scores: Dict[str, Dict[str, Any]] = {}

    # ---- Score vector results ----
    for rank, item in enumerate(vector_results, start=1):
        key = _result_key(item)
        rrf_scores[key] = {
            "rrf_score": 1.0 / (constant + rank),
            "source": "vector",
            # Keep all metadata from the vector result
            **{k: v for k, v in item.items() if k not in ("score", "index")},
        }

    # ---- Score BM25 results and merge ----
    for rank, item in enumerate(bm25_results, start=1):
        key = _result_key(item)
        contribution = 1.0 / (constant + rank)
        if key in rrf_scores:
            rrf_scores[key]["rrf_score"] += contribution
            rrf_scores[key]["source"] = "hybrid"
        else:
            rrf_scores[key] = {
                "rrf_score": contribution,
                "source": "bm25",
                # Preserve all metadata from the BM25 result
                **{k: v for k, v in item.items() if k not in ("score", "index")},
            }

    # ---- Sort by score ----
    fused = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    return fused[:top_k]


def _result_key(item: Dict[str, Any]) -> str:
    """Stable key for deduplication — group_id + chunk_idx or content hash."""
    group_id = item.get("group_id", "")
    chunk_idx = item.get("chunk_idx", None)
    if group_id and chunk_idx is not None:
        return f"{group_id}_{chunk_idx}"
    # Fallback: use content (for BM25 results that lack group_id/chunk_idx)
    return item.get("content", "")


# ---------------------------------------------------------------------------
# Convenience: build BM25 index from all ChromaDB documents
# ---------------------------------------------------------------------------

def build_index_from_collection(collection) -> BM25Index:
    """Read all documents from a ChromaDB collection and build a BM25 index.

    Args:
        collection: A ChromaDB Collection object (must support .get()).

    Returns:
        A populated BM25Index instance with doc IDs and metadata.
    """
    index = BM25Index()
    all_docs = collection.get()
    documents: List[str] = all_docs.get("documents", []) or []
    ids: List[str] = all_docs.get("ids", []) or []
    metadatas: List[Dict[str, Any]] = all_docs.get("metadatas", []) or []

    if documents:
        # Extract group_id, chunk_idx, created_at from metadata for dedup
        doc_metas = [
            {
                "group_id": m.get("group_id", "") if m else "",
                "chunk_idx": m.get("chunk_idx", 0) if m else 0,
                "total_chunks": m.get("total_chunks", 1) if m else 1,
                "created_at": m.get("created_at", "") if m else "",
                "strength": m.get("strength", 50) if m else 50,
                "memory_type": m.get("memory_type", "general") if m else "general",
                "tags": m.get("tags", "") if m else "",
            }
            for m in metadatas
        ]
        index.rebuild(documents, doc_ids=ids, doc_metas=doc_metas)
        logger.info(
            "BM25 index built from collection '%s': %d docs",
            collection.name,
            len(documents),
        )
    else:
        logger.info("Collection is empty — BM25 index is empty")
    return index


# ---------------------------------------------------------------------------
# Weighted random sampling — for recall action
# ---------------------------------------------------------------------------


def weighted_sample(
    results: List[Dict[str, Any]],
    n: int = 3,
    strength_key: str = "strength",
    default_strength: float = 50.0,
) -> List[Dict[str, Any]]:
    """Sample `n` items from `results` without replacement, weighted by strength^2.

    Uses the A-Res algorithm (Efraimidis-Spirakis) for mathematically exact
    weighted random sampling without replacement. High-strength memories appear
    more often, but all candidates have non-zero probability.

    Args:
        results: List of result dicts, each expected to have a numeric strength.
        n: How many items to draw (clamped to len(results)).
        strength_key: Dict key for the strength value.
        default_strength: Fallback when strength is missing/zero.

    Returns:
        `n` items (or fewer if results has fewer).
    """
    if not results:
        return []
    if len(results) <= n:
        return list(results)

    import math
    import random

    scored: List[tuple[float, int]] = []
    for i, r in enumerate(results):
        s = r.get(strength_key, default_strength)
        try:
            w = max(float(s), 1.0) ** 2
        except (TypeError, ValueError):
            w = max(float(default_strength), 1.0) ** 2
        u = random.random()
        u = max(u, 1e-10)
        key = math.log(u) / w
        scored.append((key, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen_indices = [idx for _, idx in scored[:n]]
    return [results[i] for i in chosen_indices]
